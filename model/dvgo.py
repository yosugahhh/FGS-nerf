import time, os, cv2
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import segment_coo
from model.extract_geometry import extract_geometry
from model import dvgo_ray
from model import grid

from torch.utils.cpp_extension import load

parent_dir = os.path.dirname(os.path.abspath(__file__))
render_utils_cuda = load(
    name='render_utils_cuda',
    sources=[
        os.path.join(parent_dir, path)
        for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']],
    verbose=True)


class dvgo(torch.nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, num_voxels_base=0,
                 alpha_init=None,
                 nearest=False,
                 mask_cache_path=None, mask_cache_thres=1e-3,
                 fast_color_thres=0,
                 rgbnet_width=128, num_space=1,
                 ref=False,
                 **kwargs):
        super(dvgo, self).__init__()

        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.fast_color_thres = fast_color_thres
        self.nearest = nearest
        # grid size and numbers
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1 / 3)

        # density bias shift
        self.alpha_init = alpha_init
        self.act_shift = np.log(1 / (1 - alpha_init) - 1)

        self.ref = True

        # init grid resolution
        self._set_grid_resolution(num_voxels)

        # init density voxel grid
        #self.density = torch.nn.Parameter(torch.zeros([1, 1, *self.world_size]))
        self.density = grid.create_grid('DenseGrid', channels=1, world_size=self.world_size,
                                        xyz_min=self.xyz_min, xyz_max=self.xyz_max)
        # density init
        x, y, z = np.mgrid[-1.0:1.0:self.world_size[0].item() * 1j, -1.0:1.0:self.world_size[1].item() * 1j,
                  -1.0:1.0:self.world_size[2].item() * 1j]
        self.density.grid.data = torch.from_numpy((x ** 2 + y ** 2 + z ** 2) ** 0.5 - 1).float()[None, None, ...]

        self.k0_dim = 3
        #self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]))
        self.k0 = grid.create_grid('DenseGrid', channels=self.k0_dim, world_size=self.world_size,
                                   xyz_min=self.xyz_min, xyz_max=self.xyz_max)

        # self.space = space(num_space)

        # Using the coarse geometry if provided (used to determine konwn free space and unknown space)
        self.mask_cache_path = mask_cache_path
        self.mask_cache_thres = mask_cache_thres
        if mask_cache_path is not None and mask_cache_thres:
            self.mask_cache = MaskCache(
                path=mask_cache_path, mask_cache_thres=mask_cache_thres
            ).to(self.xyz_min.device)
            self._set_nonempty_mask()
        else:
            self.mask_cache = None
            self.nonempty_mask = None

        self.get_rays_of_a_view = dvgo_ray.get_rays_of_a_view

    def create_mask_cache(self, mask_cache_path):
        if not self.mask_cache:
            self.mask_cache = MaskCache(path=mask_cache_path,
                                        mask_cache_thres=self.mask_cache_thres).to(self.xyz_min.device)
            self._set_nonempty_mask()

    def inside_sphere(self):
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.density.shape[2]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.density.shape[3]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.density.shape[4]),
        ), -1)
        sphere_mask = (torch.linalg.norm(self_grid_xyz, ord=2, dim=-1, keepdim=True) < 1.0).reshape(*self.density.shape)
        self.density[~sphere_mask] = -100

    def _set_grid_resolution(self, num_voxels):
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1 / 3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        print('dvgo: voxel_size      ', self.voxel_size)
        print('dvgo: world_size      ', self.world_size)
        print('dvgo: voxel_size_base ', self.voxel_size_base)
        print('dvgo: voxel_size_ratio', self.voxel_size_ratio)

    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'num_voxels_base': self.num_voxels_base,
            'alpha_init': self.alpha_init,
            'nearest': self.nearest,
            'mask_cache_path': self.mask_cache_path,
            'mask_cache_thres': self.mask_cache_thres,
            'fast_color_thres': self.fast_color_thres,
            'act_shift': self.act_shift,
            'voxel_size_ratio': self.voxel_size_ratio,
        }

    def get_MaskCache_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'act_shift': self.act_shift,
            'voxel_size_ratio': self.voxel_size_ratio,
            'nearest': self.nearest,
        }

    @torch.no_grad()
    def _set_nonempty_mask(self):
        # Find grid points that is inside nonempty (occupied) space
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.density.shape[2]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.density.shape[3]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.density.shape[4]),
        ), -1)
        nonempty_mask = self.mask_cache(self_grid_xyz)[None, None].contiguous()
        if hasattr(self, 'nonempty_mask'):
            self.nonempty_mask = nonempty_mask
        else:
            self.register_buffer('nonempty_mask', nonempty_mask)
        self.density[~self.nonempty_mask] = -100

    @torch.no_grad()
    def maskout_near_cam_vox(self, cam_o, near):
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.density.shape[2]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.density.shape[3]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.density.shape[4]),
        ), -1)
        nearest_dist = torch.stack([
            (self_grid_xyz.unsqueeze(-2) - co).pow(2).sum(-1).sqrt().amin(-1)
            for co in cam_o.split(100)  # for memory saving
        ]).amin(0)
        self.density[nearest_dist[None, None] <= near] = -100

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        print('dvgo: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        print('dvgo: scale_volume_grid scale world_size from', ori_world_size, 'to', self.world_size)

        self.density.scale_volume_grid(self.world_size)
        self.k0.scale_volume_grid(self.world_size)
        if self.mask_cache is not None:
            self._set_nonempty_mask()
        print('dvgo: scale_volume_grid finish')

    def voxel_count_views(self, rays_o_tr, rays_d_tr, imsz, near, far, stepsize, downrate=1, irregular_shape=False):
        print('dvgo: voxel_count_views start')
        eps_time = time.time()
        N_samples = int(np.linalg.norm(np.array(self.density.grid.shape[2:]) + 1) / stepsize) + 1
        rng = torch.arange(N_samples)[None].float().to(rays_o_tr.device)
        count = torch.zeros_like(self.density.grid.detach())
        device = rng.device
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = torch.ones_like(self.density.grid).requires_grad_()
            if irregular_shape:
                rays_o_ = rays_o_.split(10000)
                rays_d_ = rays_d_.split(10000)
            else:
                rays_o_ = rays_o_[::downrate, ::downrate].to(device).flatten(0, -2).split(10000)
                rays_d_ = rays_d_[::downrate, ::downrate].to(device).flatten(0, -2).split(10000)

            for rays_o, rays_d in zip(rays_o_, rays_d_):
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.xyz_max - rays_o) / vec
                rate_b = (self.xyz_min - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near, max=far)
                step = stepsize * self.voxel_size * rng
                interpx = (t_min[..., None] + step / rays_d.norm(dim=-1, keepdim=True))
                rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
                self.grid_sampler(rays_pts, ones).sum().backward()
            with torch.no_grad():
                count += (ones.grad > 1)
        eps_time = time.time() - eps_time
        print('dvgo: voxel_count_views finish (eps time:', eps_time, 'sec)')
        return count

    def density_total_variation(self):
        tv = total_variation(self.activate_density(self.density.grid, 1), self.nonempty_mask)
        return tv

    def k0_total_variation(self):
        if self.rgbnet is not None:
            v = self.k0
        else:
            v = torch.sigmoid(self.k0)
        return total_variation(v, self.nonempty_mask)

    def density_total_variation_add_grad(self, weight, dense_mode=True):
        w = weight * self.world_size.max() / 128
        self.density.total_variation_add_grad(w, w, w, dense_mode)

    def k0_total_variation_add_grad(self, weight, dense_mode=True):
        w = weight * self.world_size.max() / 128
        self.k0.total_variation_add_grad(w, w, w, dense_mode)

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        return 1 - torch.exp(-F.softplus(density + self.act_shift) * interval)

    def grid_sampler(self, xyz, *grids, mode=None, align_corners=True):
        '''Wrapper for the interp operation'''
        if mode is None:
            # bilinear is actually trilinear if 5D input is given to grid_sample
            mode = 'nearest' if self.nearest else 'bilinear'
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        ret_lst = [
            # TODO: use `rearrange' to make it readable
            F.grid_sample(grid, ind_norm, mode=mode, align_corners=align_corners).reshape(grid.shape[1], -1).T.reshape(
                *shape, grid.shape[1]).squeeze()
            for grid in grids
        ]
        if len(ret_lst) == 1:
            return ret_lst[0]
        return ret_lst

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max = render_utils_cuda.sample_pts_on_rays(
            rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        ray_id = ray_id[mask_inbbox]
        step_id = step_id[mask_inbbox]
        return ray_pts, ray_id, step_id

    def gradient(self, density=None):
        gradient = torch.zeros([1, 3] + [*self.density.grid.shape[-3:]])
        gradient[:, 0, 1:-1, :, :] = (density[:, 0, 2:, :, :] - density[:, 0, :-2, :, :]) / 2 / self.voxel_size
        gradient[:, 1, :, 1:-1, :] = (density[:, 0, :, 2:, :] - density[:, 0, :, :-2, :]) / 2 / self.voxel_size
        gradient[:, 2, :, :, 1:-1] = (density[:, 0, :, :, 2:] - density[:, 0, :, :, :-2]) / 2 / self.voxel_size

        return gradient

    def l2_normalize(self, x, eps=torch.finfo(torch.float32).eps):
        """Normalize x to unit length along last axis."""
        eps = torch.tensor(eps, device=x.device)
        return x / torch.sqrt(torch.maximum(torch.sum(x ** 2, dim=-1, keepdims=True), eps))

    def forward(self, rays_o, rays_d, viewdirs, global_step=None, **render_kwargs):
        '''Volume rendering'''

        ret_dict = {}
        N = len(rays_o)

        # sample points on rays
        ray_pts, ray_id, step_id = self.sample_ray(
            rays_o=rays_o, rays_d=rays_d, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio

        # skip known free space
        viewdirs_pts = viewdirs[ray_id]
        if self.mask_cache is not None:
            mask = self.mask_cache(ray_pts)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            viewdirs_pts = viewdirs_pts[mask]

        # query for alpha w/ post-activation
        density = self.density(ray_pts)
        alpha = self.activate_density(density, interval)

        if self.fast_color_thres > 0:
            mask = (alpha > self.fast_color_thres)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            alpha = alpha[mask]

        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask = (weights > self.fast_color_thres)
            weights = weights[mask]
            alpha = alpha[mask]
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]

        # query for color
        k0 = self.k0(ray_pts)
        rgb = torch.sigmoid(k0)

        gradient = self.grid_sampler(ray_pts, self.gradient(self.density.grid))
        normals = gradient / (gradient.norm(dim=-1, keepdim=True) + 1e-7)

        # Ray marching
        rgb_marched = segment_coo(
            src=(weights.unsqueeze(-1) * rgb),
            index=ray_id,
            out=torch.zeros([N, 3]),
            reduce='sum')
        rgb_marched += (alphainv_last.unsqueeze(-1) * render_kwargs['bg'])


        normals = segment_coo(
            src=(weights.unsqueeze(-1) * normals),
            index=ray_id, out=torch.zeros([N, 3]).to(ray_id.device), reduce='sum')

        ret_dict.update({
            'alphainv_cum': alphainv_last,
            # 'alphainv_cum': alphainv_cum,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'normal_marched': normals,
            # 'depth': depth,
            # 'disp': disp,
            'ray_id': ray_id,
            'mask': mask,
        })
        return ret_dict


class MaskCache(nn.Module):
    def __init__(self, path, mask_cache_thres, ks=3):
        super().__init__()
        st = torch.load(path)
        self.mask_cache_thres = mask_cache_thres
        self.register_buffer('xyz_min', torch.FloatTensor(st['MaskCache_kwargs']['xyz_min']))
        self.register_buffer('xyz_max', torch.FloatTensor(st['MaskCache_kwargs']['xyz_max']))
        self.register_buffer('density', F.max_pool3d(
            st['model_state_dict']['density.grid'], kernel_size=ks, padding=ks // 2, stride=1))
        self.act_shift = st['MaskCache_kwargs']['act_shift']
        self.voxel_size_ratio = st['MaskCache_kwargs']['voxel_size_ratio']
        self.nearest = st['MaskCache_kwargs'].get('nearest', False)
        self.pre_act_density = st['MaskCache_kwargs'].get('pre_act_density', False)
        self.in_act_density = st['MaskCache_kwargs'].get('in_act_density', False)

    @torch.no_grad()
    def forward(self, xyz):
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3).to(self.xyz_max.device)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        if self.nearest:
            density = F.grid_sample(self.density, ind_norm, align_corners=True, mode='nearest')
            alpha = 1 - torch.exp(-F.softplus(density + self.act_shift) * self.voxel_size_ratio.to(density.device))
        else:
            density = F.grid_sample(self.density, ind_norm, align_corners=True)
            alpha = 1 - torch.exp(-F.softplus(density + self.act_shift) * self.voxel_size_ratio.to(density.device))
        alpha = alpha.reshape(*shape)
        return (alpha >= self.mask_cache_thres)


class Alphas2Weights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, ray_id, N):
        weights, T, alphainv_last, i_start, i_end = render_utils_cuda.alpha2weight(alpha, ray_id, N)
        if alpha.requires_grad:
            ctx.save_for_backward(alpha, weights, T, alphainv_last, i_start, i_end)
            ctx.n_rays = N
        return weights, alphainv_last

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_weights, grad_last):
        alpha, weights, T, alphainv_last, i_start, i_end = ctx.saved_tensors
        grad = render_utils_cuda.alpha2weight_backward(
            alpha, weights, T, alphainv_last,
            i_start, i_end, ctx.n_rays, grad_weights, grad_last)
        return grad, None, None


def cumprod_exclusive(p):
    # Not sure why: it will be slow at the end of training if clamping at 1e-10 is not applied
    return torch.cat([torch.ones_like(p[..., [0]]), p.clamp_min(1e-10).cumprod(-1)], -1)


def get_ray_marching_ray(alpha):
    alphainv_cum = cumprod_exclusive(1 - alpha)
    weights = alpha * alphainv_cum[..., :-1]
    return weights, alphainv_cum


def total_variation(v, mask=None):
    tv2 = v.diff(dim=2).abs()
    tv3 = v.diff(dim=3).abs()
    tv4 = v.diff(dim=4).abs()
    if mask is not None:
        tv2 = tv2[mask[:, :, :-1] & mask[:, :, 1:]]
        tv3 = tv3[mask[:, :, :, :-1] & mask[:, :, :, 1:]]
        tv4 = tv4[mask[:, :, :, :, :-1] & mask[:, :, :, :, 1:]]
    return (tv2.mean() + tv3.mean() + tv4.mean()) / 3
