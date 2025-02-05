import time
from copy import deepcopy

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from torch_scatter import segment_coo

from model import nerf_ray
from model.extract_geometry import *
from model.utils import *

parent_dir = os.path.dirname(os.path.abspath(__file__))
render_utils_cuda = load(
    name='render_utils_cuda',
    sources=[
        os.path.join(parent_dir, path)
        for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']],
    verbose=True)


class nerf(torch.nn.Module):
    def __init__(self,
                 xyz_min, xyz_max,
                 num_voxels=0, num_voxels_base=0,
                 nearest=False,
                 mask_cache_path=None, mask_cache_thres=1e-5,
                 fast_color_thres=0,
                 k0_dim=12, rgbnet_depth=4, rgbnet_width=256,
                 ref=False, refnet_width=256, refnet_depth=4, sh_max_level=4,
                 posbase_pe=5, viewbase_pe=3, refbase_pe=8,
                 grad_feat=(), sdf_feat=(),
                 k_grad_feat=(1.0,), k_sdf_feat=(),
                 use_grad_norm=True, center_sdf=True,
                 grad_mode='interpolate',
                 s_ratio=2000, s_start=0.05, s_learn=False, step_start=0,
                 smooth_ksize=0, smooth_sigma=1, smooth_scale=True,
                 training=False, stage='', use_viewdir=True,
                 **kwargs):
        super(nerf, self).__init__()
        # self.logger = logger
        self.training = training
        self.stage = stage
        self.ref = ref
        self.use_viewdir = use_viewdir

        if self.stage == 'coarse' or self.stage == 'geometry_searching':
            self.forward = self.forward_coarse
        else:
            self.forward = self.forward_fine

        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.fast_color_thres = fast_color_thres
        self.nearest = nearest
        self.smooth_scale = smooth_scale

        self.s_ratio = s_ratio
        self.s_start = s_start
        self.s_learn = s_learn
        self.step_start = step_start
        self.s_val = nn.Parameter(torch.ones(1), requires_grad=s_learn).cuda()
        self.s_val.data *= s_start
        self.sdf_init_mode = "ball_init"

        # determine based grid resolution
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1 / 3)

        # determine init grid resolution
        self._set_grid_resolution(num_voxels)

        self.sdf = grid.create_grid(
            'DenseGrid', channels=1, world_size=self.world_size,
            xyz_min=self.xyz_min, xyz_max=self.xyz_max)
        if self.sdf_init_mode == "ball_init":
            x, y, z = np.mgrid[-1.0:1.0:self.world_size[0].item() * 1j, -1.0:1.0:self.world_size[1].item() * 1j,
                      -1.0:1.0:self.world_size[2].item() * 1j]
            if stage == 'geometry_searching':
                self.sdf.grid.data = torch.from_numpy((x ** 2 + y ** 2 + z ** 2) ** 0.5).float()[None, None, ...]
            else:
                self.sdf.grid.data = torch.from_numpy((x ** 2 + y ** 2 + z ** 2) ** 0.5 - 1).float()[None, None, ...]
        elif self.sdf_init_mode == "random":
            self.sdf.grid = torch.nn.Parameter(torch.rand([1, 1, *self.world_size]) * 0.05)  # random initialization
            torch.nn.init.normal_(self.sdf, 0.0, 0.5)
        else:
            raise NotImplementedError

        self.init_smooth_conv(smooth_ksize, smooth_sigma)

        # init mlp
        # init k0(feature grid)
        self.k0_dim = k0_dim
        self.k0 = grid.create_grid(
            'DenseGrid', channels=self.k0_dim, world_size=self.world_size,
            xyz_min=self.xyz_min, xyz_max=self.xyz_max)

        self.register_buffer('posfreq', torch.FloatTensor([(2 ** i) for i in range(posbase_pe)]))
        self.register_buffer('viewfreq', torch.FloatTensor([(2 ** i) for i in range(viewbase_pe)]))
        self.register_buffer('reffreq', torch.FloatTensor([(2 ** i) for i in range(refbase_pe)]))

        self.use_grad_norm = use_grad_norm
        self.center_sdf = center_sdf
        self.grad_feat = grad_feat
        self.sdf_feat = sdf_feat
        self.k_grad_feat = k_grad_feat
        self.k_sdf_feat = k_sdf_feat
        rgbnet_dim = (3 + 3 * posbase_pe * 2) + self.k0_dim + 3 + len(self.grad_feat) * 3 + len(self.sdf_feat) * 6
        if self.center_sdf:
            rgbnet_dim += 1
        if self.use_viewdir:
            rgbnet_dim += (3 + 3 * viewbase_pe * 2)

        refnet_dim = (3 + 3 * refbase_pe * 2)
        if stage == 'fine':
            refnet_dim += refnet_width
        else:
            refnet_dim += (self.k0_dim + (3 + 3 * posbase_pe * 2) + 3)
            if self.use_viewdir:
                refnet_dim += (3 + 3 * viewbase_pe * 2)

        self.refnet_width = refnet_width
        self.refnet_depth = refnet_depth
        self.refnet_dim = refnet_dim
        self.refnet = nn.Sequential(
            nn.Linear(refnet_dim, self.refnet_width), nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(self.refnet_width, self.refnet_width), nn.ReLU(inplace=True))
                for _ in range(self.refnet_depth - 2)
            ],
            nn.Linear(self.refnet_width, 3),
        )

        if self.stage == 'fine':
            self.rgbnet = nn.Sequential(
            nn.Linear(rgbnet_dim, rgbnet_width), nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                for _ in range(rgbnet_depth - 2)
            ],
            nn.Linear(rgbnet_width, rgbnet_width),
        )
        else:
            self.rgbnet = None

        self.mlp_kwargs = {
            'rgbnet_dim': rgbnet_dim, 'rgbnet_width': rgbnet_width,
            'rgbnet_depth': rgbnet_depth,
            'refnet_dim': refnet_dim, 'refnet_width': refnet_width,
            'refnet_depth': refnet_depth,
        }
        if self.rgbnet is not None:
            print('rgbnet:', self.rgbnet)
        print('refnet:', self.refnet)
        print('feature voxel grid', self.k0.grid.shape)

        # Using the coarse geometry if provided (used to determine known free space and unknown space)
        if stage == 'geometry_searching':
            self.mask_cache_thres = mask_cache_thres
            self.mask_cache = None
            self.nonempty_mask = None
        else:
            self.mask_cache_path = mask_cache_path
            self.mask_cache_thres = mask_cache_thres
            self.mask_cache = MaskCache(
                path=mask_cache_path,
                mask_cache_thres=mask_cache_thres,
                stage=self.stage).to(self.xyz_min.device)
            self._set_nonempty_mask()
            with torch.no_grad():
                if self.stage == 'coarse':
                    self.sdf.grid[~self.nonempty_mask] = 1

        # grad conv to calculate gradient
        self.init_gradient_conv()
        self.grad_mode = grad_mode

        self.get_rays_of_a_view = nerf_ray.get_rays_of_a_view
        self.integrated_dir_enc = generate_ide_fn(sh_max_level)

    def set_sdf_mask(self):
        sdf = (self.smooth_conv(self.sdf.grid) if self.smooth_sdf else self.sdf.grid)[0, 0, :]
        sdf_mask = (abs(sdf < 0.5) * 1e-3)[None, None, :]
        self.sdf_mask = grid.create_grid('DenseGrid', channels=1, world_size=self.world_size,
                                         xyz_min=self.xyz_min, xyz_max=self.xyz_max)
        self.sdf_mask.grid.data = sdf_mask

        interp = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, sdf_mask.shape[2]),
            torch.linspace(0, 1, sdf_mask.shape[3]),
            torch.linspace(0, 1, sdf_mask.shape[4]),
            indexing='ij'
        ), -1)

        dense_xyz = self.xyz_min * (1 - interp) + self.xyz_max * interp
        mask = (sdf_mask > 0)[0, 0, :]
        active_xyz = dense_xyz[mask]
        xyz_min = active_xyz.amin(0)
        xyz_max = active_xyz.amax(0)
        print(xyz_min, xyz_max)

    def sample_sdf_from_coarse(self, sdf, path):
        st = torch.load(path)
        sdf_mask = st['model_state_dict']['sdf_mask.grid']
        mask = (sdf_mask > 0)[0, 0, :]

        index = torch.stack(torch.meshgrid(
            torch.linspace(1, sdf_mask.shape[2], sdf_mask.shape[2]),
            torch.linspace(1, sdf_mask.shape[3], sdf_mask.shape[3]),
            torch.linspace(1, sdf_mask.shape[4], sdf_mask.shape[4]),
            indexing='ij'
        ), -1)
        sdf_min = index[mask].amin(0)
        sdf_max = index[mask].amax(0)

        sdf0 = sdf[0, 0, :][int(sdf_min[0]):int(sdf_max[0]),
                            int(sdf_min[1]):int(sdf_max[1]),
                            int(sdf_min[2]):int(sdf_max[2])]

        return sdf0[None, None, :]



    def init_gradient_conv(self, sigma=0):
        self.grad_conv = nn.Conv3d(1, 3, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), padding_mode='replicate')
        kernel = np.asarray([
            [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
            [[2, 4, 2], [4, 8, 4], [2, 4, 2]],
            [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
        ])
        # sigma controls the difference between naive [-1,1] and sobel kernel
        distance = np.zeros((3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    distance[i, j, k] = ((i - 1) ** 2 + (j - 1) ** 2 + (k - 1) ** 2 - 1)
        kernel0 = kernel * np.exp(-distance * sigma)

        kernel1 = kernel0 / (kernel0[0].sum() * 2 * self.voxel_size.item())
        weight = torch.from_numpy(np.concatenate([kernel1[None] for _ in range(3)])).float()
        weight[0, 1, :, :] *= 0
        weight[0, 0, :, :] *= -1
        weight[1, :, 1, :] *= 0
        weight[1, :, 0, :] *= -1
        weight[2, :, :, 1] *= 0
        weight[2, :, :, 0] *= -1
        self.grad_conv.weight.data = weight.unsqueeze(1).float()
        self.grad_conv.bias.data = torch.zeros(3)
        for param in self.grad_conv.parameters():
            param.requires_grad = False

        # smooth conv for TV
        self.tv_smooth_conv = nn.Conv3d(1, 1, (3, 3, 3), stride=1, padding=1, padding_mode='replicate')
        weight = torch.from_numpy(kernel0 / kernel0.sum()).float()
        self.tv_smooth_conv.weight.data = weight.unsqueeze(0).unsqueeze(0).float()
        self.tv_smooth_conv.bias.data = torch.zeros(1)
        for param in self.tv_smooth_conv.parameters():
            param.requires_grad = False

    def _gaussian_3dconv(self, ksize=3, sigma=1):
        x = np.arange(-(ksize // 2), ksize // 2 + 1, 1)
        y = np.arange(-(ksize // 2), ksize // 2 + 1, 1)
        z = np.arange(-(ksize // 2), ksize // 2 + 1, 1)
        xx, yy, zz = np.meshgrid(x, y, z)
        kernel = np.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2 * sigma ** 2))
        kernel = torch.from_numpy(kernel).to(self.sdf.grid)
        m = nn.Conv3d(1, 1, ksize, stride=1, padding=ksize // 2, padding_mode='replicate')
        m.weight.data = kernel[None, None, ...] / kernel.sum()
        m.bias.data = torch.zeros(1)
        for param in m.parameters():
            param.requires_grad = False
        return m

    def init_smooth_conv(self, ksize=3, sigma=1):
        self.smooth_sdf = ksize > 0
        if self.smooth_sdf:
            self.smooth_conv = self._gaussian_3dconv(ksize, sigma)
            print("- " * 10 + "init smooth conv with ksize={} and sigma={}".format(ksize, sigma) + " -" * 10)

    def init_sdf_from_sdf(self, sdf0=None, smooth=False, reduce=1., ksize=3, sigma=1., zero2neg=True):
        print("\n", "- " * 3 + "initing sdf from sdf" + " -" * 3, "\n")
        if sdf0.shape != self.sdf.grid.shape:
            sdf0 = F.interpolate(sdf0, size=tuple(self.world_size), mode='trilinear', align_corners=True)
        if smooth:
            m = self._gaussian_3dconv(ksize, sigma)
            sdf_data = m(sdf0 / reduce)
            self.sdf.grid = torch.nn.Parameter(sdf_data).to(self.sdf.grid) / reduce
        else:
            self.sdf.grid.data = sdf0.to(self.sdf.grid) / reduce  # + self.act_shift
        if self.mask_cache is not None:
            self._set_nonempty_mask()
        if self.smooth_scale:
            m = self._gaussian_3dconv(ksize=5, sigma=1)
            with torch.no_grad():
                self.sdf.grid = torch.nn.Parameter(m(self.sdf.grid.data)).cuda()
        self.gradient = self.neus_sdf_gradient()

    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1 / 3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        print('voxel_size      ', self.voxel_size)
        print('world_size      ', self.world_size)
        print('voxel_size_base ', self.voxel_size_base)
        print('voxel_size_ratio', self.voxel_size_ratio)

    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'num_voxels_base': self.num_voxels_base,
            'voxel_size': self.voxel_size,
            'nearest': self.nearest,
            'k0_dim': self.k0_dim,
            'grad_feat': self.grad_feat,
            'sdf_feat': self.sdf_feat,
            'center_sdf': self.center_sdf,
            'fast_color_thres': self.fast_color_thres,
            'stage': self.stage,
            'ref': self.ref,
            'use_viewdir': self.use_viewdir,
            's_ratio': self.s_ratio,
            's_start': self.s_start,
            **self.mlp_kwargs,
        }

    def get_MaskCache_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'voxel_size_ratio': self.voxel_size_ratio,
            'nearest': self.nearest
        }

    @torch.no_grad()
    def _set_nonempty_mask(self):
        # Find grid points that is inside nonempty (occupied) space
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.sdf.grid.shape[2]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.sdf.grid.shape[3]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.sdf.grid.shape[4]),
            indexing='ij'), -1)
        nonempty_mask = self.mask_cache(self_grid_xyz).contiguous()
        nonempty_mask = nonempty_mask.reshape(*self.sdf.grid.shape)
        if hasattr(self, 'nonempty_mask'):
            self.nonempty_mask = nonempty_mask
        else:
            self.register_buffer('nonempty_mask', nonempty_mask)
        if self.stage == 'coarse':
            self.sdf.grid[~nonempty_mask] = 1

    @torch.no_grad()
    def maskout_near_cam_vox(self, cam_o, near):
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.sdf.grid.shape[2]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.sdf.grid.shape[3]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.sdf.grid.shape[4]),
            indexing='ij'), -1)
        nearest_dist = torch.stack([
            (self_grid_xyz.unsqueeze(-2) - co).pow(2).sum(-1).sqrt().amin(-1)
            for co in cam_o.to(self_grid_xyz.device).split(100)  # for memory saving
        ]).amin(0)
        self.sdf.grid[nearest_dist[None, None] <= near] = 5

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        torch.cuda.empty_cache()
        print('scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        print(num_voxels)
        print('scale_volume_grid scale world_size from', ori_world_size, 'to', self.world_size)

        self.sdf.scale_volume_grid(self.world_size)
        self.k0.scale_volume_grid(self.world_size)
        if self.mask_cache is not None:
            self._set_nonempty_mask()
        print('scale_volume_grid finish')

    @torch.no_grad()
    def reset_voxel_and_mlp(self):
        # self.k0 = grid.create_grid(
        #     'DenseGrid', channels=self.k0_dim, world_size=self.world_size,
        #     xyz_min=self.xyz_min, xyz_max=self.xyz_max)

        self.refnet = nn.Sequential(
            nn.Linear(self.refnet_dim, self.refnet_width), nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(self.refnet_width, self.refnet_width), nn.ReLU(inplace=True))
                for _ in range(self.refnet_depth - 2)
            ],
            nn.Linear(self.refnet_width, 3),
        )

    def voxel_count_views(self, rays_o_tr, rays_d_tr, imsz, near, far, stepsize, downrate=1, irregular_shape=False):
        print('dvgo: voxel_count_views start')
        eps_time = time.time()
        N_samples = int(np.linalg.norm(np.array(self.sdf.grid.shape[2:]) + 1) / stepsize) + 1
        rng = torch.arange(N_samples)[None].float().to(rays_o_tr.device)
        count = torch.zeros_like(self.sdf.grid.detach())
        device = rng.device
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = torch.ones_like(self.sdf.grid).requires_grad_()
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

    def density_total_variation(self, sdf_tv=0, smooth_grad_tv=0, sdf_thrd=0.999):
        tv = 0
        if sdf_tv > 0:
            if self.nonempty_mask is not None:
                tv += total_variation(self.sdf.grid, self.nonempty_mask) / 2 / self.voxel_size * sdf_tv
            else:
                tv += total_variation(self.sdf.grid) / 2 / self.voxel_size * sdf_tv
        if smooth_grad_tv > 0:
            smooth_tv_error = (
                    self.tv_smooth_conv(self.gradient.permute(1, 0, 2, 3, 4)).detach() - self.gradient.permute(1, 0, 2,
                                                                                                               3, 4)
            )
            if self.nonempty_mask is not None:
                smooth_tv_error = smooth_tv_error[self.nonempty_mask.repeat(3, 1, 1, 1, 1)] ** 2
            else:
                smooth_tv_error = smooth_tv_error ** 2
            tv += smooth_tv_error.mean() * smooth_grad_tv
        return tv

    def k0_total_variation(self, k0_tv=1., k0_grad_tv=0.):
        v = self.k0.grid
        tv = 0
        if k0_tv > 0:
            if self.nonempty_mask is not None:
                tv += total_variation(v, self.nonempty_mask.repeat(1, v.shape[1], 1, 1, 1))
            else:
                tv += total_variation(v)
        if k0_grad_tv > 0:
            raise NotImplementedError
        return tv

    def k0_total_variation_add_grad(self, weight, dense_mode=True):
        w = weight * self.world_size.max() / 128
        self.k0.total_variation_add_grad(w, w, w, dense_mode)

    def sdf_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        self.sdf.total_variation_add_grad(w, w, w, dense_mode)

    def orientation_loss(self, render_result):
        """Computes the orientation loss regularizer defined in ref-NeRF."""
        zero = torch.tensor(0.0, dtype=torch.float32)
        w = render_result['weights'].detach()
        n = render_result['normal']
        # Negate viewdirs to represent normalized vectors from point to camera.
        v = -render_result['viewdirs']
        n_dot_v = (n * v).sum(dim=-1)
        zeros = torch.zeros_like(n_dot_v)
        return torch.mean((w * torch.fmin(zero, n_dot_v) ** 2).sum(dim=-1))

    def l2_normalize(self, x, eps=torch.finfo(torch.float32).eps):
        """Normalize x to unit length along last axis."""
        eps = torch.tensor(eps, device=x.device)
        return x / torch.sqrt(torch.maximum(torch.sum(x ** 2, dim=-1, keepdims=True), eps))

    def neus_sdf_gradient(self, mode=None, sdf=None):
        if sdf is None:
            sdf = self.sdf.grid
        if mode is None:
            mode = self.grad_mode
        if mode == 'interpolate':
            gradient = torch.zeros([1, 3] + [*self.sdf.grid.shape[-3:]])
            gradient[:, 0, 1:-1, :, :] = (sdf[:, 0, 2:, :, :] - sdf[:, 0, :-2, :, :]) / 2 / self.voxel_size
            gradient[:, 1, :, 1:-1, :] = (sdf[:, 0, :, 2:, :] - sdf[:, 0, :, :-2, :]) / 2 / self.voxel_size
            gradient[:, 2, :, :, 1:-1] = (sdf[:, 0, :, :, 2:] - sdf[:, 0, :, :, :-2]) / 2 / self.voxel_size
        elif mode == 'grad_conv':
            # use sobel operator for gradient seems basically the same as the naive solution
            for param in self.grad_conv.parameters():
                assert not param.requires_grad
                pass
            gradient = self.grad_conv(sdf)
        elif mode == 'raw':
            gradient = torch.zeros([1, 3] + [*self.sdf.grid.shape[-3:]]).to(self.sdf.grid.device)
            gradient[:, 0, :-1, :, :] = (sdf[:, 0, 1:, :, :] - sdf[:, 0, :-1, :, :]) / self.voxel_size
            gradient[:, 1, :, :-1, :] = (sdf[:, 0, :, 1:, :] - sdf[:, 0, :, :-1, :]) / self.voxel_size
            gradient[:, 2, :, :, :-1] = (sdf[:, 0, :, :, 1:] - sdf[:, 0, :, :, :-1]) / self.voxel_size
        else:
            raise NotImplementedError
        return gradient

    def neus_alpha_from_sdf_scatter(self, viewdirs, ray_id, dist, sdf, gradients, global_step,
                                    is_train, use_mid=True):
        if is_train:
            if not self.s_learn:
                s_val = 1. / (global_step + self.s_ratio / self.s_start - self.step_start) * self.s_ratio
                self.s_val.data = torch.ones_like(self.s_val) * s_val
            else:
                s_val = self.s_val.item()
        else:
            s_val = 0

        dirs = viewdirs[ray_id]
        inv_s = torch.ones(1).cuda() / self.s_val
        assert use_mid
        if use_mid:
            true_cos = (dirs * gradients).sum(-1, keepdim=True)
            cos_anneal_ratio = 1.0
            iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                         F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive (M, 1)

            sdf = sdf.unsqueeze(-1)  # (M, 1)

            # Estimate signed distances at section points
            estimated_next_sdf = sdf + iter_cos * dist.reshape(-1, 1) * 0.5  # (M, 1)
            estimated_prev_sdf = sdf - iter_cos * dist.reshape(-1, 1) * 0.5  # (M, 1)
        else:
            estimated_next_sdf = torch.cat([sdf[..., 1:], sdf[..., -1:]], -1).reshape(-1, 1)
            estimated_prev_sdf = torch.cat([sdf[..., :1], sdf[..., :-1]], -1).reshape(-1, 1)

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s.reshape(-1, 1))
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s.reshape(-1, 1))
        p = prev_cdf - next_cdf
        c = prev_cdf
        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0).squeeze()
        return s_val, alpha

    def neus_alpha_from_sdf(self, viewdirs, steps, sdf, gradients, global_step, is_train, use_mid=True):

        ori_shape = viewdirs.shape
        n_samples = steps.shape[-1]
        # force s_val value to change with global step
        if is_train:
            batch_size = steps.shape[0]
            if not self.s_learn:
                s_val = 1. / (global_step + self.s_ratio / self.s_start - self.step_start) * self.s_ratio
                self.s_val.data = torch.ones_like(self.s_val) * s_val
            else:
                s_val = self.s_val.item()
        else:
            dirs = viewdirs.reshape(-1, 3)
            steps = steps.reshape(-1, n_samples)
            batch_size = dirs.shape[0]
            s_val = 0
        if steps.shape[0] == 1:
            steps = steps.repeat(batch_size, 1)
        dirs = viewdirs.unsqueeze(-2)
        inv_s = torch.ones(1).cuda() / self.s_val  # * torch.exp(-inv_s)
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        if use_mid:
            true_cos = (dirs * gradients).sum(-1, keepdim=True)
            cos_anneal_ratio = 1.0
            iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                         F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive
            iter_cos = iter_cos.reshape(-1, 1)

            sdf = sdf.reshape(-1, 1)

            # calculate dist from steps / z_vals
            dists = steps[..., 1:] - steps[..., :-1]
            dists = torch.cat([dists, torch.Tensor([dists.mean()]).expand(dists[..., :1].shape)], -1)

            # Estimate signed distances at section points
            estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
            estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5
        else:
            estimated_next_sdf = torch.cat([sdf[..., 1:], sdf[..., -1:]], -1).reshape(-1, 1)
            estimated_prev_sdf = torch.cat([sdf[..., :1], sdf[..., :-1]], -1).reshape(-1, 1)

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)
        return s_val, alpha

    def sample_sdfs(self, xyz, *grids, displace_list, mode='bilinear', align_corners=True, use_grad_norm=False):

        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)

        grid = grids[0]
        # ind from xyz to zyx !!!!!
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        grid_size = grid.size()[-3:]
        size_factor_zyx = torch.tensor([grid_size[2], grid_size[1], grid_size[0]]).cuda()
        ind = ((ind_norm + 1) / 2) * (size_factor_zyx - 1)
        offset = torch.tensor([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]).cuda()
        displace = torch.tensor(displace_list).cuda()
        offset = offset[:, None, :] * displace[None, :, None]

        all_ind = ind.unsqueeze(-2) + offset.view(-1, 3)
        all_ind = all_ind.view(1, 1, 1, -1, 3)
        all_ind[..., 0] = all_ind[..., 0].clamp(min=0, max=size_factor_zyx[0] - 1)
        all_ind[..., 1] = all_ind[..., 1].clamp(min=0, max=size_factor_zyx[1] - 1)
        all_ind[..., 2] = all_ind[..., 2].clamp(min=0, max=size_factor_zyx[2] - 1)

        all_ind_norm = (all_ind / (size_factor_zyx - 1)) * 2 - 1
        feat = F.grid_sample(grid, all_ind_norm, mode=mode, align_corners=align_corners)

        all_ind = all_ind.view(1, 1, 1, -1, 6, len(displace_list), 3)
        diff = all_ind[:, :, :, :, 1::2, :, :] - all_ind[:, :, :, :, 0::2, :, :]
        diff, _ = diff.max(dim=-1)
        feat_ = feat.view(1, 1, 1, -1, 6, len(displace_list))
        feat_diff = feat_[:, :, :, :, 1::2, :] - feat_[:, :, :, :, 0::2, :]
        grad = feat_diff / diff / self.voxel_size

        feat = feat.view(shape[-1], 6, len(displace_list))
        grad = grad.view(shape[-1], 3, len(displace_list))

        if use_grad_norm:
            grad = grad / (grad.norm(dim=1, keepdim=True) + 1e-5)

        feat = feat.view(shape[-1], 6 * len(displace_list))
        grad = grad.view(shape[-1], 3 * len(displace_list))

        return feat, grad

    def grid_sampler(self, xyz, *grids, mode=None, align_corners=True, sample_ret=True, sample_grad=False, displace=0.1,
                     smooth=False):
        '''Wrapper for the interp operation'''
        if mode is None:
            # bilinear is actually trilinear if 5D input is given to grid_sample
            mode = 'nearest' if self.nearest else 'bilinear'
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)

        if smooth:
            grid = self.smooth_conv(grids[0])
            grids[0] = grid

        outs = []
        if sample_ret:
            ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
            grid = grids[0]
            ret = F.grid_sample(grid, ind_norm, mode=mode, align_corners=align_corners).reshape(
                grid.shape[1], -1).T.reshape(*shape, grid.shape[1]).squeeze(-1)
            outs.append(ret)

        if sample_grad:
            grid = grids[0]
            feat, grad = self.sample_sdfs(xyz, grid, displace_list=[1.0], use_grad_norm=False)
            feat = torch.cat([feat[:, 4:6], feat[:, 2:4], feat[:, 0:2]], dim=-1)
            grad = torch.cat([grad[:, [2]], grad[:, [1]], grad[:, [0]]], dim=-1)

            outs.append(grad)
            outs.append(feat)

        if len(outs) == 1:
            return outs[0]
        else:
            return outs

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
        # correct the cuda output N_steps, which could have a bias of 1 randomly
        N_steps = ray_id.unique(return_counts=True)[1]
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        ray_id = ray_id[mask_inbbox]
        step_id = step_id[mask_inbbox]
        return ray_pts, ray_id, step_id, mask_outbbox, N_steps

    def sample_ray_cuda(self, rays_o, rays_d, near, far, stepsize, maskout=True, use_bg=False, **render_kwargs):
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
        if not use_bg:
            stepdist = stepsize * self.voxel_size
        else:
            stepdist = stepsize * self.voxel_size_bg
        ray_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max = render_utils_cuda.sample_pts_on_rays(
            rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)
        # correct the cuda output N_steps, which could have a bias of 1 randomly
        N_steps = ray_id.unique(return_counts=True)[1]
        if maskout:
            if not use_bg:
                mask_inbbox = ~mask_outbbox
            else:
                mask_inbbox = mask_outbbox
            ray_pts = ray_pts[mask_inbbox]
            ray_id = ray_id[mask_inbbox]
            step_id = step_id[mask_inbbox]

        return ray_pts, ray_id, step_id, mask_outbbox, N_steps

    def sample_ray_ori(self, rays_o, rays_d, near, far, stepsize, is_train=False, **render_kwargs):
        '''Sample query points on rays'''
        # 1. determine the maximum number of query points to cover all possible rays
        N_samples = int(np.linalg.norm(np.array(self.sdf.grid.shape[2:]) + 1) / stepsize) + 1
        # if self.N_samples == 0:
        #     self.N_samples = (N_samples // self.group) * self.group + self.group
        # 2. determine the two end-points of ray bbox intersection
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d).to(rays_d.device)
        rate_a = (self.xyz_max - rays_o) / vec
        rate_b = (self.xyz_min - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
        t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near, max=far)
        # 3. check wheter a raw intersect the bbox or not
        mask_outbbox = (t_max <= t_min)
        # 4. sample points on each ray
        rng = torch.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2], 1)
            rng += torch.rand_like(rng[:, [0]])
        step = stepsize * self.voxel_size * rng
        interpx = (t_min[..., None] + step.to(rays_d.device) / rays_d.norm(dim=-1, keepdim=True))
        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        # 5. update mask for query points outside bbox
        mask_outbbox = mask_outbbox[..., None] | ((self.xyz_min > rays_pts) | (rays_pts > self.xyz_max)).any(dim=-1)
        return rays_pts, mask_outbbox, step

    def bg_k0_total_variation(self, bg_k0_tv=1., bg_k0_grad_tv=0.):
        nonempty_mask = self.sphere_mask if self.nonempty_mask is None else self.nonempty_mask
        if not self.tv_in_sphere:
            nonempty_mask[...] = 1

        if self.rgbnet is not None:
            v = self.bg_k0
        else:
            v = torch.sigmoid(self.bg_k0.grid)
        tv = 0
        if bg_k0_tv > 0:
            tv += total_variation(v, nonempty_mask.repeat(1, v.shape[1], 1, 1, 1))
        if bg_k0_grad_tv > 0:
            raise NotImplementedError
        return tv

    def forward_fine(self, rays_o, rays_d, viewdirs, global_step=20000, **render_kwargs):
        ret_dict = {}
        N = len(rays_o)

        ray_pts, ray_id, step_id, mask_outbbox, N_steps = self.sample_ray(
            rays_o=rays_o, rays_d=rays_d, is_train=global_step is not None, **render_kwargs)

        # skip known free space
        if self.mask_cache is not None:
            mask = self.mask_cache(ray_pts)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            mask_outbbox[~mask_outbbox] |= ~mask

        sdf_grid = self.smooth_conv(self.sdf.grid) if self.smooth_sdf else self.sdf.grid

        sdf, gradient, feat = self.grid_sampler(ray_pts, sdf_grid, sample_ret=True, sample_grad=True, displace=1.0)

        dist = render_kwargs['stepsize'] * self.voxel_size
        s_val, alpha = self.neus_alpha_from_sdf_scatter(viewdirs, ray_id, dist.to('cuda'), sdf, gradient,
                                                        global_step=global_step,
                                                        is_train=global_step is not None, use_mid=True)

        mask = None
        viewdirs_pts = viewdirs[ray_id]
        if self.fast_color_thres > 0:
            mask = (alpha > self.fast_color_thres)
            alpha = alpha[mask]
            ray_id = ray_id[mask]
            viewdirs_pts = viewdirs_pts[mask]
            ray_pts = ray_pts[mask]
            step_id = step_id[mask]
            gradient = gradient[mask]  # merge to sample once
            sdf = sdf[mask]

        # compute accumulated transmittance
        if ray_id.ndim == 2:
            print(mask, alpha, ray_id)
            mask = mask.squeeze()
            alpha = alpha.squeeze()
            ray_id = ray_id.squeeze()
            ray_pts = ray_pts.squeeze()
            step_id = step_id.squeeze()
            gradient = gradient.squeeze()
            sdf = sdf.squeeze()

        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask = (weights > self.fast_color_thres)
            weights = weights[mask]
            alpha = alpha[mask]
            ray_pts = ray_pts[mask]
            viewdirs_pts = viewdirs_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            gradient = gradient[mask]
            sdf = sdf[mask]

        normal = self.l2_normalize(gradient / (gradient.norm(dim=-1, keepdim=True) + 1e-7))

        rays_xyz = (ray_pts - self.xyz_min) / (self.xyz_max - self.xyz_min)
        xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq).flatten(-2)
        xyz_emb = torch.cat([rays_xyz, xyz_emb.sin(), xyz_emb.cos()], -1)

        k0 = self.k0(ray_pts)

        all_grad_inds = list(set(self.grad_feat + self.k_grad_feat))
        all_sdf_inds = list(set(self.sdf_feat + self.k_sdf_feat))

        assert all_grad_inds == all_sdf_inds

        if len(all_grad_inds) > 0:
            all_grad_inds = sorted(all_grad_inds)
            all_grad_inds_ = deepcopy(all_grad_inds)
            all_feat, all_grad = self.sample_sdfs(ray_pts, sdf_grid, displace_list=all_grad_inds_,
                                                  use_grad_norm=self.use_grad_norm)
        else:
            all_feat, all_grad = None, None

        self.gradient = self.neus_sdf_gradient()

        hierarchical_feats = []
        if self.center_sdf:
            hierarchical_feats.append(sdf[:, None])
        if len(all_grad_inds) > 0:
            hierarchical_feats.append(all_feat)
            hierarchical_feats.append(all_grad)

        assert len(self.k_grad_feat) == 1 and self.k_grad_feat[0] == 1.0
        assert len(self.k_sdf_feat) == 0
        all_feats_ = [gradient]
        all_feats_ = torch.cat(all_feats_, dim=-1)

        if self.use_viewdir:
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
            viewdirs_emb = viewdirs_emb.flatten(0, -2)[ray_id]
            rgb_feat = torch.cat([k0, xyz_emb, viewdirs_emb, *hierarchical_feats, all_feats_], dim=-1)
        else:
            rgb_feat = torch.cat([k0, xyz_emb, *hierarchical_feats, all_feats_], dim=-1)
        rgb_feat = self.rgbnet(rgb_feat)

        reflect_r = viewdirs_pts - 2. * torch.sum(viewdirs_pts * normal, dim=-1, keepdim=True) * normal
        reflect_emb = (reflect_r.unsqueeze(-1) * self.reffreq).flatten(-2)
        reflect_emb = torch.cat([reflect_r, reflect_emb.sin(), reflect_emb.cos()], -1)

        ref_feat = torch.cat([rgb_feat, reflect_emb], dim=-1)
        rgb = torch.sigmoid(self.refnet(ref_feat))

        sigmoid_rgb = torch.sigmoid(rgb)

        rgb_marched = segment_coo(
            src=(weights.unsqueeze(-1) * rgb),
            index=ray_id, out=torch.zeros([N, 3]).to(ray_id.device), reduce='sum')
        cum_weights = segment_coo(
            src=(weights.unsqueeze(-1)),
            index=ray_id, out=torch.zeros([N, 1]).to(ray_id.device), reduce='sum')
        sigmoid_rgb = segment_coo(
            src=(weights.unsqueeze(-1) * sigmoid_rgb),
            index=ray_id, out=torch.zeros([N, 3]).to(ray_id.device), reduce='sum')

        # Ray marching
        rgb_marched = rgb_marched + (1 - cum_weights) * render_kwargs['bg']
        rgb_marched = rgb_marched.clamp(0, 1)

        sigmoid_rgb = sigmoid_rgb + (1 - cum_weights) * render_kwargs['bg']
        sigmoid_rgb = sigmoid_rgb.clamp(0, 1)

        if gradient is not None and render_kwargs.get('render_grad', False):
            normal_marched = segment_coo(
                src=(weights.unsqueeze(-1) * normal),
                index=ray_id, out=torch.zeros([N, 3]).to(ray_id.device), reduce='sum')
        else:
            normal_marched = None

        if render_kwargs.get('render_depth', False):
            with torch.no_grad():
                depth = segment_coo(
                    src=(weights * step_id * dist),
                    index=ray_id, out=torch.zeros([N]).to(ray_id.device), reduce='sum')
                disp = 1 / depth
        else:
            depth = None
            disp = None

        ret_dict.update({
            'alphainv_cum': alphainv_last,
            'weights': weights,
            'ray_id': ray_id,
            'viewdirs': viewdirs[ray_id],
            'rgb_marched': rgb_marched,
            'sigmoid_rgb': sigmoid_rgb,
            'normal_marched': normal_marched,
            'normal': normal,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'depth': depth,
            'disp': disp,
            'mask': mask,
            'mask_outbbox': mask_outbbox,
            'gradient': gradient,
            "s_val": s_val
        })

        return ret_dict

    def forward_coarse(self, rays_o, rays_d, viewdirs, global_step=20000, **render_kwargs):
        ret_dict = {}
        N = len(rays_o)

        ray_pts, ray_id, step_id, mask_outbbox, N_steps = self.sample_ray_cuda(
            rays_o=rays_o, rays_d=rays_d, is_train=global_step is not None, **render_kwargs)

        # skip known free space
        viewdirs_pts = viewdirs[ray_id]
        if self.stage == 'coarse':
            if self.mask_cache is not None:
                mask = self.mask_cache(ray_pts)
                ray_pts = ray_pts[mask]
                ray_id = ray_id[mask]
                viewdirs_pts = viewdirs_pts[mask]
                step_id = step_id[mask]
                mask_outbbox[~mask_outbbox] |= ~mask

        # Voxel Inc
        if self.inc_mask is not None:
            mask = self.inc_mask(ray_pts)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            viewdirs_pts = viewdirs_pts[mask]
            step_id = step_id[mask]

        sdf_grid = self.smooth_conv(self.sdf.grid) if self.smooth_sdf else self.sdf.grid

        sdf = self.grid_sampler(ray_pts, sdf_grid)
        self.gradient = self.neus_sdf_gradient(sdf=self.sdf.grid)
        gradient = self.grid_sampler(ray_pts, self.gradient)
        dist = render_kwargs['stepsize'] * self.voxel_size.to(ray_id.device)
        s_val, alpha = self.neus_alpha_from_sdf_scatter(viewdirs, ray_id, dist, sdf, gradient, global_step=global_step,
                                                        is_train=global_step is not None, use_mid=True)

        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)

        mask = None
        if self.fast_color_thres > 0:
            mask = weights > self.fast_color_thres
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            viewdirs_pts = viewdirs_pts[mask]
            step_id = step_id[mask]
            alpha = alpha[mask]
            gradient = gradient[mask]

        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        normal = self.l2_normalize(gradient / (gradient.norm(dim=-1, keepdim=True) + 1e-7))

        rays_xyz = (ray_pts - self.xyz_min) / (self.xyz_max - self.xyz_min)
        xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq).flatten(-2)
        xyz_emb = torch.cat([rays_xyz, xyz_emb.sin(), xyz_emb.cos()], -1)

        k0 = self.k0(ray_pts)
        reflect_r = viewdirs_pts - 2. * torch.sum(viewdirs_pts * normal, dim=-1, keepdim=True) * normal
        reflect_emb = (reflect_r.unsqueeze(-1) * self.reffreq).flatten(-2)
        reflect_emb = torch.cat([reflect_r, reflect_emb.sin(), reflect_emb.cos()], -1)
        if self.use_viewdir:
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
            viewdirs_emb = viewdirs_emb.flatten(0, -2)[ray_id]
            ref_feat = torch.cat([k0, xyz_emb, reflect_emb, normal, viewdirs_emb], dim=-1)
        else:
            ref_feat = torch.cat([k0, xyz_emb, reflect_emb, normal], dim=-1)

        rgb = torch.sigmoid(self.refnet(ref_feat))
        sigmoid_rgb = torch.sigmoid(rgb)

        rgb_marched = segment_coo(
            src=(weights.unsqueeze(-1) * rgb),
            index=ray_id, out=torch.zeros([N, 3]).to(ray_id.device), reduce='sum')
        sigmoid_rgb = segment_coo(
            src=(weights.unsqueeze(-1) * sigmoid_rgb),
            index=ray_id, out=torch.zeros([N, 3]).to(ray_id.device), reduce='sum')
        cum_weights = segment_coo(
            src=(weights.unsqueeze(-1)),
            index=ray_id, out=torch.zeros([N, 1]).to(ray_id.device), reduce='sum')

        # Ray marching
        rgb_marched = rgb_marched + (1 - cum_weights) * render_kwargs['bg']
        rgb_marched = rgb_marched.clamp(0, 1)

        sigmoid_rgb = sigmoid_rgb + (1 - cum_weights) * render_kwargs['bg']
        sigmoid_rgb = sigmoid_rgb.clamp(0, 1)

        if gradient is not None and render_kwargs.get('render_grad', False):
            normal_marched = segment_coo(
                src=(weights.unsqueeze(-1) * normal),
                index=ray_id, out=torch.zeros([N, 3]).to(ray_id.device), reduce='sum')
        else:
            normal_marched = None

        if render_kwargs.get('render_depth', True):
            with torch.no_grad():
                depth = segment_coo(
                    src=(weights * step_id * dist),
                    index=ray_id, out=torch.zeros([N]).to(ray_id.device), reduce='sum')
                disp = 1 / depth
        else:
            depth = None
            disp = None

        ret_dict.update({
            'alphainv_cum': alphainv_last,
            'weights': weights,
            'ray_id': ray_id,
            'viewdirs': viewdirs[ray_id],
            'rgb_marched': rgb_marched,
            'sigmoid_rgb': sigmoid_rgb,
            'normal_marched': normal_marched,
            'normal': normal,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'depth': depth,
            'disp': disp,
            'mask': mask,
            'mask_outbbox': mask_outbbox,
            'gradient': gradient,
            "s_val": s_val
        })
        # with torch.no_grad():
        #     if self.ref:
        #         diffuse_marched = segment_coo(
        #             src=(weights.unsqueeze(-1) * diffuse_rgb),
        #             index=ray_id, out=torch.zeros([N, 3]), reduce='sum')
        #         ret_dict.update({'diffuse_marched': diffuse_marched})
        #
        #         specular_marched = segment_coo(
        #             src=(weights.unsqueeze(-1) * specular_rgb),
        #             index=ray_id, out=torch.zeros([N, 3]), reduce='sum')
        #         ret_dict.update({'specular_marched': specular_marched})
        return ret_dict

    @torch.no_grad()
    def set_inc_mask(self, lower, upper):
        '''set the bounding box
        @lower: [3] The lower bound for which lr is not 0 of each dimension .
        @upper: [3] The upper bound for which lr is not 0 of each dimension .
        '''
        xyz = torch.stack(torch.meshgrid(torch.linspace(0, 1, self.world_size[0]),
                                         torch.linspace(0, 1, self.world_size[1]),
                                         torch.linspace(0, 1, self.world_size[2])))
        mask = (xyz[0] >= lower[0]) & (xyz[0] <= upper[0]) & (xyz[1] >= lower[1]) & (xyz[1] <= upper[1]) & (
                    xyz[2] >= lower[2]) & (xyz[2] <= upper[2])
        self.inc_mask = grid.MaskGrid(path=None, mask=mask, xyz_min=self.xyz_min, xyz_max=self.xyz_max)

    @torch.no_grad()
    def unset_inc_mask(self):
        self.inc_mask = None

    @torch.no_grad()
    def init_cdf_mask(self, thres_mid=1.0, thres_high=0):
        print("start cdf three split")
        importance = self.importance.flatten()
        if thres_mid != 1.0:
            percent_sum = thres_mid
            vals, idx = sorted_importance = torch.sort(importance + (1e-6))
            cumsum_val = torch.cumsum(vals, dim=0)
            split_index = ((cumsum_val / vals.sum()) > (1 - percent_sum)).nonzero().min()
            split_val_nonprune = vals[split_index]
            percent_point = (importance + (1e-6) >= vals[split_index]).sum() / importance.numel()
            print(
                f'{percent_point * 100:.2f}% of most important points contribute over {(percent_sum) * 100:.2f}% importance ')
            self.non_prune_mask = importance > split_val_nonprune
        else:
            self.non_prune_mask = torch.ones_like(importance).bool()

        if thres_high != 0:
            percent_sum = thres_high
            vals, idx = sorted_importance = torch.sort(importance + (1e-6))
            cumsum_val = torch.cumsum(vals, dim=0)
            split_index = ((cumsum_val / vals.sum()) > (1 - percent_sum)).nonzero().min()
            split_val_reinclude = vals[split_index]
            percent_point = (importance + (1e-6) >= vals[split_index]).sum() / importance.numel()
            print(
                f'{percent_point * 100:.2f}% of most important points contribute over {(percent_sum) * 100:.2f}% importance ')
            self.keep_mask = importance > split_val_reinclude
        else:
            self.keep_mask = torch.zeros_like(importance).bool()
            self.keep_mask[-1] = True  # for code robustness issue

        return self.non_prune_mask, self.keep_mask

    def mesh_color_forward(self, ray_pts, **kwargs):

        ### coarse-stage geometry and texture are low in resolution

        sdf_grid = self.smooth_conv(self.sdf.grid) if self.smooth_sdf else self.sdf.grid
        self.gradient = self.neus_sdf_gradient(sdf=sdf_grid)
        gradient = self.grid_sampler(ray_pts, self.gradient).reshape(-1, 3)
        normal = gradient / (gradient.norm(dim=-1, keepdim=True) + 1e-5)
        viewdirs = -normal

        rgb_feat = []
        k0 = self.k0(ray_pts)
        rgb_feat.append(k0)

        rays_xyz = (ray_pts - self.xyz_min) / (self.xyz_max - self.xyz_min)
        xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq).flatten(-2)
        xyz_emb = torch.cat([rays_xyz, xyz_emb.sin(), xyz_emb.cos()], -1)
        rgb_feat.append(xyz_emb)

        viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
        viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
        rgb_feat.append(viewdirs_emb.flatten(0, -2))
        rgb_feat = torch.cat(rgb_feat, -1)
        if self.geo_rgb_dim == 3:
            rgb_feat = torch.cat([rgb_feat, normal], -1)
        rgb_logit = self.rgbnet(rgb_feat)
        rgb = torch.sigmoid(rgb_logit)

        return rgb

    def extract_geometry(self, bound_min, bound_max, resolution=128, threshold=0.0, **kwargs):
        if self.smooth_sdf:
            sdf_grid = self.smooth_conv(self.sdf.grid)
        else:
            sdf_grid = self.sdf.grid
        # self._set_nonempty_mask()
        query_func = lambda pts: self.grid_sampler(pts, -sdf_grid)
        if resolution is None:
            resolution = self.world_size[0]
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=query_func)


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


class MaskCache(nn.Module):
    def __init__(self, path, mask_cache_thres, stage, ks=3):
        super().__init__()
        st = torch.load(path)
        self.mask_cache_thres = mask_cache_thres
        self.register_buffer('xyz_min', torch.FloatTensor(st['MaskCache_kwargs']['xyz_min']))
        self.register_buffer('xyz_max', torch.FloatTensor(st['MaskCache_kwargs']['xyz_max']))
        self.register_buffer('sdf_mask', F.max_pool3d(st['model_state_dict']['sdf_mask.grid'],
                                                      kernel_size=ks, padding=ks // 2, stride=1))

    @torch.no_grad()
    def forward(self, xyz):
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3).to(self.xyz_max.device)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        sdf_mask = F.grid_sample(self.sdf_mask, ind_norm, align_corners=True)
        sdf_mask = sdf_mask.reshape(*shape)
        return sdf_mask >= self.mask_cache_thres


def total_variation(v, mask=None):
    tv2 = v.diff(dim=2).abs()
    tv3 = v.diff(dim=3).abs()
    tv4 = v.diff(dim=4).abs()
    if mask is not None:
        tv2 = tv2[mask[:, :, :-1] & mask[:, :, 1:]]
        tv3 = tv3[mask[:, :, :, :-1] & mask[:, :, :, 1:]]
        tv4 = tv4[mask[:, :, :, :, :-1] & mask[:, :, :, :, 1:]]
        return (tv2.sum() + tv3.sum() + tv4.sum()) / 3 / mask.sum()
    return (tv2.sum() + tv3.sum() + tv4.sum()) / 3 / v.sum()
