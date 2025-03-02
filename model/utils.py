import logging
import random
import os, tqdm, imageio, shutil

import numpy as np
import torch
import trimesh

from model import grid
from model.evaluation import *
from model.dtu_eval import eval


def get_root_logger(log_level=logging.INFO, handlers=()):
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=log_level)
    for handler in handlers:
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


def load_grid_data(model, ckpt_path, deduce=1, name='density', return_raw=False):
    ckpt = torch.load(ckpt_path)
    module = getattr(model, name)
    print(">>> {} loaded from ".format(name), ckpt_path)
    if name not in ckpt['model_state_dict']:
        name = name + '.grid'
    if return_raw:
        return ckpt['model_state_dict'][name]
    else:
        if isinstance(module, grid.DenseGrid):
            module.grid.data = ckpt['model_state_dict'][name]
        else:
            module.data = ckpt['model_state_dict'][name]
        return model


def load_checkpoint(model, optimizer, ckpt_path, no_reload_optimizer, stage='coarse', num_voxels=0, strict=True):
    ckpt = torch.load(ckpt_path)
    start = ckpt['global_step']
    if stage == 'fine':
        del ckpt['model_state_dict']['mask_cache.density']
        model.load_state_dict(ckpt['model_state_dict'], strict=strict)
        model.scale_volume_grid(num_voxels)
    else:
        model.load_state_dict(ckpt['model_state_dict'], strict=strict)
    if not no_reload_optimizer:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except:
            print("Failed to load optimizer state dict")
            if strict:
                raise ValueError
            else:
                print("Skip!")
    return model, optimizer, start


def load_model(model_class, ckpt_path, new_kwargs=None, strict=False):
    ckpt = torch.load(ckpt_path)
    if new_kwargs is not None:
        for k, v in new_kwargs.items():
            if k in ckpt['model_kwargs']:
                if ckpt['model_kwargs'][k] != v:
                    print('updating {} from {} to {}'.format(k, ckpt['model_kwargs'][k], v))
        ckpt['model_kwargs'].update(new_kwargs)
    global_step = ckpt['global_step']
    mask_cache_path = os.path.join(ckpt_path[0:-14], 'geometry_searching_last.tar')
    ckpt['model_kwargs']['mask_cache_path'] = mask_cache_path
    model = model_class(**ckpt['model_kwargs'])
    try:
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        print(">>> Checkpoint loaded successfully from {}".format(ckpt_path))
    except Exception as e:
        print(e)
        if strict:
            print(">>> Failed to load checkpoint correctly.")
            model.load_state_dict(ckpt['model_state_dict'], strict=True)
        else:
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
            print(">>> Checkpoint loaded without strict matching from {}".format(ckpt_path))
    return model, global_step


def load_weight_by_name(model, ckpt_path, deduce=1, name='density', return_raw=False):
    ckpt = torch.load(ckpt_path)
    for n, module in model.named_parameters():
        if name in n:
            if n in ckpt['model_state_dict']:
                module.data = ckpt['model_state_dict'][n]
                print('load {} to model'.format(n))
    print(">>> data with name {} are loaded from ".format(name), ckpt_path)
    return model


def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    seed = 777
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def makeMLP(in_chan, out_chan, act=torch.nn.ReLU(inplace=True), batch_norm=False):
    modules = [torch.nn.Linear(in_chan, out_chan)]
    if batch_norm == True:
        modules.append(torch.nn.BatchNorm1d(out_chan))
    if not act is None:
        modules.append(act)
    return modules


def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma) ** 2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[..., i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0 ** 2) - mu00
    sigma11 = filt_fn(img1 ** 2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


def generalized_binomial_coeff(a, k):
    """Compute generalized binomial coefficients."""
    return np.prod(a - np.arange(k)) / np.math.factorial(k)


def assoc_legendre_coeff(l, m, k):
    """Compute associated Legendre polynomial coefficients.

    Returns the coefficient of the cos^k(theta)*sin^m(theta) term in the
    (l, m)th associated Legendre polynomial, P_l^m(cos(theta)).

    Args:
      l: associated Legendre polynomial degree.
      m: associated Legendre polynomial order.
      k: power of cos(theta).

    Returns:
      A float, the coefficient of the term corresponding to the inputs.
    """
    return ((-1) ** m * 2 ** l * np.math.factorial(l) / np.math.factorial(k) /
            np.math.factorial(l - k - m) *
            generalized_binomial_coeff(0.5 * (l + k + m - 1.0), l))


def sph_harm_coeff(l, m, k):
    """Compute spherical harmonic coefficients."""
    return (np.sqrt(
        (2.0 * l + 1.0) * np.math.factorial(l - m) /
        (4.0 * np.pi * np.math.factorial(l + m))) * assoc_legendre_coeff(l, m, k))


def get_ml_array(deg_view):
    """Create a list with all pairs of (l, m) values to use in the encoding."""
    ml_list = []
    for i in range(deg_view):
        l = 2 ** i
        # Only use nonnegative m values, later splitting real and imaginary parts.
        for m in range(l + 1):
            ml_list.append((m, l))

    # Convert list into a numpy array.
    ml_array = np.array(ml_list).T
    return ml_array


def validate_image(args, cfg, stage, step, data_dict, render_viewpoints_kwargs, eval_all=True):
    testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{stage}')
    logger = render_viewpoints_kwargs['logger']
    os.makedirs(testsavedir, exist_ok=True)
    eval_lpips_alex = args.eval_lpips_alex and eval_all
    eval_lpips_vgg = args.eval_lpips_alex and eval_all
    if stage == 'eval':
        logger.info(f"validating test set idx: {data_dict['i_test']}")
        rgbs, disps, extras = render_viewpoints(
            cfg=cfg,
            render_poses=data_dict['poses'][data_dict['i_test']],
            HW=data_dict['HW'][data_dict['i_test']],
            Ks=data_dict['Ks'][data_dict['i_test']],
            gt_imgs=data_dict['images'][data_dict['i_test']].cpu().numpy(),
            masks=data_dict['masks'][data_dict['i_test']].cpu().numpy(),
            savedir=testsavedir,
            eval_ssim=args.eval_ssim, eval_lpips_alex=eval_lpips_alex, eval_lpips_vgg=eval_lpips_vgg, idx=data_dict['i_test'].tolist(),
            step=step,
            **render_viewpoints_kwargs)
    else:
        rand_idx = random.randint(0, len(data_dict['poses'][data_dict['i_test']]) - 1)
        # rand_idx = 0
        logger.info(f"validating test set idx: {rand_idx}")
        rgbs, disps, extras = render_viewpoints(
            cfg=cfg,
            render_poses=data_dict['poses'][data_dict['i_test']][rand_idx][None],
            HW=data_dict['HW'][data_dict['i_test']][rand_idx][None],
            Ks=data_dict['Ks'][data_dict['i_test']][rand_idx][None],
            gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']][rand_idx][None],
            masks=[data_dict['masks'][i].cpu().numpy() for i in data_dict['i_test']][rand_idx][None],
            savedir=testsavedir,
            eval_ssim=args.eval_ssim, eval_lpips_alex=eval_lpips_alex, eval_lpips_vgg=eval_lpips_vgg, idx=rand_idx,
            step=step,
            **render_viewpoints_kwargs)


@torch.no_grad()
@torch.no_grad()
def render_viewpoints(cfg, model, render_poses, HW, Ks, ndc, render_kwargs, logger,
                      gt_imgs=None, masks=None, savedir=None, render_factor=0, idx=None,
                      eval_ssim=True, eval_lpips_alex=True, eval_lpips_vgg=True,
                      use_bar=True, step=0, rgb_only=False):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if render_factor != 0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW //= render_factor
        Ks[:, :2, :3] //= render_factor

    rgbs = []
    ins = []
    outs = []
    disps = []
    psnrs = []
    fore_psnrs = []
    bg_psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []
    bgmaps = []
    extras = {}
    idxs = 0
    split_bg = getattr(model, "bg_density", False)
    for i, c2w in enumerate(tqdm.tqdm(render_poses)):
        H, W = HW[i]
        K = Ks[i]
        rays_o, rays_d, viewdirs = model.get_rays_of_a_view(
            H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
            flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'disp', 'depth', 'alphainv_cum']
        if split_bg:
            keys.extend(['in_marched', 'out_marched'])
        if model.ref:
            keys.extend(['diffuse_marched', 'specular_marched', 'tint_marched',
                         'normal_marched', 'roughness_marched'])
        rays_o = rays_o.flatten(0, -2).to(device)
        rays_d = rays_d.flatten(0, -2).to(device)
        viewdirs = viewdirs.flatten(0, -2).to(device)
        try:
            render_result_chunks = [
                {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
                for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
            ]
        except RuntimeError as e:
            print(e)
            idxs += 1
            continue
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H, W, -1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        rgbs.append(rgb)
        if rgb_only and savedir is not None:
            imageio.imwrite(os.path.join(savedir, '{:03d}.png'.format(i)), to8b(rgb))
            continue

        # disp = render_result['disp'].cpu().numpy()
        # disps.append(disp)

        bgmap = render_result['alphainv_cum'].cpu().numpy()
        bgmaps.append(bgmap)

        if 'normal_marched' in render_result:
            if not 'normal' in extras:
                extras['normal'] = []
            normals = render_result['normal_marched'].cpu().numpy()
            extras['normal'].append(normals)
        if 'diffuse_marched' in render_result:
            if not 'diffuse' in extras:
                extras['diffuse'] = []
            diffuse = render_result['diffuse_marched'].cpu().numpy()
            extras['diffuse'].append(diffuse)
        if 'specular_marched' in render_result:
            if not 'specular' in extras:
                extras['specular'] = []
            specular = render_result['specular_marched'].cpu().numpy()
            extras['specular'].append(specular)
        if 'tint_marched' in render_result:
            tint = render_result['tint_marched'].cpu().numpy()
            if not 'tint' in extras:
                extras['tint'] = []
            extras['tint'].append(tint)
        if 'roughness_marched' in render_result:
            if not 'roughness' in extras:
                extras['roughness'] = []
            roughness = render_result['roughness_marched'].cpu().numpy()
            extras['roughness'].append(roughness)

        if split_bg:
            inside = render_result['in_marched'].cpu().numpy()
            ins.append(inside)
            outside = render_result['out_marched'].cpu().numpy()
            outs.append(outside)

        if masks is not None:
            if isinstance(masks[i], torch.Tensor):
                mask = masks[i].cpu().numpy()  # .reshape(H, W, 1)
            else:
                mask = masks[i]  # .reshape(H, W, 1)
            if mask.ndim == 2:
                mask = mask.reshape(H, W, 1)
            bg_rgb = rgb * (1 - mask)
            bg_gt = gt_imgs[i] * (1 - mask)
        else:
            mask, bg_rgb, bg_gt = np.ones(rgb.shape[:2]), np.ones(rgb.shape), np.ones(rgb.shape)

        # if i == 0:
        #     logger.info('Testing {} {}'.format(rgb.shape, disp.shape))
        if gt_imgs is not None and render_factor == 0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            back_p, fore_p = 0., 0.
            if masks is not None:
                back_p = -10. * np.log10(np.sum(np.square(bg_rgb - bg_gt)) / np.sum(1 - mask))
                fore_p = -10. * np.log10(np.sum(np.square(rgb - gt_imgs[i])) / np.sum(mask))
            error = 1 - np.exp(-20 * np.square(rgb - gt_imgs[i]).sum(-1))[..., None].repeat(3, -1)

            logger.info("{} | full-image psnr {:.2f} | foreground psnr {:.2f} | background psnr: {:.2f} ".format(i, p, fore_p,
                                                                                                           back_p))
            psnrs.append(p)
            fore_psnrs.append(fore_p)
            bg_psnrs.append(back_p)
            if eval_ssim:
                ssims.append(rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(rgb_lpips(rgb, gt_imgs[i], net_name='alex', device='cpu'))
            if eval_lpips_vgg:
                lpips_vgg.append(rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device='cpu'))

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            if render_poses.shape[0] > 1:
                id = idx[idxs]
                idxs += 1
            else:
                id = idx if idx is not None else i
            step_pre = str(step) + '_' if step > 0 else ''
            filename = os.path.join(savedir, step_pre + '{:03d}.png'.format(i))
            rendername = os.path.join(savedir, step_pre + 'render_{:03d}.png'.format(i))
            gtname = os.path.join(savedir, step_pre + 'gt_{:03d}.png'.format(i))

            img8 = rgb8
            if gt_imgs is not None:
                error8 = to8b(error)
                gt8 = to8b(gt_imgs[i])
                imageio.imwrite(gtname, gt8)
                img8 = np.concatenate([error8, rgb8, gt8], axis=0)

            if split_bg and gt_imgs is not None:
                in8 = to8b(ins[-1])
                out8 = to8b(outs[-1])
                img8_2 = np.concatenate([in8, out8], axis=1)
                img8 = np.concatenate([rgb8, gt8], axis=1)
                img8 = np.concatenate([img8, img8_2], axis=0)

            if os.path.exists(filename):
                os.remove(filename)
            if os.path.exists(rendername):
                os.remove(rendername)
            imageio.imwrite(rendername, rgb8)
            imageio.imwrite(filename, img8)

            for key in extras:
                if key.startswith('normal'):
                    extras[key][-1] = matte(extras[key][-1] / 2. + 0.5, bgmaps[-1])
                else:
                    extras[key][-1] = matte(extras[key][-1], bgmaps[-1])

                extra8 = to8b(extras[key][-1])
                if extra8.shape[-1] == 1:
                    extra8 = np.concatenate((extra8,) * 3, axis=-1)
                filename = os.path.join(savedir, f'{step_pre}_{key}_{i:03d}.png')
                imageio.imwrite(filename, extra8)

    rgbs = np.array(rgbs)
    disps = np.array(disps)
    extras = {key: np.array(extras[key]) for key in extras}
    if len(psnrs):
        logger.info('Testing psnr {:.2f} (avg) | foreground {:.2f} | background {:.2f}'.format(
            np.mean(psnrs), np.mean(fore_psnrs), np.mean(bg_psnrs)))
        if eval_ssim: logger.info('Testing ssim {} (avg)'.format(np.mean(ssims)))
        if eval_lpips_vgg: logger.info('Testing lpips (vgg) {} (avg)'.format(np.mean(lpips_vgg)))
        if eval_lpips_alex: logger.info('Testing lpips (alex) {} (avg)'.format(np.mean(lpips_alex)))

    return rgbs, disps, extras

def matte(vis, bgmap, dark=1.0, light=1.0, width=8):
    """Set non-accumulated pixels to a Photoshop-esque checker pattern."""
    acc = 1.0 - bgmap
    bg_mask = np.logical_xor(
        (np.arange(acc.shape[0]) % (2 * width) // width)[:, None],
        (np.arange(acc.shape[1]) % (2 * width) // width)[None, :])
    bg = np.where(~bg_mask, light, dark)[..., None]
    return vis * acc + (bg * (1 - acc))


def validate_mesh(cfg, model, resolution=128, threshold=0.0, prefix="", world_space=False,
                  scale_mats_np=None, gt_eval=False, runtime=True, scene=122, smooth=True,
                  extract_color=False):
    os.makedirs(os.path.join(cfg.basedir, cfg.expname, 'meshes'), exist_ok=True)
    bound_min = model.xyz_min.clone().detach().float()
    bound_max = model.xyz_max.clone().detach().float()

    gt_path = os.path.join(cfg.data.datadir, "stl_total.ply") if gt_eval else ''
    vertices0, triangles = model.extract_geometry(bound_min, bound_max, resolution=resolution,
                                                  threshold=threshold, scale_mats_np=scale_mats_np,
                                                  gt_path=gt_path, smooth=smooth,
                                                  )

    if world_space and scale_mats_np is not None:
        vertices = vertices0 * scale_mats_np[0, 0] + scale_mats_np[:3, 3][None]
    else:
        vertices = vertices0

    if extract_color:
        # use normal direction as the viewdir
        ray_pts = torch.from_numpy(vertices0).cuda().float().split(8192 * 32, 0)
        vertex_colors = [model.mesh_color_forward(pts) for pts in ray_pts]
        vertex_colors = (torch.concat(vertex_colors).cpu().detach().numpy() * 255.).astype(np.uint8)
        mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=vertex_colors)
    else:
        mesh = trimesh.Trimesh(vertices, triangles)
    mesh_path = os.path.join(cfg.basedir, cfg.expname, 'meshes', prefix + '.ply')
    mesh.export(mesh_path)
    print("mesh saved at " + mesh_path)
    # if gt_eval:
    #     mean_d2s, mean_s2d, over_all = eval(mesh_path, scene=scene,
    #                                         eval_dir=os.path.join(cfg.basedir, cfg.expname, 'meshes'),
    #                                         dataset_dir='data/DTU', suffix=prefix + 'eval', use_o3d=False,
    #                                         runtime=runtime)
    #     res = "standard point cloud sampling" if not runtime else "down sampled point cloud for fast eval (NOT standard!):"
    #     print("mesh evaluation with {}".format(res))
    #     print(" [ d2s: {:.3f} | s2d: {:.3f} | mean: {:.3f} ]".format(mean_d2s, mean_s2d, over_all))
    #     return over_all
    return 0.


def compute_tv_norm(values, losstype='l2', weighting=None):  # pylint: disable=g-doc-args
    """Returns TV norm for input values.

  Note: The weighting / masking term was necessary to avoid degenerate
  solutions on GPU; only observed on individual DTU scenes.
  """
    v00 = values[:, :-1, :-1]
    v01 = values[:, :-1, 1:]
    v10 = values[:, 1:, :-1]

    if losstype == 'l2':
        loss = ((v00 - v01) ** 2) + ((v00 - v10) ** 2)
    elif losstype == 'l1':
        loss = abs(v00 - v01) + abs(v00 - v10)
    else:
        raise NotImplementedError
    if weighting is not None:
        loss = loss * weighting
    return loss


def generate_ide_fn(deg_view):
    """Generate integrated directional encoding (IDE) function.

    This function returns a function that computes the integrated directional
    encoding from Equations 6-8 of arxiv.org/abs/2112.03907.

    Args:
      deg_view: number of spherical harmonics degrees to use.

    Returns:
      A function for evaluating integrated directional encoding.

    Raises:
      ValueError: if deg_view is larger than 5.
    """
    if deg_view > 5:
        raise ValueError('Only deg_view of at most 5 is numerically stable.')

    ml_array = get_ml_array(deg_view)
    l_max = 2 ** (deg_view - 1)

    # Create a matrix corresponding to ml_array holding all coefficients, which,
    # when multiplied (from the right) by the z coordinate Vandermonde matrix,
    # results in the z component of the encoding.
    mat = torch.zeros(l_max + 1, ml_array.shape[1]).cuda()
    for i, (m, l) in enumerate(ml_array.T):
        for k in range(l - m + 1):
            mat[k, i] = sph_harm_coeff(l, m, k)

    def integrated_dir_enc_fn(xyz, kappa_inv):
        """Function returning integrated directional encoding (IDE).

        Args:
          xyz: [..., 3] array of Cartesian coordinates of directions to evaluate at.
          kappa_inv: [..., 1] reciprocal of the concentration parameter of the von
            Mises-Fisher distribution.

        Returns:
          An array with the resulting IDE.
        """
        x = xyz[..., 0:1]
        y = xyz[..., 1:2]
        z = xyz[..., 2:3]

        ml_array_cu = torch.from_numpy(ml_array).cuda()
        # Compute z Vandermonde matrix.
        vmz = torch.cat([z ** i for i in range(mat.shape[0])], dim=-1)
        # Compute x+iy Vandermonde matrix.
        vmxy = torch.cat([(x + 1j * y) ** m for m in ml_array_cu[0, :]], dim=-1)
        # Get spherical harmonics.
        sph_harms = vmxy * (vmz @ mat)

        # Apply attenuation function using the von Mises-Fisher distribution
        # concentration parameter, kappa.
        sigma = 0.5 * ml_array_cu[1, :] * (ml_array_cu[1, :] + 1)
        ide = sph_harms * torch.exp(-sigma * kappa_inv)
        # Split into real and imaginary parts and return
        return torch.cat([torch.real(ide), torch.imag(ide)], dim=-1)

    return integrated_dir_enc_fn
