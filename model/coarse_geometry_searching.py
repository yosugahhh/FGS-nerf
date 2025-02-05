import math
import os, time
import logging
import copy

import torch.nn as nn
import torch.nn.functional as F

from torch_efficient_distloss import flatten_eff_distloss
from tqdm import tqdm, trange
from datetime import datetime

from model.dvgo import dvgo
from model import dvgo_ray
from model.adam import *
from model.evaluation import *
from model.utils import *


def _compute_bbox_by_cam_frustrm_bounded(cfg, HW, Ks, poses, i_train, near, far):
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = dvgo_ray.get_rays_of_a_view(
            H=H, W=W, K=K, c2w=c2w,
            ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
            flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        if cfg.data.ndc:
            pts_nf = torch.stack([rays_o + rays_d * near, rays_o + rays_d * far])
        else:
            pts_nf = torch.stack([rays_o + viewdirs * near, rays_o + viewdirs * far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0, 1, 2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0, 1, 2)))
    return xyz_min, xyz_max


def _compute_bbox_by_cam_frustrm_unbounded(cfg, HW, Ks, poses, i_train, near_clip):
    # Find a tightest cube that cover all camera centers
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = dvgo_ray.get_rays_of_a_view(
            H=H, W=W, K=K, c2w=c2w,
            ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
            flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        pts = rays_o + rays_d * near_clip
        xyz_min = torch.minimum(xyz_min, pts.amin((0, 1)))
        xyz_max = torch.maximum(xyz_max, pts.amax((0, 1)))
    center = (xyz_min + xyz_max) * 0.5
    radius = (center - xyz_min).max() * cfg.data.unbounded_inner_r
    xyz_min = center - radius
    xyz_max = center + radius
    return xyz_min, xyz_max


def compute_bbox_by_cam_frustrm(cfg, HW, Ks, poses, i_train, near, far, **kwargs):
    print('compute_bbox_by_cam_frustrm: start')
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = dvgo_ray.get_rays_of_a_view(
            H=H, W=W, K=K, c2w=c2w,
            ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
            flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        pts_nf = torch.stack([rays_o + viewdirs * near, rays_o + viewdirs * far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0, 1, 2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0, 1, 2)))
    print('compute_bbox_by_cam_frustrm: xyz_min {}'.format(xyz_min))
    print('compute_bbox_by_cam_frustrm: xyz_max {}'.format(xyz_max))
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max


def create_optimizer_or_freeze_model(model, cfg_train, global_step):
    decay_steps = cfg_train.lrate_decay * 1000
    decay_factor = 0.1 ** (global_step / decay_steps)

    param_group = []
    for k in cfg_train.keys():
        if not k.startswith('lrate_'):
            continue
        k = k[len('lrate_'):]

        if not hasattr(model, k):
            continue

        param = getattr(model, k)
        if param is None:
            print(f'create_optimizer_or_freeze_model: param {k} not exist')
            continue

        lr = getattr(cfg_train, f'lrate_{k}') * decay_factor
        if lr > 0:
            print(f'create_optimizer_or_freeze_model: param {k} lr {lr}')
            if isinstance(param, nn.Module):
                param = param.parameters()
            param_group.append({'params': param, 'lr': lr, 'name': k,
                                'skip_zero_grad': (k in cfg_train.skip_zero_grad_fields)})
        else:
            print(f'create_optimizer_or_freeze_model: param {k} freeze')
            param.requires_grad = False
    return MaskedAdam(param_group, betas=(0.9, 0.99))


def geometry_searching(args, cfg, logger, cfg_model, cfg_train, xyz_min, xyz_max, data_dict, stage='coarse',
                       coarse_ckpt_path=None):
    logger.info("= " * 10 + "Begin training state [ {} ]".format(stage) + " =" * 10)

    # init
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.set_default_device('cuda')
    else:
        device = torch.device('cpu')

    if abs(cfg_model.world_bound_scale - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift
    HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images, masks = [
        data_dict[k] for k in [
            'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'images', 'masks'
        ]
    ]

    last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_dvgo_last.tar')

    # init model
    model_kwargs = copy.deepcopy(cfg_model)
    num_voxels = model_kwargs.pop('num_voxels')
    num_voxels_bg = model_kwargs.pop('num_voxels_bg', num_voxels)
    scale_ratio = getattr(cfg_train, 'scale_ratio', 2)
    if len(cfg_train.pg_scale):
        deduce = (scale_ratio ** len(cfg_train.pg_scale))
        num_voxels = int(num_voxels / deduce)
        num_voxels_bg = int(num_voxels_bg / deduce)
        logger.info("\n" + "+ " * 10 + "start with {} resolution deduction".format(deduce) + " +" * 10 + "\n")
    else:
        deduce = 1

    model = dvgo(
        xyz_min=xyz_min, xyz_max=xyz_max,
        num_voxels=num_voxels,
        num_voxels_bg=num_voxels_bg,
        mask_cache_path=coarse_ckpt_path,
        exppath=os.path.join(cfg.basedir, cfg.expname),
        **model_kwargs).to(device)
    cfg_train.weight_orientation = 0
    cfg_train.weight_pred_normals = 0

    # init optimizer
    optimizer = create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
    start = 0

    # init rendering setup
    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
        'render_depth': True,
    }

    # init batch rays sampler
    def gather_training_rays():
        rgb_tr_ori = images[i_train].to(device)
        mask_tr_ori = masks[i_train].to(device)
        rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo_ray.get_training_rays(
            rgb_tr=rgb_tr_ori,
            train_poses=poses[i_train],
            HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
            flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        index_generator = dvgo_ray.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
        if cfg_train.ray_sampler == 'patch':
            # patch sampler contains lots of empty spaces, remove them.
            index_generator = dvgo_ray.batch_indices_generator(len(rgb_tr), 1)
        batch_index_sampler = lambda: next(index_generator)
        return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler

    rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler = gather_training_rays()


    if cfg_train.pervoxel_lr:
        def per_voxel_init():
            cnt = model.voxel_count_views(
                rays_o_tr=rays_o_tr, rays_d_tr=rays_d_tr, imsz=imsz, near=near, far=far,
                stepsize=cfg_model.stepsize, downrate=cfg_train.pervoxel_lr_downrate,
                irregular_shape=data_dict['irregular_shape'])
            optimizer.set_pervoxel_lr(cnt)
            with torch.no_grad():
                model.density.grid[cnt <= 2] = -100

        per_voxel_init()

    # GOGO
    psnr_lst = []
    weight_lst = []
    mask_lst = []
    bg_mask_lst = []
    weight_sum_lst = []
    weight_nonzero_lst = []
    s_val_lst = []
    time0 = time.time()
    logger.info("start: {} end: {}".format(1 + start, 1 + cfg_train.N_iters))
    torch.cuda.empty_cache()

    for global_step in trange(1 + start, 1 + cfg_train.N_iters):
        # progress scaling checkpoint
        if global_step in cfg_train.pg_scale:
            if hasattr(model, 'num_voxels_bg'):
                model.scale_volume_grid(model.num_voxels * scale_ratio, model.num_voxels_bg * scale_ratio)
            else:
                model.scale_volume_grid(model.num_voxels * scale_ratio)
            optimizer = create_optimizer_or_freeze_model(model, cfg_train, global_step=0)

        # ray_sampler == random
        sel_b = torch.randint(rgb_tr.shape[0], [cfg_train.N_rand])
        sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
        sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand])
        target = rgb_tr[sel_b, sel_r, sel_c]
        rays_o = rays_o_tr[sel_b, sel_r, sel_c]
        rays_d = rays_d_tr[sel_b, sel_r, sel_c]
        viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]

        # volume rendering
        render_result = model(rays_o, rays_d, viewdirs, global_step=global_step, **render_kwargs)

        # gradient descent step
        optimizer.zero_grad(set_to_none=True)
        loss = cfg_train.weight_main * F.mse_loss(render_result['rgb_marched'], target)
        psnr = mse2psnr(loss.detach()).item()

        if cfg_train.weight_entropy_last > 0:
            pout = render_result['alphainv_cum'][..., -1].clamp(1e-6, 1 - 1e-6)
            entropy_last_loss = -(pout * torch.log(pout) + (1 - pout) * torch.log(1 - pout)).mean()
            loss += cfg_train.weight_entropy_last * entropy_last_loss

        if cfg_train.weight_rgbper > 0:
            rgbper = (render_result['raw_rgb'] - target[render_result['ray_id']]).pow(2).sum(-1)
            rgbper_loss = (rgbper * render_result['weights'].detach()).sum() / len(rays_o)
            loss += cfg_train.weight_rgbper * rgbper_loss

        loss.backward()

        optimizer.step()
        wm = render_result['weights'].max(-1)[0]
        ws = render_result['weights'].sum(-1)
        if (wm > 0).float().mean() > 0:
            psnr_lst.append(psnr)
            weight_lst.append(wm[wm > 0].mean().detach().cpu().numpy())
            weight_sum_lst.append(ws[ws > 0].mean().detach().cpu().numpy())
            weight_nonzero_lst.append((ws > 0).float().mean().detach().cpu().numpy())
            if render_result['mask'] is not None:
                mask_lst.append(render_result['mask'].float().mean().detach().cpu().numpy())
            if 'bg_mask' in render_result:
                bg_mask_lst.append(render_result['bg_mask'].float().mean().detach().cpu().numpy())
        s_val = render_result["s_val"] if "s_val" in render_result else 0
        s_val_lst.append(s_val)

        global_step_ = global_step - 1
        # update lr
        N_iters = cfg_train.N_iters
        if not getattr(cfg_train, 'cosine_lr', ''):
            decay_steps = cfg_train.lrate_decay * 1000
            decay_factor = 0.1 ** (1 / decay_steps)
            for i_opt_g, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = param_group['lr'] * decay_factor
        else:
            def cosine_lr_func(iter, warm_up_iters, warm_up_min_ratio, max_steps, const_warm_up=False, min_ratio=0):
                if iter < warm_up_iters:
                    if not const_warm_up:
                        lr = warm_up_min_ratio + (1 - warm_up_min_ratio) * (iter / warm_up_iters)
                    else:
                        lr = warm_up_min_ratio
                else:
                    lr = (1 + math.cos((iter - warm_up_iters) / (max_steps - warm_up_iters) * math.pi)) * 0.5 * (
                            1 - min_ratio) + min_ratio
                return lr

            def extra_warm_up_func(iter, start_iter, warm_up_iters, warm_up_min_ratio):
                if iter >= start_iter:
                    extra_lr = warm_up_min_ratio + (1 - warm_up_min_ratio) * (iter - start_iter) / warm_up_iters
                    return min(extra_lr, 1.0)
                else:
                    return 1.0

            warm_up_iters = cfg_train.cosine_lr_cfg.get('warm_up_iters', 0)
            warm_up_min_ratio = cfg_train.cosine_lr_cfg.get('warm_up_min_ratio', 1.0)
            const_warm_up = cfg_train.cosine_lr_cfg.get('const_warm_up', False)
            cos_min_ratio = cfg_train.cosine_lr_cfg.get('cos_min_ratio', False)
            if global_step == 0:
                pre_decay_factor = 1.0
            else:
                pre_decay_factor = cosine_lr_func(global_step_ - 1, warm_up_iters, warm_up_min_ratio, N_iters,
                                                  const_warm_up, cos_min_ratio)
            pos_decay_factor = cosine_lr_func(global_step_, warm_up_iters, warm_up_min_ratio, N_iters, const_warm_up,
                                              cos_min_ratio)
            decay_factor = pos_decay_factor / pre_decay_factor
            for i_opt_g, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = param_group['lr'] * decay_factor
        decay_step_module = getattr(cfg_train, 'decay_step_module', dict())
        if global_step_ in decay_step_module:
            for i_opt_g, param_group in enumerate(optimizer.param_groups):
                if param_group['name'] in decay_step_module[global_step_]:
                    decay_factor = decay_step_module[global_step_][param_group['name']]
                    param_group['lr'] = param_group['lr'] * decay_factor
                    logger.info(
                        '- ' * 10 + '[Decay lrate] for {} by {}'.format(param_group['name'], decay_factor) + ' -' * 10)

        # update tv terms
        tv_updates = getattr(cfg_train, 'tv_updates', dict())
        if global_step_ in tv_updates:
            for tv_term, value in tv_updates[global_step_].items():
                setattr(cfg_train.tv_terms, tv_term, value)
            logger.info('- ' * 10 + '[Update tv]: ' + str(tv_updates[global_step_]) + ' -' * 10)

        # update s_val func
        s_updates = getattr(cfg_model, 's_updates', dict())
        if global_step_ in s_updates:
            for s_term, value in s_updates[global_step_].items():
                setattr(model, s_term, value)
            logger.info('- ' * 10 + '[Update s]: ' + str(s_updates[global_step_]) + ' -' * 10)

        # update smooth kernel
        smooth_updates = getattr(cfg_model, 'smooth_updates', dict())
        if global_step_ in smooth_updates:
            model.init_smooth_conv(**smooth_updates[global_step_])
            logger.info('- ' * 10 + '[Update smooth conv]: ' + str(smooth_updates[global_step_]) + ' -' * 10)

        # check log & save
        # i_print: frequency of console printout and metric loggin
        i_print = args.i_print
        if global_step % i_print == 0:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time // 3600:02.0f}:{eps_time // 60 % 60:02.0f}:{eps_time % 60:02.0f}'
            bg_mask_mean = 0. if len(bg_mask_lst) == 0 else np.mean(bg_mask_lst)
            logger.info(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                        f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / '
                        f'Wmax: {np.mean(weight_lst):5.2f} / Wsum: {np.mean(weight_sum_lst):5.2f} / W>0: {np.mean(weight_nonzero_lst):5.2f}'
                        f' / s_val: {np.mean(s_val_lst):5.2g} / mask\%: {100 * np.mean(mask_lst):1.2f} / bg_mask\%: {100 * bg_mask_mean:1.2f} '
                        f'Eps: {eps_time_str}')
            psnr_lst, weight_lst, weight_sum_lst, weight_nonzero_lst, mask_lst, bg_mask_lst, s_val_lst = [], [], [], [], [], [], []

        # validate image
        if global_step == cfg_train.N_iters or global_step % args.i_validate == 0:
            render_viewpoints_kwargs = {
                'model': model,
                'ndc': cfg.data.ndc,
                'logger': logger,
                'render_kwargs': {
                    'near': data_dict['near'],
                    'far': data_dict['far'],
                    'bg': 1 if cfg.data.white_bkgd else 0,
                    'stepsize': cfg_model.stepsize,
                    'inverse_y': cfg.data.inverse_y,
                    'flip_x': cfg.data.flip_x,
                    'flip_y': cfg.data.flip_y,
                    'render_grad': True,
                    'render_depth': True,
                    'render_in_out': True,
                },
            }
            validate_image(args, cfg, stage, global_step, data_dict, render_viewpoints_kwargs,
                           eval_all=cfg_train.N_iters == global_step)

        # save checkpoints
        if global_step == cfg_train.N_iters:
            if model.ref:
                last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'geometry_searching_last.tar')
            torch.save({
                'global_step': global_step,
                'model_kwargs': model.get_kwargs(),
                'MaskCache_kwargs': model.get_MaskCache_kwargs(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, last_ckpt_path)
            logger.info(f'scene_rep_reconstruction ({stage}): saved checkpoints at ' + last_ckpt_path)
