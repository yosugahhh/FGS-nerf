import copy

from tqdm import trange

from model.adam import *
from model.nerf import *


def create_optimizer_or_freeze_model(model, logger, cfg_train, global_step):
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
            logger.info(f'create_optimizer_or_freeze_model: param {k} not exist')
            continue

        lr = getattr(cfg_train, f'lrate_{k}') * decay_factor
        if lr > 0:
            logger.info(f'create_optimizer_or_freeze_model: param {k} lr {lr}')
            if isinstance(param, nn.Module):
                param = param.parameters()
            param_group.append({'params': param, 'lr': lr, 'name': k,
                                'skip_zero_grad': (k in cfg_train.skip_zero_grad_fields)})
        else:
            logger.info(f'create_optimizer_or_freeze_model: param {k} freeze')
            param.requires_grad = False
    return MaskedAdam(param_group, betas=(0.9, 0.99))


@torch.no_grad()
def compute_bbox_by_coarse_geo(model_class, model_path, thres):
    st = torch.load(model_path)
    coarse_xyz_min = torch.tensor(st['model_kwargs']['xyz_min'])
    coarse_xyz_max = torch.tensor(st['model_kwargs']['xyz_max'])
    sdf_mask = st['model_state_dict']['sdf_mask.grid']
    interp = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, sdf_mask.shape[2]),
        torch.linspace(0, 1, sdf_mask.shape[3]),
        torch.linspace(0, 1, sdf_mask.shape[4]),
        indexing='ij'
    ), -1)

    dense_xyz = coarse_xyz_min * (1 - interp) + coarse_xyz_max * interp
    mask = (sdf_mask > 0)[0, 0, :]
    active_xyz = dense_xyz[mask]
    xyz_min = active_xyz.amin(0)
    xyz_max = active_xyz.amax(0)
    return xyz_min, xyz_max


def nerf_training(args, cfg, logger, cfg_model, cfg_train, xyz_min, xyz_max,
                  data_dict, coarse_ckpt_path, stage='', ):
    logger.info("= " * 10 + "Begin training state [ {} ]".format(stage) + " =" * 10)

    # init
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.set_default_device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
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

    # find whether there is existing checkpoint path
    last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_last.tar')
    mask_path = os.path.join(cfg.basedir, cfg.expname, 'geometry_searching_last.tar')
    sdf_ckpt_path = None
    reload_ckpt_path = None

    if coarse_ckpt_path is not None and stage == 'fine':
        sdf_ckpt_path = coarse_ckpt_path

    # init model
    model_kwargs = copy.deepcopy(cfg_model)
    scale_ratio = getattr(cfg_train, 'scale_ratio', 2)
    num_voxels = model_kwargs.pop('num_voxels')
    num_voxels_bg = model_kwargs.pop('num_voxels_bg', num_voxels)
    if len(cfg_train.pg_scale):
        deduce = (scale_ratio ** len(cfg_train.pg_scale))
        num_voxels = int(num_voxels / deduce)
        num_voxels_bg = int(num_voxels_bg / deduce)
        logger.info("\n" + "+ " * 10 + "start with {} resolution deduction".format(deduce) + " +" * 10 + "\n")
    else:
        deduce = 1

    model = nerf(
        xyz_min=xyz_min, xyz_max=xyz_max,
        num_voxels=num_voxels,
        num_voxels_bg=num_voxels_bg,
        mask_cache_path=mask_path,
        exppath=os.path.join(cfg.basedir, cfg.expname),
        training=True, stage=stage,
        **model_kwargs).to(device)

    if cfg_model.maskout_near_cam_vox:
        model.maskout_near_cam_vox(poses[i_train, :3, 3], near)

    optimizer = create_optimizer_or_freeze_model(model, logger, cfg_train, global_step=0)

    if reload_ckpt_path is None or args.no_reload:
        logger.info(f'scene_rep_reconstruction ({stage}): train from scratch')
        start = 0
        if sdf_ckpt_path:
            sdf_reduce = cfg_train.get('sdf_reduce', 1.0)
            logger.info("\n" + "+ " * 10 + "load sdf from: " + sdf_ckpt_path + "+" * 10 + "\n")
            sdf0 = load_grid_data(model, sdf_ckpt_path, name='sdf', return_raw=True)
            # if stage == 'coarse':
            #     sdf0 = model.sample_sdf_from_coarse(sdf0, sdf_ckpt_path)
            model.init_sdf_from_sdf(sdf0, smooth=False, reduce=sdf_reduce)
            optimizer = create_optimizer_or_freeze_model(model, logger, cfg_train, global_step=0)
    else:
        logger.info(f'scene_rep_reconstruction ({stage}): reload from {reload_ckpt_path}')
        model, optimizer, start = load_checkpoint(
            model, optimizer, reload_ckpt_path, args.no_reload_optimizer, stage, num_voxels, strict=False)
        logger.info("Restart from iteration {}, model sdf size: {}".format(start, model.sdf.grid.shape))

        if reload_ckpt_path.split('/')[-1].split('_')[0] != stage:
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
    }

    # init batch rays sampler
    def gather_training_rays():
        if data_dict['irregular_shape']:
            rgb_tr_ori = [images[i].to(device) for i in i_train]
            mask_tr_ori = [masks[i].to(device) for i in i_train]
        else:
            rgb_tr_ori = images[i_train].to(device)
            mask_tr_ori = masks[i_train].to(device)

        if cfg_train.ray_sampler == 'in_maskcache':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = nerf_ray.get_training_rays_in_maskcache_sampling(
                rgb_tr_ori=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train],
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                model=model, render_kwargs=render_kwargs,
            )
        elif cfg_train.ray_sampler == 'flatten':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = nerf_ray.get_training_rays_flatten(
                rgb_tr_ori=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        else:
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = nerf_ray.get_training_rays(
                rgb_tr=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        index_generator = nerf_ray.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
        if cfg_train.ray_sampler == 'patch':
            # patch sampler contains lots of empty spaces, remove them.
            index_generator = nerf_ray.batch_indices_generator(len(rgb_tr), 1)
        batch_index_sampler = lambda: next(index_generator)
        return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler

    rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler = gather_training_rays()

    def per_voxel_init():
        cnt = model.voxel_count_views(
            rays_o_tr=rays_o_tr, rays_d_tr=rays_d_tr, imsz=imsz, near=near, far=far,
            stepsize=cfg_model.stepsize, downrate=cfg_train.pervoxel_lr_downrate,
            irregular_shape=data_dict['irregular_shape'])
        optimizer.set_pervoxel_lr(cnt)
        with torch.no_grad():
            model.sdf.grid[cnt <= 2] = 1
    if cfg_train.pervoxel_lr:
        per_voxel_init()

    # Init Increment (between 0 and 1)
    if cfg_train.voxel_inc:
        x_mid = cfg_train.x_mid
        y_mid = cfg_train.y_mid
        z_mid = cfg_train.z_mid
        voxel_inc_lower_init = torch.tensor([
            x_mid-cfg_train.x_init_ratio*(x_mid),
            y_mid-cfg_train.y_init_ratio*(y_mid),
            z_mid-cfg_train.z_init_ratio*(z_mid)
        ])
        voxel_inc_upper_init = torch.tensor([
            x_mid+cfg_train.x_init_ratio*(1-x_mid),
            y_mid+cfg_train.y_init_ratio*(1-y_mid),
            z_mid+cfg_train.z_init_ratio*(1-z_mid)
        ])

    # run
    psnr_lst = []
    weight_lst = []
    mask_lst = []
    bg_mask_lst = []
    weight_sum_lst = []
    weight_nonzero_lst = []
    s_val_lst = []
    time0 = time.time()
    logger.info("start: {} end: {}".format(1 + start, 1 + cfg_train.N_iters))
    logger.info(model.num_voxels)
    torch.cuda.empty_cache()

    time_ray_sample, time_render, time_loss, time_opt = 0, 0, 0, 0
    time_log = {
        'time_ray_sample': time_ray_sample,
        'time_render': time_render,
        'time_loss': time_loss,
        'time_opt': time_opt,
    }

    for global_step in trange(1 + start, 1 + cfg_train.N_iters):
        time_start = time.time()
        if stage == 'fine' and (global_step < 1000 or global_step > 15000):
            torch.cuda.empty_cache()
        if stage == 'coarse' and global_step < 100:
            torch.cuda.empty_cache()

        if global_step in cfg_train.pg_scale:
            if hasattr(model, 'num_voxels_bg'):
                model.scale_volume_grid(model.num_voxels * scale_ratio, model.num_voxels_bg * scale_ratio)
            else:
                model.scale_volume_grid(model.num_voxels * scale_ratio)
            if global_step in cfg_train.reset_iter:
                model.reset_voxel_and_mlp()
                if cfg_model.maskout_near_cam_vox:
                    model.maskout_near_cam_vox(poses[i_train, :3, 3], near)
            optimizer = create_optimizer_or_freeze_model(model, logger, cfg_train, global_step=0)

        # random sample rays
        if cfg_train.ray_sampler in ['flatten', 'in_maskcache']:
            sel_i = batch_index_sampler()
            target = rgb_tr[sel_i]
            rays_o = rays_o_tr[sel_i]
            rays_d = rays_d_tr[sel_i]
            viewdirs = viewdirs_tr[sel_i]
        elif cfg_train.ray_sampler == 'patch':
            sel_b = batch_index_sampler()
            patch_size = cfg_train.N_patch
            sel_r_start = torch.randint(rgb_tr.shape[1] - patch_size, [1])
            sel_c_start = torch.randint(rgb_tr.shape[2] - patch_size, [1])
            sel_r, sel_c = torch.meshgrid(torch.arange(sel_r_start[0], sel_r_start[0] + patch_size),
                                          torch.arange(sel_c_start[0], sel_c_start[0] + patch_size))
            sel_r, sel_c = sel_r.reshape(-1), sel_c.reshape(-1)
            target = rgb_tr[sel_b, sel_r, sel_c]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
        elif cfg_train.ray_sampler == 'random':
            sel_b = torch.randint(rgb_tr.shape[0], [cfg_train.N_rand])
            sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
            sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand])
            target = rgb_tr[sel_b, sel_r, sel_c]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
        else:
            raise NotImplementedError

        # Voxel Increment
        if cfg_train.voxel_inc:
            if global_step <= cfg_train.inc_steps:
                weight = min(global_step * 1.0 / cfg_train.inc_steps, 1.0)
                voxel_inc_lower = voxel_inc_lower_init - weight * voxel_inc_lower_init
                voxel_inc_upper = voxel_inc_upper_init + weight * (1 - voxel_inc_upper_init)
                model.set_inc_mask(voxel_inc_lower, voxel_inc_upper)
        else:
            model.unset_inc_mask()

        time_ray_sample = time.time() - time_start
        time_log['time_ray_sample'] += time_ray_sample
        time_start = time.time()

        # volume rendering
        render_result = model(rays_o, rays_d, viewdirs, global_step=global_step, **render_kwargs)

        time_render = time.time() - time_start
        time_log['time_render'] += time_render
        time_start = time.time()

        # gradient descent step
        optimizer.zero_grad(set_to_none=True)
        loss = cfg_train.weight_main * F.mse_loss(render_result['rgb_marched'], target)
        psnr = mse2psnr(loss.detach()).item()

        if cfg_train.weight_rgbper > 0:
            rgbper = (render_result['raw_rgb'] - target[render_result['ray_id']]).pow(2).sum(-1)
            rgbper_loss = (rgbper * render_result['weights'].detach()).sum() / len(rays_o)
            loss += cfg_train.weight_rgbper * rgbper_loss

        if cfg_train.weight_entropy_last > 0:
            pout = render_result['alphainv_cum'][..., -1].clamp(1e-6, 1 - 1e-6)
            entropy_last_loss = -(pout * torch.log(pout) + (1 - pout) * torch.log(1 - pout)).mean()
            loss += cfg_train.weight_entropy_last * entropy_last_loss

        if cfg_train.weight_orientation > 0:
            ori_loss = cfg_train.weight_orientation * model.orientation_loss(render_result)
            loss += ori_loss

        if cfg_train.sigmoid_rgb_loss > 0:
            sigmoid_rgb_loss = cfg_train.sigmoid_rgb_loss * F.mse_loss(render_result['sigmoid_rgb'], target)
            loss += sigmoid_rgb_loss


        if global_step > cfg_train.tv_from and global_step < cfg_train.tv_end and global_step % cfg_train.tv_every == 0:
            if cfg_train.weight_tv_density > 0:
                tv_terms = getattr(cfg_train, 'tv_terms', dict())
                sdf_tv, smooth_grad_tv = tv_terms['sdf_tv'], tv_terms['smooth_grad_tv']
                if smooth_grad_tv > 0:
                    loss += cfg_train.weight_tv_density * model.density_total_variation(sdf_tv=0,
                                                                                        smooth_grad_tv=smooth_grad_tv)
                if getattr(cfg_train, 'ori_tv', False):
                    loss += cfg_train.weight_tv_density * model.density_total_variation(sdf_tv=sdf_tv, smooth_grad_tv=0)
                    weight_tv_k0 = getattr(cfg_train, 'weight_tv_k0')
                    if weight_tv_k0 > 0:
                        k0_tv_terms = getattr(cfg_train, 'k0_tv_terms', dict())
                        loss += cfg_train.weight_tv_k0 * model.k0_total_variation(**k0_tv_terms)
                    if getattr(tv_terms, 'bg_density_tv', 0):
                        loss += cfg_train.weight_tv_density * model.density_total_variation(sdf_tv=0, smooth_grad_tv=0,
                                                                                            bg_density_tv=tv_terms['bg_density_tv'])

        loss.backward()

        time_loss = time.time() - time_start
        time_log['time_loss'] += time_loss
        time_start = time.time()

        if global_step > cfg_train.tv_from and global_step < cfg_train.tv_end and global_step % cfg_train.tv_every == 0:
            if not getattr(cfg_train, 'ori_tv', False):
                if cfg_train.weight_tv_density > 0:
                    tv_terms = getattr(cfg_train, 'tv_terms', dict())
                    sdf_tv = tv_terms['sdf_tv']
                    if sdf_tv > 0:
                        model.sdf_total_variation_add_grad(
                            cfg_train.weight_tv_density * sdf_tv / len(rays_o), global_step < cfg_train.tv_dense_before)
                    bg_density_tv = getattr(tv_terms, 'bg_density_tv', 0)
                    if bg_density_tv > 0:
                        model.bg_density_total_variation_add_grad(
                            cfg_train.weight_tv_density * bg_density_tv / len(rays_o),
                            global_step < cfg_train.tv_dense_before)
                if cfg_train.weight_tv_k0 > 0:
                    model.k0_total_variation_add_grad(
                        cfg_train.weight_tv_k0 / len(rays_o), global_step < cfg_train.tv_dense_before)
                if getattr(cfg_train, 'weight_bg_tv_k0', 0) > 0:
                    model.bg_k0_total_variation_add_grad(
                        cfg_train.weight_bg_tv_k0 / len(rays_o), global_step < cfg_train.tv_dense_before)

        optimizer.step()
        wm = render_result['weights'].max(-1)[0]
        ws = render_result['weights'].sum(-1)
        if (wm > 0).float().mean() > 0:
            psnr_lst.append(psnr)
            weight_lst.append(wm[wm > 0].mean().detach().cpu().numpy())
            weight_sum_lst.append(ws[ws > 0].mean().detach().cpu().numpy())
            weight_nonzero_lst.append((ws > 0).float().mean().detach().cpu().numpy())
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
                    logger.info('- ' * 10 + '[Decay lrate] for {} by {}'.format(param_group['name'], decay_factor) + ' -' * 10)

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

        time_opt = time.time() - time_start
        time_log['time_opt'] += time_opt

        # check log & save
        if global_step % args.i_print == 0:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time // 3600:02.0f}:{eps_time // 60 % 60:02.0f}:{eps_time % 60:02.0f}'
            bg_mask_mean = 0. if len(bg_mask_lst) == 0 else np.mean(bg_mask_lst)
            logger.info(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                        f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / '
                        f'Wmax: {np.mean(weight_lst):5.2f} / Wsum: {np.mean(weight_sum_lst):5.2f} / W>0: {np.mean(weight_nonzero_lst):5.2f}'
                        f' / s_val: {np.mean(s_val_lst):5.2g} / mask\%: {100 * np.mean(mask_lst):1.2f} / bg_mask\%: {100 * bg_mask_mean:1.2f} '
                        f'Eps: {eps_time_str}')

            time_ray_sample, time_render, time_loss, time_opt = \
                time_log['time_ray_sample'], time_log['time_render'], time_log['time_loss'], time_log['time_opt']
            logger.info(f'ray sample time:{time_ray_sample:5.2f}s / '
                        f'render time:{time_render:5.2f}s / '
                        f'loss calculate time:{time_loss:5.2f}s / '
                        f'optimizer time:{time_opt:5.2f}s')

            psnr_lst, weight_lst, weight_sum_lst, weight_nonzero_lst, mask_lst, bg_mask_lst, s_val_lst = [], [], [], [], [], [], []

        # validate image
        if global_step == cfg_train.N_iters or global_step % args.i_validate == 0:
            render_viewpoints_kwargs = {
                'model': model,
                'logger': logger,
                'ndc': cfg.data.ndc,
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
            torch.cuda.empty_cache()
            validate_image(args, cfg, stage, global_step, data_dict, render_viewpoints_kwargs,
                           eval_all=cfg_train.N_iters == global_step)
            torch.cuda.empty_cache()

        # validate mesh
        if args.prefix:
            prefix = args.prefix
            prefix += '_' + args.suffix if args.suffix else ''
        else:
            prefix = args.expname
        if 'eval_iters' in cfg_train and stage == 'fine':
            if global_step - start in cfg_train.eval_iters and stage == 'fine':
                gt_eval = 'dtu' in cfg.basedir
                cd = validate_mesh(model, resolution=512,
                                   prefix="{}{}_fine".format(prefix, global_step),
                                   gt_eval=gt_eval,
                                   world_space=True,
                                   scale_mats_np=data_dict['scale_mats_np'],
                                   scene=args.scene)

        # save checkpoints
        if global_step == cfg_train.N_iters or global_step % cfg_train.save_iter == 0:
            model.set_sdf_mask()
            torch.save({
                'global_step': global_step,
                'model_kwargs': model.get_kwargs(),
                'MaskCache_kwargs': model.get_MaskCache_kwargs(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, last_ckpt_path)
            logger.info(f'scene_rep_reconstruction ({stage}): saved checkpoints at ' + last_ckpt_path)

        # mesh validation
        if global_step == cfg_train.N_iters:
            validate_mesh(cfg, model, 512, threshold=0.0, prefix="{}_{}".format(stage, prefix), world_space=True,
                          scale_mats_np=data_dict['scale_mats_np'], gt_eval='dtu' in cfg.basedir, runtime=False,
                          scene=args.scene)
