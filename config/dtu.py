expname = ''
basedir = ''
reso_level = 2

data = dict(
    datadir='',
    dataset_type='dtu',
    inverse_y=True,
    flip_x=False,
    flip_y=False,
    annot_path='',
    split_path='',
    sequence_name='',
    testskip=1,
    white_bkgd=True,
    half_res=False,
    factor=1,
    ndc=False,
    spherify=False,
    llffhold=8,
    load_depths=False,
    movie_render_kwargs=dict(),
    importance_prune=0.999,
    importance_include=0.6,
    codebook_size=4096,
    k_expire=10,
)

geometry_searching = dict(
    N_iters=12000,
    N_rand=8192,
    save_iter=20000,
    lrate_density=0.1,
    lrate_k0=0.1,
    lrate_sdf=0.1,
    lrate_refnet=0.001,
    lrate_decay=20,
    pervoxel_lr=False,
    pervoxel_lr_downrate=1,
    ray_sampler='random',
    weight_main=1,
    weight_entropy_last=0.001,
    weight_rgbper=0.2,
    weight_tv_density=0.01,
    weight_tv_k0=0,
    sigmoid_rgb_loss=0.1,
    weight_orientation=0.0001,
    tv_every=1,
    tv_from=0,
    tv_end=40000,
    voxel_inc=True,  # To use Incremental Voxel Training
    x_mid=0.5,  # Used to calculate $P_{min_init}$ and $P_{max_init}$ in Sec 3.2
    y_mid=0.5,  # Used to calculate $P_{min_init}$ and $P_{max_init}$ in Sec 3.2
    z_mid=0.5,  # Used to calculate $P_{min_init}$ and $P_{max_init}$ in Sec 3.2
    x_init_ratio=0.6,  # Used to calculate $P_{min_init}$ and $P_{max_init}$ in Sec 3.2
    y_init_ratio=0.6,  # Used to calculate $P_{min_init}$ and $P_{max_init}$ in Sec 3.2
    z_init_ratio=0.6,  # Used to calculate $P_{min_init}$ and $P_{max_init}$ in Sec 3.2
    inc_steps=1000,  # Maximum steps of Incremental Voxel Training
    scale_ratio=2,
    pg_scale=[1001, 2501, 4001, 5501, 7001, 8501, 10001],
    reset_iter=[1001, 2501, 4001, 5501, 7001, 8501, 10001],
    # reset_iter=[],
    tv_terms=dict(sdf_tv=0.1, grad_norm=0, grad_tv=0, smooth_grad_tv=0.05),
    tv_add_grad_new=True,
    ori_tv=True,
    tv_updates=dict({}),
    tv_dense_before=40000,
    decay_step_module=dict({
        10001: dict(sdf=0.1),
    }),
    skip_zero_grad_fields=[
        'density', 'k0', 'sdf'
    ],
    vq_finetune=False,
)

geometry_searching_model = dict(
    num_voxels=1024000,
    num_voxels_base=80**3,
    nearest=False,
    bbox_thres=0.001,
    mask_cache_thres=0.001,
    alpha_init=0.01,
    fast_color_thres=1e-4,
    ref=True,
    use_viewemb=True,
    maskout_near_cam_vox=True,
    world_bound_scale=1,
    stepsize=0.5,
    k0_dim=6,
    refnet_width=128,
    refnet_depth=3,
    sdf_refine=True,
    alpha_refine=True,
    displace_step=0.1,
    posbase_pe=5,
    viewbase_pe=1,
    refbase_pe=3,
    smooth_ksize=5,
    smooth_sigma=0.8,
    s_ratio=50,
    s_start=0.2,
)

coarse_train = dict(
    N_iters=15000,
    N_rand=8192,
    save_iter=20000,
    lrate_k0=0.1,
    lrate_sdf=0.1,
    lrate_refnet=0.001,
    lrate_decay=20,
    pervoxel_lr=False,
    pervoxel_lr_downrate=1,
    ray_sampler='in_maskcache',
    weight_main=1,
    weight_entropy_last=0.001,
    weight_rgbper=0.2,
    weight_tv_density=0.01,
    weight_tv_k0=0,
    sigmoid_rgb_loss=0.1,
    weight_orientation=0.0001,
    tv_every=1,
    tv_from=0,
    tv_end=40000,
    voxel_inc=False,
    scale_ratio=3,
    pg_scale=[1000, 2001, 3001, 4001, 5001, 8001],
    reset_iter=[],
    tv_terms=dict(sdf_tv=0.1, grad_norm=0, grad_tv=0, smooth_grad_tv=0.05),
    tv_add_grad_new=True,
    ori_tv=True,
    tv_updates=dict({
        8001: dict(sdf_tv=0.1, smooth_grad_tv=0.2)
    }),
    tv_dense_before=40000,
    decay_step_module=dict({
        5001: dict(sdf=0.2),
        8001: dict(sdf=0.1),
        12001: dict(sdf=0.2),
    }),
    skip_zero_grad_fields=[
        'density', 'k0', 'sdf'
    ],
    vq_finetune=False,
)

coarse_model = dict(
    num_voxels=1500000,
    num_voxels_base=1500000,
    nearest=False,
    bbox_thres=0.001,
    mask_cache_thres=0.001,
    alpha_init=0.01,
    fast_color_thres=1e-4,
    ref=True,
    use_viewdir=True,
    maskout_near_cam_vox=True,
    world_bound_scale=1.1,
    stepsize=0.5,
    k0_dim=12,
    rgbnet_width=192,
    rgbnet_depth=3,
    refnet_width=192,
    refnet_depth=3,
    sdf_refine=True,
    alpha_refine=True,
    displace_step=0.1,
    posbase_pe=5,
    viewbase_pe=3,
    refbase_pe=5,
    smooth_ksize=5,
    smooth_sigma=0.8,
    s_ratio=50,
    s_start=0.2,
)

fine_train = dict(
    N_iters=20000,
    N_rand=8192,
    save_iter=20000,
    lrate_k0=0.1,
    lrate_sdf=0.005,
    lrate_rgbnet=0.001,
    lrate_refnet=0.001,
    lrate_decay=20,
    pervoxel_lr=False,
    pervoxel_lr_downrate=1,
    ray_sampler='in_maskcache',
    weight_main=1,
    weight_entropy_last=0.001,
    weight_rgbper=0.0,
    weight_tv_density=0.01,
    weight_tv_k0=0.0,
    sigmoid_rgb_loss=0.02,
    weight_orientation=1e-4,
    tv_every=3,
    tv_from=0,
    tv_end=30000,
    voxel_inc=False,
    scale_ratio=4.096,
    pg_scale=[15000],
    reset_iter=[],
    tv_terms=dict(sdf_tv=0.1, grad_norm=0, grad_tv=0, smooth_grad_tv=0.05),
    tv_add_grad_new=True,
    tv_dense_before=20000,
    sdf_reduce=0.3,
    cosine_lr=True,
    cosine_lr_cfg=dict(
        warm_up_iters=0, const_warm_up=True, warm_up_min_ratio=1.0),
    decay_step_module=dict({
        15000: dict(sdf=0.1)}),
    skip_zero_grad_fields=[
        'density', 'k0', 'k1'
    ],
    vq_finetune=False,
)

fine_model = dict(
    num_voxels=256**3,
    num_voxels_base=256**3,
    nearest=False,
    bbox_thres=0.001,
    mask_cache_thres=0.001,
    alpha_init=0.01,
    fast_color_thres=0.0001,
    maskout_near_cam_vox=False,
    world_bound_scale=1.10,
    stepsize=0.5,
    ref=True,
    use_viewdir=True,
    refnet_width=256,
    refnet_depth=4,
    k0_dim=12,
    rgbnet_width=256,
    rgbnet_depth=4,
    sdf_refine=True,
    alpha_refine=True,
    center_sdf=True,
    displace_step=0.1,
    posbase_pe=5,
    viewbase_pe=3,
    refbase_pe=8,
    s_ratio=50,
    s_start=0.05,
    grad_feat=(0.5, 1.0, 1.5, 2.0),
    sdf_feat=(0.5, 1.0, 1.5, 2.0),
)

