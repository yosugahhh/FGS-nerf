from model.nerf import nerf
from model.utils import *


def nerf_eval(args, cfg, logger, cfg_model, data_dict, stage):
    logger.info("= " * 10 + "Begin [ {} ]".format(stage) + " =" * 10)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # find whether there is existing checkpoint path
    reload_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'fine_last.tar')

    # init model
    model, eval_step = load_model(nerf, reload_ckpt_path, strict=False)
    model.to(device)

    # validate image
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
    if not args.only_mesh:
        validate_image(args, cfg, stage, eval_step, data_dict, render_viewpoints_kwargs)

    # validate mesh
    prefix = args.prefix
    suffix = args.suffix
    validate_mesh(cfg, model, 1024, threshold=0.0, prefix="{}{}_eval".format(prefix, suffix), world_space=True,
                  scale_mats_np=data_dict['scale_mats_np'], gt_eval='dtu' in cfg.basedir, runtime=False,
                  scene=args.scene)