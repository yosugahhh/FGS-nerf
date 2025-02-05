import argparse
import mmcv

from model.coarse_geometry_searching import *
from model.dataset import load_dataset
from model.nerf_eval import nerf_eval
from model.nerf_training import *


def runner(args, cfg, mode='train'):
    torch.set_default_device('cuda')
    data_dict = load_dataset(cfg)
    output_dir = args.output_dir
    eps_time = time.time()

    now = datetime.now()
    time_str = now.strftime('%Y-%m-%d_%H-%M-%S')
    if mode == 'train':
        logger = get_root_logger(logging.INFO, handlers=[
            logging.FileHandler(os.path.join(args.output_dir, args.expname, '{}_train.log').format(time_str))])
    else:
        logger = get_root_logger(logging.INFO, handlers=[
            logging.FileHandler(os.path.join(args.output_dir, args.expname, '{}_eval.log').format(time_str))])

    # coarse geometry searching
    if args.geometry_searching and mode == 'train':
        xyz_min, xyz_max = compute_bbox_by_cam_frustrm(cfg=cfg, **data_dict)

        eps = time.time()
        if args.dvgo_init:
            geometry_searching(
                args=args, cfg=cfg, logger=logger,
                cfg_model=cfg.dvgo_model,
                cfg_train=cfg.dvgo,
                xyz_min=xyz_min, xyz_max=xyz_max,
                data_dict=data_dict)
        else:
            nerf_training(
                args=args, cfg=cfg, logger=logger,
                cfg_model=cfg.geometry_searching_model,
                cfg_train=cfg.geometry_searching,
                xyz_min=xyz_min, xyz_max=xyz_max,
                data_dict=data_dict,
                coarse_ckpt_path=None,
                stage='geometry_searching')

        eps = time.time() - eps
        eps_time_str = f'{eps // 3600:02.0f}:{eps // 60 % 60:02.0f}:{eps % 60:02.0f}'
        logger.info("+ " * 10 + 'coarse geometry searching complete in' + eps_time_str + " +" * 10)

    coarse_ckpt_path = os.path.join(output_dir, args.expname, f'geometry_searching_last.tar')
    if mode == 'train':
        xyz_min_train, xyz_max_train = compute_bbox_by_coarse_geo(
            model_class=nerf, model_path=coarse_ckpt_path,
            thres=cfg.coarse_model.bbox_thres)
        logger.info('xyz_min and xyz_max:' + str(xyz_min_train) + str(xyz_max_train))

        # coarse detail reconstruction
        if args.coarse_training:
            eps = time.time()
            nerf_training(
                args=args, cfg=cfg, logger=logger,
                cfg_model=cfg.coarse_model,
                cfg_train=cfg.coarse_train,
                xyz_min=xyz_min_train, xyz_max=xyz_max_train,
                data_dict=data_dict,
                coarse_ckpt_path=coarse_ckpt_path,
                stage='coarse')
            eps = time.time() - eps
            eps_time_str = f'{eps // 3600:02.0f}:{eps // 60 % 60:02.0f}:{eps % 60:02.0f}'
            logger.info("+ " * 10 + 'train: coarse detail reconstruction in' + eps_time_str + " +" * 10)

        # fine detail reconstruction
        if args.fine_training:
            coarse_ckpt_path = os.path.join(output_dir, args.expname, f'coarse_last.tar')
            eps = time.time()
            nerf_training(
                args=args, cfg=cfg, logger=logger,
                cfg_model=cfg.fine_model,
                cfg_train=cfg.fine_train,
                xyz_min=xyz_min_train, xyz_max=xyz_max_train,
                data_dict=data_dict,
                coarse_ckpt_path=coarse_ckpt_path,
                stage='fine')
            eps = time.time() - eps
            eps_time_str = f'{eps // 3600:02.0f}:{eps // 60 % 60:02.0f}:{eps % 60:02.0f}'
            logger.info("+ " * 10 + 'train: fine detail reconstruction in' + eps_time_str + " +" * 10)

            eps_time = time.time() - eps_time
            eps_time_str = f'{eps_time // 3600:02.0f}:{eps_time // 60 % 60:02.0f}:{eps_time % 60:02.0f}'
            logger.info('train: finish (eps time' + eps_time_str + ')')

    if mode == 'eval':
        nerf_eval(
            args=args, cfg=cfg, logger=logger,
            cfg_model=cfg.coarse_model,
            data_dict=data_dict,
            stage='eval')
        eps_time = time.time() - eps_time
        eps_time_str = f'{eps_time // 3600:02.0f}:{eps_time // 60 % 60:02.0f}:{eps_time % 60:02.0f}'
        logger.info('eval: finish (eps time' + eps_time_str + ')')


def config_parser():
    # path and mode
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # path
    parser.add_argument('--config', type=str, default='./config.py')
    parser.add_argument('--expname', type=str, default='lego')
    parser.add_argument('--dataset_path', type=str, default='./datasets/nerf_synthetic/lego')
    parser.add_argument('--output_dir', type=str, default='./results')

    # mode and train
    parser.add_argument('--mode', type=str, default='train', help='train, eval')
    parser.add_argument('--dataset_type', type=str, default='blender', help='custom data use llff')
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument('--dvgo_init', default=False, help='use dvgo')
    parser.add_argument('--geometry_searching', default=False)
    parser.add_argument('--coarse_training', default=False, help='coarse training')
    parser.add_argument('--fine_training', default=False, help='fine training')
    parser.add_argument('--no_reload', action='store_true', help='do not reload ckpt')
    parser.add_argument('--no_reload_optimizer', default=False)
    parser.add_argument('--i_print', type=int, default=500, help='frequency of console printout and metric loggin')
    parser.add_argument("--i_validate", type=int, default=100000)
    parser.add_argument("--i_weights", type=int, default=10000, help='frequency of weight ckpt saving')
    parser.add_argument("--suffix", type=str, default="", help='suffix for exp name')
    parser.add_argument("--prefix", type=str, default="", help='prefix for exp name')

    # eval
    parser.add_argument('--scene', type=int, default=0)
    parser.add_argument('--only_mesh', action='store_true')
    parser.add_argument("--eval_ssim", default=True)
    parser.add_argument("--eval_lpips_alex", default=True)
    parser.add_argument("--eval_lpips_vgg", default=True)

    return parser


if __name__ == "__main__":

    parser = config_parser()
    args = parser.parse_args()

    # configuration
    cfg = mmcv.Config.fromfile(args.config)
    if args.expname:
        cfg.expname = args.expname
    if args.dataset_path:
        cfg.data.datadir = args.dataset_path
    if args.output_dir:
        cfg.basedir = args.output_dir
    if args.dataset_type:
        cfg.data.dataset_type = args.dataset_type

    os.makedirs(os.path.join(args.output_dir, args.expname), exist_ok=True)
    # shutil.copyfile(args.config, os.path.join(args.output_dir, args.expname, 'config.py'), )

    seed_everything()
    if args.mode == 'train':
        runner(args, cfg, mode='train')
    if args.mode == 'eval':
        runner(args, cfg, mode='eval')

