import os
import argparse
from rembg import remove
import cv2

from lib.preprocess.convert_cameras import _load_colmap
from lib.preprocess.preprocess_cameras import *
from lib.preprocess.convert_cameras import *
from lib.preprocess.process_video import *
from lib.preprocess.colmap_poses.pose_utils import *


def config_parser():
    # path and mode
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--custom_dataset_path', type=str, default='./datasets/ref_real/gardenspheres')
    parser.add_argument('--output_path', type=str, default='./datasets/')
    parser.add_argument('--run_mode', type=str, default='images', help='images or video')
    parser.add_argument("--colmap_camera_model", default="OPENCV",
                        choices=["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV",
                                 "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "OPENCV_FISHEYE"], help="Camera model")

    # video mode
    parser.add_argument('--video_mode', type=str, default='get_frames')
    parser.add_argument('--video_source_dir', type=str, help='data source folder for preprocess')
    parser.add_argument('--video_path', type=str, help='video to process')
    parser.add_argument('--video_img_folder', type=str, default='images')
    parser.add_argument('--video_rmbg_img_folder', type=str, default='image_rmbg')
    parser.add_argument('--video_interval', type=int, default=10)
    parser.add_argument('--video_white_bg', action='store_true')

    # preprocess_cameras
    parser.add_argument('--dtu', default=False, action="store_true", help='If set, apply preprocess to all DTU scenes.')
    parser.add_argument('--root', default=None, help='If set, apply preprocess to all DTU scenes.')
    parser.add_argument('--use_linear_init', default=False, action="store_true",
                        help='If set, preprocess for linear init cameras.')

    return parser


def video(args):
    root = args.video_source_dir
    images_ori_path = os.path.join(root, args.video_img_folder)
    images_out_path = os.path.join(root, args.video_rmbg_img_folder)
    masks_out_path = os.path.join(root, 'mask')

    if args.video_mode == 'get_frames':
        get_frames(args.video_path, images_ori_path, interval=args.video_interval)
    elif args.video_mode == 'get_masks':
        add_white_bg(images_out_path, masks_out_path, args.video_white_bg)
    else:
        raise NameError


def remove_bg(args):
    images_path = os.path.join(args.custom_dataset_path, 'images')
    output_images = os.path.join(args.output_path, 'mask')
    if not os.path.exists(output_images):
        os.makedirs(output_images)

    for image in os.listdir(images_path):
        image_name = image
        image = os.path.join(images_path, image)
        image = remove(cv2.imread(image), only_mask=True)
        cv2.imwrite(os.path.join(output_images, image_name), image)

    print('Done with rembg')


def run_colmap(args):
    gen_poses(basedir=args.custom_dataset_path, match_type='exhaustive_matcher', factors=None, colmap_camera_model=args.colmap_camera_model)


def convert_cameras(args):
    _load_colmap(args.custom_dataset_path, True)


def preprocess_cameras(args):
    # see preprocess_cameras.py
    root = '/mnt/SSD/nerf_data/CO3D_data/apple'
    if args.root is not None:
        files = os.listdir(root)
        all_failed = []
        for file in files:
            path = os.path.join(root, file)
            if not os.path.isdir(path) or file == '/':
                continue
            try:
                get_normalization(path, False)
            except:
                print("failed for {}".format(path))
                all_failed.append(path.split("/")[-1])
        print(len(all_failed))
        import ipdb; ipdb.set_trace()
    if args.dtu:
        source_dir = '../data/DTU'
        scene_dirs = sorted(glob(os.path.join(source_dir, "scan*")))
        for scene_dir in scene_dirs:
            get_normalization(scene_dir, args.use_linear_init)
    else:
        get_normalization(args.custom_dataset_path, args.use_linear_init)

    print('preprocess_cameras Done!')


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    if args.run_mode == 'video':
        args.video_source_dir = args.custom_dataset_path
        while not args.video_path:
            print('please input video path')

        args.video_mode = 'get_frames'
        # get frames and remove background
        video(args)
        # get masks and run colmap
        print('Mask generating...')
        remove_bg(args)
        print('Mask generation complete, run colmap...')
        run_colmap(args)
        convert_cameras(args)
        preprocess_cameras(args)

    if args.run_mode == 'images':
        # get masks and run colmap
        print('Mask generating...')
        remove_bg(args)
        print('Mask generation complete, run colmap...')
        # run_colmap(args)
        # convert_cameras(args)
        # preprocess_cameras(args)

    print('Dataset preprocess complete.')
