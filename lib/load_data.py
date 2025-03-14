import numpy as np
import os

from lib.load_blender import load_blender_data
from lib.load_nsvf import load_nsvf_data
from lib.load_blendedmvs import load_blendedmvs_data
from lib.load_tankstemple import load_tankstemple_data, load_tankstemple_data_bound
from lib.load_nerfpp import load_nerfpp_data
from lib.load_deepvoxels import load_dv_data
from lib.load_dtu import load_dtu_data
from lib.load_volsdf_bmvs import load_vbmvs_data
from lib.load_co3d import load_co3d_data
from lib.load_scannet import load_scannet_data
from lib.load_llff import load_llff_data
from lib.load_mobilebrick import load_mobilebrick_data
from lib.load_ILSH import load_ILSH_data

def load_data(args, reso_level=2, train_all=True, wmask=True, white_bg=True):
    #print("[ resolution level {} | train all {} | wmask {} | white_bg {}]".format(reso_level, train_all, wmask, white_bg))
    print("[ resolution level {} | wmask {} | white_bg {}]".format(reso_level, wmask, white_bg))
    K, depths = None, None
    scale_mats_np = None
    masks = None
    if args.dataset_type == 'llff':
        images, depths, poses, bds, render_poses, i_test = load_llff_data(
            args.datadir, args.factor,
            recenter=True, bd_factor=1,
            spherify=args.spherify,
            load_depths=args.load_depths)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])
        print('DEFINING BOUNDS')
        if args.ndc:
            near = 0.
            far = 1.
        else:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near, far = 2., 6.

        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]*images[...,-1:]

    elif args.dataset_type == 'blendedmvs':
        images, poses, render_poses, hwf, K, i_split = load_blendedmvs_data(args.datadir)
        print('Loaded blendedmvs', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near, far = inward_nearfar_heuristic(poses[i_train, :3, 3])

        assert images.shape[-1] == 3
    elif args.dataset_type == 'dtu':
        images, poses, render_poses, hwf, K, i_split, scale_mats_np, masks = load_dtu_data(args.datadir, reso_level=reso_level, mask=wmask, white_bg=white_bg)
        print('Loaded dtu', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        if train_all:
            i_train = np.arange(int(images.shape[0]))

        near, far = inward_nearfar_heuristic(poses[i_train, :3, 3])

        assert images.shape[-1] == 3
    elif args.dataset_type == 'scannet':

        images, poses, render_poses, hwf, K, i_split, scale_mats_np, masks, depths, normals = load_scannet_data(args.datadir)
        print('Loaded scannet', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        if train_all:
            i_train = np.arange(int(images.shape[0]))

        near, far = inward_nearfar_heuristic(poses[i_train, :3, 3])

        assert images.shape[-1] == 3
    elif args.dataset_type == 'volsdf_bmvs':
        images, poses, render_poses, hwf, K, i_split = load_vbmvs_data(args.datadir)
        print('Loaded dtu', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near, far = inward_nearfar_heuristic(poses[i_train, :3, 3])

        assert images.shape[-1] == 3

    elif args.dataset_type == 'tankstemple':
        images, poses, render_poses, hwf, K, i_split = load_tankstemple_data_bound(
            args.datadir, movie_render_kwargs=args.movie_render_kwargs)
        print('Loaded tankstemple', images.shape, render_poses.shape, hwf,
              args.datadir)
        i_train, i_val, i_test = i_split

        near, far = inward_nearfar_heuristic(poses[i_train, :3, 3], ratio=0)

        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[..., :3] * images[..., -1:] + (
                            1. - images[..., -1:])
            else:
                images = images[..., :3] * images[..., -1:]

    elif args.dataset_type == 'nsvf':
        images, poses, render_poses, hwf, i_split = load_nsvf_data(args.datadir)
        print('Loaded nsvf', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near, far = inward_nearfar_heuristic(poses[i_train, :3, 3])

        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]*images[...,-1:]

    elif args.dataset_type == 'deepvoxels':
        args.scene = args.datadir.split(os.sep)[-1]
        args.datadir = os.path.join(*args.datadir.split(os.sep)[:-1])
        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.scene, basedir=args.datadir, testskip=args.testskip)
        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R - 1
        far = hemi_R + 1
        assert args.white_bkgd
        assert images.shape[-1] == 3

    elif args.dataset_type == 'co3d':
        # each image can be in different shapes and intrinsics
        images, masks, poses, render_poses, hwf, K, i_split = load_co3d_data(args)
        print('Loaded co3d', args.datadir, args.annot_path, args.sequence_name)
        i_train, i_val, i_test = i_split

        near, far = inward_nearfar_heuristic(poses[i_train, :3, 3], ratio=0)

        for i in range(len(images)):
            if args.white_bkgd:
                images[i] = images[i] * masks[i][...,None] + (1.-masks[i][...,None])
            else:
                images[i] = images[i] * masks[i][...,None]

    elif args.dataset_type == 'nerfpp':

        images, poses, render_poses, hwf, K, i_split = load_nerfpp_data(args.datadir)
        print('Loaded nerf_pp', images.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split
        near_clip, far = inward_nearfar_heuristic(poses[i_train, :3, 3], ratio=0.02)
        near = 0

    elif args.dataset_type == "mobile_brick":
        images, poses, render_poses, hwf, K, i_split, scale_mats_np, masks = load_mobilebrick_data(args.datadir, reso_level=reso_level, mask=wmask, white_bg=white_bg)
        print('Loaded mobile_brick', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        if train_all:
            i_train = np.arange(int(images.shape[0]))

        near, far = inward_nearfar_heuristic(poses[i_train, :3, 3])

        assert images.shape[-1] == 3

    elif args.dataset_type == 'ILSH':
        images, depths, poses, bds, render_poses, i_test, masks = load_ILSH_data(
            args.datadir, args.factor,
            recenter=True, bd_factor=1,
            spherify=args.spherify,
            load_depths=args.load_depths)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])
        print('DEFINING BOUNDS')
        if args.ndc:
            near = 0.
            far = 1.
        else:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        print('NEAR FAR', near, far)

    else:
        raise NotImplementedError(f'Unknown dataset type {args.dataset_type} exiting')

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    HW = np.array([im.shape[:2] for im in images])
    irregular_shape = (images.dtype is np.dtype('object'))

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if len(K.shape) == 2:
        Ks = K[None].repeat(len(poses), axis=0)
    else:
        Ks = K

    render_poses = render_poses[...,:4]
    #print("Split: train {} | validate {} | test {}".format(len(i_train), len(i_val), len(i_test)))
    print('near, far: ', near, far)
    if wmask and masks is None:
        masks = images.mean(-1) > 0

    data_dict = dict(
        hwf=hwf, HW=HW, Ks=Ks, near=near, far=far,
        i_train=i_train, i_val=i_val, i_test=i_test,
        poses=poses, render_poses=render_poses,
        images=images, depths=depths,
        irregular_shape=irregular_shape,
        scale_mats_np=scale_mats_np,
        masks=masks
    )
    if args.dataset_type == 'scannet':
        data_dict['depths'] = depths
        data_dict['normals'] = normals
    return data_dict


def inward_nearfar_heuristic(cam_o, ratio=0.05):
    dist = np.linalg.norm(cam_o[:,None] - cam_o, axis=-1)
    far = dist.max()
    near = far * ratio
    return near, far

