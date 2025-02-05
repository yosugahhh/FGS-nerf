import torch

from lib.load_data import load_data


def load_dataset(cfg):
    '''Load images / poses / camera settings / data split.
    '''
    mode = getattr(cfg.data, 'mode', dict())
    if 'reso_level' in cfg:
        mode.update(reso_level=cfg.reso_level)
    data_dict = load_data(cfg.data, **mode, white_bg=cfg.data.white_bkgd)

    # remove useless field
    kept_keys = {
        'hwf', 'HW', 'Ks', 'near', 'far',
        'i_train', 'i_val', 'i_test', 'irregular_shape',
        'poses', 'render_poses', 'images', 'scale_mats_np', 'masks'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu').cuda() for im in data_dict['images']]
        data_dict['masks'] = [torch.FloatTensor(im, device='cpu').cuda() for im in data_dict['masks']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu').cuda()
        data_dict['masks'] = torch.FloatTensor(data_dict['masks'], device='cpu').cuda()
    data_dict['poses'] = torch.Tensor(data_dict['poses']).cuda()

    return data_dict
