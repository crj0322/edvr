'''
youku dataset
'''
import os
import os.path as osp
import random
import logging
import numpy as np
import pickle
import lmdb
import torch
import torch.utils.data as data
import data.util as util

logger = logging.getLogger('base')


class YOUKUDataset(data.Dataset):
    '''
    Reading the training youku dataset
    key example: 00000_000000
    GT: Ground-Truth;
    LQ: Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames
    support reading N LQ frames, N = 1, 3, 5, 7
    '''

    def __init__(self, opt):
        super(YOUKUDataset, self).__init__()
        self.opt = opt
        # temporal augmentation
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        logger.info('Temporal augmentation interval list: [{}], with random reverse is {}.'.format(
            ','.join(str(x) for x in opt['interval_list']), self.random_reverse))

        self.half_N_frames = opt['N_frames'] // 2
        self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        if opt['down_sample']:
            self.x2GT_root, self.x2LQ_root = opt['dataroot_x2GT'], opt['dataroot_x2LQ']
        self.data_type = self.opt['data_type']
        self.LR_input = False if opt['GT_size'] == opt['LQ_size'] else True  # low resolution inputs
        #### directly load image keys
        self.paths_GT, _ = util.get_image_paths(self.data_type, opt['dataroot_GT'])
        assert self.paths_GT, 'Error: GT path is empty.'

        if self.data_type == 'lmdb':
            self._init_lmdb()
        elif self.data_type == 'img' or self.data_type == 'pkl':
            pass
        else:
            raise ValueError('Wrong data type: {}'.format(self.data_type))

        # data name_a with different resolution
        self.df_res_list = ['00031', '00044', '00054', '00101', '00121', '00142', '00177']

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                meminit=False)

    def __getitem__(self, index):
        # downsample augmentation
        if self.opt['down_sample'] and random.random() < 0.5:
            down_scale = 2
            GT_root = self.x2GT_root
            LQ_root = self.x2LQ_root
        else:
            down_scale = 1
            GT_root = self.GT_root
            LQ_root = self.LQ_root
        
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']
        key = self.paths_GT[index]
        if self.data_type == 'img' or self.data_type == 'pkl':
            name_a, name_b = key.split(os.sep)[-2:]
            name_b = name_b.split('.')[0]
        else:
            name_a, name_b = key.split('_')
        center_frame_idx = int(name_b)

        #### determine the neighbor frames
        interval = random.choice(self.interval_list)
        if self.opt['border_mode']:
            direction = 1  # 1: forward; 0: backward
            N_frames = self.opt['N_frames']
            if self.random_reverse and random.random() < 0.5:
                direction = random.choice([0, 1])
            if center_frame_idx + interval * (N_frames - 1) > 99:
                direction = 0
            elif center_frame_idx - interval * (N_frames - 1) < 0:
                direction = 1
            # get the neighbor list
            if direction == 1:
                neighbor_list = list(
                    range(center_frame_idx, center_frame_idx + interval * N_frames, interval))
            else:
                neighbor_list = list(
                    range(center_frame_idx, center_frame_idx - interval * N_frames, -interval))
            name_b = '{:06d}'.format(neighbor_list[0])
        else:
            # ensure not exceeding the borders
            while (center_frame_idx + self.half_N_frames * interval >
                   99) or (center_frame_idx - self.half_N_frames * interval < 0):
                center_frame_idx = random.randint(0, 99)
            # get the neighbor list
            neighbor_list = list(
                range(center_frame_idx - self.half_N_frames * interval,
                      center_frame_idx + self.half_N_frames * interval + 1, interval))
            if self.random_reverse and random.random() < 0.5:
                neighbor_list.reverse()
            name_b = '{:06d}'.format(neighbor_list[self.half_N_frames])
        key = '_'.join([name_a, name_b])

        assert len(
            neighbor_list) == self.opt['N_frames'], 'Wrong length of neighbor list: {}'.format(
                len(neighbor_list))

        #### get the GT image (as the center frame)
        if down_scale > 1:
            if name_a in self.df_res_list:
                GT_size_tuple = (3, 576, 1024)
            else:
                GT_size_tuple = (3, 540, 960)
        else:
            if self.data_type == 'pkl': # self.data_type == 'lmdb'
                # binary data is half croped because of driver capacity
                GT_size_tuple = (3, 540, 960)
            elif name_a in self.df_res_list:
                GT_size_tuple = (3, 1152, 2048)
            else:
                GT_size_tuple = (3, 1080, 1920)
        
        if self.data_type == 'lmdb':
            img_GT = util.read_img(self.GT_env, key, GT_size_tuple)
        elif self.data_type == 'pkl':
             with open(osp.join(GT_root, name_a, name_b + '.pkl'), 'rb') as _f:
                img_GT = pickle.load(_f).astype(np.float32) / 255.
        else:
            img_GT = util.read_img(None, osp.join(GT_root, name_a, name_b + '.png'))

        #### get LQ images
        LQ_size_tuple = (3, GT_size_tuple[1]//self.opt['scale'], GT_size_tuple[2]//self.opt['scale']) if self.LR_input else GT_size_tuple
        img_LQ_l = []
        for v in neighbor_list:
            if self.data_type == 'lmdb':
                img_LQ = util.read_img(self.LQ_env, '{}_{:06d}'.format(name_a, v), LQ_size_tuple)
            elif self.data_type == 'pkl':
                img_LQ_path = osp.join(LQ_root, name_a, '{:06d}.pkl'.format(v))
                with open(img_LQ_path, 'rb') as _f:
                    img_LQ = pickle.load(_f).astype(np.float32) / 255.
            else:
                img_LQ_path = osp.join(LQ_root, name_a, '{:06d}.png'.format(v))
                img_LQ = util.read_img(None, img_LQ_path)
            img_LQ_l.append(img_LQ)

        if self.opt['phase'] == 'train':
            C, H, W = LQ_size_tuple  # LQ size todo
            # randomly crop
            if self.LR_input:
                LQ_size = GT_size // scale
                rnd_h = random.randint(0, max(0, H - LQ_size))
                rnd_w = random.randint(0, max(0, W - LQ_size))
                img_LQ_l = [v[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :] for v in img_LQ_l]
                rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
                img_GT = img_GT[rnd_h_HR:rnd_h_HR + GT_size, rnd_w_HR:rnd_w_HR + GT_size, :]
            else:
                rnd_h = random.randint(0, max(0, H - GT_size))
                rnd_w = random.randint(0, max(0, W - GT_size))
                img_LQ_l = [v[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :] for v in img_LQ_l]
                img_GT = img_GT[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :]

            # augmentation - flip, rotate
            img_LQ_l.append(img_GT)
            rlt = util.augment(img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])
            img_LQ_l = rlt[0:-1]
            img_GT = rlt[-1]

        # stack LQ images to NHWC, N is the frame number
        img_LQs = np.stack(img_LQ_l, axis=0)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_GT = img_GT[:, :, [2, 1, 0]]
        img_LQs = img_LQs[:, :, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs,
                                                                     (0, 3, 1, 2)))).float()
        return {'LQs': img_LQs, 'GT': img_GT, 'key': key}

    def __len__(self):
        return len(self.paths_GT)
