import os
import os.path as osp
import pickle
import cv2
import sys
import random

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import utils.util as util
import data.util as data_util


def transform(mode, crop):
    scale = 4
    root = '/media/tclwh2/public/youku/train/' + mode
    root_name = root.split('/')[-1]
    if crop:
        output_root_name = root_name + '_crop_pickle'
    else:
        output_root_name = root_name + '_pickle'

    # generate random roi
    if crop:
        if mode == 'lq':
            H_dst, W_dst = 135, 240
            # def_res_l = ['00031', '00044', '00054', '00101', '00121', '00142', '00177']
            # hw_l = []
            # for folder in sorted(os.listdir(train_folder)):
            #     if folder in def_res_l:
            #         H, W = 288, 512
            #     else:
            #         H, W = 270, 480
            #     rnd_h = random.randint(0, max(0, H - H_dst))
            #     rnd_w = random.randint(0, max(0, W - W_dst))
            #     hw_l.append((rnd_h, rnd_w))

            # with open('./hw.pkl', 'wb') as f:
            #     pickle.dump(hw_l, f)
        else:
            H_dst, W_dst = 540, 960
        with open('./hw.pkl', 'rb') as f:
            hw_l = pickle.load(f)

    #### read all the image paths to a list
    print('Reading image path list ...')
    all_img_list = data_util._get_paths_from_images(root)

    pbar = util.ProgressBar(len(all_img_list))
    for idx, img_path in enumerate(all_img_list):
        file_name = osp.basename(img_path)
        out_folder = osp.dirname(img_path).replace(root_name, output_root_name)
        out_file = osp.join(out_folder, file_name)
        util.mkdir(out_folder)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        # crop
        if crop:
            H, W, C = img.shape
            rnd_h, rnd_w = hw_l[idx//100]
            if mode == 'gt':
                rnd_h, rnd_w = int(rnd_h * scale), int(rnd_w * scale)
            img = img[rnd_h:rnd_h + H_dst, rnd_w:rnd_w + W_dst, :].copy()

        pkl_path = os.path.join(out_folder, file_name.replace('.png', '.pkl'))
        pbar.update('Write {}'.format(pkl_path))
        with open(pkl_path, 'wb') as f:
            pickle.dump(img, f)

    print('Finished')


def test_dataset(data_root):
    all_img_list = data_util._get_paths_from_images(data_root)
    for i in range(10):
        idx = random.randint(0, len(all_img_list))
        with open(all_img_list[idx], 'rb') as f:
            img = pickle.load(f)

        cv2.imwrite('test/{:03}.png'.format(i), img)


if __name__ == '__main__':
    transform('x2gt', crop=False)
    # test_dataset('/media/tclwh2/public/youku/train/gt_crop_pickle')
