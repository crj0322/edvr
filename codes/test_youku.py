'''
Test youku (SR) datasets
'''

import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch

import utils.util as util
import data.util as data_util
import models.archs.EDVR_arch as EDVR_arch
from models.ssim import SSIM
from models.forward import PaddingForward, PatchedForward


def main():
    #################
    # configurations
    #################
    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    data_mode = 'youku'
    split_test = True

    if split_test:
        forward_fn = PatchedForward(3, 8, 4, 4)
    else:
        forward_fn = PaddingForward(4, 4)

    ############################################################################
    #### model
    model_path = '../experiments/pretrained_models/EDVR_YOUKU_M_woTSA.pth'
    N_in = 5

    predeblur, HR_in = False, False
    back_RBs = 10
    
    model = EDVR_arch.EDVR(64, N_in, 8, 5, 10, predeblur=predeblur, HR_in=HR_in, w_TSA=False, block_type='rcab', non_local=False)
    ssim_fn = SSIM()

    #### dataset
    test_dataset_folder = '/media/tclwh2/public/youku/val/lq'
    GT_dataset_folder = '/media/tclwh2/public/youku/val/gt'

    #### evaluation
    border_frame = N_in // 2  # border frames when evaluate
    # temporal padding mode
    padding = 'new_info'
    save_imgs = False

    save_folder = '../results/{}'.format(data_mode)
    util.mkdirs(save_folder)
    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')

    #### log info
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Split test: {}'.format(split_test))

    #### set up the models
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)
    
    ssim_fn = ssim_fn.to(device)
    ssim_fn.eval()

    avg_psnr_l, avg_ssim_l= [], []
    subfolder_name_l = []

    subfolder_l = sorted(glob.glob(osp.join(test_dataset_folder, '*')))
    subfolder_GT_l = sorted(glob.glob(osp.join(GT_dataset_folder, '*')))
    # for each subfolder
    for subfolder, subfolder_GT in zip(subfolder_l, subfolder_GT_l):
        subfolder_name = osp.basename(subfolder)
        subfolder_name_l.append(subfolder_name)
        save_subfolder = osp.join(save_folder, subfolder_name)

        img_path_l = sorted(glob.glob(osp.join(subfolder, '*')))
        max_idx = len(img_path_l)
        if save_imgs:
            util.mkdirs(save_subfolder)

        #### read LQ and GT images
        imgs_LQ = data_util.read_img_seq(subfolder)
        img_GT_l = []
        for img_GT_path in sorted(glob.glob(osp.join(subfolder_GT, '*'))):
            img_GT_l.append(data_util.read_img(None, img_GT_path))

        avg_psnr = 0
        avg_ssim = 0

        # process each image
        for img_idx, img_path in enumerate(img_path_l):
            img_name = osp.splitext(osp.basename(img_path))[0]
            select_idx = data_util.index_generation(img_idx, max_idx, N_in, padding=padding)
            imgs_in = imgs_LQ.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)

            with torch.no_grad():
                model_output = forward_fn(model, imgs_in)
                GT = np.copy(img_GT_l[img_idx][:, :, [2, 1, 0]])
                GT = torch.from_numpy(GT.transpose(2, 0, 1)).unsqueeze_(0).to(device)
                crt_ssim = ssim_fn(model_output, GT).data.cpu().item()
                output = model_output.data.float().cpu()
                output = util.tensor2img(output.squeeze(0))

            if save_imgs:
                cv2.imwrite(osp.join(save_subfolder, '{}.png'.format(img_name)), output)

            # calculate PSNR
            output = output / 255.
            GT = np.copy(img_GT_l[img_idx])
            crt_psnr = util.calculate_psnr(output * 255, GT * 255)
            logger.info('{:3d} - {:25} \tPSNR/SSIM: {:.6f}/{:.6f}'.format(img_idx + 1, img_name, crt_psnr, crt_ssim))

            avg_psnr += crt_psnr
            avg_ssim += crt_ssim

        avg_psnr /= max_idx
        avg_ssim /= max_idx
        avg_psnr_l.append(avg_psnr)
        avg_ssim_l.append(avg_ssim)

        logger.info('Folder {} - Average PSNR/SSIM: {:.6f}/{:.6f} for {} frames. '
                    .format(subfolder_name, avg_psnr, avg_ssim, max_idx))

    logger.info('################ Tidy Outputs ################')
    for subfolder_name, psnr, ssim in zip(subfolder_name_l, avg_psnr_l, avg_ssim_l):
        logger.info('Folder {} - Average PSNR/SSIM: {:.6f}/{:.6f}. '
                    .format(subfolder_name, psnr, ssim))

    logger.info('################ Final Results ################')
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Total Average PSNR/SSIM: {:.6f}/{:.6f} for {} clips. '
                .format(sum(avg_psnr_l) / len(avg_psnr_l), sum(avg_ssim_l) / len(avg_ssim_l), len(subfolder_l)))


if __name__ == '__main__':
    main()
