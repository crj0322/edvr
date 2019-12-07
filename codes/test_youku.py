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


def main():
    #################
    # configurations
    #################
    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    data_mode = 'youku'
    flip_test = False
    ############################################################################
    #### model
    model_path = '../experiments/pretrained_models/EDVR_YOUKU_M_woTSA.pth'
    N_in = 5

    predeblur, HR_in = False, False
    back_RBs = 10
    
    model = EDVR_arch.EDVR(64, N_in, 8, 5, 10, predeblur=predeblur, HR_in=HR_in, w_TSA=False, block_type='rcab')
    calc_ssim = True
    if calc_ssim:
        ssim_fn = SSIM()

    #### dataset
    test_dataset_folder = '/media/tclwh2/public/youku/val/lq'
    GT_dataset_folder = '/media/tclwh2/public/youku/val/gt'

    #### evaluation
    crop_border = 0
    border_frame = N_in // 2  # border frames when evaluate
    # temporal padding mode
    padding = 'new_info'
    save_imgs = True

    save_folder = '../results/{}'.format(data_mode)
    util.mkdirs(save_folder)
    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')

    #### log info
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip test: {}'.format(flip_test))

    #### set up the models
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)
    
    if calc_ssim:
        ssim_fn = ssim_fn.to(device)
        ssim_fn.eval()
        avg_ssim_l, avg_ssim_center_l, avg_ssim_border_l = [], [], []

    avg_psnr_l, avg_psnr_center_l, avg_psnr_border_l = [], [], []
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

        # process resolution
        mod_scale = 4
        h, w = imgs_LQ.shape[2:]
        if (h % mod_scale) != 0 or (w % mod_scale) != 0:
            crop_h = h - (h % mod_scale)
            crop_w = w - (w % mod_scale)
            imgs_LQ = imgs_LQ[:, :, :crop_h, :crop_w]
            img_GT_l = [gt[:4 * crop_h, :4 * crop_w, :] for gt in img_GT_l]

        avg_psnr, avg_psnr_border, avg_psnr_center, N_border, N_center = 0, 0, 0, 0, 0
        if calc_ssim:
            avg_ssim, avg_ssim_border, avg_ssim_center = 0, 0, 0

        # process each image
        for img_idx, img_path in enumerate(img_path_l):
            img_name = osp.splitext(osp.basename(img_path))[0]
            select_idx = data_util.index_generation(img_idx, max_idx, N_in, padding=padding)
            imgs_in = imgs_LQ.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)

            if flip_test:
                output = util.flipx4_forward(model, imgs_in)
            elif calc_ssim:
                with torch.no_grad():
                    model_output = model(imgs_in)
                    GT = np.copy(img_GT_l[img_idx][:, :, [2, 1, 0]])
                    GT = torch.from_numpy(GT.transpose(2, 0, 1)).unsqueeze_(0).to(device)
                    crt_ssim = ssim_fn(model_output, GT).data.cpu().item()
                    output = model_output.data.float().cpu()
            else:
                output = util.single_forward(model, imgs_in)
            output = util.tensor2img(output.squeeze(0))

            if save_imgs:
                cv2.imwrite(osp.join(save_subfolder, '{}.png'.format(img_name)), output)

            # calculate PSNR
            output = output / 255.
            GT = np.copy(img_GT_l[img_idx])

            output, GT = util.crop_border([output, GT], crop_border)
            crt_psnr = util.calculate_psnr(output * 255, GT * 255)
            logger.info('{:3d} - {:25} \tPSNR: {:.6f} dB'.format(img_idx + 1, img_name, crt_psnr))
            if calc_ssim:
                logger.info('{:3d} - {:25} \tSSIM: {:.6f}'.format(img_idx + 1, img_name, crt_ssim))

            if img_idx >= border_frame and img_idx < max_idx - border_frame:  # center frames
                avg_psnr_center += crt_psnr
                N_center += 1
                if calc_ssim:
                    avg_ssim_center += crt_ssim
            else:  # border frames
                avg_psnr_border += crt_psnr
                N_border += 1
                if calc_ssim:
                    avg_ssim_border += crt_ssim

        avg_psnr = (avg_psnr_center + avg_psnr_border) / (N_center + N_border)
        avg_psnr_center = avg_psnr_center / N_center
        avg_psnr_border = 0 if N_border == 0 else avg_psnr_border / N_border
        avg_psnr_l.append(avg_psnr)
        avg_psnr_center_l.append(avg_psnr_center)
        avg_psnr_border_l.append(avg_psnr_border)

        logger.info('Folder {} - Average PSNR: {:.6f} dB for {} frames; '
                    'Center PSNR: {:.6f} dB for {} frames; '
                    'Border PSNR: {:.6f} dB for {} frames.'.format(subfolder_name, avg_psnr,
                                                                   (N_center + N_border),
                                                                   avg_psnr_center, N_center,
                                                                   avg_psnr_border, N_border))

        if calc_ssim:
            avg_ssim = (avg_ssim_center + avg_ssim_border) / (N_center + N_border)
            avg_ssim_center = avg_ssim_center / N_center
            avg_ssim_border = 0 if N_border == 0 else avg_ssim_border / N_border
            avg_ssim_l.append(avg_ssim)
            avg_ssim_center_l.append(avg_ssim_center)
            avg_ssim_border_l.append(avg_ssim_border)

            logger.info('Folder {} - Average SSIM: {:.6f} for {} frames; '
                        'Center SSIM: {:.6f} for {} frames; '
                        'Border SSIM: {:.6f} for {} frames.'.format(subfolder_name, avg_ssim,
                                                                    (N_center + N_border),
                                                                    avg_ssim_center, N_center,
                                                                    avg_ssim_border, N_border))

    logger.info('################ Tidy Outputs ################')
    for subfolder_name, psnr, psnr_center, psnr_border in zip(subfolder_name_l, avg_psnr_l,
                                                              avg_psnr_center_l, avg_psnr_border_l):
        logger.info('Folder {} - Average PSNR: {:.6f} dB. '
                    'Center PSNR: {:.6f} dB. '
                    'Border PSNR: {:.6f} dB.'.format(subfolder_name, psnr, psnr_center,
                                                     psnr_border))

    if calc_ssim:
        for subfolder_name, ssim, ssim_center, ssim_border in zip(subfolder_name_l, avg_ssim_l,
                                                              avg_ssim_center_l, avg_ssim_border_l):
            logger.info('Folder {} - Average SSIM: {:.6f}. '
                        'Center SSIM: {:.6f}. '
                        'Border SSIM: {:.6f}.'.format(subfolder_name, ssim, ssim_center,
                                                        ssim_border))

    logger.info('################ Final Results ################')
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip test: {}'.format(flip_test))
    logger.info('Total Average PSNR: {:.6f} dB for {} clips. '
                'Center PSNR: {:.6f} dB. Border PSNR: {:.6f} dB.'.format(
                    sum(avg_psnr_l) / len(avg_psnr_l), len(subfolder_l),
                    sum(avg_psnr_center_l) / len(avg_psnr_center_l),
                    sum(avg_psnr_border_l) / len(avg_psnr_border_l)))

    if calc_ssim:
        logger.info('Total Average SSIM: {:.6f} for {} clips. '
                'Center SSIM: {:.6f}. Border SSIM: {:.6f}.'.format(
                    sum(avg_ssim_l) / len(avg_ssim_l), len(subfolder_l),
                    sum(avg_ssim_center_l) / len(avg_ssim_center_l),
                    sum(avg_ssim_border_l) / len(avg_ssim_border_l)))


if __name__ == '__main__':
    main()
