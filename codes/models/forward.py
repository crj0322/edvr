import torch
import torch.nn.functional as F


class PaddingForward(object):
    """Auto padding image size mod to a given size.

    Args:
        mod_size (int): Image size must mod to mod_size.
        scale (int): Super resolution scale.
    """

    def __init__(self, mod_size=4, scale=4):
        self._p_left = 0
        self._p_right = 0
        self._p_top = 0
        self._p_bottom = 0
        self._H = 0
        self._W = 0
        self._mod_size = mod_size
        self._scale = scale
        self._need_padding = False

    def _padding_border(self, H, W):
        self._H, self._W = H, W

        H_border = self._mod_size - (self._H % self._mod_size)
        W_border = self._mod_size - (self._W % self._mod_size)
        self._need_padding = False
        if H_border != self._mod_size:
            self._p_top = H_border//2
            self._p_bottom = H_border - self._p_top
            self._need_padding = True
        if W_border != self._mod_size:
            self._p_left = W_border//2
            self._p_right = W_border - self._p_left
            self._need_padding = True
    
    def _auto_padding(self, img):
        H, W = img.size()[-2:]
        if self._H != H or self._W != W:
            self._padding_border(H, W)

        if self._need_padding:
            img = F.pad(img, (self._p_left, self._p_right, self._p_top, self._p_bottom))
        return img

    def _auto_slice(self, img):
        if self._need_padding:
            h = self._scale * self._p_top
            H = self._scale * self._H
            w = self._scale * self._p_left
            W = self._scale * self._W
            img = img[..., h:h+H, w:w+W]
        return img

    def __call__(self, model, img):
        img = self._auto_padding(img)
        img = model(img)
        img = self._auto_slice(img)
        return img

class PatchedForward(object):
    """Auto split frames to patches of a given size.

    Args:
        patch_num (int): Split num of each dim.
        offset (int): Max border size of each patch.
        mod_size (int): patch size must mod to mod_size.
        scale (int): Super resolution scale.
    """

    def __init__(self, patch_num=3, offset=8, mod_size=4, scale=4):
        self._patch_num = patch_num
        self._offset = offset
        self._mod_size = mod_size
        self._scale = scale

    def __call__(self, model, frames):
        B, N, C, H, W = frames.size()

        step_w = W // self._patch_num
        step_h = H // self._patch_num

        sr_frame = torch.zeros((B, C, H * self._scale, W * self._scale), dtype=frames.dtype)
        if frames.is_cuda:
            sr_frame = sr_frame.cuda(frames.get_device())

        for indx_h in range(0, H, step_h):
            if indx_h > 0:
                offset_hl = self._offset - step_h % self._mod_size
            else:
                offset_hl = 0
            if indx_h < H:
                offset_hr = self._offset - (step_h + offset_hl) % self._mod_size
            else:
                offset_hr = 0
            for indx_w in range(0, W, step_w):
                if indx_w > 0:
                    offset_wl = self._offset - step_w % self._mod_size
                else:
                    offset_wl = 0
                if indx_w < H:
                    offset_wr = self._offset - (step_w + offset_wl) % self._mod_size
                else:
                    offset_wr = 0

                frame_patchs = frames[..., indx_h - offset_hl: indx_h + step_h + offset_hr,
                            indx_w - offset_wl: indx_w + step_w + offset_wr]

                # predict
                # print('frame_patchs h w: {} {}'.format(*frame_patchs.size()[-2:]))
                output = model(frame_patchs)
                sr_frame[..., indx_h * self._scale: (indx_h + step_h) * self._scale, indx_w * self._scale: (indx_w + step_w) * self._scale] = \
                    output[..., offset_hl * self._scale: (step_h + offset_hl) * 4, (offset_wl) * self._scale: (step_w + offset_wl) * self._scale]

        return sr_frame

