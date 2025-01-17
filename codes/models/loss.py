import torch
import torch.nn as nn
from .ssim import SSIM


class CombinedLoss(nn.Module):
    """Combined Charbonnier Loss (L1) and SSIM Loss"""

    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.cb_loss = CharbonnierLoss(sum_mode=False)
        self.ssim_loss = SSIM()

    def forward(self, x, y):
        loss = 1 - self.ssim_loss(x, y) + self.cb_loss(x, y)
        return loss


class TopkCharbonnierLoss(nn.Module):
    """Topk Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6, keep_ratio=0.5):
        super(TopkCharbonnierLoss, self).__init__()
        self.eps = eps
        self.keep_ratio = keep_ratio

    def forward(self, x, y):
        B = x.size(0)
        k = int(B * self.keep_ratio)
        if k < 1:
            k = 1
        
        diff = x - y
        l1_diff = torch.sqrt(diff * diff + self.eps)
        batch_loss = torch.sum(l1_diff, dim=[1, 2, 3])
        loss, _ = torch.topk(batch_loss, k, dim=0, sorted=False)
        loss = loss.sum()
        
        return loss


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, sum_mode=True, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.sum_mode = sum_mode
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        if self.sum_mode:
            loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        else:
            loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss


# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp,
                                          grad_outputs=grad_outputs, create_graph=True,
                                          retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss
