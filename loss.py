from math import exp
from torch import nn as nn
from torch.nn import functional as F
import torch
from einops import rearrange

class Fusion_loss(nn.Module):
    def __init__(self):
        super(Fusion_loss, self).__init__()        
        print('Building Fusion_loss() as loss function ~')
        self.loss_func_ssim = L_SSIM(window_size=13)
        self.loss_func_grad = L_Gradient_Max()
        self.loss_func_max = L_Intensity_Max()
        self.loss_func_consist = L_Intensity_Consist()
        self.loss_func_color = L_Color()
    
    def cos_consist(self, img_f, img_ir):
        num_frames = img_f.shape[1]
        diff_loss = 0
        for t in range(1, num_frames):
            output_diff = img_f[:,t] - img_f[:,t - 1]
            input_diff = img_ir[:,t] - img_ir[:,t - 1]
            # 计算余弦相似性
            cosine_sim = F.cosine_similarity(output_diff, input_diff, dim=1)
            diff_loss += (1 - cosine_sim).mean()  # 1 - 余弦相似性
        diff_loss /= (num_frames - 1)
        return diff_loss

    def forward(self, img_f, img_ir, img_vi, int_weight=10, consist_weight=1, ssim_ir_weight=1, ssim_weight=1, ir_compose=1, color_weight=20, text_weight=10, temp_weight=2,regular=False):
        b,c,t,h,w = img_f.shape
        img_vi = rearrange(img_vi, 'b t c h w -> (b t) c h w')
        img_ir = rearrange(img_ir, 'b t c h w -> (b t) c h w')
        img_f = rearrange(img_f, 'b c t h w -> (b t) c h w')
        Y_vi, Cb_vi, Cr_vi = RGB2YCrCb(img_vi)
        Y_ir, _, _ = RGB2YCrCb(img_ir)
        Y_f, Cb_f, Cr_f = RGB2YCrCb(img_f)

        loss_temp_consist = temp_weight*self.cos_consist(Y_f.reshape(b,t,1,h,w).squeeze(2).reshape(b,t,h*w), 
                                        Y_ir.reshape(b,t,1,h,w).squeeze(2).reshape(b,t,h*w))

        loss_ssim = ssim_weight * (self.loss_func_ssim(Y_vi, Y_f) + ssim_ir_weight * self.loss_func_ssim(Y_ir, Y_f))
        loss_max = int_weight * self.loss_func_max(Y_f, Y_vi, Y_ir)
        loss_consist = consist_weight * self.loss_func_consist(Y_f, Y_vi, Y_ir, ir_compose)
        loss_color = color_weight * self.loss_func_color(Cb_f, Cr_f, Cb_vi, Cr_vi)
        loss_text = text_weight * self.loss_func_grad(Y_f, Y_vi, Y_ir, regular)
        total_loss = loss_ssim + loss_max + loss_consist + loss_color + loss_text + loss_temp_consist

                   
        return {
            'loss_intensity_max': loss_max,
            'loss_color': loss_color,
            'loss_grad': loss_text,
            'loss_pixel_consist': loss_consist,
            'loss_temporary_consist': loss_temp_consist,
            'loss_ssim': loss_ssim,
            'loss_fusion': total_loss
        }

class Temporary_Consistency(nn.Module):
    def __init__(self):
        super(Temporary_Consistency, self).__init__()
    
    def cos_consist(self, img_f, img_ir, img_vi):
        b,c,num_frames,h,w = img_f.shape
        diff_loss = 0
        for t in range(1, num_frames):
            output_diff = img_f[b,c,t] - img_f[b,c,t - 1]
            input_diff = img_ir[b,c,t] - img_ir[b,c,t - 1]
            # 计算余弦相似性
            cosine_sim = F.cosine_similarity(output_diff, input_diff, dim=0)
            diff_loss += (1 - cosine_sim).mean()  # 1 - 余弦相似性
        diff_loss /= (num_frames - 1)
    
    def forward(self, img_f, img_ir):
        loss_cos = self.cos_consist(img_f, img_ir)
                   
        return loss_cos

class L_SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(L_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        (_, channel_2, _, _) = img2.size()

        if channel != channel_2 and channel == 1:
            img1 = torch.concat([img1, img1, img1], dim=1)
            channel = 3

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=24, window=None, size_average=True, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    return 1 - ret

class L_Gradient_Max(nn.Module):
    def __init__(self):
        super(L_Gradient_Max, self).__init__()
        self.grad_operator = Gradient()

    def forward(self, img_f, img_vi, img_ir, regular):
        grad_vi_x, grad_vi_y = self.grad_operator(img_vi)
        grad_ir_x, grad_ir_y = self.grad_operator(img_ir)
        # 计算融合图像的梯度
        grad_f_x, grad_f_y = self.grad_operator(img_f)
        # 计算梯度幅值最大值一致性损失
        if regular:
            loss = 5 * ((torch.abs(grad_vi_x - grad_ir_x) * F.l1_loss(grad_f_x, torch.max(grad_vi_x, grad_ir_x)) + torch.abs(grad_vi_y - grad_ir_y) * F.l1_loss(grad_f_y, torch.max(grad_vi_y, grad_ir_y))).mean())
        else:
            loss = F.l1_loss(grad_f_x, torch.max(grad_vi_x, grad_ir_x)) + F.l1_loss(grad_f_y, torch.max(grad_vi_y, grad_ir_y))
        return loss

class Gradient(nn.Module):
    def __init__(self):
        super(Gradient, self).__init__()
        self.sobel_x = nn.Parameter(torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3), requires_grad=False)
        self.sobel_y = nn.Parameter(torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3), requires_grad=False)

    def forward(self, img):
        img = F.pad(img, (1, 1, 1, 1), mode='replicate')
        grad_x = F.conv2d(img, self.sobel_x)
        grad_y = F.conv2d(img, self.sobel_y)
        return torch.abs(grad_x), torch.abs(grad_y)


class L_Intensity_Consist(nn.Module):
    def __init__(self):
        super(L_Intensity_Consist, self).__init__()

    def forward(self, img_f, img_vi, img_ir, ir_compose, consist_mode="l1"):
        loss_func = F.mse_loss if consist_mode == "l2" else F.l1_loss
        return loss_func(img_vi, img_f) + ir_compose * loss_func(img_ir, img_f)
    
class L_Color(nn.Module):
    def __init__(self):
        super(L_Color, self).__init__()

    def forward(self, Cb_f, Cr_f, Cb_vi, Cr_vi):
        return F.l1_loss(Cb_f, Cb_vi) + F.l1_loss(Cr_f, Cr_vi)
    
class L_Intensity_Max(nn.Module):
    def __init__(self):
        super(L_Intensity_Max, self).__init__()

    def forward(self, img_f, img_vi, img_ir):
        img_max = torch.max(img_vi, img_ir)
        return F.l1_loss(img_max, img_f)
    
def RGB2YCrCb(rgb_image, with_CbCr=False):
    """
    Convert RGB format to YCrCb format.
    Used in the intermediate results of the color space conversion, because the default size of rgb_image is [B, C, H, W].
    :param rgb_image: image data in RGB format
    :param with_CbCr: boolean flag to determine if Cb and Cr channels should be returned
    :return: Y, CbCr (if with_CbCr is True), otherwise Y, Cb, Cr
    """
    R, G, B = rgb_image[:, 0:1, ::], rgb_image[:, 1:2, ::], rgb_image[:, 2:3, ::]
    
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.169 * R - 0.331 * G + 0.5 * B + 128/255.0
    Cr = 0.5 * R - 0.419 * G - 0.081 * B + 128/255.0

    Y, Cb, Cr = Y.clamp(0.0, 1.0), Cb.clamp(0.0, 1.0), Cr.clamp(0.0, 1.0)
    
    if with_CbCr:
        return Y, torch.cat([Cb, Cr], dim=1)
    return Y, Cb, Cr

def YCbCr2RGB(Y, Cb, Cr):
    """
    Convert YcrCb format to RGB format
    :param Y.
    :param Cb.
    :param Cr.
    :return.
    """
    R = Y + 1.402 * (Cr - 128/255.0)
    G = Y - 0.344136 * (Cb - 128/255.0) - 0.714136 * (Cr - 128/255.0)
    B = Y + 1.772 * (Cb - 128/255.0)
    return torch.cat([R, G, B], dim=1).clamp(0, 1.0)