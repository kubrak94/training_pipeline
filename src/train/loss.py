from typing import NamedTuple

from torch.nn import *

import torch
import torch.nn.functional as F
from math import exp
import numpy as np

import src.train.alpha_scheduler as alpha_scheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LossInfo(NamedTuple):
    name: str
    loss: torch.nn.Module


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
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

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3, val_range=255):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.val_range = val_range
    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        # TODO: Fix it!
        return 1 - msssim(img1, img2, window_size=self.window_size, size_average=self.size_average, val_range=self.val_range, normalize=False)


class MSSSIM_wavelet(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3, val_range=5):
        super(MSSSIM_wavelet, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.val_range = val_range
    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        # TODO: Fix it!
        return 1 - msssim(img1, img2, window_size=self.window_size, size_average=self.size_average, val_range=self.val_range, normalize=False)


class TotalLoss(torch.nn.Module):
    def __init__(self, main_loss_params, aux_loss_params, alpha_scheduler_params={'name': 'Constant'}):
        super(TotalLoss, self).__init__()

        main_loss_name = main_loss_params['name']
        main_loss_function = parse_loss_params(main_loss_params)
        self.main_loss = LossInfo(main_loss_name, main_loss_function)
        self.main_loss_value = 0

        aux_loss_name = aux_loss_params['name']
        aux_loss_function = parse_loss_params(aux_loss_params)
        self.aux_loss = LossInfo(aux_loss_name, aux_loss_function)
        self.aux_loss_value = 0

        alpha_scheduler_name = alpha_scheduler_params.pop('name')
        self.alpha_scheduler = alpha_scheduler.__dict__[alpha_scheduler_name](**alpha_scheduler_params)

    def forward(self, output, targets):
        self.main_loss_value = self.main_loss.loss(output, targets)
        self.aux_loss_value = self.aux_loss.loss(output, targets)

        res = self.alpha_scheduler.alpha * self.main_loss_value + (1 - self.alpha_scheduler.alpha) * self.aux_loss_value
        
        return res


def parse_loss_params(loss_params):
    loss_name = loss_params.pop('name')

    if loss_name not in ['MSELoss', 'L1Loss', 'MultiLabelMarginLoss', 'MSSSIM', 'MSSSIM_wavelet', 'TotalLoss']:
        weights_path = loss_params.pop('weights_path')

        with open(weights_path, 'r') as infile:
            loss_params["weight"] = torch.Tensor(json.load(infile))

    loss_function = globals()[loss_name](**loss_params)

    if loss_name not in ['TotalLoss']:
        loss_function = loss_function.to(device)

    return loss_function
