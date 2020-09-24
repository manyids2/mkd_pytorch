import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def load_fspecial_gaussian_filter(sigma):
    # Gaussian filter of size 5x5.
    rx = np.arange(-2, 3, dtype=np.float32)
    fx = np.exp(-1 * np.square(rx / (sigma * np.sqrt(2.0))))
    fx = np.expand_dims(fx, 1)
    gx = np.dot(fx, fx.T)
    gx = gx / np.sum(gx)
    return gx.astype(np.float32)


class Gradients(nn.Module):
    def __init__(self, *, patch_size=32, do_smoothing=True, sigma=1.4, device='cuda'):
        super().__init__()
        self.patch_size = patch_size
        self.do_smoothing = do_smoothing
        self.sigma = sigma / 64.0  # 1.4 at patch_size = 64
        self.eps = 1e-8

        # Basic filter - 1, 0, -1.
        delta = np.array([1, 0, -1])
        xf = delta.reshape([1, 3])
        yf = delta.reshape([3, 1])
        xf, yf = [
            f_[np.newaxis, np.newaxis, :, :].astype(np.float32) for f_ in [xf, yf]
        ]
        self.xf = torch.Tensor(xf).to(device)
        self.yf = torch.Tensor(yf).to(device)
        self.gp = nn.ReplicationPad2d((2, 2, 2, 2))
        self.xp = nn.ReplicationPad2d((1, 1, 0, 0))
        self.yp = nn.ReplicationPad2d((0, 0, 1, 1))
        self.bias = torch.zeros([1], dtype=torch.float32).to(device)

        # Smoothing filter taken from Matlab function fspecial_gaussian.
        if do_smoothing:
            gaussian = load_fspecial_gaussian_filter(sigma)
            gaussian = gaussian[np.newaxis, np.newaxis, :, :].astype(np.float32)
            self.gaussian = torch.Tensor(gaussian).to(device)

    def forward(self, x):  # pylint: disable=W0221
        # Gaussian smoothing.
        if self.do_smoothing:
            x = self.gp(x)
            x = F.conv2d(x, self.gaussian, self.bias, 1, 0)

        # x and y gradients.
        gx = F.conv2d(self.xp(x), self.xf, self.bias, 1, 0)
        gy = F.conv2d(self.yp(x), self.yf, self.bias, 1, 0)

        # Magnitudes and orientations.
        mags = torch.sqrt(torch.pow(gx, 2) + torch.pow(gy, 2) + self.eps)
        oris = torch.atan2(gy, gx)

        # Concatenate and return.
        y = torch.cat([mags, oris], dim=1)
        return y

    def extra_repr(self):
        return f'patch_size:{self.patch_size}, pad=ReplicationPad2d'
