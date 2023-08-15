import torch
from torch.nn import *
import torch.nn.functional as F


class SSIMLoss(Module):
    def __init__(self, win_size=7, k1=0.01, k2=0.03):
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer(
            'w', torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        self.cov_norm = win_size ** 2 / (win_size ** 2 - 1)

    def forward(self, x, y, data_range):
        data_range = data_range[:, None, None, None]
        c1 = (self.k1 * data_range) ** 2
        c2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(x, self.w)
        uy = F.conv2d(y, self.w)
        uxx = F.conv2d(x * x, self.w)
        uyy = F.conv2d(y * y, self.w)
        uxy = F.conv2d(x * y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        a1 = 2 * ux * uy + c1
        a2 = 2 * vxy + c2
        b1 = ux ** 2 + uy ** 2 + c1
        b2 = vx + vy + c2
        d = b1 * b2
        s = (a1 * a2) / d
        return 1 - s.mean()
