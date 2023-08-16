import torch
from torch.nn import *
from torch.fft import *


class RealComplex(Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.unsqueeze(
            torch.complex(x.select(self.dim, 0),
                          x.select(self.dim, 1)), dim=self.dim)


class ComplexReal(Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat([x.real, x.imag], dim=self.dim)


class FFT2C(Module):
    def __init__(self, dim=(-2, -1), norm='ortho', centered=True):
        super().__init__()
        self.dim = dim
        self.norm = norm
        self.centered = centered

    def forward(self, x):
        if self.centered:
            return fftshift(fft2(ifftshift(x, dim=self.dim), dim=self.dim,
                                 norm=self.norm), dim=self.dim)
        else:
            return fft2(x, dim=self.dim, norm=self.norm)


class IFFT2C(Module):
    def __init__(self, dim=(-2, -1), norm='ortho', centered=True):
        super().__init__()
        self.dim = dim
        self.norm = norm
        self.centered = centered

    def forward(self, x):
        if self.centered:
            return fftshift(ifft2(ifftshift(x, dim=self.dim), dim=self.dim,
                                  norm=self.norm), dim=self.dim)
        else:
            return ifft2(x, dim=self.dim, norm=self.norm)


class CoilSplit(Module):
    def __init__(self, dim=2):
        super().__init__()
        self.dim = dim

    def forward(self, im, sens):
        return sens * torch.unsqueeze(im, dim=self.dim)


class CoilCombine(Module):
    def __init__(self, dim=2):
        super().__init__()
        self.dim = dim

    def forward(self, im, sens):
        return torch.sum(torch.conj_physical(sens) * im, dim=self.dim)


class DataConsistency(Module):
    def __init__(self, lamda):
        super().__init__()
        self.lamda = lamda

    def forward(self, kcnn, mask, ksub):
        return torch.where(mask == 1,
                           self.lamda * ksub + (1 - self.lamda) * kcnn,
                           kcnn)
