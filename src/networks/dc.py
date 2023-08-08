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


class SingleCoilDC(Module):
    def __init__(self, lamda):
        super().__init__()
        self.lamda = lamda
        self.real_complex = RealComplex()
        self.fft2c = FFT2C()
        self.ifft2c = IFFT2C()
        self.complex_real = ComplexReal()

    def forward(self, im, mask, ksub):
        im = self.real_complex(im)
        kcnn = self.fft2c(im)
        k = mask * (kcnn + self.lamda * ksub) / (1 + self.lamda) + \
            (1 - mask) * kcnn
        im = self.ifft2c(k)
        im = self.complex_real(im)
        return im


class MultiCoilDC(Module):
    def __init__(self, lamda):
        super().__init__()
        self.lamda = lamda
        self.real_complex = RealComplex()
        self.coil_split = CoilSplit()
        self.fft2c = FFT2C()
        self.ifft2c = IFFT2C()
        self.coil_combine = CoilCombine()
        self.complex_real = ComplexReal()

    def forward(self, im, mask, ksub, sens):
        im = self.real_complex(im)
        im = self.coil_split(im, sens)
        kcnn = self.fft2c(im)
        k = mask * (kcnn + self.lamda * ksub) / (1 + self.lamda) + \
            (1 - mask) * kcnn
        im = self.ifft2c(k)
        im = self.coil_combine(im, sens)
        im = self.complex_real(im)
        return im


class CSingleCoilDC(Module):
    def __init__(self, lamda):
        super().__init__()
        self.lamda = lamda
        self.fft2c = FFT2C()
        self.ifft2c = IFFT2C()

    def forward(self, im, mask, ksub):
        kcnn = self.fft2c(im)
        k = mask * (kcnn + self.lamda * ksub) / (1 + self.lamda) + \
            (1 - mask) * kcnn
        im = self.ifft2c(k)
        return im


class CMultiCoilDC(Module):
    def __init__(self, lamda):
        super().__init__()
        self.lamda = lamda
        self.coil_split = CoilSplit()
        self.fft2c = FFT2C()
        self.ifft2c = IFFT2C()
        self.coil_combine = CoilCombine()

    def forward(self, im, mask, ksub, sens):
        im = self.coil_split(im, sens)
        kcnn = self.fft2c(im)
        k = mask * (kcnn + self.lamda * ksub) / (1 + self.lamda) + \
            (1 - mask) * kcnn
        im = self.ifft2c(k)
        im = self.coil_combine(im, sens)
        return im
