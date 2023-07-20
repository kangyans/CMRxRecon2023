import torch
from torch.nn import *
from torch.fft import *


class RealComplex(Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.complex(x.select(self.dim, 0), x.select(self.dim, 1))


class ComplexReal(Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.stack([x.real, x.imag], self.dim)


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
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x, sens):
        return sens * torch.unsqueeze(x, dim=self.dim)


class CoilCombine(Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x, sens):
        return torch.sum(torch.conj_physical(sens) * x, dim=self.dim)


class SingleCoilDC(Module):
    def __init__(self, lamda):
        super().__init__()
        self.lamda = lamda
        self.real_complex = RealComplex()
        self.fft2c = FFT2C()
        self.ifft2c = IFFT2C()
        self.complex_real = ComplexReal()

    def forward(self, x, mask, k0):
        x = self.real_complex(x)
        kcnn = self.fft2c(x)
        k = mask * (kcnn + self.lamda * k0) / (1 + self.lamda) + \
            (1 - mask) * kcnn
        x = self.ifft2c(k)
        x = self.complex_real(x)
        return x


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

    def forward(self, x, mask, k0, sens):
        x = self.real_complex(x)
        x = self.coil_split(x, sens)
        kcnn = self.fft2c(x)
        k = mask * (kcnn + self.lamda * k0) / (1 + self.lamda) + \
            (1 - mask) * kcnn
        x = self.ifft2c(k)
        x = self.coil_combine(x, sens)
        x = self.complex_real(x)
        return x


class CSingleCoilDC(Module):
    def __init__(self, lamda):
        super().__init__()
        self.lamda = lamda
        self.fft2c = FFT2C()
        self.ifft2c = IFFT2C()

    def forward(self, x, mask, k0):
        kcnn = self.fft2c(x)
        k = mask * (kcnn + self.lamda * k0) / (1 + self.lamda) + \
            (1 - mask) * kcnn
        x = self.ifft2c(k)
        return x


class CMultiCoilDC(Module):
    def __init__(self, lamda):
        super().__init__()
        self.lamda = lamda
        self.coil_split = CoilSplit()
        self.fft2c = FFT2C()
        self.ifft2c = IFFT2C()
        self.coil_combine = CoilCombine()

    def forward(self, x, mask, k0, sens):
        x = self.coil_split(x, sens)
        kcnn = self.fft2c(x)
        k = mask * (kcnn + self.lamda * k0) / (1 + self.lamda) + \
            (1 - mask) * kcnn
        x = self.ifft2c(k)
        x = self.coil_combine(x, sens)
        return x
