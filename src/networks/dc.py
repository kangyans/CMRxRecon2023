import torch
from torch import fft, nn


class R2C(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.complex(x.select(self.dim, 0), x.select(self.dim, 1))


class C2R(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.stack([x.real, x.imag], self.dim)


class FFT(nn.Module):
    def __init__(self, dim=(-2, -1), norm='ortho', centered=True):
        super().__init__()
        self.dim = dim
        self.norm = norm
        self.centered = centered

    def forward(self, x):
        if self.centered:
            return fft.fftshift(fft.fft2(fft.ifftshift(
                x, dim=self.dim), dim=self.dim, norm=self.norm), dim=self.dim)
        else:
            return fft.fft2(x, dim=self.dim, norm=self.norm)


class IFFT(nn.Module):
    def __init__(self, dim=(-2, -1), norm='ortho', centered=True):
        super().__init__()
        self.dim = dim
        self.norm = norm
        self.centered = centered

    def forward(self, x):
        if self.centered:
            return fft.fftshift(fft.ifft2(fft.ifftshift(
                x, dim=self.dim), dim=self.dim, norm=self.norm), dim=self.dim)
        else:
            return fft.ifft2(x, dim=self.dim, norm=self.norm)


class CoilSplit(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x, sens):
        return sens * torch.unsqueeze(x, dim=self.dim)


class CoilCombine(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x, sens):
        return torch.sum(torch.conj_physical(sens) * x, dim=self.dim)


class SingleCoilDC(nn.Module):
    def __init__(self, lamda):
        super().__init__()
        self.lamda = lamda
        self.r2c = R2C()
        self.fft = FFT()
        self.ifft = IFFT()
        self.c2r = C2R()

    def forward(self, x, mask, k0):
        x = self.r2c(x)
        kcnn = self.fft(x)
        k = mask * (kcnn + self.lamda * k0) / (1 + self.lamda) + \
            (1 - mask) * kcnn
        x = self.ifft(k)
        x = self.c2r(x)
        return x


class MultiCoilDC(nn.Module):
    def __init__(self, lamda):
        super().__init__()
        self.lamda = lamda
        self.r2c = R2C()
        self.coil_split = CoilSplit()
        self.fft = FFT()
        self.ifft = IFFT()
        self.coil_combine = CoilCombine()
        self.c2r = C2R()

    def forward(self, x, mask, k0, sens):
        x = self.r2c(x)
        x = self.coil_split(x, sens)
        kcnn = self.fft(x)
        k = mask * (kcnn + self.lamda * k0) / (1 + self.lamda) + \
            (1 - mask) * kcnn
        x = self.ifft(k)
        x = self.coil_combine(x, sens)
        x = self.c2r(x)
        return x


class CSingleCoilDC(nn.Module):
    def __init__(self, lamda):
        super().__init__()
        self.lamda = lamda
        self.fft = FFT()
        self.ifft = IFFT()

    def forward(self, x, mask, k0):
        kcnn = self.fft(x)
        k = mask * (kcnn + self.lamda * k0) / (1 + self.lamda) + \
            (1 - mask) * kcnn
        x = self.ifft(k)
        return x


class CMultiCoilDC(nn.Module):
    def __init__(self, lamda):
        super().__init__()
        self.lamda = lamda
        self.coil_split = CoilSplit()
        self.fft = FFT()
        self.ifft = IFFT()
        self.coil_combine = CoilCombine()

    def forward(self, x, mask, k0, sens):
        x = self.coil_split(x, sens)
        kcnn = self.fft(x)
        k = mask * (kcnn + self.lamda * k0) / (1 + self.lamda) + \
            (1 - mask) * kcnn
        x = self.ifft(k)
        x = self.coil_combine(x, sens)
        return x
