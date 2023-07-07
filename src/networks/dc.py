import torch
from torch import fft, nn


class Real2Complex(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.complex(x.select(self.dim, 0), x.select(self.dim, 1))


class Complex2Real(nn.Module):
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


class SingleCoilDataConsistency(nn.Module):
    def __init__(self, lamda):
        super().__init__()
        self.lamda = lamda
        self.real2complex = Real2Complex()
        self.fft = FFT()
        self.ifft = IFFT()
        self.complex2real = Complex2Real()

    def forward(self, x, mask, k0):
        is_complex = torch.is_complex(x)
        if not is_complex:
            x = self.real2complex(x)
        kcnn = self.fft(x)
        k = mask * (kcnn + self.lamda * k0) / (1 + self.lamda) + \
            (1 - mask) * kcnn
        x = self.ifft(k)
        if not is_complex:
            x = self.complex2real(x)
        return x


class MultiCoilDataConsistency(nn.Module):
    def __init__(self, lamda):
        super().__init__()
        self.lamda = lamda
        self.real2complex = Real2Complex()
        self.coil_split = CoilSplit()
        self.fft = FFT()
        self.ifft = IFFT()
        self.coil_combine = CoilCombine()
        self.complex2rea = Complex2Real()

    def forward(self, x, mask, k0, sens):
        is_complex = torch.is_complex(x)
        if not is_complex:
            x = self.real2complex(x)
        x = self.coil_split(x)
        kcnn = self.fft()
        k = mask * (kcnn + self.lamda * k0) / (1 + self.lamda) + \
            (1 - mask) * kcnn
        x = self.ifft(k)
        x = self.coil_combine(x)
        if not is_complex:
            x = self.complex2real(x)
        return x
