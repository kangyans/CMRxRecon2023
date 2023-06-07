from torch import fft, nn


class FFT2d(nn.Module):
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


class IFFT2d(nn.Module):
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
