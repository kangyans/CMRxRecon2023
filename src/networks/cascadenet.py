import torch
from torch import fft, nn


class CascadeNet(nn.Module):
    def __init__(self, in_channels, out_channels, cascade_depth,
                 res_depth, lamda, num_filters=64, kernel_size=3,
                 normalization=None, activation=None):
        super().__init__()
        self.res_blocks = []
        self.dc_blocks = []
        for _ in range(cascade_depth):
            self.res_blocks.append(
                ResBlock(in_channels, out_channels, res_depth,
                         num_filters=num_filters, kernel_size=kernel_size,
                         normalization=normalization, activation=activation))
            self.dc_blocks.append(DataConsistencyBlock(lamda))

    def forward(self, x, mask, k0):
        for res_block, dc_block in zip(self.res_blocks, self.dc_blocks):
            x = res_block(x)
            x = dc_block(x, mask, k0)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 normalization=None, activation=None):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels,
                            kernel_size, padding='same')]
        if normalization is not None:
            if normalization == 'batch':
                layers.append(nn.BatchNorm2d(out_channels))
            elif normalization == 'instance':
                layers.append(nn.InstanceNorm2d(out_channels))
            else:
                raise ValueError('normalization should be None '
                                 'or batch or instance.')
        if activation is not None:
            if activation == 'LeakyReLU':
                layers.append(nn.LeakyReLU())
            elif activation == 'ReLU':
                layers.append(nn.ReLU)
            else:
                raise ValueError('activation should be None or '
                                 'LeakyReLU or ReLU.')
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth, num_filters=64,
                 kernel_size=3, normalization=None, activation='LeakyReLU'):
        super().__init__()
        layers = [ConvBlock(in_channels, num_filters, kernel_size,
                            normalization, activation)]
        if depth < 2:
            raise ValueError('depth should be greater than 2.')
        for _ in range(depth - 2):
            layers.append(ConvBlock(num_filters, num_filters, kernel_size,
                                    normalization, activation))
        layers.append(ConvBlock(num_filters, out_channels, kernel_size,
                                normalization, activation))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x) + x


class Real2ComplexBlock(nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.complex(x.select(self.dim, 0), x.select(self.dim, 1))


class Complex2RealBlock(nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.stack([x.real, x.imag], self.dim)


class FFTBlock(nn.Module):
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


class IFFTBlock(nn.Module):
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


class DataConsistencyBlock(nn.Module):
    def __init__(self, lamda):
        super().__init__()
        self.lamda = lamda
        self.real2complex = Real2ComplexBlock()
        self.fft = FFTBlock()
        self.ifft = IFFTBlock()
        self.complex2real = Complex2RealBlock()

    def forward(self, x, mask, k0):
        kcnn = self.fft(self.real2complex(x))
        k = mask * (kcnn + self.lamda * k0) / (1 + self.lamda) + \
            (1 - mask) * kcnn
        return self.complex2real(self.ifft(k))
