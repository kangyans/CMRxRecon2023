import torch
from torch import nn
import ctorch as cnn


class ConvBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True,
                 normalization=None, activation=None):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size,
                            padding='same', bias=bias)]
        if normalization is not None:
            if normalization == 'batch':
                layers.append(nn.BatchNorm2d(out_channels))
            elif normalization == 'instance':
                layers.append(nn.InstanceNorm2d(out_channels))
            else:
                raise ValueError('normalization should be None '
                                 'or batch or instance.')
        if activation is not None:
            if activation == 'ReLU':
                layers.append(nn.ReLU())
            elif activation == 'LeakyReLU':
                layers.append(nn.LeakyReLU())
            else:
                raise ValueError('activation should be None or '
                                 'ReLU or LeakyReLU.')
        self.block = nn.Sequential(*layers)

    def forward(self, input):
        return self.block(input)


class ConvBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True,
                 normalization=None, activation=None):
        super().__init__()
        layers = [nn.Conv3d(in_channels, out_channels, kernel_size,
                            padding='same', bias=bias)]
        if normalization is not None:
            if normalization == 'batch':
                layers.append(nn.BatchNorm3d(out_channels))
            elif normalization == 'instance':
                layers.append(nn.InstanceNorm3d(out_channels))
            else:
                raise ValueError('normalization should be None '
                                 'or batch or instance.')
        if activation is not None:
            if activation == 'ReLU':
                layers.append(nn.ReLU())
            elif activation == 'LeakyReLU':
                layers.append(nn.LeakyReLU())
            else:
                raise ValueError('activation should be None or '
                                 'ReLU or LeakyReLU.')
        self.block = nn.Sequential(*layers)

    def forward(self, input):
        return self.block(input)


class ConvBlock2plus1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True,
                 normalization=None, activation=None):
        super().__init__()
        mid_channels = int((kernel_size ** 3 * in_channels * out_channels) /
                           (kernel_size ** 2 * in_channels +
                            kernel_size * out_channels))
        layers = [nn.Conv3d(in_channels, mid_channels,
                            (1, kernel_size, kernel_size),
                            padding='same', bias=bias)]
        if normalization is not None:
            if normalization == 'batch':
                layers.append(nn.BatchNorm3d(mid_channels))
            elif normalization == 'instance':
                layers.append(nn.InstanceNorm3d(mid_channels))
            else:
                raise ValueError('normalization should be None '
                                 'or batch or instance.')
        if activation is not None:
            if activation == 'ReLU':
                layers.append(nn.ReLU())
            elif activation == 'LeakyReLU':
                layers.append(nn.LeakyReLU())
            else:
                raise ValueError('activation should be None or '
                                 'ReLU or LeakyReLU.')
        layers.append(nn.Conv3d(mid_channels, out_channels,
                                (kernel_size, 1, 1),
                                padding='same', bias=bias))
        if normalization is not None:
            if normalization == 'batch':
                layers.append(nn.BatchNorm3d(mid_channels))
            elif normalization == 'instance':
                layers.append(nn.InstanceNorm3d(mid_channels))
            else:
                raise ValueError('normalization should be None '
                                 'or batch or instance.')
        if activation is not None:
            if activation == 'ReLU':
                layers.append(nn.ReLU())
            elif activation == 'LeakyReLU':
                layers.append(nn.LeakyReLU())
            else:
                raise ValueError('activation should be None or '
                                 'ReLU or LeakyReLU.')
        self.block = nn.Sequential(*layers)

    def forward(self, input):
        return self.block(input)


class CConvBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True,
                 normalization=None, activation=None):
        super().__init__()
        layers = [cnn.CConv2d(in_channels, out_channels, kernel_size,
                              padding='same', bias=bias)]
        if normalization is not None:
            if normalization == 'batch':
                layers.append(cnn.CBatchNorm2d(out_channels))
            elif normalization == 'instance':
                layers.append(cnn.CInstanceNorm2d(out_channels))
            else:
                raise ValueError('normalization should be None '
                                 'or batch or instance.')
        if activation is not None:
            if activation == 'ReLU':
                layers.append(cnn.CReLU())
            elif activation == 'LeakyReLU':
                layers.append(cnn.CLeakyReLU())
            else:
                raise ValueError('activation should be None or '
                                 'ReLU or LeakyReLU.')
        self.block = nn.Sequential(*layers)

    def forward(self, input):
        return self.block(input)


class CConvBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True,
                 normalization=None, activation=None):
        super().__init__()
        layers = [cnn.CConv3d(in_channels, out_channels, kernel_size,
                              padding='same', bias=bias)]
        if normalization is not None:
            if normalization == 'batch':
                layers.append(cnn.CBatchNorm3d(out_channels))
            elif normalization == 'instance':
                layers.append(cnn.CInstanceNorm3d(out_channels))
            else:
                raise ValueError('normalization should be None '
                                 'or batch or instance.')
        if activation is not None:
            if activation == 'ReLU':
                layers.append(cnn.CReLU())
            elif activation == 'LeakyReLU':
                layers.append(cnn.CLeakyReLU())
            else:
                raise ValueError('activation should be None or '
                                 'ReLU or LeakyReLU.')
        self.block = nn.Sequential(*layers)

    def forward(self, input):
        return self.block(input)


class CConvBlock2plus1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True,
                 normalization=None, activation=None):
        super().__init__()
        mid_channels = int((kernel_size ** 3 * in_channels * out_channels) /
                           (kernel_size ** 2 * in_channels +
                            kernel_size * out_channels))
        layers = [cnn.CConv3d(in_channels, mid_channels,
                              (1, kernel_size, kernel_size),
                              padding='same', bias=bias)]
        if normalization is not None:
            if normalization == 'batch':
                layers.append(cnn.CBatchNorm3d(mid_channels))
            elif normalization == 'instance':
                layers.append(cnn.CInstanceNorm3d(mid_channels))
            else:
                raise ValueError('normalization should be None '
                                 'or batch or instance.')
        if activation is not None:
            if activation == 'ReLU':
                layers.append(cnn.CReLU())
            elif activation == 'LeakyReLU':
                layers.append(cnn.CLeakyReLU())
            else:
                raise ValueError('activation should be None or '
                                 'ReLU or LeakyReLU.')
        layers.append(cnn.CConv3d(mid_channels, out_channels,
                                  (kernel_size, 1, 1),
                                  padding='same', bias=bias))
        if normalization is not None:
            if normalization == 'batch':
                layers.append(cnn.CBatchNorm3d(mid_channels))
            elif normalization == 'instance':
                layers.append(cnn.CInstanceNorm3d(mid_channels))
            else:
                raise ValueError('normalization should be None '
                                 'or batch or instance.')
        if activation is not None:
            if activation == 'ReLU':
                layers.append(cnn.CReLU())
            elif activation == 'LeakyReLU':
                layers.append(cnn.CLeakyReLU())
            else:
                raise ValueError('activation should be None or '
                                 'ReLU or LeakyReLU.')
        self.block = nn.Sequential(*layers)

    def forward(self, input):
        return self.block(input)


class ResNet2d(nn.Module):
    def __init__(self, in_channels, out_channels, depth, num_filters=64,
                 kernel_size=3, normalization=None, activation='ReLU'):
        super().__init__()
        blocks = [ConvBlock2d(in_channels, num_filters, kernel_size,
                              normalization, activation)]
        if depth < 2:
            raise ValueError('depth should be greater than 2.')
        for _ in range(depth - 2):
            blocks.append(ConvBlock2d(num_filters, num_filters, kernel_size,
                                      normalization, activation))
        blocks.append(ConvBlock2d(num_filters, out_channels, kernel_size,
                                  normalization, activation))
        self.net = nn.Sequential(*blocks)

    def forward(self, input):
        return self.net(input) + input


class ResNet3d(nn.Module):
    def __init__(self, in_channels, out_channels, depth, num_filters=64,
                 kernel_size=3, normalization=None, activation='ReLU'):
        super().__init__()
        blocks = [ConvBlock3d(in_channels, num_filters, kernel_size,
                              normalization, activation)]
        if depth < 2:
            raise ValueError('depth should be greater than 2.')
        for _ in range(depth - 2):
            blocks.append(ConvBlock3d(num_filters, num_filters, kernel_size,
                                      normalization, activation))
        blocks.append(ConvBlock3d(num_filters, out_channels, kernel_size,
                                  normalization, activation))
        self.net = nn.Sequential(*blocks)

    def forward(self, input):
        return self.net(input) + input


class ResNet2plus1d(nn.Module):
    def __init__(self, in_channels, out_channels, depth, num_filters=64,
                 kernel_size=3, normalization=None, activation='ReLU'):
        super().__init__()
        blocks = [ConvBlock2plus1d(in_channels, num_filters, kernel_size,
                                   normalization, activation)]
        if depth < 2:
            raise ValueError('depth should be greater than 2.')
        for _ in range(depth - 2):
            blocks.append(ConvBlock2plus1d(num_filters, num_filters,
                                           kernel_size, normalization,
                                           activation))
        blocks.append(ConvBlock2plus1d(num_filters, out_channels, kernel_size,
                                       normalization, activation))
        self.net = nn.Sequential(*blocks)

    def forward(self, input):
        return self.net(input) + input
