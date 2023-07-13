from torch import nn
import ctorch as cnn


class ConvBlock(nn.Module):
    def __init__(self, dimensions, real, in_channels,
                 out_channels, kernel_size=3, bias=True,
                 normalization=None, activation=None):
        super().__init__()
        if dimensions == 1:
            conv = nn.Conv1d if real else cnn.CConv1d
        elif dimensions == 2:
            conv = nn.Conv2d if real else cnn.CConv2d
        elif dimensions == 3:
            conv = nn.Conv3d if real else cnn.CConv3d
        else:
            raise ValueError('dimensions should be 1 or 2 or 3.')
        layers = [conv(in_channels, out_channels, kernel_size,
                       padding='same', bias=bias)]
        if normalization is not None:
            if normalization == 'batch':
                if dimensions == 1:
                    norm = nn.BatchNorm1d if real else cnn.CBatchNorm1d
                elif dimensions == 2:
                    norm = nn.BatchNorm2d if real else cnn.CBatchNorm2d
                else:
                    norm = nn.BatchNorm3d if real else cnn.CBatchNorm3d
            elif normalization == 'instance':
                if dimensions == 1:
                    norm = nn.InstanceNorm1d if real else cnn.CInstanceNorm1d
                elif dimensions == 2:
                    norm = nn.InstanceNorm2d if real else cnn.CInstanceNorm2d
                else:
                    norm = nn.InstanceNorm3d if real else cnn.CInstanceNorm3d
            else:
                raise ValueError('normalization should be None '
                                 'or batch or instance.')
            layers.append(norm(out_channels))
        if activation is not None:
            if activation == 'ReLU':
                layers.append(nn.ReLU() if real else cnn.CReLU())
            elif activation == 'LeakyReLU':
                layers.append(nn.LeakyReLU() if real else cnn.CLeakyReLU())
            else:
                raise ValueError('activation should be None or '
                                 'ReLU or LeakyReLU.')
        self.block = nn.Sequential(*layers)

    def forward(self, input):
        return self.block(input)


class ConvBlock1d(ConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 bias=True, normalization=None, activation=None):
        super().__init__(1, True, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)


class ConvBlock2d(ConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 bias=True, normalization=None, activation=None):
        super().__init__(2, True, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)


class ConvBlock3d(ConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 bias=True, normalization=None, activation=None):
        super().__init__(3, True, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)


class ConvBlock2plus1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 bias=True, normalization=None, activation=None):
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


class CConvBlock1d(ConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 bias=True, normalization=None, activation=None):
        super().__init__(1, False, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)


class CConvBlock2d(ConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 bias=True, normalization=None, activation=None):
        super().__init__(2, False, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)


class CConvBlock3d(ConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 bias=True, normalization=None, activation=None):
        super().__init__(3, False, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)


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
