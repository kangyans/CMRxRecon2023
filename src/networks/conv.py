from ctorch import *


class _ConvBlock(Module):
    def __init__(self, dimensions, real, in_channels,
                 out_channels, kernel_size=3, bias=True,
                 normalization=None, activation=None):
        super().__init__()
        if dimensions == 1:
            conv = Conv1d if real else CConv1d
        elif dimensions == 2:
            conv = Conv2d if real else CConv2d
        elif dimensions == 3:
            conv = Conv3d if real else CConv3d
        else:
            raise ValueError('dimensions should be 1 or 2 or 3.')
        layers = [conv(in_channels, out_channels, kernel_size,
                       padding='same', bias=bias)]
        if normalization is not None:
            if normalization == 'batch':
                if dimensions == 1:
                    norm = BatchNorm1d if real else CBatchNorm1d
                elif dimensions == 2:
                    norm = BatchNorm2d if real else CBatchNorm2d
                else:
                    norm = BatchNorm3d if real else CBatchNorm3d
            elif normalization == 'instance':
                if dimensions == 1:
                    norm = InstanceNorm1d if real else CInstanceNorm1d
                elif dimensions == 2:
                    norm = InstanceNorm2d if real else CInstanceNorm2d
                else:
                    norm = InstanceNorm3d if real else CInstanceNorm3d
            else:
                raise ValueError('normalization should be None '
                                 'or batch or instance.')
            layers.append(norm(out_channels))
        if activation is not None:
            if activation == 'ReLU':
                layers.append(ReLU() if real else CReLU())
            elif activation == 'LeakyReLU':
                layers.append(LeakyReLU() if real else CLeakyReLU())
            else:
                raise ValueError('activation should be None or '
                                 'ReLU or LeakyReLU.')
        self.block = Sequential(*layers)

    def forward(self, input):
        return self.block(input)


class ConvBlock1d(_ConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 bias=True, normalization=None, activation=None):
        super().__init__(1, True, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)


class ConvBlock2d(_ConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 bias=True, normalization=None, activation=None):
        super().__init__(2, True, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)


class ConvBlock3d(_ConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 bias=True, normalization=None, activation=None):
        super().__init__(3, True, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)


class ConvBlock2plus1d(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 bias=True, normalization=None, activation=None):
        super().__init__()
        mid_channels = int((kernel_size ** 3 * in_channels * out_channels) /
                           (kernel_size ** 2 * in_channels +
                            kernel_size * out_channels))
        layers = [Conv3d(in_channels, mid_channels,
                         (1, kernel_size, kernel_size),
                         padding='same', bias=bias)]
        if normalization is not None:
            if normalization == 'batch':
                layers.append(BatchNorm3d(mid_channels))
            elif normalization == 'instance':
                layers.append(InstanceNorm3d(mid_channels))
            else:
                raise ValueError('normalization should be None '
                                 'or batch or instance.')
        if activation is not None:
            if activation == 'ReLU':
                layers.append(ReLU())
            elif activation == 'LeakyReLU':
                layers.append(LeakyReLU())
            else:
                raise ValueError('activation should be None or '
                                 'ReLU or LeakyReLU.')
        layers.append(Conv3d(mid_channels, out_channels,
                             (kernel_size, 1, 1),
                             padding='same', bias=bias))
        if normalization is not None:
            if normalization == 'batch':
                layers.append(BatchNorm3d(mid_channels))
            elif normalization == 'instance':
                layers.append(InstanceNorm3d(mid_channels))
            else:
                raise ValueError('normalization should be None '
                                 'or batch or instance.')
        if activation is not None:
            if activation == 'ReLU':
                layers.append(ReLU())
            elif activation == 'LeakyReLU':
                layers.append(LeakyReLU())
            else:
                raise ValueError('activation should be None or '
                                 'ReLU or LeakyReLU.')
        self.block = Sequential(*layers)

    def forward(self, input):
        return self.block(input)


class CConvBlock1d(_ConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 bias=True, normalization=None, activation=None):
        super().__init__(1, False, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)


class CConvBlock2d(_ConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 bias=True, normalization=None, activation=None):
        super().__init__(2, False, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)


class CConvBlock3d(_ConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 bias=True, normalization=None, activation=None):
        super().__init__(3, False, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)


class CConvBlock2plus1d(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True,
                 normalization=None, activation=None):
        super().__init__()
        mid_channels = int((kernel_size ** 3 * in_channels * out_channels) /
                           (kernel_size ** 2 * in_channels +
                            kernel_size * out_channels))
        layers = [CConv3d(in_channels, mid_channels,
                          (1, kernel_size, kernel_size),
                          padding='same', bias=bias)]
        if normalization is not None:
            if normalization == 'batch':
                layers.append(CBatchNorm3d(mid_channels))
            elif normalization == 'instance':
                layers.append(CInstanceNorm3d(mid_channels))
            else:
                raise ValueError('normalization should be None '
                                 'or batch or instance.')
        if activation is not None:
            if activation == 'ReLU':
                layers.append(CReLU())
            elif activation == 'LeakyReLU':
                layers.append(CLeakyReLU())
            else:
                raise ValueError('activation should be None or '
                                 'ReLU or LeakyReLU.')
        layers.append(CConv3d(mid_channels, out_channels,
                              (kernel_size, 1, 1),
                              padding='same', bias=bias))
        if normalization is not None:
            if normalization == 'batch':
                layers.append(CBatchNorm3d(mid_channels))
            elif normalization == 'instance':
                layers.append(CInstanceNorm3d(mid_channels))
            else:
                raise ValueError('normalization should be None '
                                 'or batch or instance.')
        if activation is not None:
            if activation == 'ReLU':
                layers.append(CReLU())
            elif activation == 'LeakyReLU':
                layers.append(CLeakyReLU())
            else:
                raise ValueError('activation should be None or '
                                 'ReLU or LeakyReLU.')
        self.block = Sequential(*layers)

    def forward(self, input):
        return self.block(input)


class _DoubleConvBlock(Module):
    def __init__(self, dimensions, real, in_channels,
                 out_channels, kernel_size=3, bias=True,
                 normalization=None, activation=None):
        super().__init__()
        if dimensions == '2':
            conv_block = ConvBlock2d if real else CConvBlock2d
        elif dimensions == '3':
            conv_block = ConvBlock3d if real else CConvBlock3d
        elif dimensions == '2+1':
            conv_block = ConvBlock2plus1d if real else CConvBlock2plus1d
        else:
            raise ValueError('dimensions should be 2 or 3 or 2+1.')
        self.block = Sequential(
            conv_block(in_channels, out_channels, kernel_size,
                       bias, normalization, activation),
            conv_block(out_channels, out_channels, kernel_size,
                       bias, normalization, activation))

    def forward(self, input):
        return self.block(input)


class DoubleConvBlock2d(_DoubleConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 bias=True, normalization=None, activation=None):
        super().__init__('2', True, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)


class DoubleConvBlock3d(_DoubleConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 bias=True, normalization=None, activation=None):
        super().__init__('3', True, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)


class DoubleConvBlock2plus1d(_DoubleConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 bias=True, normalization=None, activation=None):
        super().__init__('2+1', True, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)


class CDoubleConvBlock2d(_DoubleConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 bias=True, normalization=None, activation=None):
        super().__init__('2', False, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)


class CDoubleConvBlock3d(_DoubleConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True,
                 normalization=None, activation=None):
        super().__init__('3', False, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)


class CDoubleConvBlock2plus1d(_DoubleConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True,
                 normalization=None, activation=None):
        super().__init__('2+1', False, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)
