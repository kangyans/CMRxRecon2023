from .conv import *


class _ResBlock(Module):
    def __init__(self, dimensions, real, in_channels,
                 out_channels, kernel_size=3, bias=True,
                 normalization=None, activation='ReLU'):
        super().__init__()
        if dimensions == '2':
            double_conv_block = DoubleConvBlock2d if real \
                else CDoubleConvBlock2d
        elif dimensions == '3':
            double_conv_block = DoubleConvBlock3d if real \
                else CDoubleConvBlock3d
        elif dimensions == '2+1':
            double_conv_block = DoubleConvBlock2plus1d if real \
                else CDoubleConvBlock2plus1d
        else:
            raise ValueError('dimensions should be 2 or 3 or 2+1.')
        self.double_conv_block = \
            double_conv_block(in_channels, out_channels, kernel_size,
                              bias, normalization, activation)

    def forward(self, input):
        return self.double_conv_block(input) + input


class _ResBlock2d(_ResBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 bias=True, normalization=None, activation='ReLU'):
        super().__init__('2', True, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)


class _ResBlock3d(_ResBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 bias=True, normalization=None, activation='ReLU'):
        super().__init__('3', True, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)


class _ResBlock2plus1d(_ResBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 bias=True, normalization=None, activation='ReLU'):
        super().__init__('2+1', True, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)


class _CResBlock2d(_ResBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 bias=True, normalization=None, activation='ReLU'):
        super().__init__('2', False, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)


class _CResBlock3d(_ResBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 bias=True, normalization=None, activation='ReLU'):
        super().__init__('3', False, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)


class _CResBlock2plus1d(_ResBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 bias=True, normalization=None, activation='ReLU'):
        super().__init__('2+1', False, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)


class _ResNet(Module):
    def __init__(self, dimensions, real, in_channels, out_channels,
                 depth, num_filters=64, kernel_size=3, bias=True,
                 normalization=None, activation='ReLU'):
        super().__init__()
        if depth < 2:
            raise ValueError('depth should be greater than 2.')
        if dimensions == '2':
            conv_block = ConvBlock2d if real else CConvBlock2d
            res_block = _ResBlock2d if real else _CResBlock2d
        elif dimensions == '3':
            conv_block = ConvBlock3d if real else CConvBlock3d
            res_block = _ResBlock3d if real else _CResBlock3d
        elif dimensions == '2+1':
            conv_block = ConvBlock2plus1d if real else CConvBlock2plus1d
            res_block = _ResBlock2plus1d if real else _CResBlock2plus1d
        else:
            raise ValueError('dimensions should be 2 or 3 or 2+1.')
        blocks = [conv_block(in_channels, num_filters, kernel_size,
                             bias, normalization, activation)]
        for _ in range(depth):
            blocks.append(res_block(num_filters, num_filters, kernel_size,
                                    bias, normalization, activation))
        blocks.append(conv_block(num_filters, out_channels,
                                 kernel_size=1, bias=bias))
        self.net = Sequential(*blocks)

    def forward(self, input):
        return self.net(input)


class ResNet2d(_ResNet):
    def __init__(self, in_channels, out_channels, depth,
                 num_filters=64, kernel_size=3, bias=True,
                 normalization=None, activation='ReLU'):
        super().__init__('2', True, in_channels, out_channels,
                         depth, num_filters, kernel_size, bias,
                         normalization, activation)


class ResNet3d(_ResNet):
    def __init__(self, in_channels, out_channels, depth,
                 num_filters=64, kernel_size=3, bias=True,
                 normalization=None, activation='ReLU'):
        super().__init__('3', True, in_channels, out_channels,
                         depth, num_filters, kernel_size, bias,
                         normalization, activation)


class ResNet2plus1d(_ResNet):
    def __init__(self, in_channels, out_channels, depth,
                 num_filters=64, kernel_size=3, bias=True,
                 normalization=None, activation='ReLU'):
        super().__init__('2+1', True, in_channels, out_channels,
                         depth, num_filters, kernel_size, bias,
                         normalization, activation)


class CResNet2d(_ResNet):
    def __init__(self, in_channels, out_channels, depth,
                 num_filters=32, kernel_size=3, bias=True,
                 normalization=None, activation='ReLU'):
        super().__init__('2', False, in_channels, out_channels,
                         depth, num_filters, kernel_size, bias,
                         normalization, activation)


class CResNet3d(_ResNet):
    def __init__(self, in_channels, out_channels, depth,
                 num_filters=32, kernel_size=3, bias=True,
                 normalization=None, activation='ReLU'):
        super().__init__('3', False, in_channels, out_channels,
                         depth, num_filters, kernel_size, bias,
                         normalization, activation)


class CResNet2plus1d(_ResNet):
    def __init__(self, in_channels, out_channels, depth,
                 num_filters=32, kernel_size=3, bias=True,
                 normalization=None, activation='ReLU'):
        super().__init__('2+1', False, in_channels, out_channels,
                         depth, num_filters, kernel_size, bias,
                         normalization, activation)
