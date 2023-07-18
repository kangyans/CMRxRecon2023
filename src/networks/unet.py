import torch.nn.functional as F
from conv import *


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


class _DoubleConvBlock2d(_DoubleConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 bias=True, normalization=None, activation=None):
        super().__init__('2', True, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)


class _DoubleConvBlock3d(_DoubleConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 bias=True, normalization=None, activation=None):
        super().__init__('3', True, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)


class _DoubleConvBlock2plus1d(_DoubleConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 bias=True, normalization=None, activation=None):
        super().__init__('2+1', True, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)


class _CDoubleConvBlock2d(_DoubleConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 bias=True, normalization=None, activation=None):
        super().__init__('2', False, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)


class _CDoubleConvBlock3d(_DoubleConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True,
                 normalization=None, activation=None):
        super().__init__('3', False, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)


class _CDoubleConvBlock2plus1d(_DoubleConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True,
                 normalization=None, activation=None):
        super().__init__('2+1', False, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)


class _UpsampleBlock(Module):
    def __init__(self, dimensions, real, in_channels,
                 out_channels, bias=True):
        super().__init__()
        if dimensions == '2':
            upsample = Upsample if real else CUpsample
            conv_block = ConvBlock2d if real else CConvBlock2d
        elif dimensions == '3':
            upsample = Upsample if real else CUpsample
            conv_block = ConvBlock3d if real else CConvBlock3d
        elif dimensions == '2+1':
            upsample = Upsample if real else CUpsample
            conv_block = ConvBlock2plus1d if real else CConvBlock2plus1d
        else:
            raise ValueError('dimensions should be 2 or 3 or 2+1.')
        self.block = Sequential(
            upsample(scale_factor=2, mode='bilinear'),
            conv_block(in_channels, out_channels, kernel_size=1, bias=bias))

    def forward(self, input):
        return self.block(input)


class _UpsampleBlock2d(_UpsampleBlock):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__('2', True, in_channels, out_channels, bias)


class _UpsampleBlock3d(_UpsampleBlock):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__('3', True, in_channels, out_channels, bias)


class _UpsampleBlock2plus1d(_UpsampleBlock):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__('2+1', True, in_channels, out_channels, bias)


class _CUpsampleBlock2d(_UpsampleBlock):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__('2', False, in_channels, out_channels, bias)


class _CUpsampleBlock3d(_UpsampleBlock):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__('3', False, in_channels, out_channels, bias)


class _CUpsampleBlock2plus1d(_UpsampleBlock):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__('2+1', False, in_channels, out_channels, bias)


class _UNet(Module):
    def __init__(self, dimensions, real, in_channels, out_channels,
                 depth, num_filters=64, kernel_size=3, bias=True,
                 normalization=None, activation='ReLU'):
        super().__init__()
        if depth < 2:
            raise ValueError('depth should be greater than 2.')
        if dimensions == '2':
            double_conv_block = _DoubleConvBlock2d if real \
                else _CDoubleConvBlock2d
            downsample_block = MaxPool2d if real else CMaxPool2d
            upsample_block = _UpsampleBlock2d if real else _CUpsampleBlock2d
            conv_block = ConvBlock2d if real else CConvBlock2d
        elif dimensions == '3':
            double_conv_block = _DoubleConvBlock3d if real \
                else _CDoubleConvBlock3d
            downsample_block = MaxPool3d if real else CMaxPool3d
            upsample_block = _UpsampleBlock3d if real else _CUpsampleBlock3d
            conv_block = ConvBlock3d if real else CConvBlock3d
        elif dimensions == '2+1':
            double_conv_block = _DoubleConvBlock2plus1d if real \
                else _CDoubleConvBlock2plus1d
            downsample_block = MaxPool3d if real else CMaxPool3d
            upsample_block = _UpsampleBlock2plus1d if real \
                else _CUpsampleBlock2plus1d
            conv_block = ConvBlock2plus1d if real else CConvBlock2plus1d
        else:
            raise ValueError('dimensions should be 2 or 3 or 2+1.')
        self.dimensions = dimensions
        self.down_conv_blocks = [
            double_conv_block(in_channels, num_filters, kernel_size,
                              bias, normalization, activation)]
        self.downsample_blocks = [
            downsample_block(kernel_size=2)]
        for _ in range(depth - 1):
            self.down_conv_blocks.append(
                double_conv_block(num_filters, num_filters * 2, kernel_size,
                                  bias, normalization, activation))
            self.downsample_blocks.append(
                downsample_block(kernel_size=2))
            num_filters *= 2
        self.bottleneck = double_conv_block(num_filters, num_filters * 2,
                                            kernel_size, bias,
                                            normalization, activation)
        self.upsample_blocks = []
        self.up_conv_blocks = []
        for _ in range(depth - 1):
            self.upsample_blocks.append(
                upsample_block(num_filters * 2, num_filters, bias))
            self.up_conv_blocks.append(
                double_conv_block(num_filters * 2, num_filters, kernel_size,
                                  bias, normalization, activation))
            num_filters //= 2
        self.up_sample_blocks.append(
            upsample_block(num_filters * 2, num_filters, bias))
        self.up_conv_blocks.append(
            double_conv_block(num_filters * 2, num_filters, kernel_size,
                              bias, normalization, activation))
        self.last = conv_block(num_filters, out_channels,
                               kernel_size=1, bias=bias)

    def forward(self, input):
        skip_connections = []
        for down_conv_block, downsample_block in \
                zip(self.down_conv_blocks, self.downsample_blocks):
            input = down_conv_block(input)
            skip_connections.append(input)
            input = downsample_block(input)
        input = self.bottleneck(input)
        for upsample_block, up_conv_block in \
                zip(self.upsample_blocks, self.up_conv_blocks):
            input = upsample_block(input)
            skip_connection = skip_connections.pop()
            if self.dimensions == '2':
                padding = \
                    (0, int(input.shape[-1] != skip_connection.shape[-1]),
                     0, int(input.shape[-2] != skip_connection.shape[-2]))
            else:
                padding = (
                    0, int(input.shape[-1] != skip_connection.shape[-1]),
                    0, int(input.shape[-2] != skip_connection.shape[-2]),
                    0, int(input.shape[-3] != skip_connection.shape[-3]))
            if sum(padding) != 0:
                input = F.pad(input, padding)
            input = up_conv_block(torch.cat([input, skip_connection]), dim=1)
        return self.last(input)


class UNet2d(_UNet):
    def __init__(self, in_channels, out_channels, depth,
                 num_filters=64, kernel_size=3, bias=True,
                 normalization=None, activation='ReLU'):
        super().__init__('2', True, in_channels, out_channels,
                         depth, num_filters, kernel_size, bias,
                         normalization, activation)


class UNet3d(_UNet):
    def __init__(self, in_channels, out_channels, depth,
                 num_filters=64, kernel_size=3, bias=True,
                 normalization=None, activation='ReLU'):
        super().__init__('3', True, in_channels, out_channels,
                         depth, num_filters, kernel_size, bias,
                         normalization, activation)


class UNet2plus1d(_UNet):
    def __init__(self, in_channels, out_channels, depth,
                 num_filters=64, kernel_size=3, bias=True,
                 normalization=None, activation='ReLU'):
        super().__init__('2+1', True, in_channels, out_channels,
                         depth, num_filters, kernel_size, bias,
                         normalization, activation)


class CUNet2d(_UNet):
    def __init__(self, in_channels, out_channels, depth,
                 num_filters=32, kernel_size=3, bias=True,
                 normalization=None, activation='ReLU'):
        super().__init__('2', False, in_channels, out_channels,
                         depth, num_filters, kernel_size, bias,
                         normalization, activation)


class CUNet3d(_UNet):
    def __init__(self, in_channels, out_channels, depth,
                 num_filters=32, kernel_size=3, bias=True,
                 normalization=None, activation='ReLU'):
        super().__init__('3', False, in_channels, out_channels,
                         depth, num_filters, kernel_size, bias,
                         normalization, activation)


class CUNet2plus1d(_UNet):
    def __init__(self, in_channels, out_channels, depth,
                 num_filters=32, kernel_size=3, bias=True,
                 normalization=None, activation='ReLU'):
        super().__init__('2+1', False, in_channels, out_channels,
                         depth, num_filters, kernel_size, bias,
                         normalization, activation)
