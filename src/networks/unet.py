import torch
import torch.nn.functional as F
from conv import *


class DoubleConvBlock(nn.Module):
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
        self.block = nn.Sequential(
            conv_block(in_channels, out_channels, kernel_size,
                       bias, normalization, activation),
            conv_block(out_channels, out_channels, kernel_size,
                       bias, normalization, activation))

    def forward(self, input):
        return self.block(input)


class DoubleConvBlock2d(DoubleConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 bias=True, normalization=None, activation=None):
        super().__init__('2', True, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)


class DoubleConvBlock3d(DoubleConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 bias=True, normalization=None, activation=None):
        super().__init__('3', True, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)


class DoubleConvBlock2plus1d(DoubleConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 bias=True, normalization=None, activation=None):
        super().__init__('2+1', True, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)


class CDoubleConvBlock2d(DoubleConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 bias=True, normalization=None, activation=None):
        super().__init__('2', False, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)


class CDoubleConvBlock3d(DoubleConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True,
                 normalization=None, activation=None):
        super().__init__('3', False, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)


class CDoubleConvBlock2plus1d(DoubleConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True,
                 normalization=None, activation=None):
        super().__init__('2+1', False, in_channels, out_channels,
                         kernel_size, bias, normalization, activation)


class UpsampleBlock(nn.Module):
    def __init__(self, dimensions, real, in_channels,
                 out_channels, bias=True):
        super().__init__()
        if dimensions == '2':
            upsample = nn.Upsample if real else cnn.CUpsample
            conv_block = ConvBlock2d if real else CConvBlock2d
        elif dimensions == '3':
            upsample = nn.Upsample if real else cnn.CUpsample
            conv_block = ConvBlock3d if real else CConvBlock3d
        elif dimensions == '2+1':
            upsample = nn.Upsample if real else cnn.CUpsample
            conv_block = ConvBlock2plus1d if real else CConvBlock2plus1d
        else:
            raise ValueError('dimensions should be 2 or 3 or 2+1.')
        self.block = nn.Sequential(
            upsample(scale_factor=2, mode='bilinear'),
            conv_block(in_channels, out_channels, kernel_size=1, bias=bias))

    def forward(self, input):
        return self.block(input)


class UpsampleBlock2d(UpsampleBlock):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__('2', True, in_channels, out_channels, bias)


class UpsampleBlock3d(UpsampleBlock):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__('3', True, in_channels, out_channels, bias)


class UpsampleBlock2plus1d(UpsampleBlock):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__('2+1', True, in_channels, out_channels, bias)


class CUpsampleBlock2d(UpsampleBlock):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__('2', False, in_channels, out_channels, bias)


class CUpsampleBlock3d(UpsampleBlock):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__('3', False, in_channels, out_channels, bias)


class CUpsampleBlock2plus1d(UpsampleBlock):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__('2+1', False, in_channels, out_channels, bias)


class UNet(nn.Module):
    def __init__(self, dimensions, real, in_channels, out_channels,
                 depth, num_filters=64, kernel_size=3, bias=True,
                 normalization=None, activation='ReLU'):
        super().__init__()
        if depth < 2:
            raise ValueError('depth should be greater than 2.')
        if dimensions == '2':
            double_conv_block = DoubleConvBlock2d if real \
                else CDoubleConvBlock2d
            downsample_block = nn.MaxPool2d if real else cnn.CMaxPool2d
            upsample_block = UpsampleBlock2d if real else CUpsampleBlock2d
            conv_block = ConvBlock2d if real else CConvBlock2d
        elif dimensions == '3':
            double_conv_block = DoubleConvBlock3d if real \
                else CDoubleConvBlock3d
            downsample_block = nn.MaxPool3d if real else cnn.CMaxPool3d
            upsample_block = UpsampleBlock3d if real else CUpsampleBlock3d
            conv_block = ConvBlock3d if real else CConvBlock3d
        elif dimensions == '2+1':
            double_conv_block = DoubleConvBlock2plus1d if real \
                else CDoubleConvBlock2plus1d
            downsample_block = nn.MaxPool3d if real else cnn.CMaxPool3d
            upsample_block = UpsampleBlock2plus1d if real \
                else CUpsampleBlock2plus1d
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


class UNet2d(UNet):
    def __init__(self, in_channels, out_channels, depth,
                 num_filters=64, kernel_size=3, bias=True,
                 normalization=None, activation='ReLU'):
        super().__init__('2', True, in_channels, out_channels,
                         depth, num_filters, kernel_size, bias,
                         normalization, activation)


class UNet3d(UNet):
    def __init__(self, in_channels, out_channels, depth,
                 num_filters=64, kernel_size=3, bias=True,
                 normalization=None, activation='ReLU'):
        super().__init__('3', True, in_channels, out_channels,
                         depth, num_filters, kernel_size, bias,
                         normalization, activation)


class UNet2plus1d(UNet):
    def __init__(self, in_channels, out_channels, depth,
                 num_filters=64, kernel_size=3, bias=True,
                 normalization=None, activation='ReLU'):
        super().__init__('2+1', True, in_channels, out_channels,
                         depth, num_filters, kernel_size, bias,
                         normalization, activation)


class CUNet2d(UNet):
    def __init__(self, in_channels, out_channels, depth,
                 num_filters=32, kernel_size=3, bias=True,
                 normalization=None, activation='ReLU'):
        super().__init__('2', False, in_channels, out_channels,
                         depth, num_filters, kernel_size, bias,
                         normalization, activation)


class CUNet3d(UNet):
    def __init__(self, in_channels, out_channels, depth,
                 num_filters=32, kernel_size=3, bias=True,
                 normalization=None, activation='ReLU'):
        super().__init__('3', False, in_channels, out_channels,
                         depth, num_filters, kernel_size, bias,
                         normalization, activation)


class CUNet2plus1d(UNet):
    def __init__(self, in_channels, out_channels, depth,
                 num_filters=32, kernel_size=3, bias=True,
                 normalization=None, activation='ReLU'):
        super().__init__('2+1', False, in_channels, out_channels,
                         depth, num_filters, kernel_size, bias,
                         normalization, activation)
