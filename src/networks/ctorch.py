import torch
from torch.nn import *


class _CConv(Module):
    def __init__(self, dimensions, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        if dimensions == 1:
            conv = Conv1d
        elif dimensions == 2:
            conv = Conv2d
        elif dimensions == 3:
            conv = Conv3d
        else:
            raise ValueError('dimensions should be 1 or 2 or 3.')
        self.conv_r = conv(in_channels, out_channels, kernel_size, stride,
                           padding, dilation, groups, bias, padding_mode)
        self.conv_i = conv(in_channels, out_channels, kernel_size, stride,
                           padding, dilation, groups, bias, padding_mode)

    def forward(self, input):
        return (self.conv_r(input.real) - self.conv_i(input.imag)) + \
            1j * (self.conv_r(input.imag) + self.conv_i(input.real))


class CConv1d(_CConv):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super().__init__(1, in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode)


class CConv2d(_CConv):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super().__init__(2, in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode)


class CConv3d(_CConv):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super().__init__(3, in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode)


class _CConvTranspose(Module):
    def __init__(self, dimensions, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1,
                 bias=True, dilation=1, padding_mode='zeros'):
        super().__init__()
        if dimensions == 1:
            conv_transpose = ConvTranspose1d
        elif dimensions == 2:
            conv_transpose = ConvTranspose2d
        elif dimensions == 3:
            conv_transpose = ConvTranspose3d
        else:
            raise ValueError('dimensions should be 1 or 2 or 3.')
        self.conv_transpose_r = \
            conv_transpose(in_channels, out_channels, kernel_size,
                           stride, padding, output_padding, groups,
                           bias, dilation, padding_mode)
        self.conv_transpose_i = \
            conv_transpose(in_channels, out_channels, kernel_size,
                           stride, padding, output_padding, groups,
                           bias, dilation, padding_mode)

    def forward(self, input):
        return (self.conv_transpose_r(input.real) -
                self.conv_transpose_i(input.imag)) + \
            1j * (self.conv_transpose_r(input.imag) +
                  self.conv_transpose_i(input.real))


class CConvTranspose1d(_CConvTranspose):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1,
                 bias=True, dilation=1, padding_mode='zeros'):
        super().__init__(1, in_channels, out_channels, kernel_size,
                         stride, padding, output_padding, groups,
                         bias, dilation, padding_mode)


class CConvTranspose2d(_CConvTranspose):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1,
                 bias=True, dilation=1, padding_mode='zeros'):
        super().__init__(2, in_channels, out_channels, kernel_size,
                         stride, padding, output_padding, groups,
                         bias, dilation, padding_mode)


class CConvTranspose3d(_CConvTranspose):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1,
                 bias=True, dilation=1, padding_mode='zeros'):
        super().__init__(3, in_channels, out_channels, kernel_size,
                         stride, padding, output_padding, groups,
                         bias, dilation, padding_mode)


class _CBatchNorm(Module):
    def __init__(self, dimensions, num_features, eps=1e-05, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__()
        if dimensions == 1:
            batch_norm = BatchNorm1d
        elif dimensions == 2:
            batch_norm = BatchNorm2d
        elif dimensions == 3:
            batch_norm = BatchNorm3d
        else:
            raise ValueError('dimensions should be 1 or 2 or 3.')
        self.batch_norm = batch_norm(num_features, eps, momentum,
                                     affine, track_running_stats)

    def forward(self, input):
        return (self.batch_norm(input.abs()) + 1.0) * \
            torch.exp(1j * input.angle())


class CBatchNorm1d(_CBatchNorm):
    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__(1, num_features, eps, momentum,
                         affine, track_running_stats)


class CBatchNorm2d(_CBatchNorm):
    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__(2, num_features, eps, momentum,
                         affine, track_running_stats)


class CBatchNorm3d(_CBatchNorm):
    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__(3, num_features, eps, momentum,
                         affine, track_running_stats)


class _CInstanceNorm(Module):
    def __init__(self, dimensions, num_features, eps=1e-05, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__()
        if dimensions == 1:
            instance_norm = InstanceNorm1d
        elif dimensions == 2:
            instance_norm = InstanceNorm2d
        elif dimensions == 3:
            instance_norm = InstanceNorm3d
        else:
            raise ValueError('dimensions should be 1 or 2 or 3.')
        self.instance_norm = instance_norm(num_features, eps, momentum,
                                           affine, track_running_stats)

    def forward(self, input):
        return (self.instance_norm(input.abs()) + 1.0) * \
            torch.exp(1j * input.angle())


class CInstanceNorm1d(_CInstanceNorm):
    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=True, track_running_stats=False):
        super().__init__(1, num_features, eps, momentum,
                         affine, track_running_stats)


class CInstanceNorm2d(_CInstanceNorm):
    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=True, track_running_stats=False):
        super().__init__(2, num_features, eps, momentum,
                         affine, track_running_stats)


class CInstanceNorm3d(_CInstanceNorm):
    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=True, track_running_stats=False):
        super().__init__(3, num_features, eps, momentum,
                         affine, track_running_stats)


class _CMaxPool(Module):
    def __init__(self, dimensions, kernel_size, stride=None,
                 padding=0, dilation=1):
        super().__init__()
        if dimensions == 1:
            max_pool = MaxPool1d
        elif dimensions == 2:
            max_pool = MaxPool2d
        elif dimensions == 3:
            max_pool = MaxPool3d
        else:
            raise ValueError('dimensions should be 1 or 2 or 3.')
        self.max_pool = max_pool(kernel_size, stride, padding,
                                 dilation, return_indices=True)

    def forward(self, input):
        output, indices = self.max_pool(torch.abs(input))
        output_shape = output.shape
        output = input.flatten(start_dim=2).gather(
            dim=2, index=indices.flatten(start_dim=2)).reshape(output_shape)
        return output


class CMaxPool1d(_CMaxPool):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super().__init__(1, kernel_size, stride, padding, dilation)


class CMaxPool2d(_CMaxPool):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super().__init__(kernel_size, stride, padding, dilation)


class CMaxPool3d(_CMaxPool):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super().__init__(kernel_size, stride, padding, dilation)


class CReLU(Module):
    def __init__(self):
        super().__init__()
        self.relu = ReLU()

    def forward(self, input):
        return self.relu(input.real) + 1j * self.relu(input.imag)


class CLeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.leaky_relu = LeakyReLU(negative_slope)

    def forward(self, input):
        return self.leaky_relu(input.real) + 1j * self.leaky_relu(input.imag)


class CUpsample(Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest',
                 align_corners=None):
        super().__init__()
        self.upsample = Upsample(size, scale_factor, mode, align_corners)

    def forward(self, input):
        return self.upsample(input.real) + 1j * self.upsample(input.imag)
