import torch
from torch import nn


class CConv(nn.Module):
    def __init__(self, dimensions, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        if dimensions == 1:
            conv = nn.Conv1d
        elif dimensions == 2:
            conv = nn.Conv2d
        elif dimensions == 3:
            conv = nn.Conv3d
        else:
            raise ValueError('dimensions should be 1 or 2 or 3.')
        self.conv_r = conv(in_channels, out_channels, kernel_size, stride,
                           padding, dilation, groups, bias, padding_mode)
        self.conv_i = conv(in_channels, out_channels, kernel_size, stride,
                           padding, dilation, groups, bias, padding_mode)

    def forward(self, input):
        return (self.conv_r(input.real) - self.conv_i(input.imag)) + \
            1j * (self.conv_r(input.imag) + self.conv_i(input.real))


class CConv1d(CConv):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super().__init__(1, in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode)


class CConv2d(CConv):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super().__init__(2, in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode)


class CConv3d(CConv):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super().__init__(3, in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode)


class CConvTranspose(nn.Module):
    def __init__(self, dimensions, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1,
                 bias=True, dilation=1, padding_mode='zeros'):
        super().__init__()
        if dimensions == 1:
            conv_transpose = nn.ConvTranspose1d
        elif dimensions == 2:
            conv_transpose = nn.ConvTranspose2d
        elif dimensions == 3:
            conv_transpose = nn.ConvTranspose3d
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


class CConvTranspose1d(CConvTranspose):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1,
                 bias=True, dilation=1, padding_mode='zeros'):
        super().__init__(1, in_channels, out_channels, kernel_size,
                         stride, padding, output_padding, groups,
                         bias, dilation, padding_mode)


class CConvTranspose2d(CConvTranspose):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1,
                 bias=True, dilation=1, padding_mode='zeros'):
        super().__init__(2, in_channels, out_channels, kernel_size,
                         stride, padding, output_padding, groups,
                         bias, dilation, padding_mode)


class CConvTranspose3d(CConvTranspose):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1,
                 bias=True, dilation=1, padding_mode='zeros'):
        super().__init__(3, in_channels, out_channels, kernel_size,
                         stride, padding, output_padding, groups,
                         bias, dilation, padding_mode)


class CBatchNorm(nn.Module):
    def __init__(self, dimensions, num_features, eps=1e-05, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__()
        if dimensions == 1:
            batch_norm = nn.BatchNorm1d
        elif dimensions == 2:
            batch_norm = nn.BatchNorm2d
        elif dimensions == 3:
            batch_norm = nn.BatchNorm3d
        else:
            raise ValueError('dimensions should be 1 or 2 or 3.')
        self.batch_norm = batch_norm(num_features, eps, momentum,
                                     affine, track_running_stats)

    def forward(self, input):
        return (self.batch_norm(input.abs()) + 1.0) * \
            torch.exp(1j * input.angle())


class CBatchNorm1d(CBatchNorm):
    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__(1, num_features, eps, momentum,
                         affine, track_running_stats)


class CBatchNorm2d(CBatchNorm):
    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__(2, num_features, eps, momentum,
                         affine, track_running_stats)


class CBatchNorm3d(CBatchNorm):
    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__(3, num_features, eps, momentum,
                         affine, track_running_stats)


class CInstanceNorm(nn.Module):
    def __init__(self, dimensions, num_features, eps=1e-05, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__()
        if dimensions == 1:
            instance_norm = nn.InstanceNorm1d
        elif dimensions == 2:
            instance_norm = nn.InstanceNorm2d
        elif dimensions == 3:
            instance_norm = nn.InstanceNorm3d
        else:
            raise ValueError('dimensions should be 1 or 2 or 3.')
        self.instance_norm = instance_norm(num_features, eps, momentum,
                                           affine, track_running_stats)

    def forward(self, input):
        return (self.instance_norm(input.abs()) + 1.0) * \
            torch.exp(1j * input.angle())


class CInstanceNorm1d(CInstanceNorm):
    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=True, track_running_stats=False):
        super().__init__(1, num_features, eps, momentum,
                         affine, track_running_stats)


class CInstanceNorm2d(CInstanceNorm):
    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=True, track_running_stats=False):
        super().__init__(2, num_features, eps, momentum,
                         affine, track_running_stats)


class CInstanceNorm3d(CInstanceNorm):
    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=True, track_running_stats=False):
        super().__init__(3, num_features, eps, momentum,
                         affine, track_running_stats)


class CMaxPool(nn.Module):
    def __init__(self, dimensions, kernel_size, stride=None,
                 padding=0, dilation=1):
        super().__init__()
        if dimensions == 1:
            max_pool = nn.MaxPool1d
        elif dimensions == 2:
            max_pool = nn.MaxPool2d
        elif dimensions == 3:
            max_pool = nn.MaxPool3d
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


class CMaxPool1d(CMaxPool):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super().__init__(1, kernel_size, stride, padding, dilation)


class CMaxPool2d(CMaxPool):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super().__init__(kernel_size, stride, padding, dilation)


class CMaxPool3d(CMaxPool):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super().__init__(kernel_size, stride, padding, dilation)


class CReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, input):
        return self.relu(input.real) + 1j * self.relu(input.imag)


class CLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, input):
        return self.leaky_relu(input.real) + 1j * self.leaky_relu(input.imag)


class CUpsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest',
                 align_corners=None):
        super().__init__()
        self.upsample = nn.Upsample(size, scale_factor, mode, align_corners)

    def forward(self, input):
        return self.upsample(input.real) + 1j * self.upsample(input.imag)
