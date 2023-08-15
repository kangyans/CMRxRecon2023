import torch
from torch.nn import *


class CConv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super().__init__()
        self.conv_r = Conv2d(in_channels, out_channels, kernel_size,
                             stride, padding, dilation, groups,
                             bias, padding_mode)
        self.conv_i = Conv2d(in_channels, out_channels, kernel_size,
                             stride, padding, dilation, groups,
                             bias, padding_mode)

    def forward(self, input):
        return (self.conv_r(input.real) - self.conv_i(input.imag)) + \
            1j * (self.conv_r(input.imag) + self.conv_i(input.real))


class CConvTranspose2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1,
                 bias=True, dilation=1, padding_mode='zeros'):
        super().__init__()
        self.conv_transpose_r = \
            ConvTranspose2d(in_channels, out_channels, kernel_size,
                            stride, padding, output_padding, groups,
                            bias, dilation, padding_mode)
        self.conv_transpose_i = \
            ConvTranspose2d(in_channels, out_channels, kernel_size,
                            stride, padding, output_padding, groups,
                            bias, dilation, padding_mode)

    def forward(self, input):
        return (self.conv_transpose_r(input.real) -
                self.conv_transpose_i(input.imag)) + \
            1j * (self.conv_transpose_r(input.imag) +
                  self.conv_transpose_i(input.real))


class CBatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__()
        self.batch_norm = BatchNorm2d(num_features, eps, momentum,
                                      affine, track_running_stats)

    def forward(self, input):
        return (self.batch_norm(input.abs()) + 1.0) * \
            torch.exp(1j * input.angle())


class CInstanceNorm2d(Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__()
        self.instance_norm = InstanceNorm2d(num_features, eps, momentum,
                                            affine, track_running_stats)

    def forward(self, input):
        return (self.instance_norm(input.abs()) + 1.0) * \
            torch.exp(1j * input.angle())


class CMaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super().__init__()
        self.max_pool = MaxPool2d(kernel_size, stride, padding,
                                  dilation, return_indices=True)

    def forward(self, input):
        output, indices = self.max_pool(torch.abs(input))
        output_shape = output.shape
        output = input.flatten(start_dim=2).gather(
            dim=2, index=indices.flatten(start_dim=2)).reshape(output_shape)
        return output


class CAvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.avg_pool = AvgPool2d(kernel_size, stride, padding)

    def forward(self, input):
        return self.avg_pool(input.real) + 1j * self.avg_pool(input.imag)


class CReLU(Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.relu = ReLU(inplace)

    def forward(self, input):
        return self.relu(input.real) + 1j * self.relu(input.imag)


class CLeakyReLU(Module):
    def __init__(self, negative_slope=0.01, inplace=False):
        super().__init__()
        self.leaky_relu = LeakyReLU(negative_slope, inplace)

    def forward(self, input):
        return self.leaky_relu(input.real) + 1j * self.leaky_relu(input.imag)


class CUpsamplingBilinear2d(Module):
    def __init__(self, size=None, scale_factor=None):
        super().__init__()
        self.upsample = UpsamplingBilinear2d(size, scale_factor)

    def forward(self, input):
        return self.upsample(input.real) + 1j * self.upsample(input.imag)
