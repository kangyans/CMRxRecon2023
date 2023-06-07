import torch
from torch import nn


class CConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        super().__init__()
        self.conv_r = nn.Conv1d(in_channels, out_channels, kernel_size,
                                stride, padding, dilation, groups, bias,
                                padding_mode)
        self.conv_i = nn.Conv1d(in_channels, out_channels, kernel_size,
                                stride, padding, dilation, groups, bias,
                                padding_mode)

    def forward(self, input):
        return (self.conv_r(input.real) - self.conv_i(input.imag)) + \
            1j * (self.conv_r(input.imag) + self.conv_i(input.real))


class CConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        super().__init__()
        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size,
                                stride, padding, dilation, groups, bias,
                                padding_mode)
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size,
                                stride, padding, dilation, groups, bias,
                                padding_mode)

    def forward(self, input):
        return (self.conv_r(input.real) - self.conv_i(input.imag)) + \
            1j * (self.conv_r(input.imag) + self.conv_i(input.real))


class CConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        super().__init__()
        self.conv_r = nn.Conv3d(in_channels, out_channels, kernel_size,
                                stride, padding, dilation, groups, bias,
                                padding_mode)
        self.conv_i = nn.Conv3d(in_channels, out_channels, kernel_size,
                                stride, padding, dilation, groups, bias,
                                padding_mode)

    def forward(self, input):
        return (self.conv_r(input.real) - self.conv_i(input.imag)) + \
            1j * (self.conv_r(input.imag) + self.conv_i(input.real))


class CConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1,
                 padding_mode='zeros'):
        super().__init__()
        self.convtranspose_r = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, padding,
            output_padding, groups, bias, dilation, padding_mode)
        self.convtranspose_i = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, padding,
            output_padding, groups, bias, dilation, padding_mode)

    def forward(self, input):
        return (self.convtranspose_r(input.real) -
                self.convtranspose_i(input.imag)) + \
            1j * (self.convtranspose_r(input.imag) +
                  self.convtranspose_i(input.real))


class CConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1,
                 padding_mode='zeros'):
        super().__init__()
        self.convtranspose_r = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding,
            output_padding, groups, bias, dilation, padding_mode)
        self.convtranspose_i = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding,
            output_padding, groups, bias, dilation, padding_mode)
    
    def forward(self, input):
        return (self.convtranspose_r(input.real) - 
                self.convtranspose_i(input.imag)) + \
                    1j * (self.convtranspose_r(input.imag) + 
                          self.convtranspose_i(input.real))


class CConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1,
                 padding_mode='zeros'):
        super().__init__()
        self.convtranspose_r = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride, padding,
            output_padding, groups, bias, dilation, padding_mode)
        self.convtranspose_i = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride, padding,
            output_padding, groups, bias, dilation, padding_mode)

    def forward(self, input):
        return (self.convtranspose_r(input.real) -
                self.convtranspose_i(input.imag)) + \
            1j * (self.convtranspose_r(input.imag) +
                  self.convtranspose_i(input.real))


class CBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features, eps, momentum,
                                         affine, track_running_stats)

    def forward(self, input):
        return (self.batch_norm(input.abs()) + 1.0) * \
            torch.exp(1j * input.angle())


class CBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(num_features, eps, momentum,
                                         affine, track_running_stats)
    
    def forward(self, input):
        return (self.batch_norm(input.abs()) + 1.0) * \
            torch.exp(1j * input.angle())


class CBatchNorm3d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.batch_norm = nn.BatchNorm3d(num_features, eps, momentum,
                                         affine, track_running_stats)
    
    def forward(self, input):
        return (self.batch_norm(input.abs()) + 1.0) * \
            torch.exp(1j * input.angle())


class CInstanceNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True,
                 track_running_stats=False):
        super().__init__()
        self.instance_norm = nn.InstanceNorm1d(num_features, eps, momentum,
                                               affine, track_running_stats)

    def forward(self, input):
        return (self.instance_norm(input.abs()) + 1.0) * \
            torch.exp(1j * input.angle())


class CInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True,
                 track_running_stats=False):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(num_features, eps, momentum,
                                               affine, track_running_stats)

    def forward(self, input):
        return (self.instance_norm(input.abs()) + 1.0) * \
            torch.exp(1j * input.angle())


class CInstanceNorm3d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True,
                 track_running_stats=False):
        super().__init__()
        self.instance_norm = nn.InstanceNorm3d(num_features, eps, momentum,
                                               affine, track_running_stats)

    def forward(self, input):
        return (self.instance_norm(input.abs()) + 1.0) * \
            torch.exp(1j * input.angle())


class CMaxPool1d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super().__init__()
        self.max_pool = nn.MaxPool1d(kernel_size, stride, padding, dilation,
                                     return_indices=True)

    def forward(self, input):
        output, indices = self.max_pool(torch.abs(input))
        output_shape = output.shape
        output = input.flatten(start_dim=2).gather(
            dim=2, index=indices.flatten(start_dim=2)).reshape(output_shape)
        return output


class CMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size, stride, padding, dilation,
                                     return_indices=True)

    def forward(self, input):
        output, indices = self.max_pool(torch.abs(input))
        output_shape = output.shape
        output = input.flatten(start_dim=2).gather(
            dim=2, index=indices.flatten(start_dim=2)).reshape(output_shape)
        return output


class CMaxPool3d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super().__init__()
        self.max_pool = nn.MaxPool3d(kernel_size, stride, padding, dilation,
                                     return_indices=True)

    def forward(self, input):
        output, indices = self.max_pool(torch.abs(input))
        output_shape = output.shape
        output = input.flatten(start_dim=2).gather(
            dim=2, index=indices.flatten(start_dim=2)).reshape(output_shape)
        return output


class CReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, input):
        return self.relu(input.real) + 1j * self.relu(input.imag)


class CUpSample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest',
                 align_corners=None):
        super().__init__()
        self.up_sample = nn.Upsample(size, scale_factor, mode, align_corners)

    def forward(self, input):
        return self.up_sample(input.real) + 1j * self.up_sample(input.imag)
