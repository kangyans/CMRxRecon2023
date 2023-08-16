from .ctorch import *
import torch.nn.functional as F


class _DoubleConvBlock(Module):
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 kernel_size=3, bias=False, normalization='instance',
                 activation='relu'):
        super().__init__()
        if normalization != 'instance' and normalization != 'batch':
            raise ValueError('normalization should be or instance or batch.')
        if activation != 'relu' and activation != 'leakyrelu':
            raise ValueError('activation should be relu or leakyrelu.')
        if not mid_channels:
            mid_channels = out_channels
        self.block = Sequential(
            Conv2d(in_channels, mid_channels, kernel_size, bias=bias),
            InstanceNorm2d(mid_channels) if normalization == 'instance'
            else BatchNorm2d(mid_channels),
            ReLU(inplace=True) if activation == 'relu'
            else LeakyReLU(negative_slope=0.1, inplace=True),
            Conv2d(mid_channels, out_channels, kernel_size, bias=bias),
            InstanceNorm2d(out_channels) if normalization == 'instance'
            else BatchNorm2d(out_channels),
            ReLU(inplace=True) if activation == 'relu'
            else LeakyReLU(negative_slope=0.1, inplace=True))

    def forward(self, input):
        return self.block(input)


class _CDoubleConvBlock(Module):
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 kernel_size=3, bias=False, normalization='instance',
                 activation='relu'):
        super().__init__()
        if normalization != 'instance' and normalization != 'batch':
            raise ValueError('normalization should be or instance or batch.')
        if activation != 'relu' and activation != 'leakyrelu':
            raise ValueError('activation should be relu or leakyrelu.')
        if not mid_channels:
            mid_channels = out_channels
        self.block = Sequential(
            CConv2d(in_channels, mid_channels, kernel_size, bias=bias),
            CInstanceNorm2d(mid_channels) if normalization == 'instance'
            else CBatchNorm2d(mid_channels),
            CReLU(inplace=True) if activation == 'relu'
            else CLeakyReLU(negative_slope=0.1, inplace=True),
            CConv2d(mid_channels, out_channels, kernel_size, bias=bias),
            CInstanceNorm2d(out_channels) if normalization == 'instance'
            else CBatchNorm2d(out_channels),
            CReLU(inplace=True) if activation == 'relu'
            else CLeakyReLU(negative_slope=0.1, inplace=True))

    def forward(self, input):
        return self.block(input)


class _DownBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False,
                 normalization='instance', activation='relu', down='avg'):
        super().__init__()
        if down != 'avg' and down != 'max':
            raise ValueError('down should be avg or max.')
        self.down_sample = AvgPool2d(kernel_size=2) if down == 'avg' \
            else MaxPool2d(kernel_size=2)
        self.down_conv = _DoubleConvBlock(
            in_channels, out_channels, None,
            kernel_size, bias, normalization, activation)

    def forward(self, input):
        return self.down_conv(self.down_sample(input))


class _CDownBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False,
                 normalization='instance', activation='relu', down='avg'):
        super().__init__()
        if down != 'avg' and down != 'max':
            raise ValueError('down should be avg or max.')
        self.down_sample = CAvgPool2d(kernel_size=2) if down == 'avg' \
            else CMaxPool2d(kernel_size=2)
        self.down_conv = _CDoubleConvBlock(
            in_channels, out_channels, None,
            kernel_size, bias, normalization, activation)

    def forward(self, input):
        return self.down_conv(self.down_sample(input))


class _UpBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False,
                 normalization='instance', activation='relu', up='bilinear'):
        super().__init__()
        if up != 'bilinear' and up != 'convtranspose':
            raise ValueError('up should be bilinear or convtranspose.')
        if up == 'bilinear':
            self.up_sample = UpsamplingBilinear2d(scale_factor=2)
            self.up_conv = _DoubleConvBlock(
                in_channels, out_channels, in_channels // 2,
                kernel_size, bias, normalization, activation)
        else:
            self.up_sample = ConvTranspose2d(
                in_channels, in_channels // 2,
                kernel_size=2, stride=2, bias=bias)
            self.up_conv = _DoubleConvBlock(
                in_channels, out_channels, None,
                kernel_size, bias, normalization, activation)

    def forward(self, input, skip_connection):
        input = self.up_sample(input)
        diffw = skip_connection.shape[-1] - input.shape[-1]
        diffh = skip_connection.shape[-2] - input.shape[-2]
        input = F.pad(input, (diffw // 2, diffw - diffw // 2,
                              diffh // 2, diffh - diffh // 2))
        return self.up_conv(torch.cat((skip_connection, input), dim=1))


class _CUpBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False,
                 normalization='instance', activation='relu', up='bilinear'):
        super().__init__()
        if up != 'bilinear' and up != 'convtranspose':
            raise ValueError('up should be bilinear or convtranspose.')
        if up == 'bilinear':
            self.up_sample = CUpsamplingBilinear2d(scale_factor=2)
            self.up_conv = _CDoubleConvBlock(
                in_channels, out_channels, in_channels // 2,
                kernel_size, bias, normalization, activation)
        else:
            self.up_sample = CConvTranspose2d(
                in_channels, in_channels // 2,
                kernel_size=2, stride=2, bias=bias)
            self.up_conv = _CDoubleConvBlock(
                in_channels, out_channels, None,
                kernel_size, bias, normalization, activation)

    def forward(self, input, skip_connection):
        input = self.up_sample(input)
        diffw = skip_connection.shape[-1] - input.shape[-1]
        diffh = skip_connection.shape[-2] - input.shape[-2]
        input = F.pad(input, (diffw // 2, diffw - diffw // 2,
                              diffh // 2, diffh - diffh // 2))
        return self.up_conv(torch.cat((skip_connection, input), dim=1))


class UNet(Module):
    def __init__(self, in_channels, out_channels, depth=4, num_filters=64,
                 kernel_size=3, bias=False, normalization='instance',
                 activation='ReLU', down='avg', up='bilinear'):
        super().__init__()
        factor = 2 if up == 'bilinear' else 1
        self.first = _DoubleConvBlock(
            in_channels, num_filters, None,
            kernel_size, bias, normalization, activation)
        self.downs = ModuleList([_DownBlock(
            num_filters, num_filters * 2, kernel_size,
            bias, normalization, activation, down)])
        num_filters *= 2
        for _ in range(1, depth - 1):
            self.downs.append(_DownBlock(
                num_filters, num_filters * 2, kernel_size,
                bias, normalization, activation, down))
            num_filters *= 2
        self.downs.append(_DownBlock(
            num_filters, num_filters * 2 // factor, kernel_size,
            bias, normalization, activation, down))
        self.ups = ModuleList()
        for _ in range(depth - 1):
            self.ups.append(_UpBlock(
                num_filters * 2, num_filters // factor, kernel_size,
                bias, normalization, activation, up))
            num_filters //= 2
        self.ups.append(_UpBlock(
            num_filters * 2, num_filters, kernel_size,
            bias, normalization, activation, up))
        self.last = Conv2d(
            num_filters, out_channels, kernel_size=1, bias=bias)

    def forward(self, input):
        output = self.first(input)
        skip_connections = [output]
        for down in self.downs[:-1]:
            output = down(output)
            skip_connections.append(output)
        output = self.down4(output)
        for up in self.ups:
            output = up(output, skip_connections.pop())
        output = self.last(output)
        return output


class CUNet(Module):
    def __init__(self, in_channels, out_channels, depth=4, num_filters=32,
                 kernel_size=3, bias=False, normalization='instance',
                 activation='ReLU', down='avg', up='bilinear'):
        super().__init__()
        factor = 2 if up == 'bilinear' else 1
        self.first = _CDoubleConvBlock(
            in_channels, num_filters, None,
            kernel_size, bias, normalization, activation)
        self.downs = ModuleList([_CDownBlock(
            num_filters, num_filters * 2, kernel_size,
            bias, normalization, activation, down)])
        num_filters *= 2
        for _ in range(1, depth - 1):
            self.downs.append(_CDownBlock(
                num_filters, num_filters * 2, kernel_size,
                bias, normalization, activation, down))
            num_filters *= 2
        self.downs.append(_CDownBlock(
            num_filters, num_filters * 2 // factor, kernel_size,
            bias, normalization, activation, down))
        self.ups = ModuleList()
        for _ in range(depth - 1):
            self.ups.append(_CUpBlock(
                num_filters * 2, num_filters // factor, kernel_size,
                bias, normalization, activation, up))
            num_filters //= 2
        self.ups.append(_CUpBlock(
            num_filters * 2, num_filters, kernel_size,
            bias, normalization, activation, up))
        self.last = CConv2d(
            num_filters, out_channels, kernel_size=1, bias=bias)

    def forward(self, input):
        output = self.first(input)
        skip_connections = [output]
        for down in self.downs[:-1]:
            output = down(output)
            skip_connections.append(output)
        output = self.down4(output)
        for up in self.ups:
            output = up(output, skip_connections.pop())
        output = self.last(output)
        return output
