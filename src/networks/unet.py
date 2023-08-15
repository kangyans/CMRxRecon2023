from .ctorch import *
import torch.nn.functional as F


class DoubleConvBlock(Module):
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


class CDoubleConvBlock(Module):
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


class DownBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False,
                 normalization='instance', activation='relu', down='avg'):
        super().__init__()
        if down != 'avg' and down != 'max':
            raise ValueError('down should be avg or max.')
        self.down_sample = AvgPool2d(kernel_size=2) if down == 'avg' \
            else MaxPool2d(kernel_size=2)
        self.down_conv = DoubleConvBlock(
            in_channels, out_channels, None,
            kernel_size, bias, normalization, activation)

    def forward(self, input):
        return self.down_conv(self.down_sample(input))


class CDownBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False,
                 normalization='instance', activation='relu', down='avg'):
        super().__init__()
        if down != 'avg' and down != 'max':
            raise ValueError('down should be avg or max.')
        self.down_sample = CAvgPool2d(kernel_size=2) if down == 'avg' \
            else CMaxPool2d(kernel_size=2)
        self.down_conv = CDoubleConvBlock(
            in_channels, out_channels, None,
            kernel_size, bias, normalization, activation)

    def forward(self, input):
        return self.down_conv(self.down_sample(input))


class UpBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False,
                 normalization='instance', activation='relu', up='bilinear'):
        super().__init__()
        if up != 'bilinear' and up != 'convtranspose':
            raise ValueError('up should be bilinear or convtranspose.')
        if up == 'bilinear':
            self.up_sample = UpsamplingBilinear2d(scale_factor=2)
            self.up_conv = DoubleConvBlock(
                in_channels, out_channels, in_channels // 2,
                kernel_size, bias, normalization, activation)
        else:
            self.up_sample = ConvTranspose2d(
                in_channels, in_channels // 2,
                kernel_size=2, stride=2, bias=bias)
            self.up_conv = DoubleConvBlock(
                in_channels, out_channels, None,
                kernel_size, bias, normalization, activation)

    def forward(self, input, skip_connection):
        input = self.up_sample(input)
        diffw = skip_connection.shape[-1] - input.shape[-1]
        diffh = skip_connection.shape[-2] - input.shape[-2]
        input = F.pad(input, (diffw // 2, diffw - diffw // 2,
                              diffh // 2, diffh - diffh // 2))
        return self.up_conv(torch.cat((skip_connection, input), dim=1))


class CUpBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False,
                 normalization='instance', activation='relu', up='bilinear'):
        super().__init__()
        if up != 'bilinear' and up != 'convtranspose':
            raise ValueError('up should be bilinear or convtranspose.')
        if up == 'bilinear':
            self.up_sample = CUpsamplingBilinear2d(scale_factor=2)
            self.up_conv = CDoubleConvBlock(
                in_channels, out_channels, in_channels // 2,
                kernel_size, bias, normalization, activation)
        else:
            self.up_sample = CConvTranspose2d(
                in_channels, in_channels // 2,
                kernel_size=2, stride=2, bias=bias)
            self.up_conv = CDoubleConvBlock(
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
    def __init__(self, in_channels, out_channels, num_filters=64,
                 kernel_size=3, bias=False, normalization='instance',
                 activation='ReLU', down='avg', up='bilinear'):
        super().__init__()
        factor = 2 if up == 'bilinear' else 1
        self.first = DoubleConvBlock(
            in_channels, num_filters, None,
            kernel_size, bias, normalization, activation)
        self.down1 = DownBlock(
            num_filters, num_filters * 2, kernel_size,
            bias, normalization, activation, down)
        self.down2 = DownBlock(
            num_filters * 2, num_filters * 4, kernel_size,
            bias, normalization, activation, down)
        self.down3 = DownBlock(
            num_filters * 4, num_filters * 8, kernel_size,
            bias, normalization, activation, down)
        self.down4 = DownBlock(
            num_filters * 8, num_filters * 16 // factor, kernel_size,
            bias, normalization, activation, down)
        self.up4 = UpBlock(
            num_filters * 16, num_filters * 8 // factor, kernel_size,
            bias, normalization, activation, up)
        self.up3 = UpBlock(
            num_filters * 8, num_filters * 4 // factor, kernel_size,
            bias, normalization, activation, up)
        self.up2 = UpBlock(
            num_filters * 4, num_filters * 2 // factor, kernel_size,
            bias, normalization, activation, up)
        self.up1 = UpBlock(
            num_filters * 2, num_filters, kernel_size,
            bias, normalization, activation, up)
        self.last = Conv2d(
            num_filters, out_channels, kernel_size=1, bias=bias)

    def forward(self, input):
        skip_connection1 = self.first(input)
        skip_connection2 = self.down1(skip_connection1)
        skip_connection3 = self.down2(skip_connection2)
        skip_connection4 = self.down3(skip_connection3)
        output = self.down4(skip_connection4)
        output = self.up4(output, skip_connection4)
        output = self.up3(output, skip_connection3)
        output = self.up2(output, skip_connection2)
        output = self.up1(output, skip_connection1)
        output = self.last(output)
        return output


class CUNet(Module):
    def __init__(self, in_channels, out_channels, num_filters=64,
                 kernel_size=3, bias=False, normalization='instance',
                 activation='ReLU', down='avg', up='bilinear'):
        super().__init__()
        factor = 2 if up == 'bilinear' else 1
        self.first = CDoubleConvBlock(
            in_channels, num_filters, None,
            kernel_size, bias, normalization, activation)
        self.down1 = CDownBlock(
            num_filters, num_filters * 2, kernel_size,
            bias, normalization, activation, down)
        self.down2 = CDownBlock(
            num_filters * 2, num_filters * 4, kernel_size,
            bias, normalization, activation, down)
        self.down3 = CDownBlock(
            num_filters * 4, num_filters * 8, kernel_size,
            bias, normalization, activation, down)
        self.down4 = CDownBlock(
            num_filters * 8, num_filters * 16 // factor, kernel_size,
            bias, normalization, activation, down)
        self.up4 = CUpBlock(
            num_filters * 16, num_filters * 8 // factor, kernel_size,
            bias, normalization, activation, up)
        self.up3 = CUpBlock(
            num_filters * 8, num_filters * 4 // factor, kernel_size,
            bias, normalization, activation, up)
        self.up2 = CUpBlock(
            num_filters * 4, num_filters * 2 // factor, kernel_size,
            bias, normalization, activation, up)
        self.up1 = CUpBlock(
            num_filters * 2, num_filters, kernel_size,
            bias, normalization, activation, up)
        self.last = CConv2d(
            num_filters, out_channels, kernel_size=1, bias=bias)

    def forward(self, input):
        skip_connection1 = self.first(input)
        skip_connection2 = self.down1(skip_connection1)
        skip_connection3 = self.down2(skip_connection2)
        skip_connection4 = self.down3(skip_connection3)
        output = self.down4(skip_connection4)
        output = self.up4(output, skip_connection4)
        output = self.up3(output, skip_connection3)
        output = self.up2(output, skip_connection2)
        output = self.up1(output, skip_connection1)
        output = self.last(output)
        return output
