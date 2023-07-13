from conv import *


class ResNet(nn.Module):
    def __init__(self, dimensions, real, in_channels, out_channels,
                 depth, num_filters=64, kernel_size=3, bias=True,
                 normalization=None, activation='ReLU'):
        super().__init__()
        if depth < 2:
            raise ValueError('depth should be greater than 2.')
        if dimensions == '2':
            conv_block = ConvBlock2d if real else CConvBlock2d
        elif dimensions == '3':
            conv_block = ConvBlock3d if real else CConvBlock3d
        elif dimensions == '2+1':
            conv_block = ConvBlock2plus1d if real else CConvBlock2plus1d
        else:
            raise ValueError('dimensions should be 2 or 3 or 2+1.')
        blocks = [conv_block(in_channels, num_filters, kernel_size,
                             bias, normalization, activation)]
        for _ in range(depth - 2):
            blocks.append(conv_block(num_filters, num_filters, kernel_size,
                                     bias, normalization, activation))
        blocks.append(conv_block(num_filters, out_channels, kernel_size,
                                 bias, normalization, activation))
        self.net = nn.Sequential(*blocks)

    def forward(self, input):
        return self.net(input) + input


class ResNet2d(ResNet):
    def __init__(self, in_channels, out_channels, depth,
                 num_filters=64, kernel_size=3, bias=True,
                 normalization=None, activation='ReLU'):
        super().__init__('2', True, in_channels, out_channels,
                         depth, num_filters, kernel_size, bias,
                         normalization, activation)


class ResNet3d(ResNet):
    def __init__(self, in_channels, out_channels, depth,
                 num_filters=64, kernel_size=3, bias=True,
                 normalization=None, activation='ReLU'):
        super().__init__('3', True, in_channels, out_channels,
                         depth, num_filters, kernel_size, bias,
                         normalization, activation)


class ResNet2plus1d(ResNet):
    def __init__(self, in_channels, out_channels, depth,
                 num_filters=64, kernel_size=3, bias=True,
                 normalization=None, activation='ReLU'):
        super().__init__('2+1', True, in_channels, out_channels,
                         depth, num_filters, kernel_size, bias,
                         normalization, activation)


class CResNet2d(ResNet):
    def __init__(self, in_channels, out_channels, depth,
                 num_filters=32, kernel_size=3, bias=True,
                 normalization=None, activation='ReLU'):
        super().__init__('2', False, in_channels, out_channels,
                         depth, num_filters, kernel_size, bias,
                         normalization, activation)


class CResNet3d(ResNet):
    def __init__(self, in_channels, out_channels, depth,
                 num_filters=32, kernel_size=3, bias=True,
                 normalization=None, activation='ReLU'):
        super().__init__('3', False, in_channels, out_channels,
                         depth, num_filters, kernel_size, bias,
                         normalization, activation)


class CResNet2plus1d(ResNet):
    def __init__(self, in_channels, out_channels, depth,
                 num_filters=32, kernel_size=3, bias=True,
                 normalization=None, activation='ReLU'):
        super().__init__('2+1', False, in_channels, out_channels,
                         depth, num_filters, kernel_size, bias,
                         normalization, activation)
