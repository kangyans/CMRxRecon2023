from dc import *
from unet import *
from resnet import *


class _CascadeNet(Module):
    def __init__(self, net_type, dimensions, real, multi_coil, lamda,
                 net_depth, in_channels, out_channels, cascade_depth,
                 num_filters=64, kernel_size=3, bias=True,
                 normalization=None, activation='ReLU'):
        super().__init__()
        if net_type == 'unet':
            if dimensions == '2':
                net = UNet2d if real else CUNet2d
            elif dimensions == '3':
                net = UNet3d if real else CUNet3d
            elif dimensions == '2+1':
                net = UNet2plus1d if real else CUNet2plus1d
            else:
                raise ValueError('dimensions should be 2 or 3 or 2+1.')
        elif net_type == 'resnet':
            if dimensions == '2':
                net = ResNet2d if real else CResNet2d
            elif dimensions == '3':
                net = ResNet3d if real else CResNet3d
            elif dimensions == '2+1':
                net = ResNet2plus1d if real else CResNet2plus1d
            else:
                raise ValueError('dimensions should be 2 or 3 or 2+1.')
        else:
            raise ValueError('net_type should be unet or resnet')
        self.multi_coil = multi_coil
        if self.multi_coil:
            dc = MultiCoilDC if real else CMultiCoilDC
        else:
            dc = SingleCoilDC if real else CSingleCoilDC
        self.nets = []
        self.dcs = []
        for _ in range(cascade_depth):
            self.nets.append(net(in_channels, out_channels, net_depth,
                                 num_filters, kernel_size, bias,
                                 normalization, activation))
            self.dcs.append(dc(lamda))

    def forward(self, x, mask, ksub, *args):
        for net, dc in zip(self.nets, self.dcs):
            x = net(x)
            x = dc(x, mask, ksub, args[0]) if self.multi_coil \
                else dc(x, mask, ksub)
        return x


class SingleCoilCascadeResNet2d(_CascadeNet):
    def __init__(self, lamda, net_depth, in_channels, out_channels,
                 cascade_depth, num_filters=64, kernel_size=3,
                 bias=True, normalization=None, activation='ReLU'):
        super().__init__('resnet', '2', True, False, lamda,
                         net_depth, in_channels, out_channels,
                         cascade_depth, num_filters, kernel_size,
                         bias, normalization, activation)


class SingleCoilCascadeResNet3d(_CascadeNet):
    def __init__(self, lamda, net_depth, in_channels, out_channels,
                 cascade_depth, num_filters=64, kernel_size=3,
                 bias=True, normalization=None, activation='ReLU'):
        super().__init__('resnet', '3', True, False, lamda,
                         net_depth, in_channels, out_channels,
                         cascade_depth, num_filters, kernel_size,
                         bias, normalization, activation)


class SingleCoilCascadeResNet2plus1d(_CascadeNet):
    def __init__(self, lamda, net_depth, in_channels, out_channels,
                 cascade_depth, num_filters=64, kernel_size=3,
                 bias=True, normalization=None, activation='ReLU'):
        super().__init__('resnet', '2+1', True, False, lamda,
                         net_depth, in_channels, out_channels,
                         cascade_depth, num_filters, kernel_size,
                         bias, normalization, activation)


class SingleCoilCascadeCResNet2d(_CascadeNet):
    def __init__(self, lamda, net_depth, in_channels, out_channels,
                 cascade_depth, num_filters=32, kernel_size=3,
                 bias=True, normalization=None, activation='ReLU'):
        super().__init__('resnet', '2', False, False, lamda,
                         net_depth, in_channels, out_channels,
                         cascade_depth, num_filters, kernel_size,
                         bias, normalization, activation)


class SingleCoilCascadeCResNet3d(_CascadeNet):
    def __init__(self, lamda, net_depth, in_channels, out_channels,
                 cascade_depth, num_filters=32, kernel_size=3,
                 bias=True, normalization=None, activation='ReLU'):
        super().__init__('resnet', '3', False, False, lamda,
                         net_depth, in_channels, out_channels,
                         cascade_depth, num_filters, kernel_size,
                         bias, normalization, activation)


class SingleCoilCascadeCResNet2plus1d(_CascadeNet):
    def __init__(self, lamda, net_depth, in_channels, out_channels,
                 cascade_depth, num_filters=32, kernel_size=3,
                 bias=True, normalization=None, activation='ReLU'):
        super().__init__('resnet', '2+1', False, False, lamda,
                         net_depth, in_channels, out_channels,
                         cascade_depth, num_filters, kernel_size,
                         bias, normalization, activation)


class MultiCoilCascadeResNet2d(_CascadeNet):
    def __init__(self, lamda, net_depth, in_channels, out_channels,
                 cascade_depth, num_filters=64, kernel_size=3,
                 bias=True, normalization=None, activation='ReLU'):
        super().__init__('resnet', '2', True, True, lamda,
                         net_depth, in_channels, out_channels,
                         cascade_depth, num_filters, kernel_size,
                         bias, normalization, activation)


class MultiCoilCascadeResNet3d(_CascadeNet):
    def __init__(self, lamda, net_depth, in_channels, out_channels,
                 cascade_depth, num_filters=64, kernel_size=3,
                 bias=True, normalization=None, activation='ReLU'):
        super().__init__('resnet', '3', True, True, lamda,
                         net_depth, in_channels, out_channels,
                         cascade_depth, num_filters, kernel_size,
                         bias, normalization, activation)


class MultiCoilCascadeResNet2plus1d(_CascadeNet):
    def __init__(self, lamda, net_depth, in_channels, out_channels,
                 cascade_depth, num_filters=64, kernel_size=3,
                 bias=True, normalization=None, activation='ReLU'):
        super().__init__('resnet', '2+1', True, True, lamda,
                         net_depth, in_channels, out_channels,
                         cascade_depth, num_filters, kernel_size,
                         bias, normalization, activation)


class MultiCoilCascadeCResNet2d(_CascadeNet):
    def __init__(self, lamda, net_depth, in_channels, out_channels,
                 cascade_depth, num_filters=32, kernel_size=3,
                 bias=True, normalization=None, activation='ReLU'):
        super().__init__('resnet', '2', False, True, lamda,
                         net_depth, in_channels, out_channels,
                         cascade_depth, num_filters, kernel_size,
                         bias, normalization, activation)


class MultiCoilCascadeCResNet3d(_CascadeNet):
    def __init__(self, lamda, net_depth, in_channels, out_channels,
                 cascade_depth, num_filters=32, kernel_size=3,
                 bias=True, normalization=None, activation='ReLU'):
        super().__init__('resnet', '3', False, True, lamda,
                         net_depth, in_channels, out_channels,
                         cascade_depth, num_filters, kernel_size,
                         bias, normalization, activation)


class MultiCoilCascadeCResNet2plus1d(_CascadeNet):
    def __init__(self, lamda, net_depth, in_channels, out_channels,
                 cascade_depth, num_filters=32, kernel_size=3,
                 bias=True, normalization=None, activation='ReLU'):
        super().__init__('resnet', '2+1', False, True, lamda,
                         net_depth, in_channels, out_channels,
                         cascade_depth, num_filters, kernel_size,
                         bias, normalization, activation)


class SingleCoilCascadeUNet2d(_CascadeNet):
    def __init__(self, lamda, net_depth, in_channels, out_channels,
                 cascade_depth, num_filters=64, kernel_size=3,
                 bias=True, normalization=None, activation='ReLU'):
        super().__init__('unet', '2', True, False, lamda,
                         net_depth, in_channels, out_channels,
                         cascade_depth, num_filters, kernel_size,
                         bias, normalization, activation)


class SingleCoilCascadeUNet3d(_CascadeNet):
    def __init__(self, lamda, net_depth, in_channels, out_channels,
                 cascade_depth, num_filters=64, kernel_size=3,
                 bias=True, normalization=None, activation='ReLU'):
        super().__init__('unet', '3', True, False, lamda,
                         net_depth, in_channels, out_channels,
                         cascade_depth, num_filters, kernel_size,
                         bias, normalization, activation)


class SingleCoilCascadeUNet2plus1d(_CascadeNet):
    def __init__(self, lamda, net_depth, in_channels, out_channels,
                 cascade_depth, num_filters=64, kernel_size=3,
                 bias=True, normalization=None, activation='ReLU'):
        super().__init__('unet', '2+1', True, False, lamda,
                         net_depth, in_channels, out_channels,
                         cascade_depth, num_filters, kernel_size,
                         bias, normalization, activation)


class SingleCoilCascadeCUNet2d(_CascadeNet):
    def __init__(self, lamda, net_depth, in_channels, out_channels,
                 cascade_depth, num_filters=32, kernel_size=3,
                 bias=True, normalization=None, activation='ReLU'):
        super().__init__('unet', '2', False, False, lamda,
                         net_depth, in_channels, out_channels,
                         cascade_depth, num_filters, kernel_size,
                         bias, normalization, activation)


class SingleCoilCascadeCUNet3d(_CascadeNet):
    def __init__(self, lamda, net_depth, in_channels, out_channels,
                 cascade_depth, num_filters=32, kernel_size=3,
                 bias=True, normalization=None, activation='ReLU'):
        super().__init__('unet', '3', False, False, lamda,
                         net_depth, in_channels, out_channels,
                         cascade_depth, num_filters, kernel_size,
                         bias, normalization, activation)


class SingleCoilCascadeCUNet2plus1d(_CascadeNet):
    def __init__(self, lamda, net_depth, in_channels, out_channels,
                 cascade_depth, num_filters=32, kernel_size=3,
                 bias=True, normalization=None, activation='ReLU'):
        super().__init__('unet', '2+1', False, False, lamda,
                         net_depth, in_channels, out_channels,
                         cascade_depth, num_filters, kernel_size,
                         bias, normalization, activation)


class MultiCoilCascadeUNet2d(_CascadeNet):
    def __init__(self, lamda, net_depth, in_channels, out_channels,
                 cascade_depth, num_filters=64, kernel_size=3,
                 bias=True, normalization=None, activation='ReLU'):
        super().__init__('unet', '2', True, True, lamda,
                         net_depth, in_channels, out_channels,
                         cascade_depth, num_filters, kernel_size,
                         bias, normalization, activation)


class MultiCoilCascadeUNet3d(_CascadeNet):
    def __init__(self, lamda, net_depth, in_channels, out_channels,
                 cascade_depth, num_filters=64, kernel_size=3,
                 bias=True, normalization=None, activation='ReLU'):
        super().__init__('unet', '3', True, True, lamda,
                         net_depth, in_channels, out_channels,
                         cascade_depth, num_filters, kernel_size,
                         bias, normalization, activation)


class MultiCoilCascadeUNet2plus1d(_CascadeNet):
    def __init__(self, lamda, net_depth, in_channels, out_channels,
                 cascade_depth, num_filters=64, kernel_size=3,
                 bias=True, normalization=None, activation='ReLU'):
        super().__init__('unet', '2+1', True, True, lamda,
                         net_depth, in_channels, out_channels,
                         cascade_depth, num_filters, kernel_size,
                         bias, normalization, activation)


class MultiCoilCascadeCUNet2d(_CascadeNet):
    def __init__(self, lamda, net_depth, in_channels, out_channels,
                 cascade_depth, num_filters=32, kernel_size=3,
                 bias=True, normalization=None, activation='ReLU'):
        super().__init__('unet', '2', False, True, lamda,
                         net_depth, in_channels, out_channels,
                         cascade_depth, num_filters, kernel_size,
                         bias, normalization, activation)


class MultiCoilCascadeCUNet3d(_CascadeNet):
    def __init__(self, lamda, net_depth, in_channels, out_channels,
                 cascade_depth, num_filters=32, kernel_size=3,
                 bias=True, normalization=None, activation='ReLU'):
        super().__init__('unet', '3', False, True, lamda,
                         net_depth, in_channels, out_channels,
                         cascade_depth, num_filters, kernel_size,
                         bias, normalization, activation)


class MultiCoilCascadeCUNet2plus1d(_CascadeNet):
    def __init__(self, lamda, net_depth, in_channels, out_channels,
                 cascade_depth, num_filters=32, kernel_size=3,
                 bias=True, normalization=None, activation='ReLU'):
        super().__init__('unet', '2+1', False, True, lamda,
                         net_depth, in_channels, out_channels,
                         cascade_depth, num_filters, kernel_size,
                         bias, normalization, activation)
