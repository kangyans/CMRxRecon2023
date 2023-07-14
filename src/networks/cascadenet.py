from dc import *
from unet import *
from resnet import *


class SingleCoilCascadeNet(nn.Module):
    def __init__(self, net_type, dimensions, real, lamda,
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
        dc = SingleCoilDC if real else CSingleCoilDC
        self.nets = []
        self.dcs = []
        for _ in range(cascade_depth):
            self.nets.append(net(in_channels, out_channels, net_depth,
                                 num_filters, kernel_size, bias,
                                 normalization, activation))
            self.dcs.append(dc(lamda))

    def forward(self, x, mask, k0):
        for net, dc in zip(self.nets, self.dcs):
            x = net(x)
            x = dc(x, mask, k0)
        return x


class MultiCoilCascadeNet(nn.Module):
    def __init__(self, net_type, dimensions, real, lamda,
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
        dc = MultiCoilDC if real else CMultiCoilDC
        self.nets = []
        self.dcs = []
        for _ in range(cascade_depth):
            self.nets.append(net(in_channels, out_channels, net_depth,
                                 num_filters, kernel_size, bias,
                                 normalization, activation))
            self.dcs.append(dc(lamda))

    def forward(self, x, mask, k0, sens):
        for net, dc in zip(self.nets, self.dcs):
            x = net(x)
            x = dc(x, mask, k0, sens)
        return x
