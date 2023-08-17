from .dc import *
from .unet import *


class SingleCoilKspaceDomainNet(Module):
    def __init__(self, in_channels, out_channels, net_depth=4,
                 num_filters=64, kernel_size=3, bias=False,
                 normalization='instance', activation='relu',
                 down='avg', up='bilinear', channel_dim=1):
        super().__init__()
        self.knet = UNet(
            in_channels, out_channels, net_depth, num_filters,
            kernel_size, bias, normalization, activation, down, up)
        self.real_complex = RealComplex(channel_dim)
        self.complex_real = ComplexReal(channel_dim)
        self.ifft2c = IFFT2C()

    def forward(self, *args):
        ksub = args[0]
        kcnn = self.real_complex(self.knet(self.complex_real(ksub)))
        imcnn = self.ifft2c(kcnn)
        return torch.abs(imcnn)


class MultiCoilKspaceDomainNet(Module):
    def __init__(self, in_channels, out_channels, net_depth=4,
                 num_filters=64, kernel_size=3, bias=False,
                 normalization='instance', activation='relu',
                 down='avg', up='bilinear', channel_dim=1, coil_dim=2):
        super().__init__()
        self.knet = UNet(
            in_channels, out_channels, net_depth, num_filters,
            kernel_size, bias, normalization, activation, down, up)
        self.real_complex = RealComplex(channel_dim)
        self.complex_real = ComplexReal(channel_dim)
        self.ifft2c = IFFT2C()
        self.coil_combine = CoilCombine(coil_dim)
        self.coil_dim = coil_dim

    def forward(self, *args):
        ksub, sens = args[0], args[2]
        stack = []
        for i in range(ksub.shape[self.coil_dim]):
            stack.append(self.real_complex(self.knet(self.complex_real(
                ksub.select(self.coil_dim, i)))))
        kcnn = torch.stack(stack, self.coil_dim)
        imcnn = self.coil_combine(self.ifft2c(kcnn), sens)
        return torch.abs(imcnn)


class SingleCoilComplexKspaceDomainNet(Module):
    def __init__(self, in_channels, out_channels, net_depth=4,
                 num_filters=32, kernel_size=3, bias=False,
                 normalization='instance', activation='relu',
                 down='avg', up='bilinear'):
        super().__init__()
        self.knet = CUNet(
            in_channels, out_channels, net_depth, num_filters,
            kernel_size, bias, normalization, activation, down, up)
        self.ifft2c = IFFT2C()

    def forward(self, *args):
        ksub = args[0]
        kcnn = self.knet(ksub)
        imcnn = self.ifft2c(kcnn)
        return torch.abs(imcnn)


class MultiCoilComplexKspaceDomainNet(Module):
    def __init__(self, in_channels, out_channels, net_depth=4,
                 num_filters=32, kernel_size=3, bias=False,
                 normalization='instance', activation='relu',
                 down='avg', up='bilinear', coil_dim=2):
        super().__init__()
        self.knet = CUNet(
            in_channels, out_channels, net_depth, num_filters,
            kernel_size, bias, normalization, activation, down, up)
        self.ifft2c = IFFT2C()
        self.coil_combine = CoilCombine(coil_dim)
        self.coil_dim = coil_dim

    def forward(self, *args):
        ksub, sens = args[0], args[2]
        stack = []
        for i in range(ksub.shape[self.coil_dim]):
            stack.append(self.knet(ksub.select(self.coil_dim, i)))
        kcnn = torch.stack(stack, self.coil_dim)
        imcnn = self.coil_combine(self.ifft2c(kcnn), sens)
        return torch.abs(imcnn)


class SingleCoilImageDomainNet(Module):
    def __init__(self, in_channels, out_channels, net_depth=4,
                 num_filters=64, kernel_size=3, bias=False,
                 normalization='instance', activation='relu',
                 down='avg', up='bilinear', channel_dim=1):
        super().__init__()
        self.inet = UNet(
            in_channels, out_channels, net_depth, num_filters,
            kernel_size, bias, normalization, activation, down, up)
        self.real_complex = RealComplex(channel_dim)
        self.complex_real = ComplexReal(channel_dim)
        self.ifft2c = IFFT2C()

    def forward(self, *args):
        ksub = args[0]
        imsub = self.ifft2c(ksub)
        imcnn = self.real_complex(self.inet(self.complex_real(imsub)))
        return torch.abs(imcnn)


class MultiCoilImageDomainNet(Module):
    def __init__(self, in_channels, out_channels, net_depth=4,
                 num_filters=64, kernel_size=3, bias=False,
                 normalization='instance', activation='relu',
                 down='avg', up='bilinear', channel_dim=1, coil_dim=2):
        super().__init__()
        self.inet = UNet(
            in_channels, out_channels, net_depth, num_filters,
            kernel_size, bias, normalization, activation, down, up)
        self.real_complex = RealComplex(channel_dim)
        self.complex_real = ComplexReal(channel_dim)
        self.ifft2c = IFFT2C()
        self.coil_combine = CoilCombine(coil_dim)

    def forward(self, *args):
        ksub, sens = args[0], args[2]
        imsub = self.coil_combine(self.ifft2c(ksub), sens)
        imcnn = self.real_complex(self.inet(self.complex_real(imsub)))
        return torch.abs(imcnn)


class SingleCoilComplexImageDomainNet(Module):
    def __init__(self, in_channels, out_channels, net_depth=4,
                 num_filters=32, kernel_size=3, bias=False,
                 normalization='instance', activation='relu',
                 down='avg', up='bilinear'):
        super().__init__()
        self.inet = CUNet(
            in_channels, out_channels, net_depth, num_filters,
            kernel_size, bias, normalization, activation, down, up)
        self.ifft2c = IFFT2C()

    def forward(self, *args):
        ksub = args[0]
        imsub = self.ifft2c(ksub)
        imcnn = self.inet(imsub)
        return torch.abs(imcnn)


class MultiCoilComplexImageDomainNet(Module):
    def __init__(self, in_channels, out_channels, net_depth=4,
                 num_filters=32, kernel_size=3, bias=False,
                 normalization='instance', activation='relu',
                 down='avg', up='bilinear', coil_dim=2):
        super().__init__()
        self.inet = CUNet(
            in_channels, out_channels, net_depth, num_filters,
            kernel_size, bias, normalization, activation, down, up)
        self.ifft2c = IFFT2C()
        self.coil_combine = CoilCombine(coil_dim)

    def forward(self, *args):
        ksub, sens = args[0], args[2]
        imsub = self.coil_combine(self.ifft2c(ksub), sens)
        imcnn = self.inet(imsub)
        return torch.abs(imcnn)


class SingleCoilCascadeCrossDomainNet(Module):
    def __init__(self, in_channels, out_channels, cascade_depth=2,
                 net_depth=4, num_filters=64, kernel_size=3, bias=False,
                 normalization='instance', activation='relu', down='avg',
                 up='bilinear', lamda=0.99, channel_dim=1):
        super().__init__()
        self.knets = ModuleList([UNet(
            in_channels, out_channels, net_depth, num_filters, kernel_size,
            bias, normalization, activation, down, up)
            for _ in range(cascade_depth)])
        self.inets = ModuleList([UNet(
            in_channels, out_channels, net_depth, num_filters, kernel_size,
            bias, normalization, activation, down, up)
            for _ in range(cascade_depth)])
        self.real_complex = RealComplex(channel_dim)
        self.complex_real = ComplexReal(channel_dim)
        self.fft2c = FFT2C()
        self.ifft2c = IFFT2C()
        self.data_consistency = DataConsistency(lamda)

    def forward(self, *args):
        ksub, mask = args[0], args[1]
        kcnn = ksub
        for knet, inet in zip(self.knets, self.inets):
            kcnn = self.real_complex(knet(self.complex_real(kcnn)))
            imcnn = self.ifft2c(kcnn)
            imcnn = self.real_complex(inet(self.complex_real(imcnn)))
            kcnn = self.fft2c(imcnn)
            kcnn = self.data_consistency(kcnn, mask, ksub)
        imcnn = self.ifft2c(kcnn)
        return torch.abs(imcnn)


class MultiCoilCascadeCrossDomainNet(Module):
    def __init__(self, in_channels, out_channels, cascade_depth=2,
                 net_depth=4, num_filters=64, kernel_size=3, bias=False,
                 normalization='instance', activation='relu', down='avg',
                 up='bilinear', lamda=0.99, channel_dim=1, coil_dim=2):
        super().__init__()
        self.knets = ModuleList([UNet(
            in_channels, out_channels, net_depth, num_filters, kernel_size,
            bias, normalization, activation, down, up)
            for _ in range(cascade_depth)])
        self.inets = ModuleList([UNet(
            in_channels, out_channels, net_depth, num_filters, kernel_size,
            bias, normalization, activation, down, up)
            for _ in range(cascade_depth)])
        self.real_complex = RealComplex(channel_dim)
        self.complex_real = ComplexReal(channel_dim)
        self.fft2c = FFT2C()
        self.ifft2c = IFFT2C()
        self.coil_split = CoilSplit(coil_dim)
        self.coil_combine = CoilCombine(coil_dim)
        self.data_consistency = DataConsistency(lamda)
        self.coil_dim = coil_dim

    def forward(self, *args):
        ksub, mask, sens = args[0], args[1], args[2]
        kcnn = ksub
        for knet, inet in zip(self.knets, self.inets):
            stack = []
            for i in range(kcnn.shape[self.coil_dim]):
                stack.append(self.real_complex(knet(self.complex_real(
                    kcnn.select(self.coil_dim, i)))))
            kcnn = torch.stack(stack, self.coil_dim)
            imcnn = self.coil_combine(self.ifft2c(kcnn), sens)
            imcnn = self.real_complex(inet(self.complex_real(imcnn)))
            kcnn = self.fft2c(self.coil_split(imcnn, sens))
            kcnn = self.data_consistency(kcnn, mask, ksub)
        imcnn = self.coil_combine(self.ifft2c(kcnn), sens)
        return torch.abs(imcnn)


class SingleCoilComplexCascadeCrossDomainNet(Module):
    def __init__(self, in_channels, out_channels, cascade_depth=2,
                 net_depth=4, num_filters=32, kernel_size=3, bias=False,
                 normalization='instance', activation='relu', down='avg',
                 up='bilinear', lamda=0.99):
        super().__init__()
        self.knets = ModuleList([CUNet(
            in_channels, out_channels, net_depth, num_filters, kernel_size,
            bias, normalization, activation, down, up)
            for _ in range(cascade_depth)])
        self.inets = ModuleList([CUNet(
            in_channels, out_channels, net_depth, num_filters, kernel_size,
            bias, normalization, activation, down, up)
            for _ in range(cascade_depth)])
        self.fft2c = FFT2C()
        self.ifft2c = IFFT2C()
        self.data_consistency = DataConsistency(lamda)

    def forward(self, *args):
        ksub, mask = args[0], args[1]
        kcnn = ksub
        for knet, inet in zip(self.knets, self.inets):
            kcnn = knet(kcnn)
            imcnn = self.ifft2c(kcnn)
            imcnn = inet(imcnn)
            kcnn = self.fft2c(imcnn)
            kcnn = self.data_consistency(kcnn, mask, ksub)
        imcnn = self.ifft2c(kcnn)
        return torch.abs(imcnn)


class MultiCoilComplexCascadeCrossDomainNet(Module):
    def __init__(self, in_channels, out_channels, cascade_depth=2,
                 net_depth=4, num_filters=32, kernel_size=3, bias=False,
                 normalization='instance', activation='relu', down='avg',
                 up='bilinear', lamda=0.99, coil_dim=2):
        super().__init__()
        self.knets = ModuleList([CUNet(
            in_channels, out_channels, net_depth, num_filters, kernel_size,
            bias, normalization, activation, down, up)
            for _ in range(cascade_depth)])
        self.inets = ModuleList([CUNet(
            in_channels, out_channels, net_depth, num_filters, kernel_size,
            bias, normalization, activation, down, up)
            for _ in range(cascade_depth)])
        self.fft2c = FFT2C()
        self.ifft2c = IFFT2C()
        self.coil_split = CoilSplit(coil_dim)
        self.coil_combine = CoilCombine(coil_dim)
        self.data_consistency = DataConsistency(lamda)
        self.coil_dim = coil_dim

    def forward(self, *args):
        ksub, mask, sens = args[0], args[1], args[2]
        kcnn = ksub
        for knet, inet in zip(self.knets, self.inets):
            stack = []
            for i in range(kcnn.shape[self.coil_dim]):
                stack.append(knet(kcnn.select(self.coil_dim, i)))
            kcnn = torch.stack(stack, self.coil_dim)
            imcnn = self.coil_combine(self.ifft2c(kcnn), sens)
            imcnn = inet(imcnn)
            kcnn = self.fft2c(self.coil_split(imcnn, sens))
            kcnn = self.data_consistency(kcnn, mask, ksub)
        imcnn = self.coil_combine(self.ifft2c(kcnn), sens)
        return torch.abs(imcnn)
