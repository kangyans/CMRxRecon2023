import torch
from torch.fft import *


class ToTensor(object):
    def __call__(self, item):
        item['kfull'] = torch.from_numpy(item['kfull'])
        item['imfull'] = torch.from_numpy(item['imfull'])
        if 'sens' in item:
            item['sens'] = torch.from_numpy(item['sens'])
        return item


class RandomFlip(object):
    def __call__(self, item):
        if torch.rand((1,)) < 0.5:
            item['kfull'] = torch.flip(item['kfull'], (-1,))
            item['imfull'] = torch.flip(item['imfull'], (-1,))
            if 'sens' in item:
                item['sens'] = torch.flip(item['sens'], (-1,))
        if torch.rand((1,)) < 0.5:
            item['kfull'] = torch.flip(item['kfull'], (-2,))
            item['imfull'] = torch.flip(item['imfull'], (-2,))
            if 'sens' in item:
                item['sens'] = torch.flip(item['sens'], (-2,))
        return item


class Subsample(object):
    def __init__(self, center_lines, subsample_ratios):
        self.center_lines = center_lines
        self.subsample_ratios = subsample_ratios

    def __call__(self, item):
        mask = torch.zeros_like(item['kfull'], dtype=torch.float)
        center_start = (item['kfull'].shape[-2] - self.center_lines) // 2
        center_end = center_start + self.center_lines
        mask[..., center_start:center_end, :] = 1
        subsample_ratio = self.subsample_ratios[
            int(torch.randint(len(self.subsample_ratios), (1,)))]
        mask[..., ::subsample_ratio, :] = 1
        item['mask'] = mask
        item['ksub'] = mask * item['kfull']
        return item


class FFT2C(object):
    def __init__(self, dim=(-2, -1), norm='ortho', centered=True):
        self.dim = dim
        self.norm = norm
        self.centered = centered

    def __call__(self, item):
        if self.centered:
            item['ksub'] = fftshift(fft2(ifftshift(
                item['imsub'], dim=self.dim),
                dim=self.dim, norm=self.norm), dim=self.dim)
        else:
            item['ksub'] = fft2(item['imsub'], dim=self.dim, norm=self.norm)
        return item


class IFFT2C(object):
    def __init__(self, dim=(-2, -1), norm='ortho', centered=True):
        self.dim = dim
        self.norm = norm
        self.centered = centered

    def __call__(self, item):
        if self.centered:
            item['imsub'] = fftshift(ifft2(ifftshift(
                item['ksub'], dim=self.dim),
                dim=self.dim, norm=self.norm), dim=self.dim)
        else:
            item['imsub'] = ifft2(item['ksub'], dim=self.dim, norm=self.norm)
        return item


class CoilSplit(object):
    def __init__(self, dim=1):
        self.dim = dim

    def __call__(self, item):
        item['imsub'] = item['sens'] * \
                        torch.unsqueeze(item['imsub'], dim=self.dim)
        return item


class CoilCombine(object):
    def __init__(self, dim=1):
        self.dim = dim

    def __call__(self, item):
        item['imsub'] = torch.sum(torch.conj_physical(item['sens']) *
                                  item['imsub'], dim=self.dim)
        return item


class RealComplex(object):
    def __init__(self, dim=0):
        self.dim = dim

    def __call__(self, item):
        item['imfull'] = torch.unsqueeze(
            torch.complex(item['imfull'].select(self.dim, 0),
                          item['imfull'].select(self.dim, 1)), dim=self.dim)
        item['imsub'] = torch.unsqueeze(
            torch.complex(item['imsub'].select(self.dim, 0),
                          item['imsub'].select(self.dim, 1)), dim=self.dim)
        return item


class ComplexReal(object):
    def __init__(self, dim=0):
        self.dim = dim

    def __call__(self, item):
        item['imfull'] = torch.cat([item['imfull'].real,
                                    item['imfull'].imag], dim=self.dim)
        item['imsub'] = torch.cat([item['imsub'].real,
                                   item['imsub'].imag], dim=self.dim)
        return item
