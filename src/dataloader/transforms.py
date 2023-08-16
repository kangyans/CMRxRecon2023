import torch
from torch.fft import *


class ToTensor(object):
    def __call__(self, item):
        item['kfull'] = torch.from_numpy(item['kfull'])
        if 'sens' in item:
            item['sens'] = torch.from_numpy(item['sens'])
        return item


class RandomFlip(object):
    def __call__(self, item):
        if torch.rand((1,)) < 0.5:
            item['kfull'] = torch.flip(item['kfull'], (-1,))
            if 'sens' in item:
                item['sens'] = torch.flip(item['sens'], (-1,))
        if torch.rand((1,)) < 0.5:
            item['kfull'] = torch.flip(item['kfull'], (-2,))
            if 'sens' in item:
                item['sens'] = torch.flip(item['sens'], (-2,))
        return item


class TargetRecon(object):
    def __call__(self, item):
        item['imfull'] = torch.abs(fftshift(ifft2(ifftshift(
                item['kfull'], dim=(-2, -1)),
                dim=(-2, -1), norm='ortho'),
                dim=(-2, -1)))
        if 'sens' in item:
            item['imfull'] = torch.sqrt(torch.sum(
                item['full'] ** 2, dim=-3))
        return item


class Subsampling(object):
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


class DimExpansion(object):
    def __call__(self, item):
        item['ksub'] = torch.unsqueeze(item['ksub'], dim=0)
        item['imfull'] = torch.unsqueeze(item['imfull'], dim=0)
        item['mask'] = torch.unsqueeze(item['mask'], dim=0)
        if 'sens' in item:
            item['sens'] = torch.unsqueeze(item['sens'], dim=0)
        return item
