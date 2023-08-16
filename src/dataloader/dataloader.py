import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.transforms import Compose
from .transforms import *
from utils.io import *


class CMRDataset(Dataset):
    def __init__(self, dset_path, multi_coil, transform, sample_ratio=1.0):
        self.multi_coil = multi_coil
        self.transform = transform
        self.map = []
        filenames = [filename for filename in os.listdir(dset_path)
                     if filename.endswith('.h5')]
        if sample_ratio > 1.0:
            raise ValueError('sample_ratio should be smaller than 1.0.')
        filenames = filenames[:int(sample_ratio * len(filenames))]
        for filename in filenames:
            filename = os.path.join(dset_path, filename)
            num_frames, num_slices = h5shape(filename, 'kspace')[0]
            self.map += [(filename, i) for i in range(num_slices)]

    def __len__(self):
        return len(self.map)

    def __getitem__(self, idx):
        item = {}
        filename, i = self.map[idx]
        item['kfull'] = h5read(filename, 'kspace')[i]
        if self.multi_coil:
            item['sens'] = h5read(filename, 'sensitivity_map')[i]
        return self.transform(item)


def get_dataloader(dset_path, is_training, multi_coil, sample_ratio=1.0):
    if is_training:
        transform = [ToTensor(), RandomFlip(), TargetRecon(),
                     Subsampling(24, (4, 6, 8, 10, 12)), DimExpansion()]
        return DataLoader(CMRDataset(
            dset_path, multi_coil, Compose(transform), sample_ratio),
            shuffle=True)
    else:
        datasets = []
        for subsample_ratios in ((4,), (8,), (10,)):
            transform = [ToTensor(), TargetRecon(),
                         Subsampling(24, subsample_ratios), DimExpansion()]
            datasets.append(CMRDataset(
                dset_path, multi_coil, Compose(transform), sample_ratio))
        return DataLoader(ConcatDataset(datasets), shuffle=False)
