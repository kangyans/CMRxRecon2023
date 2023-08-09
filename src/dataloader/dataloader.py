import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from .transform import *
from utils.io import *


class CMRDataset(Dataset):
    def __init__(self, dset_path, dimensions, multi_coil,
                 transform, sample_ratio=1.0):
        self.dimensions = dimensions
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
            num_frames, num_slices = h5shape(filename, 'kfull')[:2]
            if self.dimensions == '2':
                self.map += [(filename, i, j) for i in range(num_frames)
                             for j in range(num_slices)]
            elif self.dimensions == '3' or self.dimensions == '2+1':
                self.map += [(filename, j) for j in range(num_slices)]
            else:
                raise ValueError('dimensions should be 2 or 3 or 2+1.')

    def __len__(self):
        return len(self.map)

    def __getitem__(self, idx):
        item = {}
        if self.dimensions == '2':
            filename, i, j = self.map[idx]
            item['kfull'] = np.expand_dims(
                h5read(filename, 'kfull')[i][j], 0)
            item['imfull'] = np.expand_dims(
                h5read(filename, 'imfull')[i][j], 0)
            if self.multi_coil:
                item['sens'] = np.expand_dims(
                    h5read(filename, 'sens')[j], 0)
        else:
            filename, j = self.map[idx]
            item['kfull'] = np.expand_dims(
                h5read(filename, 'kfull')[:, j], 0)
            item['imfull'] = np.expand_dims(
                h5read(filename, 'imfull')[:, j], 0)
            if self.multi_coil:
                item['sens'] = np.tile(h5read(filename, 'sens')[j],
                                       (1, item['kfull'].shape[1], 1, 1, 1))
                item['kfull'] = np.swapaxes(item['kfull'], 1, 2)
                item['sens'] = np.swapaxes(item['sens'], 1, 2)
        return self.transform(item)


def get_dataloader(dset_path, is_training, dimensions, real,
                   multi_coil, sample_ratio=1.0):
    transform = [ToTensor()]
    if is_training:
        transform.append(RandomFlip())
    transform.append(Subsample(center_lines=24,
                               subsample_ratios=(4, 6, 8, 10, 12)))
    transform.append(IFFT2C())
    if multi_coil:
        transform.append(CoilCombine())
    if real:
        transform.append(ComplexReal())
    return DataLoader(CMRDataset(dset_path, dimensions, multi_coil,
                                 Compose(transform), sample_ratio),
                      shuffle=is_training)
