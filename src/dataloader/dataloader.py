import os
from torch.utils.data import Dataset, DataLoader
from src.utils.io import *


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
            num_frames, num_slices = h5shape(filename, 'kspace')[:2]
            if self.dimensions == '2':
                self.map += [(filename, i, j) for i in range(num_slices)
                             for j in range(num_frames)]
            elif self.dimensions == '3' or self.dimensions == '2+1':
                self.map += [(filename, i) for i in range(num_slices)]
            else:
                raise ValueError('dimensions should be 2 or 3 or 2+1.')

    def __len__(self):
        return len(self.map)

    def __getitem__(self, idx):
        item = {}
        if self.dimensions == '2':
            filename, i, j = self.map[idx]
            item['kfull'] = h5read(filename, 'kfull')[i][j]
            item['imfull'] = h5read(filename, 'imfull')[i][j]
            if self.multi_coil:
                item['sens'] = h5read(filename, 'sens')[i]
        else:
            filename, i = self.map[idx]
            item['kfull'] = h5read(filename, 'kfull')[i]
            item['imfull'] = h5read(filename, 'imfull')[i]
            if self.multi_coil:
                item['sens'] = np.tile(h5read(filename, 'sens')[i],
                                       (item['kfull'].shape[0], 1, 1, 1))
        return self.transform(item)