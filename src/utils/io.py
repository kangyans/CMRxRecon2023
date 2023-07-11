import h5py as h5
import numpy as np


def h5read(file, dset):
    with h5.File(file, 'r') as f:
        return np.nan_to_num(f[dset][()])


def h5write(file, dset, data):
    with h5.File(file, 'a') as f:
        f.create_dataset(dset, data=data)
