import h5py as h5
import numpy as np


def h5read(file: str, dset: str):
    """Read hdf5 dataset."""
    with h5.File(file, 'r') as f:
        return np.nan_to_num(f[dset][()])


def h5write(file, dset, data):
    """Write hdf5 dataset."""
    with h5.File(file, 'a') as f:
        f.create_dataset(dset, data=data)
