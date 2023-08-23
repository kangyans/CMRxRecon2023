import os
import time
import argparse
from utils.io import *
from utils.recon import *


def get_parser():
    parser = argparse.ArgumentParser(
        description='Building training/validation dataset from raw data.')
    parser.add_argument('--old_root', type=str,
                        help='Root path of the raw data.')
    parser.add_argument('--new_root', type=str,
                        help='Root path of the new dataset.')
    return parser


def build_dataset(old_root, new_root, multi_coil):
    if not os.path.exists(new_root):
        os.makedirs(new_root)
    subjects = [subject for subject in os.listdir(old_root)
                if os.path.isdir(os.path.join(old_root, subject))]
    for i, subject in enumerate(subjects):
        print('Start processing subject {} ...'.format(subject))
        start_time = time.perf_counter()
        for data_file in ['cine_lax', 'cine_sax']:
            old_filename = os.path.join(
                old_root, subject, data_file + '.mat')
            if os.path.exists(old_filename):
                kfull = h5read(old_filename, 'kspace_full' if multi_coil
                                   else 'kspace_single_full')
                kfull = kfull['real'] + 1j * kfull['imag']
                for s in range(kfull.shape[1]):
                    new_filename = os.path.join(
                        new_root, '{}_{}_slice{:02d}.h5'.format(
                            subject, data_file, s))
                    kfull[:, s] /= np.max(np.abs(kfull[:, s]))
                    h5write(new_filename, 'kspace',
                            kfull[:, s].astype(np.csingle))
                    if multi_coil:
                        sens = espirit_map(np.mean(kfull[:, s], axis=0))
                        sens = np.tile(sens, (kfull.shape[0], 1, 1, 1))
                        h5write(new_filename, 'sensitivity_map',
                                sens.astype(np.csingle))
        print('Finish processing subject {}: {:.3f} seconds.'.format(
            subject, time.perf_counter() - start_time))


def main():
    args = get_parser().parse_args()
    old_root = args.old_root
    new_root = args.new_root
    multi_coil = 'MultiCoil' in old_root
    build_dataset(old_root, new_root, multi_coil)


if __name__ == '__main__':
    main()
