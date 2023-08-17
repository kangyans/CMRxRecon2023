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
    parser.add_argument('--trn_ratio', type=float, default=0.85,
                        help='Training ratio, validation ratio = '
                             '1 - training ratio.')
    return parser


def build_dataset(old_root, new_root, multi_coil, trn_ratio):
    trn_root = os.path.join(new_root, 'TrainingSet')
    val_root = os.path.join(new_root, 'ValidationSet')
    if not os.path.exists(trn_root):
        os.makedirs(trn_root)
    if not os.path.exists(val_root):
        os.makedirs(val_root)
    subjects = [subject for subject in os.listdir(old_root)
                if os.path.isdir(os.path.join(old_root, subject))]
    trn_num = int(trn_ratio * len(subjects))
    subjects.sort()
    rng = np.random.default_rng(seed=42)
    rng.shuffle(subjects)
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
                    kfull[:, s] /= np.max(np.abs(kfull[:, s]))
                    if multi_coil:
                        sens = espirit_map(np.mean(kfull[:, s], axis=0))
                    for f in range(kfull.shape[0]):
                        new_filename = os.path.join(
                            trn_root if i < trn_num else val_root,
                            '{}_{}_frame{:02d}_slice{:02d}.h5'.format(
                                subject, data_file, f, s))
                        h5write(new_filename, 'kspace',
                                kfull[f][s].astype(np.csingle))
                        if multi_coil:
                            h5write(new_filename, 'sensitivity_map',
                                    sens.astype(np.csingle))
        print('Finish processing subject {}: {:.3f} seconds.'.format(
            subject, time.perf_counter() - start_time))


def main():
    args = get_parser().parse_args()
    old_root = args.old_root
    new_root = args.new_root
    multi_coil = 'MultiCoil' in old_root
    trn_ratio = args.trn_ratio
    build_dataset(old_root, new_root, multi_coil, trn_ratio)


if __name__ == '__main__':
    main()
