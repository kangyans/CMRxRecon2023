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
    parser.add_argument('--old_label', type=str,
                        help='Label of the raw data.')
    parser.add_argument('--new_label', type=str,
                        help='Label of the new dataset.')
    parser.add_argument('--multi_coil', type=bool,
                        help='Single coil or multi coil.')
    return parser


def build_dataset(old_root, new_root, old_label, new_label, multi_coil):
    if not os.path.exists(new_root):
        os.makedirs(new_root)
    for subject in os.listdir(old_root):
        if os.path.isdir(os.path.join(old_root, subject)):
            print('Start processing subject {} ...'.format(subject))
            start_time = time.perf_counter()
            for data_file in ['cine_lax', 'cine_sax']:
                old_filename = os.path.join(
                    old_root, subject, data_file + '.mat')
                new_filename = os.path.join(
                    new_root, subject + '_' + data_file + '.h5')
                if os.path.exists(old_filename):
                    kspace = h5read(old_filename, old_label)
                    kspace = kspace['real'] + 1j * kspace['imag']
                    h5write(new_filename, new_label, kspace)
                    if multi_coil:
                        avg_kspace = np.average(kspace, axis=0)
                        num_slices = avg_kspace.shape[0]
                        sens = np.zeros_like(avg_kspace)
                        for i in range(num_slices):
                            sens[i] = espirit_map(avg_kspace[i])
                        h5write(new_filename, 'sens', sens)
            print('Finish processing subject {}: {:.3f} seconds.'.format(
                subject, time.perf_counter() - start_time))


def main():
    args = get_parser().parse_args()
    old_root = args.old_root
    new_root = args.new_root
    old_label = args.old_label
    new_label = args.new_label
    multi_coil = args.multi_coil
    build_dataset(old_root, new_root, old_label, new_label, multi_coil)


if __name__ == '__main__':
    main()
