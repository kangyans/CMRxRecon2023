import numpy as np
import sigpy as sp
from sigpy.mri.app import EspiritCalib, SenseRecon
from sigpy.mri.linop import Sense


def fft2c(x, axes=(-2, -1)):
    return sp.fft(x, axes=axes, center=True)


def ifft2c(k, axes=(-2, -1)):
    return sp.ifft(k, axes=axes, center=True)


def sos(x, axis=0):
    return np.sqrt(np.sum(np.abs(x) ** 2, axis=axis))


def espirit_map(k, calib_width=24, max_iter=12):
    return EspiritCalib(k, calib_width=calib_width, crop=0,
                        max_iter=max_iter, show_pbar=False).run()


def sense_recon(k, sens):
    return Sense(sens).H(k)


def iter_sense_recon(k, sens, max_iter=12):
    return SenseRecon(k, sens, max_iter=max_iter, show_pbar=False).run()
