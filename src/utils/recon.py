import numpy as np
from numpy.fft import *
from sigpy.mri.app import EspiritCalib, SenseRecon


def fft2c(x, axes=(-2, -1), norm='ortho', centered=True):
    if centered:
        return fftshift(fft2(ifftshift(
            x, axes=axes), axes=axes, norm=norm), axes=axes)
    else:
        return fft2(x, axes=axes, norm=norm)


def ifft2c(x, axes=(-2, -1), norm='ortho', centered=True):
    if centered:
        return fftshift(ifft2(ifftshift(
            x, axes=axes), axes=axes, norm=norm), axes=axes)
    else:
        return ifft2(x, axes=axes, norm=norm)


def sos(im, axis=1):
    return np.sqrt(np.sum(np.abs(im) ** 2, axis=axis))


def espirit_map(k, calib_width=24, max_iter=12):
    return EspiritCalib(k, calib_width=calib_width, crop=0,
                        max_iter=max_iter, show_pbar=False).run()


def coil_split(im, sens, axis=0):
    return sens * np.expand_dims(im, axis=axis)


def coil_combine(im, sens, axis=0):
    return np.sum(np.conj(sens) * im, axis=axis)


def iter_sense_recon(k, sens, max_iter=12):
    return SenseRecon(k, sens, max_iter=max_iter, show_pbar=False).run()
