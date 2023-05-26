import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def rmse(ref, recon):
    """Compute root mean squared error."""
    return np.sqrt(np.mean((ref - recon) ** 2))


def psnr(ref, recon, max_val):
    """Compute peak signal-to-noise ratio."""
    if max_val is None:
        max_val = ref.max()
    return peak_signal_noise_ratio(ref, recon, data_range=max_val)


def ssim(ref, recon, max_val):
    """Compute structural similarity index."""
    if max_val is None:
        max_val = ref.max()
    num_slices = ref.shape[0]
    ssim_sum = 0.0
    for s in range(num_slices):
        ssim_sum += structural_similarity(ref[s], recon[s], data_range=max_val)
    return ssim_sum / num_slices
