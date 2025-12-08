# gaussian_bank.py
import numpy as np

def make_2d_gaussian(ksize, sigma):
    """Return a normalized 2D Gaussian kernel of shape (ksize, ksize)."""
    ax = np.arange(ksize) - (ksize-1)/2.0
    xx, yy = np.meshgrid(ax, ax)
    kern = np.exp(-(xx**2 + yy**2) / (2*sigma**2))
    kern /= kern.sum() + 1e-12
    return kern

def make_cifar10_gaussian_bank(out_channels=5, in_channels=3, ksize=9,
                               sigmas=(0.6, 1.0, 1.6, 2.6, 4.2)):
    """
    Build a 4D kernel tensor for Lift2D: (out_ch, in_ch, k, k).
    Each out channel is a Gaussian with a different sigma, duplicated across RGB.
    The duplication across in_channels is normalized so total energy stays ~1.
    """
    assert out_channels == len(sigmas), "out_channels must equal len(sigmas)"
    bank = np.zeros((out_channels, in_channels, ksize, ksize), dtype=np.float32)
    for i, s in enumerate(sigmas):
        g = make_2d_gaussian(ksize, s).astype(np.float32)
        # duplicate across input channels and renormalize by in_channels
        for c in range(in_channels):
            bank[i, c] = g / in_channels
    return bank