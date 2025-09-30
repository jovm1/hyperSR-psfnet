#gaussian/motion/defocus/mixtures + utils
import math
import random
import numpy as np


def normalize_kernel_np(k):
    k = np.maximum(k, 0)
    s = k.sum()
    return k / (s + 1e-12)


def gaussian_kernel2d(k: int, sigma_x: float, sigma_y: float, theta: float) -> np.ndarray:
    """Elliptical (possibly isotropic) Gaussian kernel with rotation.
    k: odd kernel size
    sigma_x, sigma_y: std in pixels along principal axes
    theta: rotation angle in radians
    """
    assert k % 2 == 1, "kernel size must be odd"
    r = (k - 1) / 2
    xs = np.linspace(-r, r, k)
    ys = np.linspace(-r, r, k)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    # rotate coords
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    Xr = cos_t * X + sin_t * Y
    Yr = -sin_t * X + cos_t * Y
    g = np.exp(-0.5 * ((Xr / sigma_x) ** 2 + (Yr / sigma_y) ** 2))
    return normalize_kernel_np(g)


def disk_kernel2d(k: int, radius: float) -> np.ndarray:
    assert k % 2 == 1, "kernel size must be odd"
    r = (k - 1) / 2
    xs = np.linspace(-r, r, k)
    ys = np.linspace(-r, r, k)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    mask = (X ** 2 + Y ** 2) <= radius ** 2
    disk = np.zeros((k, k), dtype=np.float32)
    disk[mask] = 1.0
    return normalize_kernel_np(disk)


def sample_random_kernel(k: int, kind_probs=(0.6, 0.4)) -> np.ndarray:
    """Sample a random PSF. kinds: [elliptic Gaussian, defocus disk]."""
    p_gauss, p_disk = kind_probs
    u = random.random()
    if u < p_gauss:
        # Elliptical Gaussian: sigma in [0.6, 3.0], anisotropy up to 3×, random angle
        sigma_min, sigma_max = 0.6, 3.0
        sx = random.uniform(sigma_min, sigma_max)
        # anisotropy: sy may differ
        sy = sx * random.uniform(0.5, 3.0)
        theta = random.uniform(0, math.pi)
        return gaussian_kernel2d(k, sx, sy, theta).astype(np.float32)
    else:
        # Defocus disk: radius 1..(k//2)
        r = random.uniform(1.0, (k // 2))
        return disk_kernel2d(k, r).astype(np.float32)


