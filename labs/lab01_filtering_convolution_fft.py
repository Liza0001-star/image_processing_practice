git add .
git commit -m "Implemented lab01 functions"
git push origin main

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import numpy.typing as npt
from scipy import signal

BorderType = Literal["reflect", "constant", "wrap", "replicate"]


def conv2d(image, kernel, border="reflect") -> np.ndarray:
    image_f = image.astype(np.float32)
    kernel = kernel.astype(np.float32)

    border_map = {
        "reflect": "reflect",
        "constant": "constant",
        "wrap": "wrap",
        "replicate": "edge"
    }

    mode = border_map[border]

    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    if image.ndim == 2:
        padded = np.pad(image_f, ((pad_h, pad_h), (pad_w, pad_w)), mode=mode)
        out = signal.convolve2d(padded, kernel, mode="valid")
    else:
        out = np.zeros_like(image_f)
        for c in range(image.shape[2]):
            padded = np.pad(image_f[:, :, c],
                            ((pad_h, pad_h), (pad_w, pad_w)),
                            mode=mode)
            out[:, :, c] = signal.convolve2d(padded, kernel, mode="valid")

    return out.astype(np.float32)


def make_gaussian_kernel(ksize: int, sigma: float):
    ax = np.arange(-(ksize // 2), ksize // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def _clip_to_dtype_range(x, dtype):
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(x, info.min, info.max)
    return x


def apply_gaussian_blur(image, ksize, sigma):
    kernel = make_gaussian_kernel(ksize, sigma)
    out = conv2d(image, kernel)
    return _clip_to_dtype_range(out, image.dtype).astype(image.dtype)


def apply_box_blur(image, ksize):
    kernel = np.ones((ksize, ksize), dtype=np.float32) / (ksize * ksize)
    out = conv2d(image, kernel)
    return _clip_to_dtype_range(out, image.dtype).astype(image.dtype)


def apply_median_blur(image, ksize):
    return cv2.medianBlur(image.astype(np.uint8), ksize)


def add_salt_pepper_noise(image, amount, salt_vs_pepper=0.5, *, seed=0):
    rng = np.random.default_rng(seed)
    out = image.copy()

    H, W = image.shape[:2]
    total = int(H * W * amount)

    salt = int(total * salt_vs_pepper)
    pepper = total - salt

    coords = (rng.integers(0, H, salt), rng.integers(0, W, salt))
    out[coords] = np.max(image)

    coords = (rng.integers(0, H, pepper), rng.integers(0, W, pepper))
    out[coords] = np.min(image)

    return out


def add_gaussian_noise(image, sigma, *, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, image.shape)
    out = image.astype(np.float32) + noise
    return _clip_to_dtype_range(out, image.dtype).astype(image.dtype)


def sobel_edges(image, ksize=3):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = image.astype(np.float32)

    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = np.sqrt(gx**2 + gy**2)

    return gx, gy, mag


def laplacian_edges(image, ksize=3):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    lap = cv2.Laplacian(image.astype(np.float32), cv2.CV_32F, ksize=ksize)
    return np.abs(lap)


def fft2_image(image):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)


def fftshift2(spectrum):
    return np.fft.fftshift(spectrum, axes=(0, 1))


def magnitude_spectrum(spectrum, log_scale=True):
    mag = cv2.magnitude(spectrum[:, :, 0], spectrum[:, :, 1])
    if log_scale:
        mag = np.log1p(mag)
    return mag.astype(np.float32)


def ideal_low_pass_filter(shape, cutoff_radius):
    H, W = shape[:2]
    Y, X = np.ogrid[:H, :W]
    center = (H // 2, W // 2)

    dist = np.sqrt((Y - center[0])**2 + (X - center[1])**2)
    mask = (dist <= cutoff_radius).astype(np.float32)

    if len(shape) == 3:
        mask = np.stack([mask, mask], axis=-1)

    return mask


def ideal_high_pass_filter(shape, cutoff_radius):
    return 1.0 - ideal_low_pass_filter(shape, cutoff_radius)


def apply_frequency_filter(image, filter_mask):
    spec = fft2_image(image)
    spec_shift = fftshift2(spec)

    if filter_mask.ndim == 2:
        filter_mask = np.stack([filter_mask, filter_mask], axis=-1)

    filtered = spec_shift * filter_mask

    ishift = np.fft.ifftshift(filtered)
    img_back = cv2.idft(ishift, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)

    return img_back.astype(np.float32)


def normalize_to_uint8(x):
    arr = np.asarray(x, dtype=np.float32)
    mn, mx = arr.min(), arr.max()

    if mx <= mn:
        return np.zeros_like(arr, dtype=np.uint8)

    y = (arr - mn) * 255.0 / (mx - mn)
    return np.clip(y, 0, 255).astype(np.uint8)
