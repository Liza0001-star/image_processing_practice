from __future__ import annotations


import argparse
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import numpy.typing as npt
from scipy import signal

BorderType = Literal["reflect", "constant", "wrap", "replicate"]


def conv2d(
    image: npt.NDArray[np.generic],
    kernel: npt.NDArray[np.generic],
    border: BorderType = "reflect",
) -> np.ndarray:
    image_f = np.asarray(image, dtype=np.float32)
    kernel_f = np.asarray(kernel, dtype=np.float32)

    border_map = {
        "reflect": "reflect",
        "constant": "constant",
        "wrap": "wrap",
        "replicate": "edge",
    }
    mode = border_map[border]

    kh, kw = kernel_f.shape
    pad_h, pad_w = kh // 2, kw // 2

    if image_f.ndim == 2:
        padded = np.pad(image_f, ((pad_h, pad_h), (pad_w, pad_w)), mode=mode)
        out = signal.convolve2d(padded, kernel_f, mode="valid")
    else:
        out = np.zeros_like(image_f, dtype=np.float32)
        for c in range(image_f.shape[2]):
            padded = np.pad(
                image_f[:, :, c],
                ((pad_h, pad_h), (pad_w, pad_w)),
                mode=mode,
            )
            out[:, :, c] = signal.convolve2d(padded, kernel_f, mode="valid")

    return out.astype(np.float32)


def make_gaussian_kernel(ksize: int, sigma: float) -> npt.NDArray[np.float32]:
    if ksize <= 0 or ksize % 2 == 0:
        raise ValueError("ksize must be a positive odd integer")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    ax = np.arange(-(ksize // 2), ksize // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def _clip_to_dtype_range(x: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(x, info.min, info.max)
    return x


def apply_gaussian_blur(image: npt.NDArray[np.generic], ksize: int, sigma: float) -> np.ndarray:
    kernel = make_gaussian_kernel(ksize, sigma)
    out = conv2d(image, kernel, border="reflect")
    out = _clip_to_dtype_range(out, image.dtype)
    return out.astype(image.dtype)


def apply_box_blur(image: npt.NDArray[np.generic], ksize: int) -> np.ndarray:
    if ksize <= 0 or ksize % 2 == 0:
        raise ValueError("ksize must be a positive odd integer")

    kernel = np.ones((ksize, ksize), dtype=np.float32) / float(ksize * ksize)
    out = conv2d(image, kernel, border="reflect")
    out = _clip_to_dtype_range(out, image.dtype)
    return out.astype(image.dtype)


def apply_median_blur(image: npt.NDArray[np.generic], ksize: int) -> np.ndarray:
    if ksize <= 0 or ksize % 2 == 0:
        raise ValueError("ksize must be a positive odd integer")
    return cv2.medianBlur(np.asarray(image).astype(np.uint8), ksize)


def add_salt_pepper_noise(
    image: npt.NDArray[np.generic],
    amount: float,
    salt_vs_pepper: float = 0.5,
    *,
    seed: int = 0,
) -> np.ndarray:
    if not (0.0 <= amount <= 1.0):
        raise ValueError("amount must be in [0, 1]")
    if not (0.0 <= salt_vs_pepper <= 1.0):
        raise ValueError("salt_vs_pepper must be in [0, 1]")

    rng = np.random.default_rng(seed)
    out = np.array(image, copy=True)

    h, w = out.shape[:2]
    n_total = int(round(amount * h * w))
    n_salt = int(round(n_total * salt_vs_pepper))
    n_pepper = n_total - n_salt

    if np.issubdtype(out.dtype, np.integer):
        info = np.iinfo(out.dtype)
        salt_value = info.max
        pepper_value = info.min
    else:
        salt_value = 1.0
        pepper_value = 0.0

    if n_salt > 0:
        ys = rng.integers(0, h, size=n_salt)
        xs = rng.integers(0, w, size=n_salt)
        if out.ndim == 2:
            out[ys, xs] = salt_value
        else:
            out[ys, xs, :] = salt_value

    if n_pepper > 0:
        ys = rng.integers(0, h, size=n_pepper)
        xs = rng.integers(0, w, size=n_pepper)
        if out.ndim == 2:
            out[ys, xs] = pepper_value
        else:
            out[ys, xs, :] = pepper_value

    return out


def add_gaussian_noise(image: npt.NDArray[np.generic], sigma: float, *, seed: int = 0) -> np.ndarray:
    if sigma < 0:
        raise ValueError("sigma must be >= 0")

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=image.shape)
    out = np.asarray(image, dtype=np.float32) + noise
    out = _clip_to_dtype_range(out, image.dtype)
    return out.astype(image.dtype)


def sobel_edges(image: npt.NDArray[np.generic], ksize: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    gray_f = np.asarray(gray, dtype=np.float32)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=ksize)
    magnitude = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    return gx.astype(np.float32), gy.astype(np.float32), magnitude


def laplacian_edges(image: npt.NDArray[np.generic], ksize: int = 3) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    lap = cv2.Laplacian(np.asarray(gray, dtype=np.float32), cv2.CV_32F, ksize=ksize)
    return np.abs(lap).astype(np.float32)


def fft2_image(image: npt.NDArray[np.generic]) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    gray_f = np.asarray(gray, dtype=np.float32)
    spectrum = cv2.dft(gray_f, flags=cv2.DFT_COMPLEX_OUTPUT)
    return spectrum.astype(np.float32)


def fftshift2(spectrum: npt.NDArray[np.floating]) -> np.ndarray:
    return np.fft.fftshift(spectrum, axes=(0, 1))


def magnitude_spectrum(spectrum: npt.NDArray[np.floating], log_scale: bool = True) -> np.ndarray:
    mag = cv2.magnitude(spectrum[:, :, 0], spectrum[:, :, 1])
    if log_scale:
        mag = np.log1p(mag)
    return mag.astype(np.float32)


def ideal_low_pass_filter(shape: tuple[int, int] | tuple[int, int, int], cutoff_radius: float) -> np.ndarray:
    h, w = shape[:2]
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    mask2d = (dist <= cutoff_radius).astype(np.float32)

    if len(shape) == 3:
        return np.stack([mask2d, mask2d], axis=-1)
    return mask2d


def ideal_high_pass_filter(shape: tuple[int, int] | tuple[int, int, int], cutoff_radius: float) -> np.ndarray:
    return 1.0 - ideal_low_pass_filter(shape, cutoff_radius)


def apply_frequency_filter(image: npt.NDArray[np.generic], filter_mask: npt.NDArray[np.floating]) -> np.ndarray:
    spectrum = fft2_image(image)
    spectrum_shift = fftshift2(spectrum)

    mask = np.asarray(filter_mask, dtype=np.float32)
    if mask.ndim == 2:
        mask = np.stack([mask, mask], axis=-1)

    filtered = spectrum_shift * mask
    inv_shift = np.fft.ifftshift(filtered, axes=(0, 1))
    img_back = cv2.idft(inv_shift, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
    return img_back.astype(np.float32)


def normalize_to_uint8(x: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    arr = np.asarray(x, dtype=np.float32)
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx <= mn:
        return np.zeros_like(arr, dtype=np.uint8)
    y = (arr - mn) * (255.0 / (mx - mn))
    return np.clip(y, 0.0, 255.0).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description="Lab 01 skeleton (implement functions first).")
    parser.add_argument("--img1", type=str, default="lenna.png", help="First image from ./imgs/")
    parser.add_argument("--img2", type=str, default="airplane.bmp", help="Second image from ./imgs/")
    parser.add_argument("--out", type=str, default="out/lab01", help="Output directory (relative to repo root)")
    args = parser.parse_args()

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    def save_figure(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    repo_root = Path(__file__).resolve().parents[1]
    imgs_dir = repo_root / "imgs"
    out_dir = (repo_root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    img1 = cv2.imread(str(imgs_dir / args.img1), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(imgs_dir / args.img2), cv2.IMREAD_GRAYSCALE)
    if img1 is None:
        raise FileNotFoundError(str(imgs_dir / args.img1))
    if img2 is None:
        raise FileNotFoundError(str(imgs_dir / args.img2))

    missing: list[str] = []

    try:
        sp_noisy = add_salt_pepper_noise(img1, amount=0.08, salt_vs_pepper=0.55, seed=0)
        g_noisy = add_gaussian_noise(img1, sigma=15.0, seed=0)

        median_sp = apply_median_blur(sp_noisy, 5)
        gauss_sp = apply_gaussian_blur(sp_noisy, 5, 1.2)
        box_sp = apply_box_blur(sp_noisy, 5)

        gauss_g = apply_gaussian_blur(g_noisy, 5, 1.2)
        box_g = apply_box_blur(g_noisy, 5)

        plt.figure(figsize=(12, 6))
        for i, (title, im) in enumerate(
            [
                ("Original", img1),
                ("Salt & pepper", sp_noisy),
                ("Median (5x5)", median_sp),
                ("Gaussian (5,σ=1.2)", gauss_sp),
                ("Box (5x5)", box_sp),
                ("Gaussian noise", g_noisy),
            ],
            start=1,
        ):
            plt.subplot(2, 3, i)
            plt.title(title)
            plt.imshow(im, cmap="gray")
            plt.axis("off")
        save_figure(out_dir / "denoise_sp_and_examples.png")

        plt.figure(figsize=(12, 4))
        for i, (title, im) in enumerate(
            [
                ("Gaussian noise", g_noisy),
                ("Gaussian blur", gauss_g),
                ("Box blur", box_g),
            ],
            start=1,
        ):
            plt.subplot(1, 3, i)
            plt.title(title)
            plt.imshow(im, cmap="gray")
            plt.axis("off")
        save_figure(out_dir / "denoise_gaussian_noise.png")
    except NotImplementedError as exc:
        missing.append(str(exc))

    try:
        gx, gy, mag = sobel_edges(img2, ksize=3)
        _ = (gx, gy)
        lap = laplacian_edges(img2, ksize=3)

        plt.figure(figsize=(12, 4))
        for i, (title, im) in enumerate(
            [
                ("Input", img2),
                ("Sobel magnitude", normalize_to_uint8(mag)),
                ("Laplacian |·|", normalize_to_uint8(lap)),
            ],
            start=1,
        ):
            plt.subplot(1, 3, i)
            plt.title(title)
            plt.imshow(im, cmap="gray")
            plt.axis("off")
        save_figure(out_dir / "edges.png")

        cv2.imwrite(str(out_dir / "sobel_mag.png"), normalize_to_uint8(mag))
        cv2.imwrite(str(out_dir / "laplacian_abs.png"), normalize_to_uint8(lap))
    except NotImplementedError as exc:
        missing.append(str(exc))

    try:
        spec = fft2_image(img2)
        spec_shift = fftshift2(spec)
        mag = magnitude_spectrum(spec_shift, log_scale=True)

        lp = ideal_low_pass_filter(spec_shift.shape, cutoff_radius=30.0)
        hp = ideal_high_pass_filter(spec_shift.shape, cutoff_radius=30.0)
        lowpassed = apply_frequency_filter(img2, lp)
        highpassed = apply_frequency_filter(img2, hp)

        plt.figure(figsize=(12, 4))
        for i, (title, im) in enumerate(
            [
                ("Input", img2),
                ("Magnitude spectrum (log)", normalize_to_uint8(mag)),
                ("LPF result", normalize_to_uint8(lowpassed)),
                ("HPF result", normalize_to_uint8(highpassed)),
            ],
            start=1,
        ):
            plt.subplot(1, 4, i)
            plt.title(title)
            plt.imshow(im, cmap="gray")
            plt.axis("off")
        save_figure(out_dir / "fft_frequency_filters.png")
    except NotImplementedError as exc:
        missing.append(str(exc))

    if missing:
        (out_dir / "STATUS.txt").write_text(
            "Lab 01 demo is incomplete. Implement the TODO functions in labs/lab01_filtering_convolution_fft.py.\n\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {out_dir / 'STATUS.txt'}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
