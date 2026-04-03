from __future__ import annotations

"""Lab 02 (skeleton): Wavelets (Haar) + STFT bridge."""

import argparse
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import numpy.typing as npt
from scipy import signal

ThresholdMode = Literal["soft", "hard"]


def haar_dwt1(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute one-level 1D Haar DWT.

    For odd-length inputs, pad one sample by repeating the last value.

    Args:
        x: 1D numeric signal.

    Returns:
        (approx, detail): each length ~N/2.
    """
    x = np.asarray(x, dtype=np.float32).ravel()

    if x.ndim != 1:
        raise ValueError("x must be a 1D signal")

    if x.size % 2 == 1:
        x = np.concatenate([x, x[-1:]])

    even = x[0::2]
    odd = x[1::2]

    s = np.sqrt(2.0).astype(np.float32)
    approx = (even + odd) / s
    detail = (even - odd) / s

    return approx.astype(np.float32), detail.astype(np.float32)


def haar_idwt1(approx: np.ndarray, detail: np.ndarray) -> np.ndarray:
    """
    Invert one-level 1D Haar DWT.

    Args:
        approx: Approximation coefficients.
        detail: Detail coefficients.

    Returns:
        Reconstructed signal.
    """
    approx = np.asarray(approx, dtype=np.float32).ravel()
    detail = np.asarray(detail, dtype=np.float32).ravel()

    if approx.shape != detail.shape:
        raise ValueError("approx and detail must have the same shape")

    s = np.sqrt(2.0).astype(np.float32)

    even = (approx + detail) / s
    odd = (approx - detail) / s

    out = np.empty(approx.size * 2, dtype=np.float32)
    out[0::2] = even
    out[1::2] = odd
    return out


def haar_dwt2(image: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute one-level 2D separable Haar DWT for grayscale images.

    For odd sizes, last row/column is repeated before transform.

    Args:
        image: 2D grayscale image.

    Returns:
        LL, (LH, HL, HH).
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("image must be 2D grayscale")

    h, w = image.shape

    if h % 2 == 1:
        image = np.vstack([image, image[-1:, :]])
    if w % 2 == 1:
        image = np.hstack([image, image[:, -1:]])

    # Row transform
    low_rows = []
    high_rows = []
    for r in range(image.shape[0]):
        a, d = haar_dwt1(image[r, :])
        low_rows.append(a)
        high_rows.append(d)

    low_rows = np.stack(low_rows, axis=0)
    high_rows = np.stack(high_rows, axis=0)

    # Column transform on low_rows -> LL, HL
    ll_cols = []
    hl_cols = []
    for c in range(low_rows.shape[1]):
        a, d = haar_dwt1(low_rows[:, c])
        ll_cols.append(a)
        hl_cols.append(d)

    LL = np.stack(ll_cols, axis=1)
    HL = np.stack(hl_cols, axis=1)

    # Column transform on high_rows -> LH, HH
    lh_cols = []
    hh_cols = []
    for c in range(high_rows.shape[1]):
        a, d = haar_dwt1(high_rows[:, c])
        lh_cols.append(a)
        hh_cols.append(d)

    LH = np.stack(lh_cols, axis=1)
    HH = np.stack(hh_cols, axis=1)

    return LL.astype(np.float32), (LH.astype(np.float32), HL.astype(np.float32), HH.astype(np.float32))


def haar_idwt2(LL: np.ndarray, bands: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Invert one-level 2D Haar DWT.

    Crop policy for odd sizes: inverse reconstructs padded size; caller may crop if needed.

    Args:
        LL: Low-low sub-band.
        bands: Tuple `(LH, HL, HH)`.

    Returns:
        Reconstructed image.
    """
    LH, HL, HH = bands
    LL = np.asarray(LL, dtype=np.float32)
    LH = np.asarray(LH, dtype=np.float32)
    HL = np.asarray(HL, dtype=np.float32)
    HH = np.asarray(HH, dtype=np.float32)

    if not (LL.shape == LH.shape == HL.shape == HH.shape):
        raise ValueError("All sub-bands must have the same shape")

    # Inverse column transform
    low_rows_cols = []
    high_rows_cols = []

    for c in range(LL.shape[1]):
        col_low = haar_idwt1(LL[:, c], HL[:, c])
        col_high = haar_idwt1(LH[:, c], HH[:, c])
        low_rows_cols.append(col_low)
        high_rows_cols.append(col_high)

    low_rows = np.stack(low_rows_cols, axis=1)
    high_rows = np.stack(high_rows_cols, axis=1)

    # Inverse row transform
    rows = []
    for r in range(low_rows.shape[0]):
        row = haar_idwt1(low_rows[r, :], high_rows[r, :])
        rows.append(row)

    out = np.stack(rows, axis=0)
    return out.astype(np.float32)


def wavelet_threshold(coeffs: Any, threshold: float, mode: ThresholdMode = "soft") -> Any:
    """
    Apply thresholding to coefficient arrays.

    Args:
        coeffs: Array or nested tuples/lists of arrays.
        threshold: Non-negative threshold value.
        mode: `"soft"` or `"hard"`.

    Returns:
        Thresholded coefficients with same structure.
    """
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if mode not in ("soft", "hard"):
        raise ValueError("mode must be 'soft' or 'hard'")

    if isinstance(coeffs, tuple):
        return tuple(wavelet_threshold(c, threshold, mode) for c in coeffs)
    if isinstance(coeffs, list):
        return [wavelet_threshold(c, threshold, mode) for c in coeffs]

    arr = np.asarray(coeffs, dtype=np.float32)

    if mode == "hard":
        out = np.where(np.abs(arr) >= threshold, arr, 0.0)
    else:
        out = np.sign(arr) * np.maximum(np.abs(arr) - threshold, 0.0)

    return out.astype(np.float32)


def wavelet_denoise(image: np.ndarray, levels: int, threshold: float, mode: ThresholdMode = "soft") -> np.ndarray:
    """
    Denoise image via multi-level Haar thresholding.

    Args:
        image: 2D grayscale image.
        levels: Number of decomposition levels.
        threshold: Coefficient threshold.
        mode: `"soft"` or `"hard"`.

    Returns:
        Denoised image with deterministic behavior.
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("image must be 2D grayscale")
    if levels < 1:
        raise ValueError("levels must be >= 1")

    original_shape = image.shape
    current = image.copy()
    coeff_stack: list[tuple[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]] = []

    # Forward multi-level decomposition
    for _ in range(levels):
        shape_before = current.shape
        LL, (LH, HL, HH) = haar_dwt2(current)
        LH_t = wavelet_threshold(LH, threshold, mode)
        HL_t = wavelet_threshold(HL, threshold, mode)
        HH_t = wavelet_threshold(HH, threshold, mode)
        coeff_stack.append((shape_before, (LH_t, HL_t, HH_t)))
        current = LL

    # Inverse reconstruction
    recon = current
    for shape_before, (LH_t, HL_t, HH_t) in reversed(coeff_stack):
        recon = haar_idwt2(recon, (LH_t, HL_t, HH_t))
        recon = recon[: shape_before[0], : shape_before[1]]

    recon = recon[: original_shape[0], : original_shape[1]]
    return recon.astype(np.float32)


def stft1(
    x: np.ndarray,
    fs_hz: float,
    frame_len: int,
    hop_len: int,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute STFT for 1D signal using SciPy.

    Returns:
        `(freqs_hz, times_s, Zxx)` where `Zxx` is complex.
    """
    x = np.asarray(x, dtype=np.float64).ravel()

    if x.ndim != 1:
        raise ValueError("x must be 1D")
    if frame_len <= 0 or hop_len <= 0:
        raise ValueError("frame_len and hop_len must be positive")
    if hop_len > frame_len:
        raise ValueError("hop_len must be <= frame_len")

    noverlap = frame_len - hop_len

    freqs, times, zxx = signal.stft(
        x,
        fs=fs_hz,
        window=window,
        nperseg=frame_len,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )
    return freqs.astype(np.float32), times.astype(np.float32), zxx


def spectrogram_magnitude(Zxx: np.ndarray, log_scale: bool = True) -> np.ndarray:
    """
    Convert STFT matrix to magnitude spectrogram.

    Args:
        Zxx: Complex STFT matrix.
        log_scale: If True, return `log(1 + magnitude)`.

    Returns:
        Non-negative finite magnitude matrix.
    """
    mag = np.abs(Zxx).astype(np.float32)
    if log_scale:
        mag = np.log1p(mag)
    mag = np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)
    return mag.astype(np.float32)


def normalize_to_uint8(x: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    """Min-max normalize an array to `[0,255]` for visualization."""
    arr = np.asarray(x, dtype=np.float32)
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx <= mn:
        return np.zeros_like(arr, dtype=np.uint8)
    y = (arr - mn) * (255.0 / (mx - mn))
    return np.clip(y, 0.0, 255.0).astype(np.uint8)


def main() -> int:
    """
    Lab 02 demo (skeleton).

    Expected behavior after implementation:
    - wavelet denoising demo on image from `./imgs/`
    - LL/LH/HL/HH band visualization
    - STFT spectrogram demo on synthetic chirp signal
    - save outputs to `./out/lab02/` (no GUI windows)
    """
    parser = argparse.ArgumentParser(description="Lab 02 skeleton (implement functions first).")
    parser.add_argument("--img", type=str, default="lenna.png", help="Image from ./imgs/")
    parser.add_argument("--out", type=str, default="out/lab02", help="Output directory (relative to repo root)")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save_figure(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    repo_root = Path(__file__).resolve().parents[1]
    imgs_dir = repo_root / "imgs"
    out_dir = (repo_root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(imgs_dir / args.img), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(imgs_dir / args.img))

    missing: list[str] = []

    # --- Wavelet demo ---
    try:
        rng = np.random.default_rng(0)
        noisy = img.astype(np.float32) + rng.normal(0.0, 20.0, size=img.shape).astype(np.float32)
        noisy = np.clip(noisy, 0.0, 255.0)
        den = wavelet_denoise(noisy, levels=2, threshold=20.0, mode="soft")

        ll, (lh, hl, hh) = haar_dwt2(img.astype(np.float32))

        plt.figure(figsize=(12, 4))
        for i, (title, im) in enumerate(
            [
                ("Original", img),
                ("Noisy (Gaussian)", noisy),
                ("Wavelet denoised", den),
            ],
            start=1,
        ):
            plt.subplot(1, 3, i)
            plt.title(title)
            plt.imshow(normalize_to_uint8(im), cmap="gray")
            plt.axis("off")
        save_figure(out_dir / "wavelet_denoise.png")

        plt.figure(figsize=(10, 8))
        for i, (title, band) in enumerate(
            [
                ("LL", ll),
                ("LH", lh),
                ("HL", hl),
                ("HH", hh),
            ],
            start=1,
        ):
            plt.subplot(2, 2, i)
            plt.title(title)
            plt.imshow(normalize_to_uint8(band), cmap="gray")
            plt.axis("off")
        save_figure(out_dir / "wavelet_bands.png")
    except NotImplementedError as exc:
        missing.append(str(exc))

    # --- STFT bridge demo ---
    try:
        fs = 400.0
        duration_s = 2.0
        t = np.arange(int(fs * duration_s), dtype=np.float64) / fs
        f0, f1 = 15.0, 120.0
        k = (f1 - f0) / duration_s
        phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t)
        x = np.sin(phase)

        freqs, times, zxx = stft1(x, fs_hz=fs, frame_len=128, hop_len=32, window="hann")
        mag = spectrogram_magnitude(zxx, log_scale=True)

        plt.figure(figsize=(8, 4))
        plt.pcolormesh(times, freqs, mag, shading="gouraud")
        plt.title("STFT Spectrogram (log-magnitude)")
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency [Hz]")
        plt.colorbar(label="log(1 + |Zxx|)")
        save_figure(out_dir / "stft_spectrogram.png")
    except NotImplementedError as exc:
        missing.append(str(exc))

    if missing:
        (out_dir / "STATUS.txt").write_text(
            "Lab 02 demo is incomplete. Implement the TODO functions in labs/lab02_wavelets_stft.py.\n\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {out_dir / 'STATUS.txt'}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
