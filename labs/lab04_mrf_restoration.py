from __future__ import annotations

"""Lab 04: Markov Random Field (MRF) image restoration."""

import argparse
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

PenaltyType = Literal["quadratic", "huber"]


def _huber_penalty(d: np.ndarray, delta: float) -> np.ndarray:
    abs_d = np.abs(d)
    return np.where(abs_d <= delta, 0.5 * d**2, delta * (abs_d - 0.5 * delta))


def _huber_grad(d: np.ndarray, delta: float) -> np.ndarray:
    return np.where(np.abs(d) <= delta, d, delta * np.sign(d))


def mrf_energy(
    x: np.ndarray,
    y: np.ndarray,
    lambda_smooth: float,
    penalty: PenaltyType = "quadratic",
    huber_delta: float = 1.0,
) -> float:
    """
    Compute pairwise MRF energy for grayscale image restoration.
    """
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    data_term = np.sum((x - y) ** 2)

    dx = x[:, 1:] - x[:, :-1]
    dy = x[1:, :] - x[:-1, :]

    if penalty == "quadratic":
        smooth_term = np.sum(dx**2) + np.sum(dy**2)
    elif penalty == "huber":
        smooth_term = np.sum(_huber_penalty(dx, huber_delta)) + np.sum(
            _huber_penalty(dy, huber_delta)
        )
    else:
        raise ValueError(f"Unknown penalty: {penalty}")

    return float(data_term + lambda_smooth * smooth_term)


def mrf_denoise(
    y: np.ndarray,
    lambda_smooth: float,
    num_iters: int,
    step: float = 0.1,
    penalty: PenaltyType = "quadratic",
    huber_delta: float = 1.0,
) -> np.ndarray:
    """
    Restore grayscale image by minimizing MRF energy.
    """
    y = y.astype(np.float32)
    x = y.copy()

    for _ in range(num_iters):
        grad = 2.0 * (x - y)

        # Horizontal neighbours
        d = x[:, 1:] - x[:, :-1]

        if penalty == "quadratic":
            g = 2.0 * d
        elif penalty == "huber":
            g = _huber_grad(d, huber_delta)
        else:
            raise ValueError(f"Unknown penalty: {penalty}")

        grad[:, 1:] += lambda_smooth * g
        grad[:, :-1] -= lambda_smooth * g

        # Vertical neighbours
        d = x[1:, :] - x[:-1, :]

        if penalty == "quadratic":
            g = 2.0 * d
        elif penalty == "huber":
            g = _huber_grad(d, huber_delta)
        else:
            raise ValueError(f"Unknown penalty: {penalty}")

        grad[1:, :] += lambda_smooth * g
        grad[:-1, :] -= lambda_smooth * g

        x -= step * grad
        x = np.clip(x, 0.0, 255.0)

    return x.astype(np.float32)


def normalize_to_uint8(x: np.ndarray) -> np.ndarray:
    """Min-max normalize array to [0,255] uint8 for visualization."""
    x = x.astype(np.float32)
    min_val = float(np.min(x))
    max_val = float(np.max(x))

    if max_val - min_val < 1e-8:
        return np.zeros_like(x, dtype=np.uint8)

    normalized = (x - min_val) / (max_val - min_val)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def main() -> int:
    """
    Lab 04 demo.

    Behavior:
    - load grayscale image from ./imgs/
    - add Gaussian noise
    - denoise with MRF quadratic and Huber penalties
    - save side-by-side result to ./out/lab04/mrf_denoise.png
    """
    parser = argparse.ArgumentParser(description="Lab 04 MRF image restoration.")
    parser.add_argument("--img", type=str, default="lenna.png", help="Input image from ./imgs/")
    parser.add_argument("--out", type=str, default="out/lab04", help="Output directory")
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

    clean = img.astype(np.float32)

    rng = np.random.default_rng(0)
    noisy = clean + rng.normal(0.0, 18.0, size=clean.shape).astype(np.float32)
    noisy = np.clip(noisy, 0.0, 255.0)

    den_quad = mrf_denoise(
        noisy,
        lambda_smooth=0.25,
        num_iters=80,
        step=0.1,
        penalty="quadratic",
    )

    den_hub = mrf_denoise(
        noisy,
        lambda_smooth=0.25,
        num_iters=80,
        step=0.1,
        penalty="huber",
        huber_delta=8.0,
    )

    e_noisy_q = mrf_energy(noisy, noisy, lambda_smooth=0.25, penalty="quadratic")
    e_quad = mrf_energy(den_quad, noisy, lambda_smooth=0.25, penalty="quadratic")

    e_noisy_h = mrf_energy(
        noisy,
        noisy,
        lambda_smooth=0.25,
        penalty="huber",
        huber_delta=8.0,
    )
    e_hub = mrf_energy(
        den_hub,
        noisy,
        lambda_smooth=0.25,
        penalty="huber",
        huber_delta=8.0,
    )

    plt.figure(figsize=(12, 4))

    panels = [
        ("Original", clean),
        ("Noisy (seed=0)", noisy),
        (f"MRF quadratic\nE: {e_noisy_q:.1f} -> {e_quad:.1f}", den_quad),
        (f"MRF huber\nE: {e_noisy_h:.1f} -> {e_hub:.1f}", den_hub),
    ]

    for i, (title, im) in enumerate(panels, start=1):
        plt.subplot(1, 4, i)
        plt.title(title)
        plt.imshow(normalize_to_uint8(im), cmap="gray")
        plt.axis("off")

    save_figure(out_dir / "mrf_denoise.png")

    print(f"Saved result to: {out_dir / 'mrf_denoise.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
