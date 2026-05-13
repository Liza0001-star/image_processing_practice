from __future__ import annotations

"""Lab 05: motion estimation with dense optical flow."""

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def optical_flow_farneback(
    prev_gray: np.ndarray,
    next_gray: np.ndarray,
    **params: Any
) -> np.ndarray:
    """
    Compute dense optical flow using Farneback algorithm.
    """
    if prev_gray.ndim != 2 or next_gray.ndim != 2:
        raise ValueError("prev_gray and next_gray must be grayscale images")

    if prev_gray.shape != next_gray.shape:
        raise ValueError("prev_gray and next_gray must have the same shape")

    prev = prev_gray.astype(np.uint8)
    nxt = next_gray.astype(np.uint8)

    default_params = {
        "pyr_scale": 0.5,
        "levels": 3,
        "winsize": 15,
        "iterations": 3,
        "poly_n": 5,
        "poly_sigma": 1.2,
        "flags": 0,
    }

    default_params.update(params)

    flow = cv2.calcOpticalFlowFarneback(
        prev,
        nxt,
        None,
        default_params["pyr_scale"],
        default_params["levels"],
        default_params["winsize"],
        default_params["iterations"],
        default_params["poly_n"],
        default_params["poly_sigma"],
        default_params["flags"],
    )

    return flow.astype(np.float32)


def flow_to_hsv(flow_xy: np.ndarray) -> np.ndarray:
    """
    Convert flow field to BGR visualization via HSV mapping.
    """
    if flow_xy.ndim != 3 or flow_xy.shape[2] != 2:
        raise ValueError("flow_xy must have shape (H, W, 2)")

    dx = flow_xy[..., 0]
    dy = flow_xy[..., 1]

    magnitude, angle = cv2.cartToPolar(dx, dy, angleInDegrees=True)

    hsv = np.zeros((flow_xy.shape[0], flow_xy.shape[1], 3), dtype=np.uint8)

    hsv[..., 0] = (angle / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr


def main() -> int:
    """
    Lab 05 demo.
    """
    parser = argparse.ArgumentParser(description="Lab 05: motion estimation with dense optical flow.")
    parser.add_argument("--img", type=str, default="airplane.bmp", help="Input image from ./imgs/")
    parser.add_argument("--out", type=str, default="out/lab05", help="Output directory relative to repo root")
    parser.add_argument("--dx", type=float, default=5.0, help="Horizontal translation in pixels")
    parser.add_argument("--dy", type=float, default=3.0, help="Vertical translation in pixels")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")

    repo_root = Path(__file__).resolve().parents[1]
    imgs_dir = repo_root / "imgs"
    out_dir = (repo_root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(imgs_dir / args.img), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(str(imgs_dir / args.img))

    prev = img
    h, w = prev.shape

    M = np.array(
        [
            [1.0, 0.0, float(args.dx)],
            [0.0, 1.0, float(args.dy)],
        ],
        dtype=np.float32,
    )

    nxt = cv2.warpAffine(
        prev,
        M,
        dsize=(w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    flow = optical_flow_farneback(prev, nxt)
    vis = flow_to_hsv(flow)

    cv2.imwrite(str(out_dir / "prev.png"), prev)
    cv2.imwrite(str(out_dir / "next.png"), nxt)
    cv2.imwrite(str(out_dir / "flow_vis.png"), vis)

    mean_dx = float(np.mean(flow[..., 0]))
    mean_dy = float(np.mean(flow[..., 1]))
    mean_mag = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))

    status_text = (
        "Lab 05 completed successfully.\n\n"
        f"Input image: {args.img}\n"
        f"Expected translation dx: {args.dx}\n"
        f"Expected translation dy: {args.dy}\n"
        f"Estimated mean dx: {mean_dx:.4f}\n"
        f"Estimated mean dy: {mean_dy:.4f}\n"
        f"Mean flow magnitude: {mean_mag:.4f}\n"
    )

    (out_dir / "STATUS.txt").write_text(status_text, encoding="utf-8")

    print(status_text)
    print(f"Saved outputs to: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
