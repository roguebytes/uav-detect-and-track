"""Generate a seamless procedural grass texture (no third-party assets, no licence questions).

    python tools/make_grass_texture.py --out sim/models/grass_ground/grass.png --size 4096 --seed 1

The texture is not committed (see .gitignore); scripts/setup.sh regenerates it.
Layers: low-frequency colour patches (mown stripes and dry spots), mid-frequency clumps,
high-frequency blade noise, all made tileable by wrapping the noise. Colours are picked to
sit near the mean of real DJI Mini 4 Pro grass frames (green channel dominant, muted).
"""

from __future__ import annotations

__author__ = "Frank Loewenich"

import argparse

import cv2
import numpy as np


def tileable_noise(size: int, scale: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth noise in [0,1] that tiles seamlessly, built by upsampling a wrapped coarse grid."""
    n = max(2, size // scale)
    coarse = rng.random((n, n), dtype=np.float32)
    # wrap by tiling 3x3 and cropping the centre after resize so edges match
    big = np.tile(coarse, (3, 3))
    up = cv2.resize(big, (size * 3, size * 3), interpolation=cv2.INTER_CUBIC)
    out = up[size:2 * size, size:2 * size]
    out -= out.min()
    return out / max(out.max(), 1e-6)


def blade_strokes(size: int, count: int, rng: np.random.Generator) -> np.ndarray:
    """Short bright and dark strokes in random directions, wrapped so the tile stays seamless."""
    layer = np.zeros((size, size), dtype=np.float32)
    xs = rng.integers(0, size, count)
    ys = rng.integers(0, size, count)
    ang = rng.uniform(0, np.pi, count)
    ln = rng.uniform(3, 9, count)
    val = rng.uniform(-1, 1, count)
    for x, y, a, length, v in zip(xs, ys, ang, ln, val):
        dx, dy = int(round(np.cos(a) * length)), int(round(np.sin(a) * length))
        for ox in (-size, 0, size):
            for oy in (-size, 0, size):
                cv2.line(layer, (int(x) + ox, int(y) + oy), (int(x) + dx + ox, int(y) + dy + oy), float(v), 1, cv2.LINE_AA)
    return layer


def make_grass(size: int, seed: int) -> np.ndarray:
    """Compose the grass texture from patch, clump, blade and grain layers."""
    rng = np.random.default_rng(seed)
    patches = tileable_noise(size, size // 5, rng)     # dry and lush patches, about 3 m across
    clumps = tileable_noise(size, max(8, size // 64), rng)   # grass clumps, about 25 cm
    blades = blade_strokes(size, size * size // 40, rng)
    fine = rng.random((size, size), dtype=np.float32)  # per-pixel grain

    lush = np.array([66, 124, 84], dtype=np.float32)  # BGR, muted green; the renderer darkens by about 25%
    dry = np.array([78, 134, 108], dtype=np.float32)  # BGR, slightly drier grass (kept close to lush so tiles do not read as a checkerboard)
    base = lush[None, None, :] * (1 - patches[..., None]) + dry[None, None, :] * patches[..., None]
    shade = 0.88 + 0.14 * clumps + 0.22 * blades + 0.18 * (fine - 0.5)
    img = base * shade[..., None]
    soil = tileable_noise(size, max(3, size // 800), rng) > 0.94   # sparse darker soil specks
    img[soil] *= 0.6
    return np.clip(img, 0, 255).astype(np.uint8)


def main() -> None:
    """Parse the command line and write the texture."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="sim/models/grass_ground/grass.png")
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    img = make_grass(a.size, a.seed)
    cv2.imwrite(a.out, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"wrote {a.out} {img.shape[1]}x{img.shape[0]} mean BGR {img.reshape(-1, 3).mean(axis=0).round(1)}")


if __name__ == "__main__":
    main()
