"""A panel of plain random noise, drawn the same way as the poster's ERA5 and BLE panels, for the 'where wind comes from' slide.

  noise.png  smoothed random u, v: wind speed as colour, white arrows on top

Run from the repo root:  uv run python talk/figures/noise.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "talk" / "public" / "figures"
rng = np.random.default_rng(0)


def smooth(a: np.ndarray, passes: int = 2) -> np.ndarray:
    """A light box blur, so the noise isn't pure static but still has no structure of real wind"""
    for _ in range(passes):
        a = (a + np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1)) / 5
    return a


u, v = (smooth(rng.normal(size=(24, 32))) * 25 for _ in range(2))
speed = np.hypot(u, v)

figure, ax = plt.subplots(figsize=(8, 6))
ax.imshow(speed, cmap="viridis", interpolation="nearest", extent=(0, 32, 0, 24), origin="lower")
ys, xs = np.mgrid[2:24:4, 2:32:4]
iu, iv = u[ys.astype(int), xs.astype(int)], v[ys.astype(int), xs.astype(int)]
ax.quiver(xs, ys, iu, iv, color="white", scale=320, width=0.004, headwidth=4)
ax.set_xlim(0, 32)
ax.set_ylim(0, 24)
ax.axis("off")
figure.savefig(OUT / "noise.png", dpi=200, bbox_inches="tight", pad_inches=0)
print(f"wrote {OUT / 'noise.png'}")
