"""The truth and a coarser copy for the fidelity-experiment slide, in the poster's ERA5 panel style.

  split-truth-px.png   held-out ERA5, 64 x 64 window at 0.25 degrees (the poster's intro panel: 2023-04-09,
                       level index 9), raw grid cells, an arrow every 7th cell
  split-coarse-px.png  the same field keeping only every 8th grid point (0.25 -> 2 degrees): each kept point
                       fills its 8 x 8 block, one arrow per block

Both share one colour scale and one arrow scale (arrow length = speed, never longer than the arrow spacing), so
they compare directly. In the experiment the controller sees a linear blend between kept points; the blocks show
what is kept. Data and rendering follow balloon-env-dev/src/eval/windeval/scripts/intro_comparison_panels.py.

Run:  cd ~/Dev/balloon-sim && uv run --with xarray --with zarr python <this file>
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

DATA = Path.home() / "Dev" / "balloon-env-dev" / "src" / "eval" / "windeval" / "data"
OUT = Path(__file__).resolve().parents[1] / "public" / "fig"
BLOCK, FRAME, LEVEL = 30, 3, 9  # the poster's chosen timestamp and level
K = 8  # keep every 8th grid point: 0.25 -> 2 degrees
ARROW_STEP = 7
PX, ASPECT = 1600, 4 / 3

times = np.load(DATA / "idiff_m2cond_blocks_4yr_250000.npz")["times"]
t = times[BLOCK, FRAME]
ref = xr.open_zarr(DATA / "era5_heldout.zarr", consolidated=False, zarr_format=2)
y0, x0 = (ref.sizes["y"] - 64) // 2, (ref.sizes["x"] - 64) // 2
ti = int(np.searchsorted(ref.time.values, t))
u = ref["u"].isel(time=ti, level=LEVEL, y=slice(y0, y0 + 64), x=slice(x0, x0 + 64)).values
v = ref["v"].isel(time=ti, level=LEVEL, y=slice(y0, y0 + 64), x=slice(x0, x0 + 64)).values

# coarse copy: each kept point fills its K x K block
ku, kv = u[::K, ::K], v[::K, ::K]
cu = np.repeat(np.repeat(ku, K, 0), K, 1)
cv = np.repeat(np.repeat(kv, K, 0), K, 1)

# 4:3 crop, centred vertically, aligned to block edges so the coarse blocks stay whole
H, W = u.shape
ch = int(round(W / ASPECT))
r0 = ((H - ch) // 2) // K * K
rows = slice(r0, r0 + ch)

speed = np.hypot(u, v)
vmin, vmax = np.percentile(speed[rows], [2, 98])  # the poster's per-panel stretch, shared by both
amax = max(np.hypot(u, v).max(), 1e-6)
scale = amax / (0.9 * ARROW_STEP)  # one arrow scale for both panels; fastest arrow spans 0.9 of the truth spacing


def render(name, su, sv, arrows):
    s = np.hypot(su, sv)[rows]
    fig = plt.figure(figsize=(PX / 300, PX / ASPECT / 300), dpi=300, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(s, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
    yy, xx, au, av = arrows
    ax.quiver(xx, yy - r0, au, -av, color="white", angles="xy", scale_units="xy", scale=scale,
              width=0.004, headwidth=3.2, headlength=4.0, headaxislength=3.3)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(ch - 0.5, -0.5)
    ax.set_axis_off()
    fig.savefig(OUT / name, dpi=300, facecolor="white", pad_inches=0)
    plt.close(fig)
    print(f"wrote {OUT / name}")


yy, xx = np.mgrid[ARROW_STEP // 2:H:ARROW_STEP, ARROW_STEP // 2:W:ARROW_STEP]
keep = (yy >= r0) & (yy < r0 + ch)
render("split-truth-px.png", u, v, (yy[keep], xx[keep], u[yy, xx][keep], v[yy, xx][keep]))

by, bx = np.mgrid[0:H:K, 0:W:K]
cy, cx = by + (K - 1) / 2, bx + (K - 1) / 2
keep = (cy >= r0) & (cy < r0 + ch)
render("split-coarse-px.png", cu, cv, (cy[keep], cx[keep], ku[keep], kv[keep]))
print(f"{t}  truth 64x64 at 0.25 deg; kept {ku.shape[0]}x{ku.shape[1]} at 2 deg")
