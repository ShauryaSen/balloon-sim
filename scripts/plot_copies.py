"""One picture of the fidelity experiment's coarser copies: a row per property, a column per step down the ladder

    uv run scripts/plot_copies.py records/era5_global_jul2021.npz --out figures/fidelity/copies.png

Each row cuts the wind along the axis it coarsens, so the loss is visible:
    altitude levels   altitude against longitude at 35N, at launch
    frames in time    time against longitude at 35N, at the middle of the band
    grid              latitude against longitude, at the middle of the band, at launch
Colour is the eastward wind, blue westward and red eastward, so where the colours flip within a column is where
the winds at different altitudes oppose, which is what the balloon steers on. Every panel shows the copy as the
controller reads it: blended linearly between kept points.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from balloon_sim import degrade
from balloon_sim.wind import load_record

LAT_CUT, LON_RANGE, LAT_RANGE = 35.0, (150.0, 230.0), (20.0, 50.0)
WIND = LinearSegmentedColormap.from_list("wind", ["#173f73", "#2a78d6", "#f0efec", "#e34948", "#8f2526"])
INK, MUTED = "#1a1a1a", "#6b6b6b"


def rungs(n_levels: int) -> list[tuple[str, list[tuple[str, object]]]]:
    """Each row's label and its ladder of (column title, copy)"""
    level_steps = [n for n in (18, 9, 6, 3, 2, 1) if n <= n_levels]
    return [
        ("Altitude levels", [(f"{n} level{'s' if n > 1 else ''}" + (" (truth)" if n == n_levels else ""), ("levels", n)) for n in level_steps]),
        ("Frames in time", [("every 3 h (truth)", ("frames", 1)), ("every 6 h", ("frames", 2)), ("every 12 h", ("frames", 4)),
                            ("every 24 h", ("frames", 8)), ("frozen at launch", ("frames", None))]),
        ("Grid", [("2° (truth)", ("grid", 1)), ("4°", ("grid", 2)), ("8°", ("grid", 4)), ("16°", ("grid", 8))]),
    ]


def build(truth, kind: str, setting):
    return {"levels": degrade.levels, "frames": degrade.frames, "grid": degrade.grid}[kind](truth, setting)


def main() -> None:
    ask = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ask.add_argument("record")
    ask.add_argument("--out", default="figures/fidelity/copies.png")
    a = ask.parse_args()

    truth = load_record(a.record)
    lat, lon, alt, hours = (np.asarray(x) for x in (truth.lat, truth.lon, truth.altitude_km, truth.hours))
    i_lat = int(np.argmin(abs(lat - LAT_CUT)))
    cols = (lon >= LON_RANGE[0]) & (lon <= LON_RANGE[1])
    rows = (lat >= LAT_RANGE[0]) & (lat <= LAT_RANGE[1])
    middle = len(alt) // 2
    fine_alt = np.linspace(alt[0], alt[-1], 120)  # read between levels as wind_at does: a straight line

    def cut(field, kind):
        u = np.asarray(field.uv[..., 0])
        if kind == "levels":
            column = u[0, :, i_lat][:, cols]  # (alt, lon)
            return np.stack([np.interp(fine_alt, alt, column[:, j]) for j in range(column.shape[1])], 1)
        if kind == "frames":
            return u[:, middle, i_lat][:, cols]  # (time, lon)
        return u[0, middle][rows][:, cols]  # (lat, lon)

    ladder = rungs(len(alt))
    n_cols = max(len(steps) for _, steps in ladder)
    cuts = {(r, c): cut(build(truth, *copy), copy[0]) for r, (_, steps) in enumerate(ladder) for c, (_, copy) in enumerate(steps)}
    limit = np.percentile(np.abs(np.concatenate([x.ravel() for x in cuts.values()])), 98)

    fig, axes = plt.subplots(len(ladder), n_cols, figsize=(2.6 * n_cols + 1.2, 2.3 * len(ladder) + 0.9), squeeze=False)
    extent = {"levels": (LON_RANGE[0], LON_RANGE[1], alt[0], alt[-1]),
              "frames": (LON_RANGE[0], LON_RANGE[1], hours[-1] / 24, 0),
              "grid": (LON_RANGE[0], LON_RANGE[1], LAT_RANGE[0], LAT_RANGE[1])}
    y_label = {"levels": "altitude (km)", "frames": "days after launch", "grid": "latitude"}
    for r, (label, steps) in enumerate(ladder):
        for c in range(n_cols):
            ax = axes[r, c]
            if c >= len(steps):
                ax.axis("off")
                continue
            title, (kind, _) = steps[c]
            shown = ax.imshow(cuts[r, c], cmap=WIND, vmin=-limit, vmax=limit, aspect="auto", interpolation="nearest",
                              origin="upper" if kind == "frames" else "lower", extent=extent[kind])
            ax.set_title(title, fontsize=10, color=INK, pad=4)
            ax.tick_params(labelsize=7, colors=MUTED, length=2)
            for side in ax.spines.values():
                side.set_visible(False)
            if c == 0:
                ax.set_ylabel(y_label[kind], fontsize=8, color=MUTED)
                ax.annotate(label, xy=(-0.42, 0.5), xycoords="axes fraction", rotation=90, ha="center", va="center",
                            fontsize=12, fontweight="bold", color=INK)
            else:
                ax.set_yticklabels([])
            if r == len(ladder) - 1 or c >= len(ladder[r + 1][1]):
                ax.set_xlabel("longitude (°E)", fontsize=8, color=MUTED)
            else:
                ax.set_xticklabels([])

    bar = fig.colorbar(shown, ax=axes, shrink=0.55, pad=0.015, aspect=30)
    bar.set_label("eastward wind (m/s)", fontsize=9, color=MUTED)
    bar.ax.tick_params(labelsize=7, colors=MUTED)
    bar.outline.set_visible(False)
    fig.suptitle("Degraded copies of ERA5, July 2021, North Pacific", fontsize=13, color=INK, x=0.45, y=0.995)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"wrote {a.out}; colour limit ±{limit:.1f} m/s")


if __name__ == "__main__":
    main()
