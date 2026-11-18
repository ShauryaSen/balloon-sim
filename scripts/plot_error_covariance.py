"""Draw the measured correlation of a day-old forecast's error in each direction, against our fitted kernel

    uv run scripts/plot_error_covariance.py results/error-covariance-2026-11-18.json --out figures/error-covariance/k.png

Four panels: horizontal (east, north, along and across the wind), altitude (every pair of levels), time, and the east
error with the north error. Lines are the mean over the records (weeks), the band their range; the dashed grey
line is the kernel every sweep so far assumed (BeliefConfig: Matern 5/2 over 544 km and 1.3 km, Matern 3/2 over
9.1 h, u and v unrelated). Also prints each direction's reach: where the correlation first falls below 1/e.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from balloon_sim.config import BeliefConfig

INK, MUTED, GRIDLINE, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"


def matern52(r):
    r = np.abs(r)
    return (1 + np.sqrt(5) * r + 5 * r**2 / 3) * np.exp(-np.sqrt(5) * r)


def matern32(r):
    r = np.abs(r)
    return (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)


def reach(x, y) -> float:
    """Where y first falls below 1/e, by straight-line interpolation; nan if it never does"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    below = np.where(y < np.exp(-1))[0]
    if len(below) == 0 or below[0] == 0:
        return float("nan")
    i = below[0]
    return float(x[i - 1] + (np.exp(-1) - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1]))


def altitude_curves(results, key, bins):
    """Correlation against altitude gap, pooled over every starting level, per record"""
    out = []
    for r in results:
        pairs = np.array([p for level in r[key] for p in level])
        idx = np.digitize(pairs[:, 0], bins) - 1
        out.append([pairs[idx == i, 1].mean() if np.any(idx == i) else np.nan for i in range(len(bins) - 1)])
    return np.array(out)


def style(ax, xlabel):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRIDLINE)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    ax.grid(color=GRIDLINE, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.axhline(np.exp(-1), color=MUTED, linewidth=0.7, linestyle=(0, (2, 3)))
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylim(-0.4, 1.02)


def band(ax, x, ys, color, label):
    ys = np.array(ys, float)
    ax.fill_between(x, np.nanmin(ys, 0), np.nanmax(ys, 0), color=color, alpha=0.18, linewidth=0)
    ax.plot(x, np.nanmean(ys, 0), color=color, linewidth=2, label=label)


def main() -> None:
    ask = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ask.add_argument("measured")
    ask.add_argument("--out", default="figures/error-covariance/k.png")
    a = ask.parse_args()
    results = json.loads(Path(a.measured).read_text())
    k = BeliefConfig()
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.9), facecolor=SURFACE, sharey=True)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.76, bottom=0.15, wspace=0.12)
    fig.text(0.01, 0.97, "Correlation of a day-old forecast's error, by direction", fontsize=13, color=INK, weight="bold", va="top")
    fig.text(0.01, 0.89, f"{len(results)} weeks, 10S to 10N; lines are the mean over weeks, bands their range. "
             "Dashed grey: the kernel assumed so far. Dotted: 1/e.", fontsize=9.5, color=MUTED, va="top")
    reaches = {}

    ax = axes[0]
    style(ax, "distance (km)")
    for key, xkey, color, name in (("east", "east_km", BLUE, "east"), ("north", "north_km", ORANGE, "north"),
                                   ("along", "direction_km", AQUA, "along the wind"), ("across", "direction_km", YELLOW, "across the wind")):
        x = results[0][xkey]
        ys = [r[key] for r in results]
        if key in ("along", "across"):  # the first bin holds no offsets at zero distance; anchor it at 1
            x, ys = [0.0] + list(x), [[1.0] + list(y) for y in ys]
        band(ax, x, ys, color, name)
        reaches[name] = reach(x, np.nanmean(np.array(ys, float), 0))
    d = np.linspace(0, 1800, 100)
    ax.plot(d, matern52(d / k.horizontal_km), color=MUTED, linestyle="--", linewidth=1.2, label="assumed")
    ax.set_title("horizontal", fontsize=10, color=INK, loc="left")
    ax.set_ylabel("correlation", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")

    ax = axes[1]
    style(ax, "altitude gap (km)")
    bins = np.arange(-0.175, 6.2, 0.35)
    mid = bins[:-1] + 0.175
    ys = altitude_curves(results, "up", bins)  # every pair of levels once; up and down are the same pairs pooled
    band(ax, mid, ys, BLUE, "measured")
    reaches["altitude"] = reach(mid, np.nanmean(ys, 0))
    z = np.linspace(0, 6, 100)
    ax.plot(z, matern52(z / k.altitude_km), color=MUTED, linestyle="--", linewidth=1.2, label="assumed")
    ax.set_title("altitude", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")

    ax = axes[2]
    style(ax, "hours later")
    band(ax, results[0]["hours"], [r["time"] for r in results], BLUE, "measured")
    reaches["time"] = reach(results[0]["hours"], np.mean([r["time"] for r in results], 0))
    h = np.linspace(0, 36, 100)
    ax.plot(h, matern32(h / k.hours), color=MUTED, linestyle="--", linewidth=1.2, label="assumed")
    ax.set_title("time (dips near 24 h by construction)", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")

    ax = axes[3]
    style(ax, "distance east (km)")
    band(ax, results[0]["east_km"], [r["u_with_v"] for r in results], BLUE, "measured")
    ax.axhline(0, color=MUTED, linestyle="--", linewidth=1.2, label="assumed: 0")
    ax.set_title("east error with north error", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200)
    print(f"wrote {a.out}")
    print("reach (correlation falls to 1/e):")
    for name, value in reaches.items():
        unit = "h" if name == "time" else "km"
        print(f"  {name:16s} {value:7.1f} {unit}")
    print(f"assumed: horizontal {reach(d, matern52(d / k.horizontal_km)):.0f} km, altitude {reach(z, matern52(z / k.altitude_km)):.2f} km, "
          f"time {reach(h, matern32(h / k.hours)):.1f} h")
    print("RMS by week:", ", ".join(f"{r['record'].split('_')[2]} {r['rms_ms']:.2f}" for r in results))


if __name__ == "__main__":
    main()
