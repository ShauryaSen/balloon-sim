"""How far one reading's information reaches, as a picture: the measured forecast-error correlation against the
kernel assumed so far

    uv run scripts/plot_error_reach.py results/error-covariance-2026-11-18.json --out figures/error-covariance/reach.png

    from above    a reading at the centre; where the error stays correlated above 1/e. Measured
                  as an ellipse through the east and north reaches (results only go along those two axes);
                  assumed as the circle of BeliefConfig's 544 km kernel. The grey ring is how far the balloon
                  drifts in a 12 h plan at the band's mean wind, 13 m/s, for scale
    from the side a reading at the level nearest 17.2 km; each level coloured by its error's correlation with
                  that level's, measured and assumed side by side
    level by level  the measured correlation between every pair of levels, mean over the weeks
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Ellipse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from balloon_sim.config import BeliefConfig

INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
CORR = LinearSegmentedColormap.from_list("corr", ["#8f2526", "#e34948", "#f0efec", "#2a78d6", "#173f73"])
ANCHOR_KM = 17.2


def matern52(r):
    r = np.abs(r)
    return (1 + np.sqrt(5) * r + 5 * r**2 / 3) * np.exp(-np.sqrt(5) * r)


def crossing(x, y, level) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    i = np.where(y < level)[0][0]
    return float(x[i - 1] + (level - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1]))


def level_matrix(results) -> tuple[np.ndarray, np.ndarray]:
    """(levels,) altitudes bottom to top and (levels, levels) correlation, mean over records"""
    alt = np.array(results[0]["levels_km"])
    n = len(alt)
    mats = []
    for r in results:
        m = np.eye(n)
        for i, row in enumerate(r["up"]):
            for k, (_, c) in enumerate(row):
                m[i, i + k] = m[i + k, i] = c
        mats.append(m)
    return alt, np.mean(mats, axis=0)


def main() -> None:
    ask = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ask.add_argument("measured")
    ask.add_argument("--out", default="figures/error-covariance/reach.png")
    a = ask.parse_args()
    results = json.loads(Path(a.measured).read_text())
    k = BeliefConfig()
    east = np.mean([r["east"] for r in results], 0)
    north = np.mean([r["north"] for r in results], 0)
    x_e, x_n = results[0]["east_km"], results[0]["north_km"]
    alt, mat = level_matrix(results)

    fig = plt.figure(figsize=(14, 5.2), facecolor=SURFACE)
    fig.text(0.01, 0.97, "How far one reading's information reaches", fontsize=13, color=INK, weight="bold", va="top")
    fig.text(0.01, 0.91, f"Correlation of a day-old forecast's error, measured on {len(results)} weeks (10S to 10N) "
             "against the kernel assumed so far.", fontsize=9.5, color=MUTED, va="top")
    grid = fig.add_gridspec(1, 3, left=0.04, right=0.97, top=0.8, bottom=0.11, wspace=0.28, width_ratios=[1.25, 0.9, 1.1])

    ax = fig.add_subplot(grid[0])
    ax.set_facecolor(SURFACE)
    drift_km = 13 * 12 * 3.6
    ae, an = crossing(x_e, east, np.exp(-1)), crossing(x_n, north, np.exp(-1))
    d = np.linspace(0, 3000, 600)
    assumed_km = crossing(d, matern52(d / k.horizontal_km), np.exp(-1))
    ax.add_patch(Ellipse((0, 0), 2 * ae, 2 * an, facecolor="#2a78d6", alpha=0.25, edgecolor="#173f73", linewidth=1.2))
    ax.add_patch(Circle((0, 0), assumed_km, fill=False, edgecolor=INK, linewidth=1.3, linestyle="--"))
    ax.add_patch(Circle((0, 0), drift_km, fill=False, edgecolor="#D85A30", linewidth=1.3, linestyle=(0, (1, 2))))
    ax.plot(0, 0, "o", color="#D85A30", markersize=8)
    ax.text(-1550, -1180, f"blue: measured reach, {ae:,.0f} km east, {an:,.0f} km north\n"
            f"dashed: assumed, {assumed_km:,.0f} km every way\norange dots: 12 h of drift at 13 m/s ({drift_km:,.0f} km)\n"
            "reach = where correlation falls to 1/e", fontsize=8.5, color=MUTED, va="top")
    ax.set_xlim(-1600, 1600)
    ax.set_ylim(-1700, 1300)
    ax.set_aspect("equal")
    ax.set_xlabel("km east", color=MUTED, fontsize=9)
    ax.set_ylabel("km north", color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title("from above: a reading at the centre", fontsize=10, color=INK, loc="left")

    ax = fig.add_subplot(grid[1])
    i = int(np.argmin(abs(alt - ANCHOR_KM)))
    measured = mat[i]
    assumed = matern52((alt - alt[i]) / k.altitude_km)
    edges = np.concatenate([[alt[0] - 0.17], (alt[1:] + alt[:-1]) / 2, [alt[-1] + 0.17]])
    for col, values in ((0, assumed), (1, measured)):
        for j in range(len(alt)):
            ax.add_patch(plt.Rectangle((col, edges[j]), 0.8, edges[j + 1] - edges[j], facecolor=CORR((values[j] + 1) / 2), edgecolor=SURFACE, linewidth=0.6))
    ax.plot([0.4, 1.4], [alt[i], alt[i]], "o", color="#D85A30", markersize=7)
    ax.set_xlim(-0.1, 1.9)
    ax.set_ylim(edges[0], edges[-1])
    ax.set_xticks([0.4, 1.4], ["assumed", "measured"])
    ax.set_ylabel("altitude (km)", color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_facecolor(SURFACE)
    ax.set_title(f"from the side: a reading at {alt[i]:.1f} km", fontsize=10, color=INK, loc="left")

    ax = fig.add_subplot(grid[2])
    shown = ax.pcolormesh(edges, edges, mat, cmap=CORR, vmin=-1, vmax=1)
    ax.set_aspect("equal")
    ax.set_xlabel("altitude (km)", color=MUTED, fontsize=9)
    ax.set_ylabel("altitude (km)", color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.set_title("level by level, measured", fontsize=10, color=INK, loc="left")
    bar = fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.04)
    bar.set_label("correlation (red: opposite sign)", color=MUTED, fontsize=8.5)
    bar.ax.tick_params(colors=MUTED, labelsize=8, length=0)
    bar.outline.set_visible(False)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
