"""The fidelity sweep's results as four figures, for a meeting: the main test, the ladders, the error shapes, the
two interactions

    uv run scripts/plot_fidelity.py results/rows-fidelity-2026-11-17.jsonl --out figures/fidelity

Every number is the score of balloon_sim.fidelity.score: the share of MPC's time within 50 km a copy costs,
against coast (0 nothing lost, 1 no better than never steering), with its 95% paired bootstrap interval over
the seeds. Each copy is scored against MPC on the truth in its own arm.

    test.png          score against the wind error, and against the level-difference error, one dot a copy
    ladders.png       score at each step of the levels, frames and grid ladders
    shapes.png        score of each error shape at 1x, and the two offsets at 2x, a day-old forecast's error
    interactions.png  levels at two climb limits; frames at three horizons

The dashed line at 0.10 is a reading aid, not a test: a loss under a tenth of MPC's control (assumed).
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from balloon_sim.fidelity import FRAMES, GRID, LEVELS, SHAPES, score

INK, MUTED, GRIDLINE, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3, validated all-pairs
SMALL = 0.10

LABEL = {
    "levels:10": "10", "levels:9": "9", "levels:6": "6", "levels:3": "3", "levels:2": "2", "levels:1": "1",
    "frames:3": "3 h", "frames:6": "6 h", "frames:12": "12 h", "frames:24": "24 h", "frames:frozen": "frozen",
    "grid:2": "4°", "grid:4": "8°", "grid:8": "16°",
    "offset": "uniform offset", "offset_altitude": "offset, ramps with altitude", "random": "random, same at every altitude",
    "random_altitude": "random, varies with altitude", "persistence": "real: the wind 24 h earlier",
}


def load(path: Path) -> dict:
    """{(arm, climb, horizon, copy, size): score dict with the copy's mean errors}"""
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    rows = [json.loads(line) for f in files for line in f.read_text().splitlines() if line.strip()]
    coast = {r["seed"]: r["covered_fraction"] for r in rows if r["controller"] == "coast"}
    groups: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for r in rows:
        if r["controller"] == "mpc":
            groups[(r["arm"], r["climb_scale"], r["horizon_hours"], r["copy"], r["size"])][r["seed"]] = r
    out = {}
    for key, by_seed in groups.items():
        arm, climb, horizon, copy, size = key
        reference = groups.get((arm, climb, horizon, "truth", 0.0), {})
        seeds = sorted(set(by_seed) & set(reference) & set(coast))
        if copy == "truth" or len(seeds) < 2:
            continue
        s = score(np.array([reference[i]["covered_fraction"] for i in seeds]), np.array([by_seed[i]["covered_fraction"] for i in seeds]),
                  np.array([coast[i] for i in seeds]))
        s["wind"] = float(np.mean([by_seed[i]["wind_error_ms"] for i in seeds]))
        s["steer"] = float(np.mean([by_seed[i]["level_difference_error_ms"] for i in seeds]))
        out[key] = s
    return out


def style(ax, ylabel=True) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRIDLINE)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    if ylabel:
        ax.set_ylabel("control lost", color=MUTED, fontsize=9)
    ax.axhline(SMALL, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))


def bars(ax, xs, s, color) -> None:
    y = np.array([v["score"] for v in s])
    err = np.array([[v["score"] - v["low"] for v in s], [v["high"] - v["score"] for v in s]])
    ax.errorbar(xs, y, yerr=err, fmt="none", ecolor=color, elinewidth=1.5, capsize=0, alpha=0.55)
    ax.plot(xs, y, "-", color=color, linewidth=2)
    ax.plot(xs, y, "o", color=color, markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)


def title(fig, text, sub) -> None:
    fig.text(0.01, 0.98, text, ha="left", va="top", fontsize=13, color=INK, weight="bold")
    fig.text(0.01, 0.915, sub, ha="left", va="top", fontsize=9.5, color=MUTED)


def test_figure(r: dict, out: Path) -> None:
    base = {k[3] + (f"@{k[4]:g}" if k[4] not in (0.0, 1.0) else ""): v for k, v in r.items() if k[0] == "base"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, facecolor=SURFACE)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.14, wspace=0.08)
    title(fig, "Control loss follows the wind error, not the level-difference error",
          "Station keeping, 32 launches. One dot per degraded copy. Rank correlation: wind error +0.90, level differences +0.39.")
    for ax, field, name in ((axes[0], "wind", "wind error (m/s, RMS)"), (axes[1], "steer", "level-difference error (m/s, RMS)")):
        style(ax, ylabel=ax is axes[0])
        for name_, v in base.items():
            coarse = name_.split(":")[0] in ("levels", "frames", "grid")
            ax.plot(v[field], v["score"], "o", color=BLUE if coarse else ORANGE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        o = base["offset"]
        ax.annotate("uniform offset\nloses 0.90", (o[field], o["score"]), xytext=(14, -62) if field == "steer" else (-120, 6),
                    textcoords="offset points", fontsize=8.5, color=INK, va="top", arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
        ax.set_xlabel(name, color=MUTED, fontsize=9)
    axes[0].plot([], [], "o", color=BLUE, label="coarser copy (levels, frames, grid)")
    axes[0].plot([], [], "o", color=ORANGE, label="error shape at 1x or 2x")
    axes[0].legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK)
    fig.savefig(out / "test.png", dpi=200)


def ladders_figure(r: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True, facecolor=SURFACE, gridspec_kw=dict(width_ratios=[6, 5, 3]))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.76, bottom=0.16, wspace=0.12)
    title(fig, "How coarse a wind model can be", "Share of MPC's control lost at each step down; bars are 95% intervals. Dashed: a tenth lost.")
    for ax, (name, ladder, note) in zip(axes, (("altitude levels (truth: 18)", LEVELS, ""), ("frames in time (truth: hourly)", FRAMES, ""),
                                                ("grid (truth: 2°)", GRID, ""))):
        style(ax, ylabel=ax is axes[0])
        s = [r[("base", 1.0, 12.0, c, 0.0)] for c in ladder]
        bars(ax, np.arange(len(ladder)), s, BLUE)
        ax.set_xticks(np.arange(len(ladder)), [LABEL[c] for c in ladder])
        ax.set_xlim(-0.4, len(ladder) - 0.6)
        ax.set_title(name, fontsize=10, color=INK, loc="left")
    fig.savefig(out / "ladders.png", dpi=200)


def shapes_figure(r: dict, out: Path) -> None:
    items = [(LABEL[c], r[("base", 1.0, 12.0, c, 0.0 if c == "persistence" else 1.0)]) for c in SHAPES]
    items += [(LABEL[c] + ", 2x", r[("base", 1.0, 12.0, c, 2.0)]) for c in ("offset", "offset_altitude")]
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=SURFACE)
    fig.subplots_adjust(left=0.31, right=0.97, top=0.78, bottom=0.14)
    title(fig, "Any error the size of a day-old forecast costs most of the control",
          "Each shape scaled to 5.0 m/s RMS (1x), the error of the wind 24 h earlier; two offsets also at 2x.")
    ax.set_facecolor(SURFACE)
    y = np.arange(len(items))[::-1]
    for yi, (name, v) in zip(y, items):
        color = AQUA if name.startswith("real") else ORANGE
        ax.barh(yi, v["score"], height=0.55, color=color)
        ax.plot([v["low"], v["high"]], [yi, yi], color=INK, linewidth=1.2)
        ax.text(1.2, yi, f"{v['score']:.2f}", va="center", ha="right", fontsize=9, color=INK)
    ax.set_yticks(y, [n for n, _ in items], fontsize=9, color=INK)
    ax.set_xlim(0, 1.2)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("control lost", color=MUTED, fontsize=9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRIDLINE)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.savefig(out / "shapes.png", dpi=200)


def interactions_figure(r: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True, facecolor=SURFACE)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.76, bottom=0.16, wspace=0.1)
    title(fig, "The levels loss is information, not reach; coarse frames stop a long horizon from paying",
          "Left: the levels ladder at the base and twice the climb limit. Right: the frames ladder at three planning horizons.")
    ax = axes[0]
    style(ax)
    x = np.arange(len(LEVELS))
    for (arm, climb), color, name in ((("base", 1.0), BLUE, "base climb limit"), (("climb", 2.0), ORANGE, "twice the climb limit")):
        bars(ax, x, [r[(arm, climb, 12.0, c, 0.0)] for c in LEVELS], color)
        ax.plot([], [], "o-", color=color, label=name)
    ax.set_xticks(x, [LABEL[c] for c in LEVELS])
    ax.set_title("altitude levels", fontsize=10, color=INK, loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
    ax = axes[1]
    style(ax, ylabel=False)
    x = np.arange(len(FRAMES))
    for (arm, horizon), color, name in ((("horizon", 6.0), AQUA, "6 h horizon"), (("base", 12.0), BLUE, "12 h (base)"), (("horizon", 24.0), ORANGE, "24 h horizon")):
        bars(ax, x, [r[(arm, 1.0, horizon, c, 0.0)] for c in FRAMES], color)
        ax.plot([], [], "o-", color=color, label=name)
    ax.set_xticks(x, [LABEL[c] for c in FRAMES])
    ax.set_title("frames in time", fontsize=10, color=INK, loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
    fig.savefig(out / "interactions.png", dpi=200)


def main() -> None:
    ask = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ask.add_argument("rows")
    ask.add_argument("--out", default="figures/fidelity")
    a = ask.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    r = load(Path(a.rows))
    test_figure(r, out)
    ladders_figure(r, out)
    shapes_figure(r, out)
    interactions_figure(r, out)
    print("wrote", ", ".join(str(out / n) for n in ("test.png", "ladders.png", "shapes.png", "interactions.png")))


if __name__ == "__main__":
    main()
