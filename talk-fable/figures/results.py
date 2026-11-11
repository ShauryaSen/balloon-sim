"""The result figures of Part 4, drawn from the sweep's own rows (results/rows-2026-10-24.jsonl).

Everything is on `summed_km`, distance to the goal summed over the 50 moves, divided by 50 so the axis reads
as the average distance from the goal over the flight. That is the metric the tables in
docs/results-2026-10-29.md use for replanning, readings, the horizon and hedging. Differences are paired
within a seed; bars are 95% intervals on the paired mean (1.96 standard errors over 32 seeds).

    uv run python talk-fable/figures/results.py        (from the repo root)
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

from balloon_sim.results import best_hindsight, one_arm, read_rows
from style import ACCENT, FAINT, FIXED, GOAL, HINDSIGHT, INK, MPC, MPC_LIGHT, MUTED, REPO, save, save_layers

sys.path.insert(0, str(REPO / "scripts"))
from plot_scout import scouted  # noqa: E402

MOVES = 50
SIGMAS = [0.0, 0.8, 1.6, 3.2, 6.4, 12.8]
X = np.arange(len(SIGMAS))
REAL = SIGMAS.index(3.2)  # a real day-old forecast's error
rows = best_hindsight(one_arm(read_rows([REPO / "results" / "rows-2026-10-24.jsonl"])))


def cells(controller: str, horizon: int | None = None, n_winds: int = 0) -> dict[tuple[int, float], float]:
    """{(seed, sigma): average distance from the goal in km} for one controller setting"""
    out = {}
    for r in rows:
        if r["controller"] != controller or r["n_winds"] != n_winds:
            continue
        if controller.startswith("mpc") and r["planner.horizon"] != horizon:
            continue
        out[(r["seed"], r["belief.sigma"])] = r["summed_km"] / MOVES
    return out


HIND = {seed: km for (seed, _), km in cells("hindsight").items()}  # one flight per seed: it never sees the forecast


def paired(a: dict, b: dict, sigma: float) -> np.ndarray:
    """a minus b, seed by seed, at one forecast error"""
    return np.array([a[(seed, s)] - b[(seed, s)] for (seed, s) in a if s == sigma and (seed, s) in b])


def above_hindsight(a: dict, sigma: float) -> np.ndarray:
    return np.array([km - HIND[seed] for (seed, s), km in a.items() if s == sigma and seed in HIND])


def mean_ci(values: np.ndarray) -> tuple[float, float]:
    return float(values.mean()), float(1.96 * values.std(ddof=1) / np.sqrt(values.size))


def error_axis(ax, label: bool = True) -> None:
    ax.set_xticks(X, [f"{s:g}" for s in SIGMAS])
    ax.set_xlim(-0.35, len(SIGMAS) - 0.65)
    if label:
        ax.set_xlabel("forecast error (m/s)")
    ax.get_xticklabels()[REAL].set_fontweight(700)


def mark_real(ax, y: float, va: str = "bottom"):
    """Says which rung of the ladder is a real forecast's error"""
    line = ax.axvline(REAL, color=FAINT, linewidth=10, zorder=0)
    text = ax.text(REAL, y, "a real day-old\nforecast", ha="center", va=va, fontsize=10.5, color=MUTED, linespacing=1.15)
    return [line, text]


# --- three controllers against forecast error: where MPC sits, and what replanning is worth -----------------

def draw_bounds() -> None:
    fixed, mpc = cells("fixed_plan"), cells("mpc", 20)
    figure, ax = plt.subplots(figsize=(7.0, 4.5))
    hind_mean = np.mean(list(HIND.values()))
    fixed_mean = np.array([np.mean([km for (_, s), km in fixed.items() if s == sigma]) for sigma in SIGMAS])
    mpc_mean = np.array([np.mean([km for (_, s), km in mpc.items() if s == sigma]) for sigma in SIGMAS])

    real = mark_real(ax, 2180, va="top")
    h = [ax.plot(X, [hind_mean] * len(X), color=HINDSIGHT, marker="o", markersize=7)[0],
         ax.text(X[-1] + 0.12, hind_mean, "hindsight", color=HINDSIGHT, va="center", fontweight=600)]
    f = [ax.plot(X, fixed_mean, color=FIXED, marker="o", markersize=7)[0],
         ax.text(X[-1] + 0.12, fixed_mean[-1], "fixed plan", color=FIXED, va="center", fontweight=600)]
    m = [ax.plot(X, mpc_mean, color=MPC, marker="o", markersize=7)[0],
         ax.text(X[-1] + 0.12, mpc_mean[-1], "MPC", color=MPC, va="center", fontweight=600)]
    gap = [ax.fill_between(X, mpc_mean, fixed_mean, color=MPC, alpha=0.10, linewidth=0),
           ax.annotate("", (REAL, mpc_mean[REAL] + 25), (REAL, fixed_mean[REAL] - 25),
                       arrowprops=dict(arrowstyle="<->", color=INK, linewidth=1.4)),
           ax.text(REAL + 0.12, (mpc_mean[REAL] + fixed_mean[REAL]) / 2, "what replanning\nis worth", va="center", fontsize=11.5,
                   linespacing=1.15)]
    ax.set_ylabel("average distance from the goal (km)")
    ax.set_ylim(800, 2200)
    error_axis(ax)
    ax.set_xlim(-0.35, len(SIGMAS) + 0.6)
    figure.tight_layout()
    save_layers(figure, "bounds", [real, h, f, m, gap])
    plt.close(figure)
    worth = [mean_ci(paired(mpc, fixed, s)) for s in SIGMAS]
    print("  MPC minus fixed plan, km:", ", ".join(f"{m:.0f}±{c:.0f}" for m, c in worth))


# --- what the balloon's own readings are worth ---------------------------------------------------------------

def draw_readings() -> None:
    mpc, blind = cells("mpc", 20), cells("mpc_blind", 20)
    figure, ax = plt.subplots(figsize=(7.0, 4.4))
    rng = np.random.default_rng(0)
    ax.axhline(0, color=INK, linewidth=1.0)
    limit = 100 * np.ceil(max(np.abs(paired(mpc, blind, s)).max() for s in SIGMAS) / 100) + 50
    real = mark_real(ax, 0.94 * limit, va="top")
    dots, means = [], []
    for x, sigma in zip(X, SIGMAS):
        d = paired(mpc, blind, sigma)
        dots.append(ax.scatter(x + rng.uniform(-0.16, 0.16, d.size), d, s=42, color=MPC, alpha=0.35, linewidth=0))
        mean, half = mean_ci(d)
        means.append(ax.errorbar(x, mean, yerr=half, color=INK, marker="D", markersize=8, capsize=6, linewidth=2.0, zorder=5))
    ax.text(-0.3, 0.7 * limit, "readings hurt", color=MUTED, fontsize=11.5, va="center")
    ax.text(-0.3, -0.7 * limit, "readings helped", color=MUTED, fontsize=11.5, va="center")
    ax.set_ylabel("MPC with readings − without\naverage distance from the goal (km)")
    ax.set_ylim(-limit, limit)
    error_axis(ax)
    figure.tight_layout()
    save_layers(figure, "readings", [real, dots, [part for e in means for part in (e.lines[0], *e.lines[1], *e.lines[2])]])
    plt.close(figure)


# --- the breaking point: where planning further ahead stops paying -------------------------------------------

def draw_breaking() -> None:
    horizons = [5, 10, 20, 50]
    shades = ["#b9cdf5", "#7fa5ec", MPC, "#143a8a"]
    figure, ax = plt.subplots(figsize=(7.0, 4.5))
    real = mark_real(ax, 690, va="top")
    layers = [real]
    for slot, (horizon, shade) in enumerate(zip(horizons, shades)):
        stats = [mean_ci(above_hindsight(cells("mpc", horizon), s)) for s in SIGMAS]
        means, half = np.array([m for m, _ in stats]), np.array([c for _, c in stats])
        bar = ax.errorbar(X, means, yerr=half, color=shade, marker="o", markersize=7, capsize=0, elinewidth=1.2, alpha=1.0)
        label = ax.text(0.03, 0.80 - 0.085 * slot, f"{horizon * 3} hours", color=shade, fontweight=600, transform=ax.transAxes)
        layers.append([bar.lines[0], *bar.lines[2], label])
        print(f"  horizon {horizon}: " + ", ".join(f"{m:.0f}" for m in means))
    title = ax.text(0.03, 0.89, "MPC plans ahead", color=MUTED, fontsize=11.5, transform=ax.transAxes)
    layers[1].append(title)
    ax.set_ylabel("MPC above hindsight\naverage distance from the goal (km)")
    ax.set_ylim(0, 700)
    error_axis(ax)
    figure.tight_layout()
    save_layers(figure, "breaking", layers)
    plt.close(figure)


# --- hedging: planning over many possible winds ---------------------------------------------------------------

def draw_hedging() -> None:
    figure, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.axhline(0, color=INK, linewidth=1.0)
    real = mark_real(ax, 235, va="top")
    layers = [real]
    for offset, name, colour, many, few in (
        (-0.09, "fixed plan, 64 winds", FIXED, cells("fixed_plan", n_winds=64), cells("fixed_plan")),
        (+0.09, "MPC, 16 winds", MPC, cells("mpc", 20, 16), cells("mpc", 20)),
    ):
        stats = [mean_ci(paired(many, few, s)) for s in SIGMAS]
        means, half = np.array([m for m, _ in stats]), np.array([c for _, c in stats])
        bar = ax.errorbar(X + offset, means, yerr=half, color=colour, marker="o", markersize=8, capsize=5, linestyle="none", elinewidth=2.0)
        label = ax.text(0.02 if colour == FIXED else 0.02, 0.13 if colour == FIXED else 0.04, name, color=colour, fontweight=600,
                        transform=ax.transAxes)
        layers.append([bar.lines[0], *bar.lines[1], *bar.lines[2], label])
        print(f"  {name}: " + ", ".join(f"{m:.0f}±{c:.0f}" for m, c in stats))
    ax.text(len(SIGMAS) - 0.7, -330, "many winds helped", color=MUTED, fontsize=11.5, va="center", ha="right")
    ax.set_ylabel("many winds − one forecast\naverage distance from the goal (km)")
    ax.set_ylim(-420, 250)
    error_axis(ax)
    figure.tight_layout()
    save_layers(figure, "hedging", layers)
    plt.close(figure)


# --- which tasks are even possible ----------------------------------------------------------------------------

def draw_scouting() -> None:
    found = scouted(REPO / "docs" / "scouting-rule.md")
    distances = sorted({r["goal"] for r in found})
    colours = {"downwind": MPC, "random": MUTED, "crosswind": "#b7bcc6", "upwind": "#d3d6dc"}
    figure, (reach, room) = plt.subplots(1, 2, figsize=(9.6, 4.1))
    reach_lines, room_lines, picked = [], [], []
    for direction, colour in colours.items():
        mine = sorted((r for r in found if r["direction"] == direction), key=lambda r: r["goal"])
        x = [distances.index(r["goal"]) for r in mine]
        weight = 600 if direction == "downwind" else 400
        reach_lines += [reach.plot(x, [r["arrived"] for r in mine], color=colour, marker="o", markersize=7)[0],
                        reach.text(x[-1] + 0.15, mine[-1]["arrived"], direction, color=colour if direction == "downwind" else MUTED,
                                   va="center", fontsize=11.5, fontweight=weight)]
        room_lines += [room.plot(x, [r["percent"] for r in mine], color=colour, marker="o", markersize=7)[0]]
        if direction == "downwind":
            chosen = next(r for r in mine if r["goal"] == 2000)
            at = distances.index(2000)
            picked += [reach.plot(at, chosen["arrived"], "o", markersize=17, markerfacecolor="none", markeredgecolor=GOAL, markeredgewidth=2.2)[0],
                       room.plot(at, chosen["percent"], "o", markersize=17, markerfacecolor="none", markeredgecolor=GOAL, markeredgewidth=2.2)[0]]
    for ax, floor, said in ((reach, 24, "too hard below"), (room, 25, "too easy below")):
        ax.axhspan(0, floor, color=FAINT, alpha=0.6, linewidth=0, zorder=0)
        ax.text(*((-0.15, 1.2, said) if ax is reach else (len(distances) - 0.2, 1.8, said)), color=MUTED, fontsize=10.5,
                va="bottom", ha="left" if ax is reach else "right")
        ax.set_xticks(range(len(distances)), [f"{d:,}" for d in distances])
        ax.set_xlabel("distance to the goal (km)")
        ax.set_xlim(-0.3, len(distances) - 0.1)
    heading = dict(fontsize=13.5, fontweight=600, va="bottom")  # plain text, not set_title: a title will not reappear once hidden
    reach.text(0.0, 1.05, "Can the best possible controller get there?", transform=reach.transAxes, **heading)
    room.text(0.0, 1.05, "Does steering matter?", transform=room.transAxes, **heading)
    reach.set_ylabel("launches where hindsight arrives (of 32)")
    reach.set_ylim(0, 33)
    reach.set_xlim(-0.3, len(distances) + 0.5)
    room.set_ylabel("a blind plan's cost above hindsight (%)")
    room.set_ylim(0, 80)
    figure.tight_layout(w_pad=3.0, rect=(0, 0, 1, 0.94))
    room_parts = [room.xaxis, room.yaxis, room.spines["left"], room.spines["bottom"], *room.texts, *room.patches, *room_lines]
    save_layers(figure, "scouting", [reach_lines, room_parts, picked])
    plt.close(figure)


if __name__ == "__main__":
    draw_bounds()
    draw_readings()
    draw_breaking()
    draw_hedging()
    draw_scouting()
