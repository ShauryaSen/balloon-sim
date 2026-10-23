"""Read result rows and draw the figures

    uv run scripts/plot.py results/rows --into figures/      # a directory of per-cell files, or .jsonl files

For each metric and each MPC draw count, the headline: MPC's excess over hindsight against sigma, a line per
horizon (where they converge, a longer horizon stops helping). Then for each metric and each MPC horizon, three
figures:

    lines     the metric against sigma, a line per controller, with the spread across seeds
    dots      the same numbers seed by seed, one dot per seed and no summary line
    score     where MPC sits between the fixed plan and hindsight, paired within each seed

Everything is paired. Within a seed the start, the goal and the true wind are the same for every controller
and every sigma, so a seed that drew a hard flight makes every line worse together. Comparing seed by seed
and then averaging the differences removes that, where averaging each line separately leaves it in. The
dots are drawn on the raw numbers, because the point of them is to show the spread the pairing removes.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from balloon_sim.results import best_hindsight, read_rows

METRICS = {"summed_km": "distance to the goal, summed (km)", "final_km": "distance to the goal at the end (km)"}
INK, COLOURS = "#12161b", ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00")
STYLE = {"hindsight": "-", "mpc": "-", "fixed_plan": "--"}
"""A line's colour says how many winds it planned on, its style which controller it is: hindsight black,
MPC solid, the fixed plan dashed, so the same draw count can be compared across the two"""


def colour(controller: str, n_winds: int) -> str:
    if controller == "hindsight":
        return INK
    return COLOURS[{0: 0, 1: 1, 4: 2, 16: 3, 64: 4}.get(n_winds, 0)]


def label(controller: str, n_winds: int) -> str:
    """What a line is called: hindsight reads the truth, so its draws mean nothing"""
    if controller == "hindsight":
        return "hindsight (best search on the true wind)"
    drawn = "mean" if n_winds == 0 else f"{n_winds} draws"
    return f"{'fixed plan' if controller == 'fixed_plan' else 'MPC'}, {drawn}"


def series(rows: list[dict], horizon: int) -> dict[tuple[str, int], dict[float, dict[int, float]]]:
    """The rows this horizon's figures use, as {(controller, n_winds): {sigma: {seed: metric}}}

    Hindsight and the fixed plan commit to the whole episode, so the MPC horizon does not reach them and
    their rows are shared by every horizon's figures. Hindsight never reads the forecast either, so its one
    run per seed is repeated across the ladder rather than run again at each sigma.
    """
    out: dict = {}
    for r in rows:
        if r["controller"] == "mpc" and r["planner.horizon"] != horizon:
            continue
        out.setdefault((r["controller"], r["n_winds"]), {}).setdefault(r["belief.sigma"], {})[r["seed"]] = r
    return out


def spread(by_seed: dict[int, dict], metric: str) -> tuple[float, float]:
    """The mean of a metric over seeds and the standard error of that mean"""
    values = np.array([r[metric] for r in by_seed.values()], dtype=float)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0


def at_every_sigma(line: dict[float, dict[int, dict]], sigmas: list[float], metric: str) -> tuple[np.ndarray, np.ndarray]:
    """One line's mean and standard error at each sigma, carrying a sigma-independent run across the ladder"""
    only = next(iter(line.values())) if len(line) == 1 else None  # hindsight: one run, repeated along the ladder
    means, errors = [], []
    for sigma in sigmas:
        by_seed = only if only is not None else line.get(sigma, {})
        mean, error = spread(by_seed, metric) if by_seed else (np.nan, np.nan)
        means.append(mean)
        errors.append(error)
    return np.array(means), np.array(errors)


def draw_lines(lines: dict, sigmas: list[float], metric: str, into: Path, title: str) -> None:
    """The metric against sigma, a line per controller"""
    figure, panel = plt.subplots(figsize=(7.0, 4.6), dpi=200)
    for slot, (which, line) in enumerate(sorted(lines.items())):
        means, errors = at_every_sigma(line, sigmas, metric)
        panel.errorbar(sigmas, means, yerr=errors, label=label(*which), color=colour(*which), linestyle=STYLE[which[0]],
                       marker="o", markersize=4, capsize=3, linewidth=1.5)
    panel.set_xlabel("forecast error $\\sigma$ (m/s)")
    panel.set_ylabel(METRICS[metric])
    panel.set_title(title, fontsize=11, loc="left", color=INK)
    panel.spines[["top", "right"]].set_visible(False)
    panel.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(into)
    plt.close(figure)


def draw_dots(lines: dict, sigmas: list[float], metric: str, into: Path, title: str) -> None:
    """The same numbers seed by seed: one column of dots per controller at each sigma, grouped along the axis.

    Only the dots, no smoothed shape or summary line: at 32 seeds the points are the data, and a violin's
    kernel or a median tick adds a shape the seeds do not have. Each column is jittered sideways so points at
    the same height can be told apart; the jitter is cosmetic and drawn from a fixed key, so the figure is the
    same every time it is made.
    """
    which = sorted(lines)
    figure, panel = plt.subplots(figsize=(1.9 * len(sigmas) + 4.0, 4.6), dpi=200)
    width = 0.8 / max(len(which), 1)
    shake = np.random.default_rng(0)
    for slot, key in enumerate(which):
        offset = (slot - (len(which) - 1) / 2) * width
        only = next(iter(lines[key].values())) if len(lines[key]) == 1 else None  # hindsight: one run per seed
        for i, sigma in enumerate(sigmas):
            by_seed = only if only is not None else lines[key].get(sigma, {})
            if not by_seed:
                continue
            values = [r[metric] for r in by_seed.values()]
            jitter = shake.uniform(-width * 0.35, width * 0.35, len(values))
            hollow = key[0] == "fixed_plan"  # the fixed plan's dots are open, as its lines are dashed elsewhere
            panel.scatter(i + offset + jitter, values, s=7.0, linewidths=0.8, zorder=3,
                          facecolors="none" if hollow else colour(*key), edgecolors=colour(*key), alpha=0.8)
        panel.scatter([], [], s=14.0, linewidths=0.8, facecolors="none" if key[0] == "fixed_plan" else colour(*key),
                      edgecolors=colour(*key), label=label(*key))
    panel.set_xticks(range(len(sigmas)), [f"{s:g}" for s in sigmas])
    panel.set_xlabel("forecast error $\\sigma$ (m/s)")
    panel.set_ylabel(METRICS[metric])
    panel.set_title(title, fontsize=11, loc="left", color=INK)
    panel.spines[["top", "right"]].set_visible(False)
    panel.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))  # outside, off the dots
    figure.tight_layout()
    figure.savefig(into)
    plt.close(figure)


def draw_score(lines: dict, sigmas: list[float], metric: str, into: Path, title: str) -> None:
    """Where MPC sits between the fixed plan and hindsight, as a fraction, paired within each seed

        score = (fixed plan - MPC) / (fixed plan - hindsight)

    1 means MPC did as well as a controller that knew the wind, and 0 means it did no better than committing
    to a plan and flying it blind. The score is a ratio, so it is summarised by its median over seeds with the
    quartiles as the bar, and a seed whose fixed plan is within 5% of hindsight has no gap worth dividing by
    and is left out: near sigma = 0 the two agree, and a mean of ratios there swings to +-20.
    """
    hind = lines.get(("hindsight", 0))
    if hind is None:
        return
    hind_by_seed = next(iter(hind.values()))
    figure, panel = plt.subplots(figsize=(7.0, 4.6), dpi=200)
    for slot, (which, line) in enumerate(sorted(lines.items())):
        if which[0] != "mpc":
            continue
        floor = lines.get(("fixed_plan", which[1]))
        if floor is None:
            continue
        medians, low, high = [], [], []
        for sigma in sigmas:
            scores = []
            for seed, mpc_row in line.get(sigma, {}).items():
                if seed not in floor.get(sigma, {}) or seed not in hind_by_seed:
                    continue
                gap = floor[sigma][seed][metric] - hind_by_seed[seed][metric]
                if gap > 0.05 * hind_by_seed[seed][metric]:  # a gap worth dividing by
                    scores.append((floor[sigma][seed][metric] - mpc_row[metric]) / gap)
            values = np.array(scores, dtype=float)
            q = np.percentile(values, [25, 50, 75]) if values.size >= 4 else np.array([np.nan] * 3)
            medians.append(q[1]); low.append(q[1] - q[0]); high.append(q[2] - q[1])
        panel.errorbar(sigmas, medians, yerr=[low, high], label=label(*which), color=colour(*which), linestyle=STYLE[which[0]],
                       marker="o", markersize=4, capsize=3, linewidth=1.5)
    panel.axhline(1.0, color="#999999", linewidth=0.9, linestyle=":", zorder=0)  # as good as knowing the wind
    panel.axhline(0.0, color="#999999", linewidth=0.9, zorder=0)  # no better than a blind plan
    panel.set_xlabel("forecast error $\\sigma$ (m/s)")
    panel.set_ylabel("(fixed plan $-$ MPC) / (fixed plan $-$ hindsight)")
    panel.set_title(title, fontsize=11, loc="left", color=INK)
    panel.spines[["top", "right"]].set_visible(False)
    panel.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(into)
    plt.close(figure)


def draw_horizons(rows: list[dict], sigmas: list[float], horizons: list[int], metric: str, n_winds: int, into: Path) -> None:
    """The headline: MPC's excess over hindsight against sigma, a line per horizon, paired within each seed.

    Where the lines converge, a longer horizon no longer recovers what the noise takes away. The bar is a 95%
    interval on the paired mean (1.96 standard errors over seeds)."""
    hind = {r["seed"]: r[metric] for r in rows if r["controller"] == "hindsight"}
    if not hind:
        return
    figure, panel = plt.subplots(figsize=(7.0, 4.6), dpi=200)
    shades = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(horizons)))
    for shade, horizon in zip(shades, horizons):
        means, half = [], []
        for sigma in sigmas:
            gaps = np.array([r[metric] - hind[r["seed"]] for r in rows if r["controller"] == "mpc" and r["planner.horizon"] == horizon
                             and r["n_winds"] == n_winds and r["belief.sigma"] == sigma and r["seed"] in hind], dtype=float)
            scale = 1.0 / rows[0]["task.n_moves"] if metric == "summed_km" else 1.0  # summed shown as an average per move
            means.append(gaps.mean() * scale if gaps.size else np.nan)
            half.append(1.96 * gaps.std(ddof=1) / np.sqrt(gaps.size) * scale if gaps.size > 1 else np.nan)
        panel.errorbar(sigmas, means, yerr=half, label=f"horizon {horizon} ({horizon * rows[0]['balloon.step_hours']:g} h)",
                       color=shade, marker="o", markersize=4, capsize=3, linewidth=1.8)
    panel.axhline(0.0, color=INK, linewidth=0.9)  # hindsight
    what = "average distance over the episode" if metric == "summed_km" else METRICS[metric].split(" (")[0]
    panel.set_xlabel("forecast error $\\sigma$ (m/s)")
    panel.set_ylabel(f"MPC minus hindsight, {what} (km)")
    drawn = "the belief's mean" if n_winds == 0 else f"{n_winds} belief draws"
    panel.set_title(f"MPC above hindsight, by horizon (MPC on {drawn})", fontsize=11, loc="left", color=INK)
    panel.spines[["top", "right"]].set_visible(False)
    panel.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(into)
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    ask = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ask.add_argument("rows", nargs="+", help="the .jsonl files to read")
    ask.add_argument("--into", default="figures", help="where the figures are written")
    asked = ask.parse_args(argv)

    rows = best_hindsight(read_rows(asked.rows))  # hindsight: the best search found on the true wind
    if not rows:
        sys.exit("no rows in those files")
    into = Path(asked.into)
    into.mkdir(parents=True, exist_ok=True)

    sigmas = sorted({r["belief.sigma"] for r in rows if r["controller"] != "hindsight"})
    horizons = sorted({r["planner.horizon"] for r in rows if r["controller"] == "mpc"}) or [0]
    for horizon in horizons:
        lines = series(rows, horizon)
        for metric in METRICS:
            stem = f"{metric}_h{horizon}"
            named = f"{METRICS[metric].split(' (')[0]}, MPC horizon {horizon}"
            draw_lines(lines, sigmas, metric, into / f"lines_{stem}.png", named)
            draw_dots(lines, sigmas, metric, into / f"dots_{stem}.png", named)
            draw_score(lines, sigmas, metric, into / f"score_{stem}.png", f"MPC between the bounds: {named}")
    for metric in METRICS:
        for n in sorted({r["n_winds"] for r in rows if r["controller"] == "mpc"}):
            draw_horizons(rows, sigmas, horizons, metric, n, into / f"horizons_{metric}_n{n}.png")
    print(f"wrote {len(list(into.glob('*.png')))} figures under {into}")


if __name__ == "__main__":
    main()
