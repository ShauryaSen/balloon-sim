"""Read result rows and draw the figures

    uv run scripts/plot.py results/rows.jsonl --into figures/

For each metric (summed distance and final distance) and each MPC horizon, three figures:

    lines     the metric against sigma, a line per controller, with the spread across seeds
    violins   the same numbers as distributions, since the spread is what the lines hide
    score     where MPC sits between the fixed plan and hindsight, paired within each seed

Everything is paired. Within a seed the start, the goal and the true wind are the same for every controller
and every sigma, so a seed that drew a hard flight makes every line worse together. Comparing seed by seed
and then averaging the differences removes that, where averaging each line separately leaves it in. The
violins are drawn on the raw numbers, because the point of them is to show the spread the pairing removes.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from balloon_sim.results import read_rows

METRICS = {"summed_km": "distance to the goal, summed (km)", "final_km": "distance to the goal at the end (km)"}
INK, COLOURS = "#12161b", ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00")


def label(controller: str, n_winds: int) -> str:
    """What a line is called: hindsight reads the truth, so its draws mean nothing"""
    if controller == "hindsight":
        return "hindsight (knows the wind)"
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
        panel.errorbar(sigmas, means, yerr=errors, label=label(*which), color=COLOURS[slot % len(COLOURS)],
                       marker="o", markersize=4, capsize=3, linewidth=1.5)
    panel.set_xlabel("forecast error $\\sigma$ (m/s)")
    panel.set_ylabel(METRICS[metric])
    panel.set_title(title, fontsize=11, loc="left", color=INK)
    panel.spines[["top", "right"]].set_visible(False)
    panel.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(into)
    plt.close(figure)


def draw_violins(lines: dict, sigmas: list[float], metric: str, into: Path, title: str) -> None:
    """The same numbers as distributions: one violin per controller at each sigma, grouped along the axis"""
    which = sorted(lines)
    figure, panel = plt.subplots(figsize=(1.9 * len(sigmas) + 2.0, 4.6), dpi=200)
    width = 0.8 / max(len(which), 1)
    for slot, key in enumerate(which):
        offset = (slot - (len(which) - 1) / 2) * width
        at, data = [], []
        only = next(iter(lines[key].values())) if len(lines[key]) == 1 else None
        for i, sigma in enumerate(sigmas):
            by_seed = only if only is not None else lines[key].get(sigma, {})
            if by_seed:
                at.append(i + offset)
                data.append([r[metric] for r in by_seed.values()])
        if not data:
            continue
        drawn = panel.violinplot(data, positions=at, widths=width * 0.9, showmedians=True, showextrema=False)
        for body in drawn["bodies"]:
            body.set_facecolor(COLOURS[slot % len(COLOURS)])
            body.set_alpha(0.55)
        drawn["cmedians"].set_color(INK)
        panel.plot([], [], color=COLOURS[slot % len(COLOURS)], linewidth=6, alpha=0.55, label=label(*key))
    panel.set_xticks(range(len(sigmas)), [f"{s:g}" for s in sigmas])
    panel.set_xlabel("forecast error $\\sigma$ (m/s)")
    panel.set_ylabel(METRICS[metric])
    panel.set_title(title, fontsize=11, loc="left", color=INK)
    panel.spines[["top", "right"]].set_visible(False)
    panel.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(into)
    plt.close(figure)


def draw_score(lines: dict, sigmas: list[float], metric: str, into: Path, title: str) -> None:
    """Where MPC sits between the fixed plan and hindsight, as a fraction, paired within each seed

        score = (fixed plan - MPC) / (fixed plan - hindsight)

    1 means MPC did as well as a controller that knew the wind, and 0 means it did no better than committing
    to a plan and flying it blind. A seed whose fixed plan and hindsight agree has no gap to divide by and is
    left out of the average rather than allowed to explode it.
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
        means, errors = [], []
        for sigma in sigmas:
            scores = []
            for seed, mpc_row in line.get(sigma, {}).items():
                if seed not in floor.get(sigma, {}) or seed not in hind_by_seed:
                    continue
                gap = floor[sigma][seed][metric] - hind_by_seed[seed][metric]
                if abs(gap) > 1e-9:  # nothing to capture, so the fraction is undefined rather than huge
                    scores.append((floor[sigma][seed][metric] - mpc_row[metric]) / gap)
            values = np.array(scores, dtype=float)
            means.append(values.mean() if values.size else np.nan)
            errors.append(values.std(ddof=1) / np.sqrt(values.size) if values.size > 1 else np.nan)
        panel.errorbar(sigmas, means, yerr=errors, label=label(*which), color=COLOURS[slot % len(COLOURS)],
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


def main(argv: list[str] | None = None) -> None:
    ask = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ask.add_argument("rows", nargs="+", help="the .jsonl files to read")
    ask.add_argument("--into", default="figures", help="where the figures are written")
    asked = ask.parse_args(argv)

    rows = read_rows(asked.rows)
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
            draw_violins(lines, sigmas, metric, into / f"violins_{stem}.png", named)
            draw_score(lines, sigmas, metric, into / f"score_{stem}.png", f"Where MPC sits between the bounds: {named}")
    print(f"wrote {len(list(into.glob('*.png')))} figures under {into}")


if __name__ == "__main__":
    main()
