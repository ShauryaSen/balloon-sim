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

from balloon_sim.cells import BASE_GOAL_KM, BASE_MOVES
from balloon_sim.results import best_hindsight, one_arm, read_rows

METRICS = {
    "cost_km": "the objective: distance summed until arrival (km)",
    "summed_km": "distance to the goal, summed over every move (km)",
    "final_km": "distance to the goal at the end (km)",
    "closest_km": "the nearest the balloon came to the goal (km)",
}
"""What gets a figure. cost_km is the objective under the arrival rule and summed_km the plain sum, so on a
sweep run with arrival_km = 0 the two are the same and their figures agree. closest_km is how near the
balloon ever came, which is the honest reachability measure once the goal is far: a balloon still flying
toward a distant goal at the last move can end nearer than one that reached a near goal and was blown past.

cost_km and summed_km are sums over the moves, and were once divided by the episode's length and drawn as
an average distance per move. That is a constant rescale, so no comparison changed, but under the arrival
rule the episode does not last the whole budget: a flight that arrived on move 17 had its cost spread over
50 moves it never flew, and calling that an average distance was wrong for exactly the flights that went
well. They are drawn as what they are, the objective in km, and the axis is large because the objective
is large."""

SHORT = {
    "cost_km": "the objective",
    "summed_km": "summed distance to the goal",
    "final_km": "distance at the end",
    "closest_km": "closest approach",
}
"""The axis label says what a metric is; a title only has to say which one it is, and the long form was
running off the right edge of the score figures"""

INK, COLOURS = "#12161b", ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00")
STYLE = {"hindsight": "-", "mpc": "-", "mpc_blind": ":", "fixed_plan": "--"}
MARKER = {"hindsight": "o", "mpc": "o", "mpc_blind": "^", "fixed_plan": "o"}
"""A line's style tells the controllers apart where they share a colour, and there are no lines on the dots
figure to carry it, so the shape does the same job there: MPC a circle, MPC that does not learn a triangle"""
"""A line per controller: solid replans and learns, dotted replans without learning, dashed commits once.
Anything a later sweep adds falls back to solid rather than stopping the figure"""
"""A line's colour says how many winds it planned on, its style which controller it is: hindsight black,
MPC solid, the fixed plan dashed, so the same draw count can be compared across the two"""


def colour(controller: str, n_winds: int) -> str:
    if controller == "hindsight":
        return INK
    return COLOURS[{0: 0, 1: 1, 4: 2, 16: 3, 64: 4}.get(n_winds, 0)]


NAMES = {"fixed_plan": "fixed plan", "mpc": "MPC", "mpc_blind": "MPC not learning"}


def label(controller: str, n_winds: int) -> str:
    """What a line is called: hindsight reads the truth, so its draws mean nothing"""
    if controller == "hindsight":
        return "hindsight (best search on the true wind)"
    drawn = "mean" if n_winds == 0 else f"{n_winds} draws"
    return f"{NAMES.get(controller, controller)}, {drawn}"


def plainly(lines: dict) -> dict:
    """Just the controllers planning on the belief's mean.

    The full figure draws every draw count, which is eleven lines and unreadable anywhere but on paper. The
    comparison the results actually rest on is four: the ceiling, the floor, MPC, and MPC not learning. The
    cluttered version is still written beside this one, since it is the record of what was run.
    """
    return {which: line for which, line in lines.items() if which[1] == 0}


def series(rows: list[dict], horizon: int) -> dict[tuple[str, int], dict[float, dict[int, float]]]:
    """The rows this horizon's figures use, as {(controller, n_winds): {sigma: {seed: metric}}}

    Hindsight and the fixed plan commit to the whole episode, so the MPC horizon does not reach them and
    their rows are shared by every horizon's figures. Hindsight never reads the forecast either, so its one
    run per seed is repeated across the ladder rather than run again at each sigma.

    Every controller that replans is held to the horizon, mpc_blind included. Keeping only "mpc" to it left
    mpc_blind's four horizons averaged into one line, so the figure's not-learning line was a blend that a
    short horizon dragged upward, and it read as a gap that the paired comparison does not support.
    """
    out: dict = {}
    for r in rows:
        if r["controller"].startswith("mpc") and r["planner.horizon"] != horizon:
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
        panel.errorbar(sigmas, means, yerr=errors, label=label(*which), color=colour(*which), linestyle=STYLE.get(which[0], "-"),
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
    # wide enough for the columns it actually has: eleven series per sigma need the room, four do not
    wide = 4.0 + 0.9 * len(sigmas) * (0.3 + 0.12 * max(len(which), 1))
    figure, panel = plt.subplots(figsize=(wide, 4.6), dpi=200)
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
            panel.scatter(i + offset + jitter, values, s=7.0, linewidths=0.8, zorder=3, marker=MARKER.get(key[0], "o"),
                          facecolors="none" if hollow else colour(*key), edgecolors=colour(*key), alpha=0.8)
        panel.scatter([], [], s=14.0, linewidths=0.8, marker=MARKER.get(key[0], "o"),
                      facecolors="none" if key[0] == "fixed_plan" else colour(*key),
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
    drawn_medians: list[float] = []  # what the axis is scaled to
    drawn_whiskers: list[float] = []  # and what may run past it
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
            drawn_medians.append(q[1]); drawn_whiskers.extend((q[0], q[2]))
        panel.errorbar(sigmas, medians, yerr=[low, high], label=label(*which), color=colour(*which), linestyle=STYLE.get(which[0], "-"),
                       marker="o", markersize=4, capsize=3, linewidth=1.5)
    panel.axhline(1.0, color="#999999", linewidth=0.9, linestyle=":", zorder=0)  # as good as knowing the wind
    panel.axhline(0.0, color="#999999", linewidth=0.9, zorder=0)  # no better than a blind plan
    # A score is a ratio and near sigma = 0 the gap it divides by is nearly zero, so one seed's quartile
    # reaches -8 and squashes every line onto one another. The axis is held to where the medians live,
    # padded, and never tighter than the 0 and 1 lines, which are the two heights that mean anything. A
    # whisker running past the edge is said in the corner rather than left to flatten the figure.
    shown = np.array([m for m in drawn_medians if np.isfinite(m)], dtype=float)
    if shown.size:
        pad = 0.15 * max(float(shown.max() - shown.min()), 0.5)
        bottom, top = min(float(shown.min()) - pad, -0.1), max(float(shown.max()) + pad, 1.1)
        panel.set_ylim(bottom, top)
        if any(np.isfinite(w) and (w < bottom or w > top) for w in drawn_whiskers):
            panel.text(0.99, 0.02, "a quartile runs past the axis", transform=panel.transAxes,
                       ha="right", va="bottom", fontsize=7, color="#999999")
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
            means.append(gaps.mean() if gaps.size else np.nan)
            half.append(1.96 * gaps.std(ddof=1) / np.sqrt(gaps.size) if gaps.size > 1 else np.nan)
        panel.errorbar(sigmas, means, yerr=half, label=f"horizon {horizon} ({horizon * rows[0]['balloon.step_hours']:g} h)",
                       color=shade, marker="o", markersize=4, capsize=3, linewidth=1.8)
    panel.axhline(0.0, color=INK, linewidth=0.9)  # hindsight
    what = SHORT.get(metric, metric)
    panel.set_xlabel("forecast error $\\sigma$ (m/s)")
    panel.set_ylabel(f"MPC minus hindsight, {what} (km)")
    drawn = "the belief's mean" if n_winds == 0 else f"{n_winds} belief draws"
    panel.set_title(f"MPC above hindsight, by horizon (MPC on {drawn})", fontsize=11, loc="left", color=INK)
    panel.spines[["top", "right"]].set_visible(False)
    panel.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(into)
    plt.close(figure)


def draw_replanning(rows: list[dict], sigmas: list[float], horizons: list[int], metric: str, n_winds: int, into: Path) -> None:
    """What replanning is worth: MPC minus the fixed plan, paired within each seed, a line per horizon.

    Both controllers plan on the same belief, so the only difference between them is that MPC re-solves from
    where the balloon really is. Below zero MPC is nearer the goal. The comparison is paired because
    subtracting two separately averaged lines would carry the seed-to-seed spread of the task into a
    difference that does not contain it.

    No interval is drawn: with four horizons the bars overlapped enough to hide the lines, and this figure is
    for the shape, where each horizon crosses zero and how the gap grows. The intervals behind it are in
    `docs/results-2026-10-22.md` and in `horizons_*.png`, and they are what any claim should be read off.
    """
    floor = {(r["seed"], r["belief.sigma"]): r[metric] for r in rows if r["controller"] == "fixed_plan" and r["n_winds"] == n_winds}
    if not floor:
        return
    figure, panel = plt.subplots(figsize=(7.0, 4.6), dpi=200)
    shades = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(horizons)))
    for shade, horizon in zip(shades, horizons):
        means = []
        for sigma in sigmas:
            gaps = np.array([r[metric] - floor[(r["seed"], sigma)] for r in rows
                             if r["controller"] == "mpc" and r["planner.horizon"] == horizon
                             and r["n_winds"] == n_winds and r["belief.sigma"] == sigma
                             and (r["seed"], sigma) in floor], dtype=float)
            means.append(gaps.mean() if gaps.size else np.nan)
        panel.plot(sigmas, means, label=f"horizon {horizon} ({horizon * rows[0]['balloon.step_hours']:g} h)",
                   color=shade, marker="o", markersize=4, linewidth=1.8)
    panel.axhline(0.0, color=INK, linewidth=0.9)  # the fixed plan: above it, replanning cost more than it gave
    what = SHORT.get(metric, metric)
    panel.set_xlabel("forecast error $\\sigma$ (m/s)")
    panel.set_ylabel(f"MPC $-$ fixed plan, {what} (km)")
    drawn = "the belief's mean" if n_winds == 0 else f"{n_winds} belief draws"
    panel.set_title(f"What replanning is worth (both planning on {drawn})", fontsize=11, loc="left", color=INK)
    panel.spines[["top", "right"]].set_visible(False)
    panel.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(into)
    plt.close(figure)


def draw_surface(rows: list[dict], sigmas: list[float], horizons: list[int], metric: str, n_winds: int, into: Path) -> None:
    """The same numbers as `horizons_*`, as one surface over forecast error and horizon.

    Both axes are drawn evenly spaced rather than to scale: sigma doubles and the horizon roughly doubles, so
    on a true scale the low end, where the horizons separate, would be squashed into a corner. Ticks carry the
    real values. Height is MPC's excess over hindsight, paired within each seed, so the floor is the ceiling
    controller and lower is better. The grid is 4 horizons by 6 noise levels, which is coarse for a surface:
    the points drawn on it are the measurements, everything between them is interpolation.
    """
    hind = {r["seed"]: r[metric] for r in rows if r["controller"] == "hindsight"}
    if not hind:
        return
    height = np.full((len(horizons), len(sigmas)), np.nan)
    for i, horizon in enumerate(horizons):
        for j, sigma in enumerate(sigmas):
            gaps = [r[metric] - hind[r["seed"]] for r in rows if r["controller"] == "mpc" and r["planner.horizon"] == horizon
                    and r["n_winds"] == n_winds and r["belief.sigma"] == sigma and r["seed"] in hind]
            if gaps:
                height[i, j] = float(np.mean(gaps))

    figure = plt.figure(figsize=(8.0, 6.0), dpi=200)
    panel = figure.add_subplot(projection="3d")
    x, y = np.meshgrid(np.arange(len(sigmas)), np.arange(len(horizons)))
    panel.plot_surface(x, y, height, cmap="viridis", edgecolor=INK, linewidth=0.35, alpha=0.9, rstride=1, cstride=1)
    panel.scatter(x, y, height, color=INK, s=8, depthshade=False)
    panel.set_xticks(range(len(sigmas)), [f"{s:g}" for s in sigmas])
    panel.set_yticks(range(len(horizons)), [f"{h} ({h * rows[0]['balloon.step_hours']:g} h)" for h in horizons])
    panel.set_xlabel("forecast error $\\sigma$ (m/s)", labelpad=8)
    panel.set_ylabel("MPC horizon", labelpad=14)
    what = SHORT.get(metric, metric)
    panel.set_zlabel(f"MPC $-$ hindsight, {what} (km)", labelpad=10)
    drawn = "the belief's mean" if n_winds == 0 else f"{n_winds} belief draws"
    panel.set_title(f"MPC above hindsight over error and horizon (MPC on {drawn})", fontsize=11, color=INK, y=0.97)
    panel.view_init(elev=24, azim=-128)  # looking in along both axes, so the fan at low sigma is face on
    panel.tick_params(labelsize=8)
    panel.set_box_aspect((1.35, 1.0, 0.78), zoom=1.08)
    figure.subplots_adjust(left=0.02, right=0.97, bottom=0.04, top=0.99)
    figure.savefig(into)
    plt.close(figure)


def arm(ask: argparse.ArgumentParser) -> None:
    """Which arm of the sweep to read. The arms overlap in every other setting, so one must be chosen"""
    ask.add_argument("--goal-km", type=float, default=None, help=f"goal distance (default {BASE_GOAL_KM:g}, the main arm)")
    ask.add_argument("--belief-scale", type=float, default=1.0, help="what the belief assumed the error was, as a multiple")
    ask.add_argument("--nested", action="store_true", help="read the nested-draw arm instead of the fresh-draw one")
    ask.add_argument("--n-moves", type=int, default=None, help=f"episode length (default {BASE_MOVES:g}; the length arm is longer)")
    ask.add_argument("--objective", choices=("reach", "escape"), default="reach", help="which objective the arm flew")


def main(argv: list[str] | None = None) -> None:
    ask = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ask.add_argument("rows", nargs="+", help="the .jsonl files to read")
    arm(ask)
    ask.add_argument("--into", default="figures", help="where the figures are written")
    asked = ask.parse_args(argv)

    if asked.objective == "escape":  # cost_km is then minus the distance from the start, summed
        METRICS["cost_km"] = "the objective: minus distance from the start, summed (km)"
    chosen = one_arm(read_rows(asked.rows), asked.goal_km, asked.belief_scale, asked.nested, asked.n_moves, asked.objective)
    if not chosen:
        sys.exit("no rows in that arm; --goal-km, --n-moves, --objective, --belief-scale and --nested pick one")
    print(f"# arm: {asked.objective}, goal {asked.goal_km or BASE_GOAL_KM:g} km, "
          f"{asked.n_moves or BASE_MOVES:g} moves, belief scale {asked.belief_scale:g}, "
          f"draws {'nested' if asked.nested else 'fresh'}")
    rows = best_hindsight(chosen)  # hindsight: the best search found on the true wind
    drawn_metrics = {m: name for m, name in METRICS.items() if any(m in r for r in rows)}  # older rows lack some
    if not rows:
        sys.exit("no rows in those files")
    into = Path(asked.into)
    into.mkdir(parents=True, exist_ok=True)

    sigmas = sorted({r["belief.sigma"] for r in rows if r["controller"] != "hindsight"})
    horizons = sorted({r["planner.horizon"] for r in rows if r["controller"] == "mpc"}) or [0]
    for horizon in horizons:
        lines = series(rows, horizon)
        for metric in drawn_metrics:
            stem = f"{metric}_h{horizon}"
            named = f"{SHORT.get(metric, metric)}, MPC horizon {horizon}"
            # every draw count, which is the record, and then the four lines the results rest on, which is
            # what goes on a slide: the ceiling, the floor, MPC, and MPC not learning
            for prefix, chosen in (("", lines), ("plain_", plainly(lines))):
                draw_lines(chosen, sigmas, metric, into / f"{prefix}lines_{stem}.png", named)
                draw_dots(chosen, sigmas, metric, into / f"{prefix}dots_{stem}.png", named)
                draw_score(chosen, sigmas, metric, into / f"{prefix}score_{stem}.png", f"MPC between the bounds: {named}")
    for metric in drawn_metrics:
        for n in sorted({r["n_winds"] for r in rows if r["controller"] == "mpc"}):
            draw_horizons(rows, sigmas, horizons, metric, n, into / f"horizons_{metric}_n{n}.png")
            draw_surface(rows, sigmas, horizons, metric, n, into / f"surface_{metric}_n{n}.png")
            draw_replanning(rows, sigmas, horizons, metric, n, into / f"replanning_{metric}_n{n}.png")
    print(f"wrote {len(list(into.glob('*.png')))} figures under {into}")


if __name__ == "__main__":
    main()
