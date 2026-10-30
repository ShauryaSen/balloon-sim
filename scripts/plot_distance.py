"""Two figures the per-arm plots cannot draw: arrival rate, and the trend across goal distances

    uv run scripts/plot_distance.py results/rows-2026-10-24.jsonl --into figures/2026-10-24-distance

scripts/plot.py draws one arm at a time and knows four metrics, none of which is whether the balloon ever
arrived. So the distance arm could only be read as three separate directories compared by eye, and the
arrival counts only out of the summary tables. These two put the goal distance on an axis of its own.

    arrival_rate   the share of seeds that ever came within the arrival radius, against forecast error,
                   one panel per goal distance. Bars are the standard error of a proportion.
    by_distance    the objective against the goal distance, a line per controller, at one forecast error.

Both are paired like everything else: within a seed the start, the goal and the true wind are the same for
every controller, so the controllers in one panel are flying the same weather. Goal distances are not paired
with each other, since a different goal is a different task, and each carries its own hindsight.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from balloon_sim.cells import BASE_HORIZON, BASE_MOVES
from balloon_sim.results import read_rows

INK = "#12161b"
ORDER = ("hindsight", "mpc", "mpc_blind", "fixed_plan")
NAMES = {"hindsight": "hindsight (knows the wind)", "mpc": "MPC", "mpc_blind": "MPC not learning", "fixed_plan": "fixed plan"}
STYLE = {"hindsight": (INK, "-", "o"), "mpc": ("#0072b2", "-", "o"),
         "mpc_blind": ("#0072b2", ":", "^"), "fixed_plan": ("#0072b2", "--", "s")}


def cells(rows: list[dict], goal_km: float, controller: str, sigma: float, horizon: int, n_winds: int) -> dict[int, dict]:
    """One cell of the base arm, by seed. Hindsight reads the truth, so sigma and the horizon do not reach it"""
    out = {}
    for r in rows:
        if r.get("goal_km") != goal_km or r.get("belief_scale", 1.0) != 1.0 or r.get("nested"):
            continue
        if r.get("objective", "reach") != "reach" or r.get("task.n_moves", BASE_MOVES) != BASE_MOVES:
            continue
        if r["controller"] != controller or r["n_winds"] != n_winds:
            continue
        if controller != "hindsight":
            if r["belief.sigma"] != sigma:
                continue
            if controller.startswith("mpc") and r["planner.horizon"] != horizon:
                continue
        out[r["seed"]] = r
    return out


def draw_arrival(rows, goals, sigmas, horizon, into: Path) -> None:
    """The share of seeds that ever arrived, against forecast error, one panel per goal distance"""
    figure, panels = plt.subplots(1, len(goals), figsize=(4.0 * len(goals), 4.2), dpi=200, sharey=True)
    panels = np.atleast_1d(panels)
    for panel, goal in zip(panels, goals):
        for controller in ORDER:
            colour, style, marker = STYLE[controller]
            share, error = [], []
            for sigma in sigmas:
                got = cells(rows, goal, controller, sigma, horizon, 0)
                if not got:
                    share.append(np.nan); error.append(np.nan); continue
                hits = np.array([bool(r["arrived"]) for r in got.values()], dtype=float)
                share.append(hits.mean())
                error.append(np.sqrt(hits.mean() * (1 - hits.mean()) / hits.size))  # a proportion's standard error
            panel.errorbar(sigmas, share, yerr=error, color=colour, linestyle=style, marker=marker,
                           markersize=4, capsize=3, linewidth=1.5, label=NAMES[controller])
        seeds = len(cells(rows, goal, "hindsight", 0.0, horizon, 0))
        panel.set_title(f"goal {goal:.0f} km, {seeds} seeds", fontsize=10, color=INK)
        panel.set_xlabel("forecast error $\\sigma$ (m/s)")
        panel.set_ylim(-0.03, 1.03)
        panel.spines[["top", "right"]].set_visible(False)
    panels[0].set_ylabel("share of seeds that ever reached the goal")
    panels[0].legend(frameon=False, fontsize=8)
    figure.suptitle(f"Did it get there at all? MPC horizon {horizon}, planning on the belief's mean",
                    fontsize=11, color=INK, x=0.01, ha="left")
    figure.tight_layout()
    figure.savefig(into)
    plt.close(figure)
    print(f"wrote {into}")


def draw_by_distance(rows, goals, sigma, horizon, into: Path) -> None:
    """The objective against the goal distance, a line per controller, at one forecast error"""
    figure, panel = plt.subplots(figsize=(7.0, 4.6), dpi=200)
    for controller in ORDER:
        colour, style, marker = STYLE[controller]
        means, errors = [], []
        for goal in goals:
            got = cells(rows, goal, controller, sigma, horizon, 0)
            per_move = np.array([r["cost_km"] / r["task.n_moves"] for r in got.values()]) if got else np.array([np.nan])
            means.append(per_move.mean())
            errors.append(per_move.std(ddof=1) / np.sqrt(per_move.size) if per_move.size > 1 else np.nan)
        panel.errorbar(goals, means, yerr=errors, color=colour, linestyle=style, marker=marker,
                       markersize=5, capsize=3, linewidth=1.5, label=NAMES[controller])
    panel.set_xlabel("how far away the goal is (km)")
    panel.set_ylabel("the objective, average distance over the episode (km)")
    panel.set_xticks(goals, [f"{g:.0f}" for g in goals])
    panel.set_title(f"The task gets harder: forecast error {sigma:g} m/s, MPC horizon {horizon}",
                    fontsize=11, loc="left", color=INK)
    panel.spines[["top", "right"]].set_visible(False)
    panel.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(into)
    plt.close(figure)
    print(f"wrote {into}")


def main(argv: list[str] | None = None) -> None:
    ask = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ask.add_argument("rows", nargs="+", help="the .jsonl files to read")
    ask.add_argument("--sigma", type=float, default=3.2, help="the error the distance figure is drawn at")
    ask.add_argument("--horizon", type=int, default=BASE_HORIZON)
    ask.add_argument("--into", default="figures")
    asked = ask.parse_args(argv)

    rows = read_rows(asked.rows)
    goals = sorted({r["goal_km"] for r in rows if "goal_km" in r and r.get("objective", "reach") == "reach"})
    sigmas = sorted({r["belief.sigma"] for r in rows if r["controller"] != "hindsight"})
    if not goals:
        sys.exit("no rows carry a goal distance")
    into = Path(asked.into)
    into.mkdir(parents=True, exist_ok=True)
    print(f"# goals {', '.join(f'{g:g}' for g in goals)} km; sigma ladder {', '.join(f'{s:g}' for s in sigmas)} m/s")
    draw_arrival(rows, goals, sigmas, asked.horizon, into / "arrival_rate.png")
    draw_by_distance(rows, goals, asked.sigma, asked.horizon, into / f"by_distance_sigma{asked.sigma:g}.png")


if __name__ == "__main__":
    main()
