"""Print the sweep's tables and the paired comparisons the write-up rests on

    uv run scripts/summarise.py results/rows

Every comparison is paired within a seed (the same truth, start, goal and error pattern), so a difference is the
controller or the noise, never the weather. A comparison prints the mean difference over seeds, its 95% interval
(1.96 standard errors), and how many seeds went each way. "No detectable difference" means the interval
contains zero; it is not evidence of no difference.
"""

import argparse
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from balloon_sim.cells import BASE_GOAL_KM
from balloon_sim.results import best_hindsight, one_arm, read_rows


def keyed(rows: list[dict]) -> dict:
    """{(controller, horizon or None, n_winds, sigma): {seed: row}}

    Hindsight and the fixed plan plan the whole episode at once, so a horizon does not describe them and they
    carry None. Every controller that replans carries its own, mpc_blind included: without that its four
    horizons land on one key and overwrite each other seed by seed.
    """
    out: dict = {}
    for r in rows:
        horizon = r["planner.horizon"] if r["controller"].startswith("mpc") else None
        sigma = 0.0 if r["controller"] == "hindsight" else round(r["belief.sigma"], 4)
        out.setdefault((r["controller"], horizon, r["n_winds"], sigma), {})[r["seed"]] = r
    return out


def per_move(row: dict, metric: str) -> float:
    """Summed distance as an average per move, so it reads as km; other metrics as they are"""
    return row[metric] / row["task.n_moves"] if metric in ("summed_km", "cost_km") else row[metric]


def compare(a: dict, b: dict, metric: str) -> str:
    """a minus b over the seeds both have: mean [95% interval], and seeds where a was lower / higher"""
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return "too few seeds"
    d = np.array([per_move(a[s], metric) - per_move(b[s], metric) for s in seeds])
    half = 1.96 * d.std(ddof=1) / np.sqrt(d.size)
    verdict = "lower" if d.mean() + half < 0 else "higher" if d.mean() - half > 0 else "no detectable difference"
    return f"{d.mean():+7.0f} km [{d.mean() - half:+.0f}, {d.mean() + half:+.0f}]  a lower on {(d < 0).sum()}/{d.size}  -> {verdict}"


def arm(ask: argparse.ArgumentParser) -> None:
    """Which arm of the sweep to read. The arms overlap in every other setting, so one must be chosen"""
    ask.add_argument("--goal-km", type=float, default=None, help=f"goal distance (default {BASE_GOAL_KM:g}, the main arm)")
    ask.add_argument("--belief-scale", type=float, default=1.0, help="what the belief assumed the error was, as a multiple")
    ask.add_argument("--nested", action="store_true", help="read the nested-draw arm instead of the fresh-draw one")


def main(argv: list[str] | None = None) -> None:
    ask = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ask.add_argument("rows", nargs="+")
    arm(ask)
    asked = ask.parse_args(argv)
    chosen = one_arm(read_rows(asked.rows), asked.goal_km, asked.belief_scale, asked.nested)
    if not chosen:
        sys.exit("no rows in that arm; --goal-km, --belief-scale and --nested pick one")
    print(f"# arm: goal {asked.goal_km or BASE_GOAL_KM:g} km, belief scale {asked.belief_scale:g}, "
          f"draws {'nested' if asked.nested else 'fresh'}")
    rows = best_hindsight(chosen)  # hindsight: the best search found on the true wind
    by = keyed(rows)
    hind = by[("hindsight", None, 0, 0.0)]
    sigmas = sorted({k[3] for k in by if k[0] != "hindsight"})
    horizons = sorted({k[1] for k in by if k[0] == "mpc"})
    mpc_draws = sorted({k[2] for k in by if k[0] == "mpc"})
    fixed_draws = sorted({k[2] for k in by if k[0] == "fixed_plan"})
    blind_horizons = sorted({k[1] for k in by if k[0] == "mpc_blind"})

    print(f"{len(rows)} rows, {len(hind)} seeds\n")
    has_closest = all("closest_km" in r for r in hind.values())  # rows written before 2026-10-22 lack it
    present = [m for m in ("closest_km", "final_km", "summed_km", "cost_km") if all(m in r for r in hind.values())]
    for metric in present:
        values = [per_move(r, metric) for r in hind.values()]
        print(f"hindsight {metric:11s}: median {st.median(values):6.0f} km, quartiles {np.percentile(values, 25):.0f} to {np.percentile(values, 75):.0f}")
    if has_closest:
        closest = [r["closest_km"] for r in hind.values()]
        print(f"hindsight within 200 km of the goal at some point: {sum(c < 200 for c in closest)}/{len(closest)}; "
              f"median move of closest approach {st.median(r['closest_move'] for r in hind.values()):.0f}")
    if all("arrived" in r for r in hind.values()):  # the arrival rule: how often, and how soon
        reached = [r for r in hind.values() if r["arrived"]]
        print(f"hindsight arrived (within task.arrival_km, ending the episode): {len(reached)}/{len(hind)}"
              + (f"; median move {st.median(r['arrived_move'] for r in reached):.0f}" if reached else ""))
    print()

    for metric in [m for m in ("cost_km", "summed_km", "final_km") if m in present]:
        what = {"cost_km": "the objective (distance summed until arrival)",
                "summed_km": "average distance over the episode",
                "final_km": "final distance"}[metric]
        print(f"## {what}, km above hindsight, paired mean over seeds\n")
        print("| controller | " + " | ".join(f"σ={s:g}" for s in sigmas) + " |")
        print("|---|" + "---|" * len(sigmas))
        lines = [(f"fixed plan, {n} draws", ("fixed_plan", None, n)) for n in fixed_draws]
        lines += [(f"MPC, horizon {h}, {n} draws", ("mpc", h, n)) for h in horizons for n in mpc_draws]
        # the same controller without its readings: replanning on the forecast alone
        lines += [(f"MPC not learning, horizon {h}, mean", ("mpc_blind", h, 0)) for h in blind_horizons]
        for name, (c, h, n) in lines:
            cells = []
            for s in sigmas:
                line = by.get((c, h, n, s), {})
                seeds = set(line) & set(hind)
                cells.append(f"{st.mean(per_move(line[x], metric) - per_move(hind[x], metric) for x in seeds):.0f}" if seeds else "-")
            print(f"| {name} | " + " | ".join(cells) + " |")
        print()

    metric = "summed_km"
    print("## paired comparisons, average distance over the episode\n")
    print("horizon: each horizon minus the next longer one, MPC on the mean")
    for s in sigmas:
        for short, long in zip(horizons, horizons[1:]):
            print(f"  sigma {s:5g}  H{short:<2d} - H{long:<2d}: {compare(by[('mpc', short, 0, s)], by[('mpc', long, 0, s)], metric)}")
    # an arm need not carry every draw count: the nested one starts at 1 and the distance ones have only the
    # mean, so each comparison is against the fewest draws that arm actually flew, and is skipped when that is
    # the same cell as the most
    if len(mpc_draws) > 1:
        print(f"\ndraws: MPC with {mpc_draws[-1]} draws minus MPC with {mpc_draws[0]}, at each horizon")
        for s in sigmas:
            for h in horizons:
                most, fewest = by.get(("mpc", h, mpc_draws[-1], s)), by.get(("mpc", h, mpc_draws[0], s))
                if most and fewest:
                    print(f"  sigma {s:5g}  H{h:<2d}  N{mpc_draws[-1]} - N{mpc_draws[0]}: {compare(most, fewest, metric)}")
    if blind_horizons:
        print("\nreadings: MPC minus the same controller not learning from them, on the mean")
        print("  below zero, what the balloon learned from its own anemometer was worth")
        for s in sigmas:
            for h in blind_horizons:
                if ("mpc", h, 0, s) in by and ("mpc_blind", h, 0, s) in by:
                    print(f"  sigma {s:5g}  H{h:<2d}: {compare(by[('mpc', h, 0, s)], by[('mpc_blind', h, 0, s)], metric)}")
        print("\nreplanning alone: MPC not learning, minus the fixed plan, both on the mean")
        print("  below zero, what re-solving every move was worth without knowing any more than the forecast said")
        for s in sigmas:
            for h in blind_horizons:
                if ("mpc_blind", h, 0, s) in by and ("fixed_plan", None, fixed_draws[0], s) in by:
                    print(f"  sigma {s:5g}  H{h:<2d}: {compare(by[('mpc_blind', h, 0, s)], by[('fixed_plan', None, fixed_draws[0], s)], metric)}")

    if len(fixed_draws) > 1:
        print(f"\ndraws: the fixed plan with {fixed_draws[-1]} draws minus the fixed plan with {fixed_draws[0]}")
        for s in sigmas:
            most, fewest = by.get(("fixed_plan", None, fixed_draws[-1], s)), by.get(("fixed_plan", None, fixed_draws[0], s))
            if most and fewest:
                print(f"  sigma {s:5g}  N{fixed_draws[-1]} - N{fixed_draws[0]}: {compare(most, fewest, metric)}")

    base = horizons[len(horizons) // 2] if horizons else None  # the middle horizon the arm flew
    if base is not None:
        print(f"\nreplanning: MPC (horizon {base}, {mpc_draws[0]} draws) minus the fixed plan ({fixed_draws[0]} draws)")
        for s in sigmas:
            mpc_cell, floor = by.get(("mpc", base, mpc_draws[0], s)), by.get(("fixed_plan", None, fixed_draws[0], s))
            if mpc_cell and floor:
                print(f"  sigma {s:5g}: {compare(mpc_cell, floor, metric)}")


if __name__ == "__main__":
    main()
