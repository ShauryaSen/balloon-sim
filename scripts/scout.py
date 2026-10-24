"""Read the scouting arm and apply the rule in docs/scouting-rule.md

    uv run scripts/scout.py results/rows-scout.jsonl

One line per (direction, distance): whether hindsight can reach the goal, how much room there is between a
controller that knows the wind and one flying blind, and whether the pair qualifies. The task the sweep
should use is the qualifying one with the largest gap.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from balloon_sim.results import read_rows

REACH = 24  # seeds of 32 hindsight must arrive on
ROOM = 0.25  # the fixed plan must be at least this much above hindsight


def paired(a: dict, b: dict, metric: str) -> tuple[float, float, int]:
    """a minus b over the seeds both flew: the mean, half a 95% interval, and how many seeds were compared"""
    seeds = sorted(set(a) & set(b))
    gaps = np.array([a[s][metric] - b[s][metric] for s in seeds], dtype=float)
    if gaps.size < 2:
        return float("nan"), float("nan"), gaps.size
    return float(gaps.mean()), float(1.96 * gaps.std(ddof=1) / np.sqrt(gaps.size)), gaps.size


def main(argv: list[str] | None = None) -> None:
    ask = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ask.add_argument("rows", nargs="+")
    asked = ask.parse_args(argv)
    rows = read_rows(asked.rows)
    if not rows:
        sys.exit("no rows in those files")

    tasks: dict[tuple[str, float], dict[str, dict[int, dict]]] = {}
    for r in rows:
        tasks.setdefault((r["direction"], r["goal_km"]), {}).setdefault(r["controller"], {})[r["seed"]] = r

    print(f"{len(rows)} rows over {len(tasks)} tasks. Rule: hindsight arrives on >= {REACH}/32, "
          f"and the fixed plan is >= {ROOM:.0%} above it with the interval clear of zero.\n")
    header = f"{'direction':10s} {'goal':>6s}  {'arrives':>8s} {'move':>5s}  {'hindsight':>10s} {'blind above':>22s}  {'gap':>6s}  verdict"
    print(header)
    print("-" * len(header))

    qualified = []
    for (direction, goal_km), by in sorted(tasks.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        hind, blind = by.get("hindsight", {}), by.get("fixed_plan", {})
        if not hind or not blind:
            continue
        arrived = [r for r in hind.values() if r.get("arrived")]
        moves = sorted(r["arrived_move"] for r in arrived)
        median_move = moves[len(moves) // 2] if moves else 0
        mean, half, n = paired(blind, hind, "cost_km")
        floor = float(np.mean([r["cost_km"] for r in hind.values()]))
        share = mean / floor if floor else float("nan")

        reachable = len(arrived) >= REACH
        room = share >= ROOM and (mean - half) > 0
        verdict = "qualifies" if reachable and room else ("too free" if reachable else "unreachable")
        if reachable and room:
            qualified.append((mean, goal_km, direction, share, len(arrived), median_move))
        print(f"{direction:10s} {goal_km:6.0f}  {len(arrived):4d}/{len(hind):<3d} {median_move:5d}  "
              f"{floor:10.0f}  {mean:9.0f} [{mean - half:6.0f},{mean + half:6.0f}]  {share:5.0%}  {verdict}")

    print()
    if not qualified:
        print("Nothing qualifies. Per the rule, the episode length and the wind window are what change next,")
        print("not the thresholds: a longer episode, or a season with weaker flow.")
        return
    qualified.sort(key=lambda q: (-q[0], -q[1]))
    gap, goal_km, direction, share, arrived, median_move = qualified[0]
    print(f"The sweep takes {direction} at {goal_km:.0f} km: the largest gap among the qualifying tasks, "
          f"{gap:.0f} km ({share:.0%} above hindsight),")
    print(f"hindsight arriving on {arrived}/32 seeds at a median move {median_move} of 50.")
    print(f'\nSet BASE_DIRECTION = "{direction}" and BASE_GOAL_KM = {goal_km:.0f} in balloon_sim/cells.py, '
          "and TaskConfig's defaults to match.")


if __name__ == "__main__":
    main()
