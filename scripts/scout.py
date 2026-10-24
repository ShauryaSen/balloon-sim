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
ROOM = 0.25  # the fixed plan must be at least this much above hindsight, seed by seed
TYPICAL = 16  # seeds of 32 where that gap must hold: the rule reads the typical seed, not the mean


def gaps(a: dict, b: dict, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """a minus b on each seed both flew, in km and as a share of b's own cost on that seed.

    Seed by seed, because the two are not the same question. A task where a blind plan is typically a little
    worse and occasionally catastrophic has a huge mean gap and a small median one, and it is the median that
    says whether a controller has room to show something on a seed you would actually draw.
    """
    seeds = sorted(set(a) & set(b))
    km = np.array([a[s][metric] - b[s][metric] for s in seeds], dtype=float)
    floor = np.array([b[s][metric] for s in seeds], dtype=float)
    return km, np.divide(km, floor, out=np.zeros_like(km), where=floor > 0)


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

    print(f"{len(rows)} rows over {len(tasks)} tasks (docs/scouting-rule.md).")
    print(f"Qualifies if hindsight arrives on >= {REACH} seeds AND the blind plan is >= {ROOM:.0%} above it "
          f"on >= {TYPICAL} seeds.\n")
    header = (f"{'direction':10s} {'goal':>6s}  {'h.arr':>6s} {'move':>5s} {'f.arr':>6s}  {'hindsight':>10s}  "
              f"{'median gap':>13s} {'seeds':>6s}  {'mean gap':>10s}  verdict")
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
        blind_arrived = sum(1 for r in blind.values() if r.get("arrived"))
        km, share = gaps(blind, hind, "cost_km")
        if km.size == 0:
            continue
        floor = float(np.mean([r["cost_km"] for r in hind.values()]))
        material = int((share >= ROOM).sum())  # seeds where the blind plan is materially worse

        reachable = len(arrived) >= REACH
        room = material >= TYPICAL
        verdict = "QUALIFIES" if reachable and room else ("too free" if reachable else "unreachable")
        if reachable and room:
            qualified.append((float(np.median(km)), goal_km, direction, float(np.median(share)),
                              len(arrived), median_move, material))
        print(f"{direction:10s} {goal_km:6.0f}  {len(arrived):3d}/{len(hind):<2d} {median_move:5d} "
              f"{blind_arrived:3d}/{len(blind):<2d}  {floor:10.0f}  "
              f"{np.median(km):8.0f} {np.median(share):4.0%} {material:3d}/{km.size:<2d}  "
              f"{km.mean():10.0f}  {verdict}")

    print()
    if not qualified:
        print("Nothing qualifies. Per the rule, the episode length and the wind window are what change next,")
        print("not the thresholds: a longer episode, or a season with weaker flow.")
        return
    qualified.sort(key=lambda q: (-q[0], -q[1]))
    gap, goal_km, direction, share, arrived, median_move, material = qualified[0]
    print(f"The sweep takes {direction} at {goal_km:.0f} km: the largest median gap among the qualifying "
          f"tasks, {gap:.0f} km ({share:.0%} above hindsight on a typical seed, material on {material} of them),")
    print(f"hindsight arriving on {arrived} seeds at a median move {median_move} of 50.")
    print(f'\nSet BASE_DIRECTION = "{direction}" and BASE_GOAL_KM = {goal_km:.0f} in balloon_sim/cells.py, '
          "and TaskConfig's defaults to match.")


if __name__ == "__main__":
    main()
