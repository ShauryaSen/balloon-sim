"""Score every copy of the fidelity sweep, and run the primary test

    uv run scripts/summarise_fidelity.py results/fidelity-sweep/rows --out docs/fidelity-results.md

Score (decision 55): the share of MPC's time within 50 km a copy costs, against coast, as a ratio of means
over the seeds with a paired bootstrap interval (balloon_sim.fidelity.score). Each copy is scored against
MPC on the truth in its own arm, at the same climb limit and horizon.

Primary test (pre-registered, docs/fidelity-design-2026-11-06.md): over the base arm's copies, the rank
correlation of the score with the wind error, and with the level-difference error. Which one control loss
follows is the result; the bootstrap over copies says how sure.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from balloon_sim.fidelity import score


def ranks(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x)).astype(float)


def spearman(x, y) -> float:
    return float(np.corrcoef(ranks(np.asarray(x)), ranks(np.asarray(y)))[0, 1])


def main() -> None:
    ask = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ask.add_argument("rows", help="a folder of .jsonl row files, or one file")
    ask.add_argument("--out", default=None)
    a = ask.parse_args()

    files = sorted(Path(a.rows).glob("*.jsonl")) if Path(a.rows).is_dir() else [Path(a.rows)]
    rows = [json.loads(line) for f in files for line in f.read_text().splitlines() if line.strip()]
    coast = {r["seed"]: r["covered_fraction"] for r in rows if r["controller"] == "coast"}
    groups: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for r in rows:
        if r["controller"] == "mpc":
            groups[(r["arm"], r["climb_scale"], r["horizon_hours"], r["copy"], r["size"])][r["seed"]] = r

    out = ["| arm | climb | horizon h | copy | size | score [95%] | TWR50 drop | wind err | level-diff err | n |", "|" + "---|" * 10]
    base_points = []
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2], k[3] != "truth", k[3], k[4])):
        arm, climb, horizon, copy, size = key
        if copy == "truth":
            continue
        reference = groups.get((arm, climb, horizon, "truth", 0.0), {})
        seeds = sorted(set(groups[key]) & set(reference) & set(coast))
        if len(seeds) < 2:
            continue
        s = score(np.array([reference[i]["covered_fraction"] for i in seeds]), np.array([groups[key][i]["covered_fraction"] for i in seeds]),
                  np.array([coast[i] for i in seeds]))
        wind = np.mean([groups[key][i]["wind_error_ms"] for i in seeds])
        steer = np.mean([groups[key][i]["level_difference_error_ms"] for i in seeds])
        out.append(f"| {arm} | {climb:g} | {horizon:g} | {copy} | {size:g} | {s['score']:+.2f} [{s['low']:+.2f}, {s['high']:+.2f}] | "
                   f"{s['twr_drop']:+.3f} | {wind:.2f} | {steer:.2f} | {s['n']} |")
        if arm == "base":
            base_points.append((s["score"], wind, steer))

    if len(base_points) >= 3:
        scores, wind, steer = (np.array(x) for x in zip(*base_points))
        rng = np.random.default_rng(0)
        picks = rng.integers(0, len(scores), (10_000, len(scores)))
        gap = np.array([spearman(steer[i], scores[i]) - spearman(wind[i], scores[i]) for i in picks])
        out += ["", "**Primary test** (base arm, one point per copy):", "",
                f"- rank correlation of the score with the wind error: {spearman(wind, scores):+.2f}",
                f"- rank correlation with the level-difference error: {spearman(steer, scores):+.2f}",
                f"- level-difference minus wind, bootstrap over copies: {np.mean(gap):+.2f} "
                f"[{np.percentile(gap, 2.5):+.2f}, {np.percentile(gap, 97.5):+.2f}]"]
    out += ["", f"coast TWR50 mean {np.mean(list(coast.values())):.3f} over {len(coast)} seeds"]
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
