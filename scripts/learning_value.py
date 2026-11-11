"""What learning from readings is worth, seed by seed and move by move (the 2026-11-06 meeting's questions)

    uv run scripts/learning_value.py results/rows-2026-10-24.jsonl                       # the main reach arm
    uv run scripts/learning_value.py rows.jsonl --objective station --metric-into figures/x
    uv run scripts/learning_value.py rows.jsonl --objective targets --n-agents 4 --horizon 20

For one arm, at one MPC horizon on the belief's mean, it pairs every seed's MPC with the same seed's MPC not
learning (or rl with rl_blind, with --learned) and asks:

    the distribution over seeds of the difference, learning minus not, in the arm's own cost
    how that difference builds up over the episode: each move's cost is recomputed from the flight's saved
        path, summed as the objective sums it, and the running total of the difference is mapped move by move
    whether it is ever worse to learn: on how many seeds, by how much, and where in the episode

Each move's cost is rebuilt from the path, and the rebuilt total must equal the row's cost_km (it is checked,
and a row that does not match is reported and left out), so the move-by-move map adds up to the number every
other reader uses. Below zero, learning did better.

Writes a text summary to stdout and two figures into --into: the per-seed differences at each noise level,
and the running difference over the episode (median and quartiles over seeds), one line per noise level.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from balloon_sim.results import one_arm, read_rows

RADIUS_EARTH_KM = 6371.0088


def _distance_km(lat1, lon1, lat2, lon2):
    """Haversine, as dynamics.distance_km, with its 1e-12 so the two agree to the metre"""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2.0 * RADIUS_EARTH_KM * np.arcsin(np.sqrt(h + 1e-12))


def _mission(r: dict):
    """(starts_lat, starts_lon, targets_lat, targets_lon) as arrays; a single-balloon reach or escape row
    carries only its start and goal"""
    if "starts_lat" in r:
        return (np.array(r["starts_lat"]), np.array(r["starts_lon"]), np.array(r["targets_lat"]), np.array(r["targets_lon"]))
    return np.array([r["start_lat"]]), np.array([r["start_lon"]]), np.array([r["goal_lat"]]), np.array([r["goal_lon"]])


def per_move_cost(r: dict, objective: str) -> np.ndarray | None:
    """(T,) each move's share of the objective, summing to cost_km; None if the row kept no path"""
    if "path_lat" not in r:
        return None
    lat, lon = np.array(r["path_lat"], float), np.array(r["path_lon"], float)
    if lat.ndim == 1:
        lat, lon = lat[None], lon[None]  # (N, T)
    s_lat, s_lon, t_lat, t_lon = _mission(r)
    radius = float(r.get("task.arrival_km", 200.0))
    T = lat.shape[1]
    if objective == "reach":
        d = _distance_km(lat, lon, t_lat[0], t_lon[0])  # (N, T)
        cost = np.zeros(T)
        for one in d:
            hit = np.nonzero(one < radius)[0]
            last = hit[0] if hit.size else T - 1
            cost[: last + 1] += one[: last + 1]
        return cost / d.shape[0]
    if objective == "escape":
        return -np.mean(_distance_km(lat, lon, s_lat[:, None], s_lon[:, None]), axis=0)
    if objective == "station":
        return np.min(np.maximum(_distance_km(lat, lon, t_lat[0], t_lon[0]) - radius, 0.0), axis=0)
    if objective == "targets":
        d = np.min(_distance_km(lat[None], lon[None], t_lat[:, None, None], t_lon[:, None, None]), axis=1)  # (K, T)
        cost = np.zeros(T)
        for one in d:
            hit = np.nonzero(one < radius)[0]
            last = hit[0] if hit.size else T - 1
            cost[: last + 1] += one[: last + 1]
        return cost
    raise ValueError(objective)


def main(argv: list[str] | None = None) -> None:
    ask = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ask.add_argument("rows", nargs="+")
    ask.add_argument("--objective", choices=("reach", "escape", "station", "targets"), default="reach")
    ask.add_argument("--n-agents", type=int, default=1)
    ask.add_argument("--n-moves", type=int, default=None)
    ask.add_argument("--goal-km", type=float, default=None)
    ask.add_argument("--horizon", type=int, default=20, help="the MPC horizon compared (ignored with --learned)")
    ask.add_argument("--learned", action="store_true", help="compare rl with rl_blind instead of mpc with mpc_blind")
    ask.add_argument("--into", default=None, help="where to write the two figures (default: no figures)")
    asked = ask.parse_args(argv)

    rows = one_arm(read_rows(asked.rows), goal_km=asked.goal_km, n_moves=asked.n_moves, objective=asked.objective, n_agents=asked.n_agents)
    learn, blind = ("rl", "rl_blind") if asked.learned else ("mpc", "mpc_blind")

    def wanted(r):
        if r["controller"] not in (learn, blind):
            return False
        return asked.learned or (r.get("planner.horizon") == asked.horizon and r.get("n_winds", 0) == 0)

    by: dict[tuple, dict] = {}  # (controller, sigma) -> {seed: row}
    for r in rows:
        if wanted(r):
            by.setdefault((r["controller"], round(r["belief.sigma"], 4)), {})[r["seed"]] = r
    sigmas = sorted({s for (_, s) in by})
    if not sigmas:
        raise SystemExit(f"no {learn}/{blind} rows in this arm")

    label = "RL minus RL not learning" if asked.learned else f"MPC minus MPC not learning, horizon {asked.horizon}"
    print(f"# {asked.objective}, {asked.n_agents} balloon(s): {label}, in km of the objective. Below zero, learning did better.\n")
    totals, curves, mismatched = {}, {}, 0
    for s in sigmas:
        a, b = by.get((learn, s), {}), by.get((blind, s), {})
        seeds = sorted(set(a) & set(b))
        totals[s] = np.array([a[k]["cost_km"] - b[k]["cost_km"] for k in seeds])
        runs = []
        for k in seeds:
            ca, cb = per_move_cost(a[k], asked.objective), per_move_cost(b[k], asked.objective)
            if ca is None or cb is None:
                continue
            if abs(ca.sum() - a[k]["cost_km"]) > 1.0 + 1e-3 * abs(a[k]["cost_km"]) or abs(cb.sum() - b[k]["cost_km"]) > 1.0 + 1e-3 * abs(b[k]["cost_km"]):
                mismatched += 1
                continue
            runs.append(np.cumsum(ca - cb))
        curves[s] = np.array(runs) if runs else None

        d = totals[s]
        if d.size == 0:
            continue
        worse, better = int(np.sum(d > 1.0)), int(np.sum(d < -1.0))
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        print(f"sigma {s:5g}  seeds {d.size:2d}  mean {d.mean():+9.0f}  median {med:+9.0f}  quartiles [{q1:+.0f}, {q3:+.0f}]  "
              f"learning better on {better}, worse on {worse}, tied on {d.size - better - worse}")
        if worse:
            print(f"             worse: by up to {d.max():+.0f} km; the seeds {[seeds[i] for i in np.nonzero(d > 1.0)[0]]}")

    print("\n# over the episode: the running difference, median over seeds, at each tenth of the episode")
    for s in sigmas:
        c = curves.get(s)
        if c is None or not c.size:
            print(f"sigma {s:5g}  no saved paths")
            continue
        T = c.shape[1]
        marks = [max(0, int(round(T * f)) - 1) for f in np.linspace(0.1, 1.0, 10)]
        med = np.median(c, axis=0)
        ahead = np.mean(c > 1.0, axis=0)  # the share of seeds where learning is behind at each move
        print(f"sigma {s:5g}  " + " ".join(f"{med[m]:+8.0f}" for m in marks) + f"   most seeds behind at once: {ahead.max():.0%} (move {int(ahead.argmax()) + 1})")
    if mismatched:
        print(f"\n{mismatched} flights whose rebuilt cost did not match cost_km were left out of the move-by-move map")

    if asked.into:
        into = Path(asked.into)
        into.mkdir(parents=True, exist_ok=True)
        colours = plt.cm.viridis(np.linspace(0.05, 0.9, len(sigmas)))
        tag = f"{asked.objective}_n{asked.n_agents}_" + ("rl" if asked.learned else f"h{asked.horizon}")

        figure, axis = plt.subplots(figsize=(7, 4.6))
        rng = np.random.default_rng(0)
        for i, s in enumerate(sigmas):
            d = totals[s]
            axis.scatter(i + rng.uniform(-0.18, 0.18, d.size), d, s=14, color=colours[i], alpha=0.8)
            if d.size:
                axis.hlines(np.median(d), i - 0.28, i + 0.28, color="#12161b", lw=1.5)
        axis.axhline(0.0, color="#888888", lw=0.8)
        axis.set_xticks(range(len(sigmas)), [f"{s:g}" for s in sigmas])
        axis.set_xlabel("forecast error σ (m/s)")
        axis.set_ylabel("learning minus not, km of the objective")
        axis.set_title(f"{label}: every seed (bar: median)", fontsize=10)
        figure.tight_layout()
        figure.savefig(into / f"learning_per_seed_{tag}.png", dpi=150)

        figure, axis = plt.subplots(figsize=(7, 4.6))
        for i, s in enumerate(sigmas):
            c = curves.get(s)
            if c is None or not c.size:
                continue
            moves = np.arange(1, c.shape[1] + 1)
            q1, med, q3 = np.percentile(c, [25, 50, 75], axis=0)
            axis.plot(moves, med, color=colours[i], lw=1.6, label=f"σ {s:g}")
            axis.fill_between(moves, q1, q3, color=colours[i], alpha=0.12)
        axis.axhline(0.0, color="#888888", lw=0.8)
        axis.set_xlabel("move")
        axis.set_ylabel("running difference, km (median, quartiles)")
        axis.set_title(f"{label}: over the episode", fontsize=10)
        axis.legend(fontsize=8, frameon=False)
        figure.tight_layout()
        figure.savefig(into / f"learning_over_episode_{tag}.png", dpi=150)
        print(f"\nwrote {into}/learning_per_seed_{tag}.png and learning_over_episode_{tag}.png")


if __name__ == "__main__":
    main()
