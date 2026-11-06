"""Draw one seed's flights for the station, targets and fleet arms, from the rows the sweep wrote

    uv run scripts/map_fleet.py results/rows.jsonl --objective station --n-agents 4 --seed 9 --sigma 3.2 --into figures/

One panel per controller, side by side on the same axes: every balloon's track, the launch points, and the
objective's points with a ring at the radius that counts (the goal, the station, the stops). Escape has no
point; its panel shows the launches only. Below, every balloon's altitude against time. Read from the rows'
saved paths, so this is the sweep's own flight, as with scripts/map_rows.py; that script draws the
single-balloon reach and escape arms and this one draws everything fleet.py flew.
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

INK = "#12161b"
SHADES = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9", "#f0e442", "#999999")
NAMES = {"hindsight": "hindsight", "mpc": "MPC", "mpc_blind": "MPC not learning", "fixed_plan": "fixed plan", "rl": "RL (PPO)", "rl_blind": "RL not learning"}
DONE = {
    "reach": lambda r: f"{r['arrived_count']}/{r['n_agents']} arrived",
    "escape": lambda r: f"farthest {r['farthest_km']:,.0f} km",
    "station": lambda r: f"held {r['covered_fraction']:.0%} of moves",
    "targets": lambda r: f"{r['visited_count']}/{len(r['targets_lat'])} visited",
}


def chosen(rows: list[dict], controller: str, seed: int, sigma: float, horizon: int) -> dict | None:
    for r in rows:
        if r["controller"] != controller or r["seed"] != seed or r["n_winds"] != 0:
            continue
        if controller == "hindsight":
            return r
        if r["belief.sigma"] == sigma and (not controller.startswith("mpc") or r["planner.horizon"] == horizon):
            return r
    return None


def tracks(row: dict, key: str) -> list[list[float]]:
    """A path column as one list per balloon, whether the row is a fleet's or a single balloon's"""
    path = row[key]
    return path if path and isinstance(path[0], list) else [path]


def main(argv: list[str] | None = None) -> None:
    ask = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ask.add_argument("rows", nargs="+")
    ask.add_argument("--objective", choices=tuple(DONE), required=True)
    ask.add_argument("--n-agents", type=int, default=1)
    ask.add_argument("--seed", type=int, default=9)
    ask.add_argument("--sigma", type=float, default=3.2)
    ask.add_argument("--horizon", type=int, default=20)
    ask.add_argument("--n-moves", type=int, default=None, help="episode length, if not the base 50")
    ask.add_argument("--controllers", default="hindsight,mpc,mpc_blind,fixed_plan,rl,rl_blind")
    ask.add_argument("--into", default="figures")
    asked = ask.parse_args(argv)

    rows = one_arm(read_rows(asked.rows), n_moves=asked.n_moves, objective=asked.objective, n_agents=asked.n_agents)
    flights = {}
    for controller in [c.strip() for c in asked.controllers.split(",") if c.strip()]:
        row = chosen(rows, controller, asked.seed, asked.sigma, asked.horizon)
        if row is not None and "path_lat" in row:
            flights[controller] = row
    if not flights:
        sys.exit(f"no {asked.objective} rows with paths for {asked.n_agents} balloon(s) at seed {asked.seed}, sigma {asked.sigma:g}")

    figure, panels = plt.subplots(2, len(flights), figsize=(4.2 * len(flights), 7.0), dpi=200, squeeze=False,
                                  gridspec_kw={"height_ratios": [3, 2]}, sharex="row", sharey="row")
    ring = np.linspace(0, 2 * np.pi, 120)
    for column, (controller, row) in enumerate(flights.items()):
        where, when = panels[0, column], panels[1, column]
        radius, step_hours = row["task.arrival_km"], row["balloon.step_hours"]
        centre = row["start_lon"]
        near = lambda lon: centre + ((np.asarray(lon, dtype=float) - centre + 180) % 360 - 180)  # no jump at the seam
        starts_lat = row.get("starts_lat", [row["start_lat"]])
        starts_lon = row.get("starts_lon", [row["start_lon"]])
        for i, (lat, lon, alt) in enumerate(zip(tracks(row, "path_lat"), tracks(row, "path_lon"), tracks(row, "path_altitude_km"))):
            shade = SHADES[i % len(SHADES)]
            lon = near(np.degrees(np.unwrap(np.radians(np.concatenate([[starts_lon[i]], lon])))))
            where.plot(lon, np.concatenate([[starts_lat[i]], lat]), color=shade, linewidth=1.3)
            where.plot(lon[0], starts_lat[i], "o", color=shade, markersize=4)
            when.plot(np.arange(len(alt) + 1) * step_hours, np.concatenate([[alt[0]], alt]), color=shade, linewidth=1.1)
        if asked.objective != "escape":
            for k, (lat, lon) in enumerate(zip(row["targets_lat"], row["targets_lon"])):
                lon = float(near(lon))
                where.plot(lon, lat, "*", color=INK, markersize=11)
                if asked.objective == "targets":
                    where.annotate(str(k + 1), (lon, lat), textcoords="offset points", xytext=(6, 6), fontsize=8, color=INK)
                where.plot(lon + np.degrees(radius / 6371) * np.sin(ring) / np.cos(np.radians(lat)),
                           lat + np.degrees(radius / 6371) * np.cos(ring), color=INK, linewidth=0.7, linestyle=":")
        where.set_title(f"{NAMES.get(controller, controller)}\ncost {row['cost_km']:,.0f} km, {DONE[asked.objective](row)}",
                        fontsize=9, loc="left", color=INK)
        where.set_aspect(1 / np.cos(np.radians(row["start_lat"])))
        where.spines[["top", "right"]].set_visible(False)
        when.spines[["top", "right"]].set_visible(False)
        where.set_xlabel("longitude (degrees east)")
        when.set_xlabel("hours since launch")
    panels[0, 0].set_ylabel("latitude (degrees north)")
    panels[1, 0].set_ylabel("altitude (km)")
    figure.suptitle(f"{asked.objective}, {asked.n_agents} balloon(s), seed {asked.seed}, forecast error {asked.sigma:g} m/s "
                    f"(one colour per balloon; rings at {radius:.0f} km)", fontsize=10, x=0.01, ha="left", color=INK)
    figure.tight_layout()
    into = Path(asked.into)
    into.mkdir(parents=True, exist_ok=True)
    path = into / f"flight_{asked.objective}_n{asked.n_agents}_seed{asked.seed}_sigma{asked.sigma:g}.png"
    figure.savefig(path)
    plt.close(figure)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
