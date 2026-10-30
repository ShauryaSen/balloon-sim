"""Draw one seed's flights from the rows a sweep already wrote

    uv run scripts/map_rows.py results/rows-2026-10-24.jsonl --seed 9 --sigma 3.2 --into figures/

Top: the flights on a map, with the start, the goal and a ring at the arrival radius. Bottom: altitude
against time. One seed is an illustration, not evidence; the sweep's figures are the evidence.

This reads the trajectory out of the row's path_lat, path_lon and path_altitude_km columns, which a sweep
run with --keep-path saves. scripts/map_flight.py re-flies the seed instead, which needs the wind record and
has to be given the same settings the sweep used; this cannot disagree with the sweep because it is the
sweep's own flight. Prefer this one whenever the rows carry a path.

The episode ends the first time the balloon is inside the ring, and the moves after that cost nothing. The
simulator still flies them, because the rule is applied to the cost rather than by stopping the scan, so the
trajectory always has every move in it. What is drawn faint is those moves: flown, recorded, charged nothing.
Reading the solid part is reading the episode.
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
from balloon_sim.results import one_arm, read_rows

INK = "#12161b"
STYLE = {"hindsight": (INK, "-"), "mpc": ("#0072b2", "-"), "mpc_blind": ("#0072b2", ":"), "fixed_plan": ("#0072b2", "--")}
NAMES = {"hindsight": "hindsight", "mpc": "MPC", "mpc_blind": "MPC not learning", "fixed_plan": "fixed plan"}


def chosen(rows: list[dict], controller: str, seed: int, sigma: float, horizon: int, n_winds: int) -> dict | None:
    """The one row for this cell. Hindsight reads the truth, so its sigma is 0 and its horizon is the episode"""
    for r in rows:
        if r["controller"] != controller or r["seed"] != seed or r["n_winds"] != n_winds:
            continue
        if controller == "hindsight":
            return r
        if r["belief.sigma"] != sigma:
            continue
        if controller.startswith("mpc") and r["planner.horizon"] != horizon:
            continue
        return r
    return None


def main(argv: list[str] | None = None) -> None:
    ask = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ask.add_argument("rows", nargs="+", help="the .jsonl files to read")
    ask.add_argument("--seed", type=int, default=9)
    ask.add_argument("--sigma", type=float, default=3.2)
    ask.add_argument("--horizon", type=int, default=20, help="MPC's horizon")
    ask.add_argument("--controllers", default="hindsight,mpc,fixed_plan",
                     help="comma separated, in the order they are drawn; mpc_blind may be added")
    ask.add_argument("--goal-km", type=float, default=None)
    ask.add_argument("--n-moves", type=int, default=None)
    ask.add_argument("--into", default="figures")
    asked = ask.parse_args(argv)

    rows = one_arm(read_rows(asked.rows), asked.goal_km, 1.0, False, asked.n_moves)
    wanted = [c.strip() for c in asked.controllers.split(",") if c.strip()]
    flights = {}
    for controller in wanted:
        row = chosen(rows, controller, asked.seed, asked.sigma, asked.horizon, 0)
        if row is None:
            sys.exit(f"no row for {controller} at seed {asked.seed}, sigma {asked.sigma:g}, horizon {asked.horizon}")
        if "path_lat" not in row:
            sys.exit(f"{controller}'s row has no trajectory; the sweep needed --keep-path")
        flights[controller] = row

    any_row = next(iter(flights.values()))
    radius, n = any_row["task.arrival_km"], any_row["task.n_moves"]
    step_hours = any_row["balloon.step_hours"]
    start_lat, start_lon = any_row["start_lat"], any_row["start_lon"]
    goal_lat, goal_lon_raw = any_row["goal_lat"], any_row["goal_lon"]

    figure, (where, when) = plt.subplots(2, 1, figsize=(9.0, 7.5), dpi=200, gridspec_kw={"height_ratios": [2, 2]})
    unwrap = lambda lon: np.degrees(np.unwrap(np.radians(np.asarray(lon, dtype=float))))  # no jump at the 0/360 seam
    for controller, row in flights.items():
        colour, style = STYLE.get(controller, ("#666666", "-"))
        lon = unwrap(np.concatenate([[start_lon], row["path_lon"]]))
        lat = np.concatenate([[start_lat], row["path_lat"]])
        altitude = np.concatenate([[row["path_altitude_km"][0]], row["path_altitude_km"]])
        hours = np.arange(len(lon)) * step_hours
        counted = int(row["arrived_move"]) + 1 if row.get("arrived") else len(lon)
        average = row["cost_km"] / n
        ended = f"arrived at {hours[counted - 1]:.0f} h" if row.get("arrived") else "never arrived"
        where.plot(lon[:counted], lat[:counted], color=colour, linestyle=style, linewidth=1.6,
                   label=f"{NAMES.get(controller, controller)}: {average:.0f} km on average, {ended}")
        when.plot(hours[:counted], altitude[:counted], color=colour, linestyle=style, linewidth=1.4)
        if counted < len(lon):  # flown after the episode ended, and charged nothing
            where.plot(lon[counted - 1:], lat[counted - 1:], color=colour, linestyle=style, linewidth=1.0, alpha=0.25)
            when.plot(hours[counted - 1:], altitude[counted - 1:], color=colour, linestyle=style, linewidth=1.0, alpha=0.25)

    goal_lon = start_lon + ((goal_lon_raw - start_lon + 180) % 360 - 180)
    where.plot(start_lon, start_lat, "o", color=INK, markersize=6)
    where.plot(goal_lon, goal_lat, "*", color="#d55e00", markersize=14, label="goal")
    ring = np.linspace(0, 2 * np.pi, 120)
    where.plot(goal_lon + np.degrees(radius / 6371) * np.sin(ring) / np.cos(np.radians(goal_lat)),
               goal_lat + np.degrees(radius / 6371) * np.cos(ring), color="#d55e00", linewidth=0.8, linestyle=":")
    where.set_xlabel("longitude (degrees east)")
    where.set_ylabel("latitude (degrees north)")
    # to scale, so a degree of longitude is shorter than a degree of latitude at this latitude. The track is
    # far wider than it is tall, so the panel comes out a strip; that is the shape the flight has, and
    # stretching the latitudes to fill the panel would only invent room the flight never used
    where.set_aspect(1 / np.cos(np.radians(start_lat)))
    where.set_title(f"Seed {asked.seed}: goal {row['goal_km']:.0f} km downwind, forecast error {asked.sigma:g} m/s\n"
                    f"faint: flown after the episode ended inside the {radius:.0f} km ring, and charged nothing",
                    fontsize=10, loc="left", color=INK)
    where.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))  # beside the map, off the track
    where.spines[["top", "right"]].set_visible(False)
    when.set_xlabel("hours since launch")
    when.set_ylabel("altitude (km)")
    when.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    into = Path(asked.into)
    into.mkdir(parents=True, exist_ok=True)
    path = into / f"flight_seed{asked.seed}_sigma{asked.sigma:g}.png"
    figure.savefig(path)
    plt.close(figure)
    print(f"wrote {path}")
    for controller, row in flights.items():
        reached = f"move {int(row['arrived_move']):2d}" if row.get("arrived") else "  never"
        print(f"  {NAMES.get(controller, controller):18s} objective {row['cost_km'] / n:6.0f} km/move, "
              f"closest {row['closest_km']:6.0f} km, arrived {reached}")


if __name__ == "__main__":
    main()
