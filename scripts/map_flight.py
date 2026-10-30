"""Fly one seed with each controller and draw where they went

    uv run scripts/map_flight.py --seed 3 --sigma 3.2 --into figures/

Top: the three flights on a map (hindsight, MPC, fixed plan), with the start and the goal and a ring at the
arrival radius. Bottom: each one's altitude against time. One seed is an illustration, not evidence; the
sweep's figures are the evidence.

The episode ends the first time the balloon is inside the ring, and the moves after that cost nothing. The
simulator still flies them, because JAX wants a fixed number of moves and the rule is applied to the cost
rather than by stopping the scan, so the trajectory always has every move in it. What is drawn faint is
those moves: flown, recorded, and charged for nothing. Reading the solid part is reading the episode.
"""

import argparse
import dataclasses
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from balloon_sim.belief import GPBelief, forecast
from balloon_sim.cells import keys
from balloon_sim.config import BalloonConfig, BeliefConfig, PlannerConfig, TaskConfig, WindConfig
from balloon_sim.controllers import fixed_plan, hindsight, mpc
from balloon_sim.dynamics import distance_km
from balloon_sim.episode import fly
from balloon_sim.task import draw_task
from balloon_sim.wind import load_record

INK = "#12161b"
STYLE = {"hindsight": (INK, "-"), "MPC": ("#0072b2", "-"), "fixed plan": ("#0072b2", "--")}


def main(argv: list[str] | None = None) -> None:
    ask = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ask.add_argument("--seed", type=int, default=0)
    ask.add_argument("--sigma", type=float, default=BeliefConfig().sigma)
    ask.add_argument("--horizon", type=int, default=20, help="MPC's horizon")
    ask.add_argument("--into", default="figures")
    asked = ask.parse_args(argv)

    balloon, task_config = BalloonConfig(), TaskConfig()
    planner = dataclasses.replace(PlannerConfig(), horizon=asked.horizon)
    belief_config = dataclasses.replace(BeliefConfig(), sigma=asked.sigma)
    truth = load_record(WindConfig().path)
    k_task, k_noise, k_fly = keys(asked.seed)  # the same keys run_cell.py uses, so this is the sweep's flight
    task = draw_task(task_config, k_task, truth)
    believed = GPBelief(belief_config, forecast(belief_config, k_noise, truth))
    n, radius = task_config.n_moves, task_config.arrival_km
    # the arrival radius has to reach the controllers and the simulator, exactly as run_cell.py passes it:
    # without it these would plan and be scored on the plain sum, which is a different controller from the
    # one the sweep flew, on a different objective, and the map would not be of the sweep at all
    controllers = {
        "hindsight": hindsight(balloon, planner, truth, task.goal_lat, task.goal_lon, n, radius),
        "MPC": mpc(balloon, planner, believed, task.goal_lat, task.goal_lon, 0, n, radius),
        "fixed plan": fixed_plan(balloon, planner, believed, task.goal_lat, task.goal_lon, n, 0, radius),
    }
    flights = {name: fly(balloon, c, truth, task.start, task.goal_lat, task.goal_lon, n, k_fly, radius)
               for name, c in controllers.items()}

    figure, (where, when) = plt.subplots(2, 1, figsize=(7.0, 8.0), dpi=200, gridspec_kw={"height_ratios": [3, 2]})
    unwrap = lambda lon: np.degrees(np.unwrap(np.radians(np.asarray(lon))))  # no jump at the 0/360 seam
    lon0 = float(task.start.lon)
    for name, flight in flights.items():
        colour, style = STYLE[name]
        lon = unwrap(np.concatenate([[lon0], np.asarray(flight.visited.lon)]))
        lat = np.concatenate([[float(task.start.lat)], np.asarray(flight.visited.lat)])
        hours = np.concatenate([[0.0], np.asarray(flight.visited.hours)])
        altitude = np.concatenate([[float(task.start.altitude_km)], np.asarray(flight.visited.altitude_km)])
        # the episode is over once it arrives, so everything past that move is drawn faint: still flown, charged nothing
        counted = int(flight.arrived_move) + 1 if bool(flight.arrived) else len(lon)
        average = float(flight.cost_km) / n
        ended = f"arrived at {hours[counted - 1]:.0f} h" if bool(flight.arrived) else "never arrived"
        where.plot(lon[:counted], lat[:counted], color=colour, linestyle=style, linewidth=1.6,
                   label=f"{name}: {average:.0f} km on average, {ended}")
        when.plot(hours[:counted], altitude[:counted], color=colour, linestyle=style, linewidth=1.4)
        if counted < len(lon):  # the moves after the episode ended
            where.plot(lon[counted - 1:], lat[counted - 1:], color=colour, linestyle=style, linewidth=1.0, alpha=0.25)
            when.plot(hours[counted - 1:], altitude[counted - 1:], color=colour, linestyle=style, linewidth=1.0, alpha=0.25)
    goal_lon = lon0 + ((float(task.goal_lon) - lon0 + 180) % 360 - 180)
    where.plot(lon0, float(task.start.lat), "o", color=INK, markersize=6)
    where.plot(goal_lon, float(task.goal_lat), "*", color="#d55e00", markersize=14, label="goal")
    ring = np.linspace(0, 2 * np.pi, 120)
    lat_ring = float(task.goal_lat) + np.degrees(radius / 6371) * np.cos(ring)
    lon_ring = goal_lon + np.degrees(radius / 6371) * np.sin(ring) / np.cos(np.radians(float(task.goal_lat)))
    where.plot(lon_ring, lat_ring, color="#d55e00", linewidth=0.8, linestyle=":")
    where.set_xlabel("longitude (degrees east)")
    where.set_ylabel("latitude (degrees north)")
    where.set_aspect(1 / np.cos(np.radians(float(task.start.lat))))
    where.set_title(f"Seed {asked.seed}: goal {task_config.distance_km:.0f} km downwind, forecast error {asked.sigma:g} m/s\n"
                    f"faint: flown after the episode ended inside the {radius:.0f} km ring, and charged nothing",
                    fontsize=10, loc="left", color=INK)
    where.legend(frameon=False, fontsize=8)
    where.spines[["top", "right"]].set_visible(False)
    when.set_xlabel("hours since launch")
    when.set_ylabel("altitude (km)")
    when.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    into = Path(asked.into)
    into.mkdir(parents=True, exist_ok=True)
    path = into / f"flight_seed{asked.seed}_sigma{asked.sigma:g}.png"
    figure.savefig(path)
    print(f"wrote {path}")
    for name, flight in flights.items():
        closest = float(np.min(np.asarray(distance_km(flight.visited.lat, flight.visited.lon, task.goal_lat, task.goal_lon))))
        reached = f"move {int(flight.arrived_move):2d}" if bool(flight.arrived) else "  never"
        print(f"  {name:10s} objective {float(flight.cost_km) / n:6.0f} km/move, plain sum {float(flight.summed_km) / n:6.0f}, "
              f"closest {closest:6.0f} km, arrived {reached}")


if __name__ == "__main__":
    main()
