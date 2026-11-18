"""Dry run of station keeping with Pasetsky et al.'s control setup, before the fidelity sweep (decision 54)

    uv run scripts/dry_run_station.py --seeds 0 1 2 3 4 5 6 7 --out results/dry-run-station.jsonl

Their setup (arXiv 2511.07761, Sec. VI and Fig. 3): an action every 3 min, a 12 h horizon (240 steps),
a replan every 72 min (24 steps), 100 random starting plans, 2-day episodes (960 steps), scored by time within
50 km (TWR50), launches within 10 degrees of the equator. Our climb limit is kept, 1.1 km per 3 h, which is
18.3 m per 3-min step.

Flown on the undegraded truth, so it says whether the setup leaves room to measure anything:
    coast                    never changes altitude: the zero-control floor
    mpc_truth_<optimizer>    MPC planning on the true wind, with L-BFGS and with Myles's normalised descent
    mpc_persistence_<opt>    MPC planning on the truth 24 h earlier: 1x a real day-old error
Pass: MPC on the truth well above coast, persistence in between, a flight in minutes.
"""

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import jax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from balloon_sim import degrade, fleet
from balloon_sim.belief import GPBelief, Truth
from balloon_sim.cells import keys
from balloon_sim.config import BalloonConfig, BeliefConfig, PlannerConfig, TaskConfig
from balloon_sim.results import write_rows
from balloon_sim.wind import load_record

STEP_HOURS = 0.05  # 3 min
BALLOON = BalloonConfig(step_hours=STEP_HOURS, substeps=1, climb_cap_km=1.1 * STEP_HOURS / 3.0)
PLANNER = PlannerConfig(horizon=240, replan_every=24, candidates=100, iterations=100, step_size=1.0, tolerance=1e-7)
TASK = TaskConfig(objective="station", n_agents=1, n_moves=960, arrival_km=50.0, lat_band=(-10.0, 10.0), distance_km=300.0)
RUNS = ("coast", "mpc_truth_lbfgs", "mpc_truth_normalised", "mpc_persistence_lbfgs", "mpc_persistence_normalised")


def main() -> None:
    ask = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ask.add_argument("--record", default="records/era5_global_jul2021.npz")
    ask.add_argument("--seeds", type=int, nargs="+", default=[0])
    ask.add_argument("--runs", nargs="+", default=list(RUNS), choices=RUNS)
    ask.add_argument("--out", default="results/dry-run-station.jsonl")
    a = ask.parse_args()

    truth, earlier = load_record(a.record), load_record(a.record, "forecast_")
    print(f"persistence error in the band: {degrade.error_size(truth, earlier, TASK.lat_band):.2f} m/s")
    for seed in a.seeds:
        k_task, _, k_fly = keys(seed)
        mission, task = fleet.draw_mission(TASK, k_task, truth)
        for run in a.runs:
            if run == "coast":
                controller, planner = fleet.coast(mission), PLANNER
            else:
                _, wind, optimizer = run.split("_")
                planner = dataclasses.replace(PLANNER, optimizer=optimizer)
                believed = Truth(truth) if wind == "truth" else GPBelief(BeliefConfig(), degrade.persistence(truth, earlier))
                controller = fleet.mpc(BALLOON, planner, believed, "station", mission, 0, TASK.n_moves, TASK.arrival_km, learns=False)
            began = time.perf_counter()
            flight = fleet.fly(BALLOON, controller, truth, mission, "station", TASK.n_moves, k_fly, TASK.arrival_km)
            jax.block_until_ready(flight.cost_km)
            seconds = time.perf_counter() - began
            row = fleet.fleet_row(flight, mission, task, controller=run, n_winds=0, seed=seed, seconds=seconds,
                                  settings={"balloon": BALLOON, "planner": planner, "task": TASK}, extra={"run": run})
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            write_rows(a.out, [row])
            print(f"seed {seed:2d} {run:28s} TWR50 {float(flight.covered_fraction):6.1%}  cost {float(flight.cost_km):10.0f} km  "
                  f"{seconds:6.1f} s ({1000 * seconds / TASK.n_moves:5.1f} ms/step)", flush=True)


if __name__ == "__main__":
    main()
