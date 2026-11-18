"""Fly fidelity cells and write one row each

    uv run scripts/run_fidelity.py --record records/era5_tropics_jul2021_l18_1h.npz --controller mpc --seed 0 --copy levels:3
    uv run scripts/run_fidelity.py --record ... --cells cells.txt --first 0 --count 4 --out rows/0.jsonl

Several cells per process (--cells), so the record is loaded once. Each row carries the cell, the full
configuration, and the copy's errors against the truth in the launch band: the wind error and the
level-difference error the primary test ranks the copies by (docs/fidelity-design-2026-11-06.md).
"""

import argparse
import shlex
import sys
import time
from pathlib import Path

import jax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from balloon_sim import degrade, fleet
from balloon_sim.belief import Truth
from balloon_sim.cells import keys
from balloon_sim.config import BeliefConfig
from balloon_sim.fidelity import BASE_HORIZON_HOURS, LAUNCH_BAND, Cell, configs
from balloon_sim.results import write_rows
from balloon_sim.wind import load_record


def cell_parser() -> argparse.ArgumentParser:
    ask = argparse.ArgumentParser(add_help=False)
    ask.add_argument("--controller", choices=("mpc", "coast"))
    ask.add_argument("--seed", type=int)
    ask.add_argument("--copy", default="truth")
    ask.add_argument("--size", type=float, default=0.0)
    ask.add_argument("--climb-scale", type=float, default=1.0)
    ask.add_argument("--horizon-hours", type=float, default=BASE_HORIZON_HOURS)
    ask.add_argument("--arm", default="base")
    return ask


def as_cell(asked: argparse.Namespace) -> Cell:
    return Cell(asked.controller, asked.seed, asked.copy, asked.size, asked.climb_scale, asked.horizon_hours, asked.arm)


def fly_one(cell: Cell, truth, earlier, one_x: float, out: str) -> None:
    balloon, planner, task_config = configs(cell)
    k_task, k_error, k_fly = keys(cell.seed)
    mission, task = fleet.draw_mission(task_config, k_task, truth)
    size_ms = cell.size if cell.arm == "errors" else cell.size * one_x  # the errors arm gives its sizes in m/s
    copied = degrade.copy(cell.copy, truth, k_error, BeliefConfig(), size_ms, earlier, LAUNCH_BAND)
    if cell.controller == "coast":
        controller = fleet.coast(mission)
    else:  # readings off: the copy is the whole of what MPC knows, so it is planned on as a plain field
        controller = fleet.mpc(balloon, planner, Truth(copied), "station", mission, 0, task_config.n_moves, task_config.arrival_km, learns=False)
    began = time.perf_counter()
    flight = fleet.fly(balloon, controller, truth, mission, "station", task_config.n_moves, k_fly, task_config.arrival_km)
    jax.block_until_ready(flight.cost_km)
    seconds = time.perf_counter() - began
    errors = degrade.errors(truth, copied, LAUNCH_BAND)
    row = fleet.fleet_row(flight, mission, task, controller=cell.controller, n_winds=0, seed=cell.seed, seconds=seconds,
                          settings={"balloon": balloon, "planner": planner, "task": task_config},
                          extra={**cell._asdict(), "size_ms": size_ms, "one_x_ms": one_x, **errors})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_rows(out, [row])
    print(f"{cell.controller:5s} seed {cell.seed:2d} {cell.copy:16s} size {cell.size:g} climb {cell.climb_scale:g} "
          f"horizon {cell.horizon_hours:g} h  TWR50 {float(flight.covered_fraction):6.1%}  "
          f"wind err {errors['wind_error_ms']:5.2f}  level-diff err {errors['level_difference_error_ms']:5.2f}  {seconds:6.1f} s", flush=True)


def main() -> None:
    ask = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, parents=[cell_parser()])
    ask.add_argument("--record", required=True)
    ask.add_argument("--cells", help="a file of cell argument lines, as submit_fidelity.py writes")
    ask.add_argument("--first", type=int, default=0)
    ask.add_argument("--count", type=int, default=1)
    ask.add_argument("--out", default="results/fidelity.jsonl")
    asked = ask.parse_args()

    truth, earlier = load_record(asked.record), load_record(asked.record, "forecast_")
    one_x = degrade.error_size(truth, earlier, LAUNCH_BAND)  # 1x a day-old forecast's error, in this record's band
    if asked.cells:
        lines = Path(asked.cells).read_text().splitlines()[asked.first: asked.first + asked.count]
        todo = [as_cell(cell_parser().parse_args(shlex.split(line))) for line in lines]
    else:
        todo = [as_cell(asked)]
    for cell in todo:
        fly_one(cell, truth, earlier, one_x, asked.out)


if __name__ == "__main__":
    main()
