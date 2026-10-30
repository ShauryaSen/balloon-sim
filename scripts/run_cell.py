"""Fly one cell and write one row

    uv run scripts/run_cell.py --controller mpc --seed 0 --sigma 2.8 --horizon 20 --n-winds 8

Every setting is a command-line flag named `<piece>.<field>`, so `--planner.iterations 200` or
`--belief.horizontal_km 600` override a default without touching the code. The row carries the full
configuration it ran with, so a result can always be traced back to the settings that produced it.
"""

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from balloon_sim.belief import GPBelief, Nested, forecast
from balloon_sim.cells import keys
from balloon_sim.config import BalloonConfig, BeliefConfig, PlannerConfig, TaskConfig, WindConfig
from balloon_sim.controllers import fixed_plan, hindsight, mpc
from balloon_sim.episode import fly
from balloon_sim.results import row, write_rows
from balloon_sim.task import draw_task
from balloon_sim.wind import load_record

PIECES = {"wind": WindConfig, "balloon": BalloonConfig, "planner": PlannerConfig, "belief": BeliefConfig, "task": TaskConfig}
"""The configs a flight is built from, by the name their fields are prefixed with"""


def reader(annotation) -> type:
    """How a flag's text becomes a value. A field annotated `float | None` reads as a float, and anything that
    is not plainly a number stays text for the dataclass to make sense of"""
    if annotation in (int, float, str):
        return annotation
    for kind in (int, float):
        if annotation in (kind | None, f"{kind.__name__} | None"):
            return kind
    return str


def parser() -> argparse.ArgumentParser:
    """The cell's own arguments, then one flag per field of every config"""
    ask = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ask.add_argument("--controller", required=True, choices=("hindsight", "fixed_plan", "mpc", "mpc_blind"))
    ask.add_argument("--seed", type=int, required=True)
    ask.add_argument("--sigma", type=float, required=True, help="the forecast's error, overriding belief.sigma")
    ask.add_argument("--horizon", type=int, help="moves the MPC plans ahead, overriding planner.horizon")
    ask.add_argument("--n-winds", type=int, default=0, help="belief draws a plan is averaged over; 0 is its mean")
    ask.add_argument("--goal-km", type=float, help="how far away the goal is, overriding task.distance_km")
    ask.add_argument("--direction", choices=("downwind", "crosswind", "upwind", "random"),
                     help="where the goal sits relative to the flow, overriding task.direction")
    ask.add_argument("--belief-scale", type=float, default=1.0,
                     help="what the belief assumes the error is, as a multiple of --sigma; 1 is calibrated")
    ask.add_argument("--nested", action="store_true",
                     help="draw one pool of --nested-pool winds and use the first --n-winds of it")
    ask.add_argument("--nested-pool", type=int, default=16, help="the pool the nested draws come from")
    ask.add_argument("--objective", choices=("reach", "escape"), help="what the episode is for, overriding task.objective")
    ask.add_argument("--n-moves", type=int, help="length of the episode in moves, overriding task.n_moves")
    ask.add_argument("--keep-path", action="store_true", help="also record the trajectory in the row")
    ask.add_argument("--out", default="results/rows.jsonl", help="the .jsonl the row is added to")
    for piece, config in PIECES.items():
        for field in dataclasses.fields(config):
            kind = reader(field.type)
            ask.add_argument(f"--{piece}.{field.name}", type=kind, default=None, dest=f"{piece}.{field.name}")
    return ask


def configured(asked: dict) -> dict:
    """One config per piece, its defaults overridden by whatever was given on the command line"""
    built = {}
    for piece, config in PIECES.items():
        given = {f.name: asked[f"{piece}.{f.name}"] for f in dataclasses.fields(config) if asked.get(f"{piece}.{f.name}") is not None}
        built[piece] = dataclasses.replace(config(), **given)
    return built


def build(asked: dict):
    """The cell's own flags land on the configs they belong to, so a row records one consistent set"""
    pieces = configured(asked)
    pieces["belief"] = dataclasses.replace(pieces["belief"], sigma=asked["sigma"])
    if asked["horizon"] is not None:
        pieces["planner"] = dataclasses.replace(pieces["planner"], horizon=asked["horizon"])
    if asked.get("goal_km") is not None:
        pieces["task"] = dataclasses.replace(pieces["task"], distance_km=asked["goal_km"])
    if asked.get("direction") is not None:
        pieces["task"] = dataclasses.replace(pieces["task"], direction=asked["direction"])
    if asked.get("objective") is not None:
        pieces["task"] = dataclasses.replace(pieces["task"], objective=asked["objective"])
    if asked.get("n_moves") is not None:
        pieces["task"] = dataclasses.replace(pieces["task"], n_moves=asked["n_moves"])
    return pieces


def main(argv: list[str] | None = None) -> None:
    asked = vars(parser().parse_args(argv))
    pieces = build(asked)
    wind, balloon, planner, belief_config, task_config = (pieces[name] for name in PIECES)

    truth = load_record(wind.path)
    k_task, k_noise, k_fly = keys(asked["seed"])
    task = draw_task(task_config, k_task, truth)

    # The error actually put into the forecast is belief_config's sigma. What the controller believes about
    # that error is the same kernel with its amplitude scaled, so a scale other than 1 is a belief that is
    # wrong about its own uncertainty while the world is unchanged.
    assumed = dataclasses.replace(belief_config, sigma=belief_config.sigma * asked["belief_scale"])
    believed = GPBelief(assumed, forecast(belief_config, k_noise, truth))
    if asked["nested"]:
        believed = Nested(believed, asked["nested_pool"])

    n_moves, n_winds = task_config.n_moves, asked["n_winds"]
    # Escaping has nowhere to arrive, so the arrival rule is off, and the point the planner is pointed at is
    # the start rather than the goal: with `away` the sign is flipped, so it runs from there instead of to it.
    away = task_config.objective == "escape"
    radius = 0.0 if away else task_config.arrival_km
    aim_lat, aim_lon = (task.start.lat, task.start.lon) if away else (task.goal_lat, task.goal_lon)
    if asked["controller"] == "hindsight":
        controller = hindsight(balloon, planner, truth, aim_lat, aim_lon, n_moves, radius, away)
    elif asked["controller"] == "fixed_plan":
        controller = fixed_plan(balloon, planner, believed, aim_lat, aim_lon, n_moves, n_winds, radius, away)
    else:
        controller = mpc(balloon, planner, believed, aim_lat, aim_lon, n_winds, n_moves, radius,
                         learns=asked["controller"] == "mpc", away=away)

    began = time.perf_counter()
    flight = fly(balloon, controller, truth, task.start, task.goal_lat, task.goal_lon, n_moves, k_fly, radius, away)
    jax.block_until_ready(flight.summed_km)  # JAX is lazy, so the clock would otherwise stop before the flight
    seconds = time.perf_counter() - began

    written = row(
        flight, task, controller=asked["controller"], n_winds=n_winds, seed=asked["seed"], seconds=seconds, settings=pieces,
        extra={"goal_km": task_config.distance_km, "direction": task_config.direction,
               "belief_scale": asked["belief_scale"], "nested": asked["nested"],
               "objective": task_config.objective},
        keep_path=asked["keep_path"],
    )
    Path(asked["out"]).parent.mkdir(parents=True, exist_ok=True)
    write_rows(asked["out"], [written])
    common = (f"{asked['controller']:10s} seed {asked['seed']:3d} sigma {asked['sigma']:5.2f} "
              f"horizon {planner.horizon:3d} n_winds {n_winds:2d} moves {n_moves:3d} "
              f"scale {asked['belief_scale']:3.1f}  cost {float(flight.cost_km):10.1f} km")
    if away:
        print(f"{common}  farthest {float(flight.farthest_km):8.1f} km  "
              f"ended {float(flight.escape_final_km):8.1f} km out  {seconds:6.1f} s")
    else:
        reached = f"move {int(flight.arrived_move):2d}" if bool(flight.arrived) else "   never"
        print(f"{common}  goal {task_config.distance_km:5.0f} km {task_config.direction:9s}  "
              f"arrived {reached}  final {float(flight.final_km):8.1f} km  {seconds:6.1f} s")


if __name__ == "__main__":
    main()
