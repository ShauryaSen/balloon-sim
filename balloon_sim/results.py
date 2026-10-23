"""The result row: its columns, writing and reading

One row is one flight. Its columns:
    controller            "hindsight", "fixed_plan" or "mpc"
    n_winds               winds the controller planned on (0: the belief's mean; hindsight: 0, it uses the truth)
    seed
    <piece>.<field>       every setting the flight ran with, e.g. "planner.horizon", "belief.sigma"
    start_lat, start_lon, goal_lat, goal_lon
    summed_km, final_km   distance to the goal summed over every move, and after the last move
    cost_km               the objective: summed until the balloon arrives, nothing after (the arrival rule)
    arrived, arrived_move whether it ever came within task.arrival_km, and the move it first did (0 if never)
    closest_km, closest_move   the nearest the balloon came to the goal, and after which move (from 1)
    goal_km               how far the goal was placed, in km
    belief_scale          what the belief assumed the error was, as a multiple of the real one; 1 is calibrated
    nested                whether the draws were a subset of one shared pool rather than drawn fresh
    path_lat, path_lon, path_altitude_km   the trajectory, when the cell was asked to keep it
    seconds               how long the flight took
Rows are stored as JSON lines: one row per line of a .jsonl file. A sweep writes one file per cell under a
directory, since many cells appending to one file at once on NFS can tear a line; read_rows takes either.
"""

import dataclasses
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp

from balloon_sim.dynamics import distance_km
from balloon_sim.episode import Flight
from balloon_sim.task import Task


def row(
    flight: Flight,
    task: Task,
    *,
    controller: str,
    n_winds: int,
    seed: int,
    seconds: float,
    settings: dict[str, Any],
    extra: dict[str, Any] | None = None,
    keep_path: bool = False,
) -> dict:
    """settings maps a piece's name to its config, e.g. {"planner": PlannerConfig(...), "belief": BeliefConfig(...)}.

    `extra` carries what the cell knows and the configs do not, such as which goal distance was drawn or
    whether the draws were nested. `keep_path` adds the trajectory, which is three numbers per move: worth it
    for the arms whose shape is in question, and about 2 kB a row, so not for every cell of every sweep.
    """
    columns: dict[str, Any] = {"controller": controller, "n_winds": n_winds, "seed": seed}
    along = distance_km(flight.visited.lat, flight.visited.lon, task.goal_lat, task.goal_lon)  # after each move
    for piece, config in settings.items():
        for field, value in dataclasses.asdict(config).items():
            columns[f"{piece}.{field}"] = value
    columns.update(
        start_lat=float(task.start.lat),
        start_lon=float(task.start.lon),
        goal_lat=float(task.goal_lat),
        goal_lon=float(task.goal_lon),
        summed_km=float(flight.summed_km),
        final_km=float(flight.final_km),
        cost_km=float(flight.cost_km),
        arrived=bool(flight.arrived),
        arrived_move=int(flight.arrived_move),
        closest_km=float(jnp.min(along)),
        closest_move=int(jnp.argmin(along)) + 1,
        seconds=seconds,
    )
    if keep_path:
        columns.update(
            path_lat=[round(float(x), 4) for x in flight.visited.lat],
            path_lon=[round(float(x), 4) for x in flight.visited.lon],
            path_altitude_km=[round(float(x), 4) for x in flight.visited.altitude_km],
        )
    columns.update(extra or {})
    return columns


def write_rows(path: str | Path, rows: list[dict]) -> None:
    """Add rows to the end of a .jsonl file, creating it if needed"""
    with open(path, "a") as file:
        for r in rows:
            file.write(json.dumps(r) + "\n")


def read_rows(paths: list[str | Path]) -> list[dict]:
    """Every row in the given .jsonl files, or in every .jsonl under a given directory. A line that is not a
    row (two cells once appended to one file at the same moment on NFS and tore a line) is skipped and
    reported, not fatal"""
    files = [f for p in paths for f in (sorted(Path(p).glob("*.jsonl")) if Path(p).is_dir() else [Path(p)])]
    rows, torn = [], 0
    for path in files:
        for line in path.read_text().splitlines():
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                torn += 1
    if torn:
        print(f"skipped {torn} torn line(s); the cells they belonged to need rerunning")
    return rows


CLAIRVOYANT_METRICS = ("summed_km", "final_km", "closest_km", "cost_km")


def best_hindsight(rows: list[dict]) -> list[dict]:
    """The rows with each seed's hindsight replaced by the best any search found on the true wind.

    Every cell at sigma 0 plans on the truth, as hindsight does, so each is a clairvoyant flight; they differ only
    in how well their optimiser searched. Hindsight is meant to be the best a clairvoyant controller can do, and the
    best of these searches is the closest we have to it (DESIGN.md 34, 36). Each metric takes its own best, since
    each is bounded on its own. Rows that lack a metric (written before it existed) keep hindsight's own value.
    """
    clairvoyant: dict[int, list[dict]] = {}
    for r in rows:
        if r["controller"] == "hindsight" or r["belief.sigma"] == 0.0:
            clairvoyant.setdefault(r["seed"], []).append(r)
    out = []
    for r in rows:
        if r["controller"] != "hindsight":
            out.append(r)
            continue
        best = dict(r)
        for metric in CLAIRVOYANT_METRICS:
            values = [c[metric] for c in clairvoyant[r["seed"]] if metric in c]
            if values:
                best[metric] = min(values)
        out.append(best)
    return out
