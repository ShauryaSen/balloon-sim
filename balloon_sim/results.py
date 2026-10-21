"""The result row: its columns, writing and reading

One row is one flight. Its columns:
    controller            "hindsight", "fixed_plan" or "mpc"
    n_winds               winds the controller planned on (0: the belief's mean; hindsight: 0, it uses the truth)
    seed
    <piece>.<field>       every setting the flight ran with, e.g. "planner.horizon", "belief.sigma"
    start_lat, start_lon, goal_lat, goal_lon
    summed_km, final_km   the two metrics
    seconds               how long the flight took
Rows are stored as JSON lines: one row per line of a .jsonl file. A sweep writes one file per cell under a
directory, since many cells appending to one file at once on NFS can tear a line; read_rows takes either.
"""

import dataclasses
import json
from pathlib import Path
from typing import Any

from balloon_sim.episode import Flight
from balloon_sim.task import Task


def row(flight: Flight, task: Task, *, controller: str, n_winds: int, seed: int, seconds: float, settings: dict[str, Any]) -> dict:
    """settings maps a piece's name to its config, e.g. {"planner": PlannerConfig(...), "belief": BeliefConfig(...)}"""
    columns: dict[str, Any] = {"controller": controller, "n_winds": n_winds, "seed": seed}
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
        seconds=seconds,
    )
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
