"""List the cells of a sweep and submit them as a Slurm array

    uv run scripts/submit.py --dry-run          # print the cells and stop
    uv run scripts/submit.py                    # write the array script and sbatch it

One array task runs one cell, so a failure loses one flight rather than the sweep. The task's index picks its
line out of a cell list written beside the results, which keeps the sbatch script short and lets a failed
subset be resubmitted by hand. Slurm caps an array at 1,001 tasks, so the list is submitted as arrays of
1,000, each starting at its own offset into the one list.

No --account is passed. Naming an account the submitter is not associated with is refused outright, while
naming none charges their own default association, which is what an interactive job already does. The
partition, which is the thing that decides the nodes, is still named.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from balloon_sim.cells import cells
from balloon_sim.config import BeliefConfig, ExperimentConfig, TaskConfig

ARRAY = """#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem_gb}G
#SBATCH --time={minutes}
#SBATCH --array=0-{last}%{at_once}
#SBATCH --output={logs}/{offset}-%a.out
#SBATCH --error={logs}/{offset}-%a.out

set -euo pipefail
cd {root}
CELL=$(sed -n "$(({offset} + SLURM_ARRAY_TASK_ID + 1))p" {list})
uv run scripts/run_cell.py $CELL --out {out}
"""


ARRAY_LIMIT = 1000
"""Tasks per array: Slurm's MaxArraySize on Unicorn is 1,001"""


def flags(cell) -> str:
    """One cell as the arguments run_cell.py takes"""
    return (f"--controller {cell.controller} --seed {cell.seed} --sigma {cell.sigma} "
            f"--horizon {cell.horizon} --n-winds {cell.n_winds}")


def main(argv: list[str] | None = None) -> None:
    ask = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ask.add_argument("--out", default="results/rows.jsonl")
    ask.add_argument("--name", default="balloon")
    ask.add_argument("--partition", default="dean")
    ask.add_argument("--cpus", type=int, default=1)
    ask.add_argument("--mem-gb", type=int, default=8)
    ask.add_argument("--minutes", default="120")
    ask.add_argument("--at-once", type=int, default=48, help="array tasks allowed to run at the same time")
    ask.add_argument("--dry-run", action="store_true", help="print the cells and do not submit")
    asked = ask.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    grid = cells(ExperimentConfig(), TaskConfig().n_moves, BeliefConfig().sigma)
    lines = [flags(cell) for cell in grid]

    counted: dict[str, int] = {}
    for cell in grid:
        counted[cell.controller] = counted.get(cell.controller, 0) + 1
    print(f"{len(grid)} cells: " + ", ".join(f"{n} {name}" for name, n in sorted(counted.items())))
    if asked.dry_run:
        print("\n".join(lines[:10]) + ("\n..." if len(lines) > 10 else ""))
        return

    runs = root / "results"
    logs = runs / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    listing = runs / "cells.txt"
    listing.write_text("\n".join(lines) + "\n")
    (runs / "cells.json").write_text(json.dumps([cell._asdict() for cell in grid], indent=1))

    for offset in range(0, len(grid), ARRAY_LIMIT):
        chunk = min(ARRAY_LIMIT, len(grid) - offset)
        script = runs / f"array-{offset}.sh"
        script.write_text(ARRAY.format(
            name=asked.name, partition=asked.partition, cpus=asked.cpus, mem_gb=asked.mem_gb, minutes=asked.minutes,
            last=chunk - 1, at_once=asked.at_once, logs=logs, root=root, list=listing, out=asked.out, offset=offset,
        ))
        submitted = subprocess.run(["sbatch", str(script)], check=True, capture_output=True, text=True).stdout.strip()
        print(f"cells {offset}-{offset + chunk - 1}: {submitted}")


if __name__ == "__main__":
    main()
