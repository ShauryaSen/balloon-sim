"""List the cells of a sweep and submit them as a Slurm array

    uv run scripts/submit.py --dry-run          # print the cells and stop
    uv run scripts/submit.py                    # write the array scripts and sbatch them
    uv run scripts/submit.py -- --task.distance_km 2000      # a setting changed for every cell

One array task runs one cell and writes its own row file, so a failure loses one flight rather than the sweep
and no two tasks ever write the same file (appends to one file from many tasks tore a line on NFS). The task's index picks its
line out of a cell list written beside the results, which keeps the sbatch script short and lets a failed
subset be resubmitted by hand. Slurm caps an array at 1,001 tasks, so the list is submitted as arrays of
1,000, each starting at its own offset into the one list. Cells that average over many belief draws (the
fixed plan's 64) need far more memory than the rest, so they go in arrays of their own with --big-mem-gb.
Anything after `--` is a run_cell.py flag added to every cell, so a whole sweep can change one setting.

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

from balloon_sim.cells import ARMS, cells
from balloon_sim.config import BeliefConfig, ExperimentConfig, TaskConfig

ARRAY = """#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem_gb}G
#SBATCH --time={minutes}
#SBATCH --array=0-{last}%{at_once}
#SBATCH --output={logs}/{prefix}{offset}-%a.out
#SBATCH --error={logs}/{prefix}{offset}-%a.out

set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
cd {root}
LINE=$(sed -n "$(({offset} + SLURM_ARRAY_TASK_ID + 1))p" {list})
INDEX=${{LINE%% *}}
uv run scripts/run_cell.py ${{LINE#* }} --out {out}/$INDEX.jsonl
"""


ARRAY_LIMIT = 1000
"""Tasks per array: Slurm's MaxArraySize on Unicorn is 1,001"""


def flags(cell, pool: int, keep_path: bool) -> str:
    """One cell as the arguments run_cell.py takes"""
    out = (f"--controller {cell.controller} --seed {cell.seed} --sigma {cell.sigma} "
           f"--horizon {cell.horizon} --n-winds {cell.n_winds} --goal-km {cell.goal_km} "
           f"--belief-scale {cell.belief_scale}")
    if cell.nested:
        out += f" --nested --nested-pool {pool}"
    if keep_path:
        out += " --keep-path"
    return out


def drawn(cell, pool: int) -> int:
    """How many winds the cell actually draws, which is what its memory scales with. A nested cell draws the
    whole pool however few of them its plan ends up averaging over"""
    return pool if cell.nested else cell.n_winds


def main(argv: list[str] | None = None) -> None:
    ask = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ask.add_argument("--out", default="results/rows", help="a directory: one .jsonl per cell, named by its index")
    ask.add_argument("--name", default="balloon")
    ask.add_argument("--partition", default="dean")
    ask.add_argument("--cpus", type=int, default=1)
    ask.add_argument("--mem-gb", type=int, default=8)
    ask.add_argument("--minutes", default="120")
    ask.add_argument("--at-once", type=int, default=48, help="array tasks allowed to run at the same time")
    ask.add_argument("--big-mem-gb", type=int, default=40, help="memory for cells averaging over many draws")
    ask.add_argument("--big-draws", type=int, default=16, help="a cell drawing at least this many winds is big")
    ask.add_argument("--arms", default=",".join(ARMS), help=f"which arms to run, comma separated: {', '.join(ARMS)}")
    ask.add_argument("--keep-path", action="store_true", help="record every cell's trajectory in its row")
    ask.add_argument("--dry-run", action="store_true", help="print the cells and do not submit")
    ask.add_argument("overrides", nargs="*", help="after --: run_cell.py flags added to every cell")
    asked = ask.parse_args(argv)
    extra = " ".join(asked.overrides)

    root = Path(__file__).resolve().parents[1]
    experiment = ExperimentConfig()
    arms = tuple(name.strip() for name in asked.arms.split(",") if name.strip())
    unknown = [name for name in arms if name not in ARMS]
    if unknown:
        raise SystemExit(f"no such arm: {', '.join(unknown)}. Known arms: {', '.join(ARMS)}")
    grid = cells(experiment, TaskConfig().n_moves, BeliefConfig().sigma, arms)
    pool = experiment.nested_pool
    lines = [f"{flags(cell, pool, asked.keep_path)} {extra}".strip() for cell in grid]

    counted: dict[str, int] = {}
    for cell in grid:
        counted[cell.controller] = counted.get(cell.controller, 0) + 1
    print(f"{len(grid)} cells over arms {', '.join(arms)}: " + ", ".join(f"{n} {name}" for name, n in sorted(counted.items())))
    for name in arms:  # what each arm costs on its own, before the overlap between them is removed
        print(f"  {name:12s} {len(cells(experiment, TaskConfig().n_moves, BeliefConfig().sigma, (name,))):5d} cells")
    if asked.dry_run:
        print("\n".join(lines[:10]) + ("\n..." if len(lines) > 10 else ""))
        return

    runs = root / "results"
    logs = runs / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (root / asked.out).mkdir(parents=True, exist_ok=True)
    listing = runs / "cells.txt"
    listing.write_text("\n".join(lines) + "\n")
    (runs / "cells.json").write_text(json.dumps([cell._asdict() for cell in grid], indent=1))

    big = [i for i, cell in enumerate(grid) if drawn(cell, pool) >= asked.big_draws]
    groups = [("", [i for i in range(len(grid)) if i not in set(big)], asked.mem_gb), ("big-", big, asked.big_mem_gb)]
    for prefix, indices, mem_gb in groups:
        if not indices:
            continue
        group = runs / f"{prefix}cells.txt"  # each line: the cell's index in cells.txt, then its flags
        group.write_text("".join(f"{i} {lines[i]}\n" for i in indices))
        for offset in range(0, len(indices), ARRAY_LIMIT):
            chunk = min(ARRAY_LIMIT, len(indices) - offset)
            script = runs / f"{prefix}array-{offset}.sh"
            script.write_text(ARRAY.format(
                name=asked.name, partition=asked.partition, cpus=asked.cpus, mem_gb=mem_gb, minutes=asked.minutes,
                last=chunk - 1, at_once=asked.at_once, logs=logs, root=root, list=group, out=asked.out, offset=offset,
                prefix=prefix,
            ))
            submitted = subprocess.run(["sbatch", str(script)], check=True, capture_output=True, text=True).stdout.strip()
            print(f"{prefix or 'regular '}cells {offset}-{offset + chunk - 1} at {mem_gb} GB: {submitted}")

if __name__ == "__main__":
    main()
