"""Write the fidelity sweep's cell list and sbatch it as one array, a few cells per task

    uv run scripts/submit_fidelity.py --record records/era5_tropics_jul2021_l18_1h.npz --dry-run
    uv run scripts/submit_fidelity.py --record records/era5_tropics_jul2021_l18_1h.npz --seeds 0 1 --name smoke

Each task flies `--per-task` consecutive cells in one process, so the record loads once, and writes its own
row file (one file per task, so concurrent writers never share one on NFS).
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from balloon_sim.fidelity import cells, error_cells, readings_cells

SCRIPT = """#!/bin/bash
#SBATCH --job-name=fid-{name}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task=4
#SBATCH --mem={mem_gb}G
#SBATCH --time={minutes}
#SBATCH --array=0-{last}%{at_once}
#SBATCH --output={logs}/%a.out
#SBATCH --error={logs}/%a.out
cd {repo}
~/.local/bin/uv run scripts/run_fidelity.py --record {record} --cells {cell_file} \\
    --first $((SLURM_ARRAY_TASK_ID * {per_task})) --count {per_task} --out {rows}/$SLURM_ARRAY_TASK_ID.jsonl
"""


def main() -> None:
    ask = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ask.add_argument("--record", required=True)
    ask.add_argument("--seeds", type=int, nargs="+", default=list(range(32)))
    ask.add_argument("--name", default="sweep")
    ask.add_argument("--per-task", type=int, default=4)
    ask.add_argument("--partition", default="dean")
    ask.add_argument("--mem-gb", type=int, default=8)
    ask.add_argument("--minutes", type=int, default=240)
    ask.add_argument("--at-once", type=int, default=200)
    ask.add_argument("--arm", choices=("sweep", "errors", "readings"), default="sweep", help="the 2026-11-17 sweep, the errors arm or the readings arm")
    ask.add_argument("--dry-run", action="store_true")
    a = ask.parse_args()

    todo = {"sweep": cells, "errors": error_cells, "readings": readings_cells}[a.arm](tuple(a.seeds))
    repo = Path(__file__).resolve().parents[1]
    base = repo / "results" / f"fidelity-{a.name}"
    logs, rows = base / "logs", base / "rows"
    n_tasks = -(-len(todo) // a.per_task)
    print(f"{len(todo)} cells in {n_tasks} tasks of {a.per_task}")
    if n_tasks > 1000:
        raise SystemExit("more than 1,000 tasks: raise --per-task (MaxArraySize is 1001)")
    if a.dry_run:
        for cell in todo[:12]:
            print(" ", cell.args())
        return
    for folder in (logs, rows):
        folder.mkdir(parents=True, exist_ok=True)
    cell_file = base / "cells.txt"
    cell_file.write_text("\n".join(cell.args() for cell in todo) + "\n")
    script = base / "array.sh"
    script.write_text(SCRIPT.format(name=a.name, partition=a.partition, mem_gb=a.mem_gb, minutes=a.minutes, last=n_tasks - 1,
                                    at_once=a.at_once, logs=logs, repo=repo, record=a.record, cell_file=cell_file,
                                    per_task=a.per_task, rows=rows))
    print(subprocess.run(["sbatch", str(script)], capture_output=True, text=True, check=True).stdout.strip())


if __name__ == "__main__":
    main()
