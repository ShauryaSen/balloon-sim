#!/usr/bin/env bash
# The 2026-10-23 sweep, start to finish. Run it from the repo root on a Unicorn interactive job:
#
#     srun --partition=dean --cpus-per-task=2 --mem=8G --time=60 --pty bash
#     cd ~/balloon-sim && ./run_all.sh
#
# It checks the code, flies one cell of each kind, prints the grid, asks once, then submits. Nothing reaches
# Slurm until the checks have passed and you have said yes, so it is safe to run and read.
set -euo pipefail

RECORD=${RECORD:-records/era5_global_jul2021.npz}
OUT=${OUT:-results/rows-2026-10-24}
ARMS=${ARMS:-main,distance,calibration,nested}
#
# The scouting arm comes first and runs on its own, since it is what decides the task the sweep flies:
#     ARMS=scout OUT=results/rows-scout ./run_all.sh
#     uv run scripts/scout.py results/rows-scout/*.jsonl
# then set BASE_DIRECTION and BASE_GOAL_KM in balloon_sim/cells.py to what it chose and run this again.

say() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

say "1. the record is there"
test -f "$RECORD" || { echo "no record at $RECORD; set RECORD=... if it lives elsewhere"; exit 1; }

say "2. the tests"
uv run pytest -q

say "3. can the balloon hold a point in this record at all?"
uv run scripts/wind_report.py "$RECORD" --lat 20 50 | tee results/wind-report.txt

# the task the sweep will actually fly, read from the config rather than written out here, so these cells
# cannot quietly go on testing a task the sweep has moved away from
GOAL=$(uv run python -c "from balloon_sim.cells import BASE_GOAL_KM, BASE_DIRECTION; print(BASE_GOAL_KM, BASE_DIRECTION)")
read -r GOAL_KM DIRECTION <<< "$GOAL"

say "4. one cell of each controller at ${GOAL_KM} km ${DIRECTION}, to see it fly before 8,000 of them do"
for CONTROLLER in hindsight fixed_plan mpc mpc_blind; do
  uv run scripts/run_cell.py --controller "$CONTROLLER" --seed 0 --sigma 3.2 --horizon 20 \
    --n-winds 0 --goal-km "$GOAL_KM" --direction "$DIRECTION" --keep-path --out results/smoke.jsonl
done
say "   and one with a miscalibrated belief, and one with nested draws"
uv run scripts/run_cell.py --controller mpc --seed 0 --sigma 3.2 --horizon 20 --n-winds 4 \
  --goal-km "$GOAL_KM" --direction "$DIRECTION" --belief-scale 2.0 --out results/smoke.jsonl
uv run scripts/run_cell.py --controller mpc --seed 0 --sigma 3.2 --horizon 20 --n-winds 4 \
  --goal-km "$GOAL_KM" --direction "$DIRECTION" --nested --nested-pool 16 --out results/smoke.jsonl

say "5. the grid"
uv run scripts/submit.py --arms "$ARMS" --out "$OUT" --dry-run

say "6. submit?"
read -r -p "   type yes to send it to Slurm: " ANSWER
[ "$ANSWER" = "yes" ] || { echo "   nothing submitted"; exit 0; }

uv run scripts/submit.py --arms "$ARMS" --out "$OUT" --keep-path

cat <<'NOTE'

Submitted. To watch it:
    squeue -u "$USER"
    sacct -X --format=JobID,State,Elapsed --starttime today | tail -20

When it is done:
    cat results/rows-2026-10-23/*.jsonl > results/rows-2026-10-23.jsonl
    uv run scripts/summarise.py results/rows-2026-10-23.jsonl
    uv run scripts/plot.py results/rows-2026-10-23.jsonl --into figures/2026-10-23
    git add results/rows-2026-10-23.jsonl figures/2026-10-23 && git commit && git push
NOTE
