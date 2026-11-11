#!/usr/bin/env bash
# The sweep, start to finish. Run it from the repo root on a Unicorn interactive job:
#
#     srun --partition=dean --cpus-per-task=2 --mem=8G --time=60 --pty bash
#     cd ~/balloon-sim && ./run_all.sh
#
# It checks the code, flies one cell of each kind, prints the grid, asks once, then submits. Nothing reaches
# Slurm until the checks have passed and you have said yes, so it is safe to run and read.
set -euo pipefail

RECORD=${RECORD:-records/era5_global_jul2021.npz}
PYTHON=${PYTHON:-uv run python}
OUT=${OUT:-results/rows-$(date +%F)}
ARMS=${ARMS:-station,targets,fleet}
MINUTES=${MINUTES:-120}  # the time limit of each sweep cell; the flight arm needs more (MINUTES=480)
RL=${RL:-yes}  # also train and evaluate the learned policies (balloon_sim/rl.py); RL=no skips them
EARLIER=${EARLIER:-results/rows-2026-10-24.jsonl results/rows-2026-10-30.jsonl}
# rows of earlier sweeps the analysis reads beside this one: the single-balloon reach (main) and escape arms
# the learned policies are compared against are there, and are not flown again
# The 2026-11-05 default is the three new arms only: everything else ran on 2026-10-30 and is in
# results/rows-2026-10-30.jsonl. ARMS=main,distance,calibration,nested,length,escape,station,targets,fleet
# runs the whole sweep again.
#
# The scouting arm comes first and runs on its own, since it is what decides the task the sweep flies:
#     ARMS=scout OUT=results/rows-scout ./run_all.sh
#     uv run scripts/scout.py results/rows-scout/*.jsonl
# then set BASE_DIRECTION and BASE_GOAL_KM in balloon_sim/cells.py to what it chose and run this again.
#
# The greedy arm (MPC with a one-move horizon, decision 51) flies the July record beside the 2026-11-05 rows:
#     ARMS=greedy RL=no EARLIER="results/rows-2026-10-24.jsonl results/rows-2026-10-30.jsonl results/rows-2026-11-05.jsonl" ./run_all.sh
# The flight arm (long flights and long horizons, decision 51) needs a record at least 1,500 h long, flown alone:
#     ARMS=flight RL=no MINUTES=480 RECORD=records/<the long record>.npz EARLIER="" ./run_all.sh
# Every cell flies $RECORD: it is passed to each one, not only checked here.

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

say "4. does the record cover the longest episode?"
$PYTHON - "$RECORD" "$ARMS" <<'CHECK'
import sys
import numpy as np
from balloon_sim.config import BalloonConfig, ExperimentConfig

record = np.load(sys.argv[1])
covered = float(record["hours"][-1] - record["hours"][0])
lengths = ExperimentConfig().flight_lengths if "flight" in sys.argv[2].split(",") else ExperimentConfig().episode_lengths
needed = max(lengths) * BalloonConfig().step_hours
print(f"   record covers {covered:.0f} h; the longest episode needs {needed:.0f} h")
if needed > covered:
    sys.exit(f"   the longest flight would fly past the end of the record, where the wind is held at the last\n"
             f"   frame. Use a longer record, or lower the longest length to {int(covered // BalloonConfig().step_hours)} or fewer moves.")
CHECK

say "5. one cell of each controller at ${GOAL_KM} km ${DIRECTION}, to see one fly before the whole sweep does"
for CONTROLLER in hindsight fixed_plan mpc mpc_blind; do
  uv run scripts/run_cell.py --controller "$CONTROLLER" --seed 0 --sigma 3.2 --horizon 20 \
    --n-winds 0 --goal-km "$GOAL_KM" --direction "$DIRECTION" --keep-path --wind.path "$RECORD" --out results/smoke.jsonl
done
say "   and one with a miscalibrated belief, one with nested draws, and one of each new arm"
uv run scripts/run_cell.py --controller mpc --seed 0 --sigma 3.2 --horizon 20 --n-winds 4 \
  --goal-km "$GOAL_KM" --direction "$DIRECTION" --belief-scale 2.0 --wind.path "$RECORD" --out results/smoke.jsonl
uv run scripts/run_cell.py --controller mpc --seed 0 --sigma 3.2 --horizon 20 --n-winds 4 \
  --goal-km "$GOAL_KM" --direction "$DIRECTION" --nested --nested-pool 16 --wind.path "$RECORD" --out results/smoke.jsonl
LONG=$(uv run python -c "from balloon_sim.config import ExperimentConfig; print(max(ExperimentConfig().episode_lengths))")
uv run scripts/run_cell.py --controller mpc --seed 0 --sigma 3.2 --horizon 20 --n-winds 0 \
  --goal-km "$GOAL_KM" --direction "$DIRECTION" --n-moves "$LONG" --wind.path "$RECORD" --out results/smoke.jsonl
uv run scripts/run_cell.py --controller mpc --seed 0 --sigma 3.2 --horizon 20 --n-winds 0 \
  --goal-km "$GOAL_KM" --direction "$DIRECTION" --objective escape --wind.path "$RECORD" --out results/smoke.jsonl
say "   and the new objectives, alone and as a fleet"
FLEET=$(uv run python -c "from balloon_sim.config import ExperimentConfig; print(max(ExperimentConfig().fleet_sizes))")
for OBJECTIVE in station targets; do
  uv run scripts/run_cell.py --controller mpc --seed 0 --sigma 3.2 --horizon 20 --n-winds 0 \
    --goal-km "$GOAL_KM" --direction "$DIRECTION" --objective "$OBJECTIVE" --keep-path --wind.path "$RECORD" --out results/smoke.jsonl
done
for OBJECTIVE in reach escape station targets; do
  uv run scripts/run_cell.py --controller mpc --seed 0 --sigma 3.2 --horizon 20 --n-winds 0 \
    --goal-km "$GOAL_KM" --direction "$DIRECTION" --objective "$OBJECTIVE" --n-agents "$FLEET" --keep-path --wind.path "$RECORD" --out results/smoke.jsonl
done
uv run scripts/run_cell.py --controller hindsight --seed 0 --sigma 0 --horizon 50 --n-winds 0 \
  --goal-km "$GOAL_KM" --direction "$DIRECTION" --objective targets --n-agents "$FLEET" --keep-path --wind.path "$RECORD" --out results/smoke.jsonl

if [[ ",$ARMS," == *",greedy,"* ]]; then
  say "   and the greedy controller: MPC planning one move ahead"
  uv run scripts/run_cell.py --controller mpc --seed 0 --sigma 3.2 --horizon 1 --n-winds 0 \
    --goal-km "$GOAL_KM" --direction "$DIRECTION" --objective station --wind.path "$RECORD" --out results/smoke.jsonl
fi
if [[ ",$ARMS," == *",flight,"* ]]; then
  say "   and the longest flight, greedy and at the base horizon, to see one run to the end of the record"
  FLIGHT=$(uv run python -c "from balloon_sim.config import ExperimentConfig; print(max(ExperimentConfig().flight_lengths))")
  for HORIZON in 1 20; do
    uv run scripts/run_cell.py --controller mpc --seed 0 --sigma 3.2 --horizon "$HORIZON" --n-winds 0 \
      --goal-km "$GOAL_KM" --direction "$DIRECTION" --objective station --n-moves "$FLIGHT" --wind.path "$RECORD" --out results/smoke.jsonl
  done
fi

if [ "$RL" = "yes" ]; then
  say "   and a learned policy: two iterations on this record, then flown on one seed, to see it run end to end"
  uv run scripts/train_rl.py --objective targets --n-agents "$FLEET" --rl.iterations 2 --rl.envs 4 \
    --rl.train_seeds 4 --rl.minibatches 2 --record "$RECORD" --out results/smoke-policies
  uv run scripts/eval_rl.py "results/smoke-policies/targets-n$FLEET-learns.pkl" --seeds 0 --sigmas 3.2 \
    --out results/smoke-rl.jsonl
fi

say "6. the grid"
uv run scripts/submit.py --arms "$ARMS" --out "$OUT" --dry-run -- --wind.path "$RECORD"
if [ "$RL" = "yes" ]; then
  N_POLICIES=$(uv run python -c "from balloon_sim import rl; from balloon_sim.config import ExperimentConfig; print(len(rl.policies(ExperimentConfig())))")
  echo "   and $N_POLICIES learned policies, each trained, then flown on every seed and noise level:"
  uv run python -c "from balloon_sim import rl; from balloon_sim.config import ExperimentConfig; [print('     ', i, rl.policy_name(*p)) for i, p in enumerate(rl.policies(ExperimentConfig()))]"
fi

say "7. submit?"
read -r -p "   type yes to send it to Slurm: " ANSWER
[ "$ANSWER" = "yes" ] || { echo "   nothing submitted"; exit 0; }

SUBMITTED=$(uv run scripts/submit.py --arms "$ARMS" --out "$OUT" --keep-path --minutes "$MINUTES" -- --wind.path "$RECORD" | tee /dev/stderr)
JOBS=$(grep -o 'Submitted batch job [0-9]*' <<< "$SUBMITTED" | awk '{print $4}' | paste -sd: -)
[ -n "$JOBS" ] || { echo "   no job ids came back from sbatch"; exit 1; }
STAMP=${OUT##*rows-}
mkdir -p results/logs policies "$OUT"

if [ "$RL" = "yes" ]; then
  say "   the learned policies: one training job each, then one evaluation job each once its training is done"
  # aftercorr: evaluation task i starts when training task i has finished, whatever the others are doing
  cat > "results/rl-train-$STAMP.sh" <<JOB
#!/bin/bash
#SBATCH --job-name=balloon-rl-train
#SBATCH --partition=${PARTITION:-dean}
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=720
#SBATCH --array=0-$((N_POLICIES - 1))
#SBATCH --output=$PWD/results/logs/rl-train-$STAMP-%a.out
set -euo pipefail
export PATH=\$HOME/.local/bin:\$PATH
cd $PWD
uv run scripts/train_rl.py --index \$SLURM_ARRAY_TASK_ID --record $RECORD --out policies
JOB
  TRAIN=$(sbatch --parsable "results/rl-train-$STAMP.sh")
  echo "   training: job $TRAIN"
  cat > "results/rl-eval-$STAMP.sh" <<JOB
#!/bin/bash
#SBATCH --job-name=balloon-rl-eval
#SBATCH --partition=${PARTITION:-dean}
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=240
#SBATCH --array=0-$((N_POLICIES - 1))
#SBATCH --dependency=aftercorr:$TRAIN
#SBATCH --output=$PWD/results/logs/rl-eval-$STAMP-%a.out
set -euo pipefail
export PATH=\$HOME/.local/bin:\$PATH
cd $PWD
uv run scripts/eval_rl.py --index \$SLURM_ARRAY_TASK_ID --policies policies --out-dir $OUT
JOB
  EVAL=$(sbatch --parsable "results/rl-eval-$STAMP.sh")
  echo "   evaluation: job $EVAL, each task waiting on its own policy"
  JOBS="$JOBS:$TRAIN:$EVAL"
fi

FLIGHT_ONLY=no
if [[ ",$ARMS," == *",flight,"* ]]; then FLIGHT_ONLY=yes; fi
say "8. the analysis, queued to start when everything above has finished"
# afterany, not afterok: a cell that fails loses one row, and the analysis should still run on the rest.
# It joins this sweep's rows into one file (the one to commit), adds the earlier sweeps' rows beside them in
# a second file for reading, summarises and plots every arm, draws the flight maps and the training curves.
cat > "results/analyse-$STAMP.sh" <<JOB
#!/bin/bash
#SBATCH --job-name=balloon-analyse
#SBATCH --partition=${PARTITION:-dean}
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=240
#SBATCH --dependency=afterany:$JOBS
#SBATCH --output=$PWD/results/logs/analyse-$STAMP.out
set -uo pipefail
export PATH=\$HOME/.local/bin:\$PATH
cd $PWD
cat $OUT/*.jsonl > $OUT.jsonl
echo "\$(wc -l < $OUT.jsonl) rows: \$(wc -l < results/cells.txt) sweep cells, and the learned policies' flights"
cat $EARLIER $OUT.jsonl > results/rows-$STAMP-with-earlier.jsonl
FLIGHT=$FLIGHT_ONLY ./analyse_all.sh results/rows-$STAMP-with-earlier.jsonl $STAMP
if [ "$RL" = "yes" ]; then uv run scripts/plot_rl.py policies --into figures/$STAMP-rl; fi
JOB
sbatch "results/analyse-$STAMP.sh"

cat <<NOTE

Submitted. Nothing else needs running: the analysis job waits for all of it, then writes
    results/summary-$STAMP-*.txt     one per arm (RL rows are in the arms they flew)
    figures/$STAMP-*/                one folder per arm
    figures/$STAMP-maps/             flight maps for every new arm
    figures/$STAMP-rl/               the policies' training curves: look at these first
    results/logs/analyse-$STAMP.out  what it printed

To watch it:
    squeue -u "\$USER"
    tail -f results/logs/analyse-$STAMP.out

When the analysis job is gone from squeue, commit what it made:
    git add -f $OUT.jsonl policies results/summary-$STAMP-*.txt figures/$STAMP* && git commit -m "The $STAMP sweep" && git push
NOTE
