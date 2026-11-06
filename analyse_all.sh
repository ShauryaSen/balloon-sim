#!/usr/bin/env bash
# Every arm of a sweep, summarised and plotted, in one pass.
#
#     ./analyse_all.sh results/rows-2026-10-30.jsonl 2026-10-30
#
# The arms overlap in every setting but the one each of them moves, so a reader that is not told which arm to
# read silently mixes them and the later cells overwrite the earlier ones. Every call below therefore names
# its arm in full. Each writes a summary under results/ and its figures under figures/<stamp>-<arm>/, and the
# main arm keeps the bare stamp, since that is the one the headline figures are copied from.
set -euo pipefail

RUN=${RUN:-uv run}           # how a script is started here
PYTHON=${PYTHON:-uv run python}  # and how a one-liner is; both are set when uv is not what runs python
ROWS=${1:?usage: ./analyse_all.sh <rows.jsonl> [stamp]}
STAMP=${2:-$(date +%F)}
LONG=$($PYTHON -c "from balloon_sim.config import ExperimentConfig; print(max(ExperimentConfig().episode_lengths))")
FAR=$($PYTHON -c "from balloon_sim.config import ExperimentConfig; print(f'{max(ExperimentConfig().goal_distances):g}')")
FLEETS=$($PYTHON -c "from balloon_sim.config import ExperimentConfig; print(*ExperimentConfig().fleet_sizes)")

# name, then the flags that pick the arm out, then (after a second |) flags for the summary alone. The new
# arms compare on cost_km, the objective itself: summed_km there is distance to the objective's point, which
# is not what any of them is scored on
ARMS=(
  "main|"
  "1000km|--goal-km 1000"
  "${FAR}km|--goal-km $FAR"
  "belief05|--belief-scale 0.5"
  "belief2|--belief-scale 2.0"
  "nested|--nested"
  "long|--n-moves $LONG"
  "long-${FAR}km|--n-moves $LONG --goal-km $FAR"
  "escape|--objective escape"
  "station|--objective station|--metric cost_km"
  "targets|--objective targets|--metric cost_km"
)
for N in $FLEETS; do
  for OBJECTIVE in reach escape station targets; do
    ARMS+=("fleet$N-$OBJECTIVE|--objective $OBJECTIVE --n-agents $N|--metric cost_km")
  done
done

for ARM in "${ARMS[@]}"; do
  NAME=${ARM%%|*}
  REST=${ARM#*|}
  FLAGS=${REST%%|*}
  ONLY=""
  if [[ "$REST" == *"|"* ]]; then ONLY=${REST#*|}; fi
  if [ "$NAME" = "main" ]; then INTO="figures/$STAMP"; else INTO="figures/$STAMP-$NAME"; fi
  printf '\n\033[1m== %s\033[0m\n' "$NAME"
  # an arm that was not run is reported and skipped, so one missing arm does not stop the rest
  if ! $RUN scripts/summarise.py "$ROWS" $FLAGS $ONLY > "results/summary-$STAMP-$NAME.txt" 2>&1; then
    echo "   no rows in this arm; skipped"
    rm -f "results/summary-$STAMP-$NAME.txt"
    continue
  fi
  head -2 "results/summary-$STAMP-$NAME.txt"
  # a figure that fails is reported and the rest go on: the escape arm's score figure is known to divide
  # by zero (a guard written for positive costs), and that must not stop every arm after it
  if ! $RUN scripts/plot.py "$ROWS" $FLAGS --into "$INTO" > "results/plot-$STAMP-$NAME.log" 2>&1; then
    echo "   plot.py failed on this arm; the figures before the failure were written. See results/plot-$STAMP-$NAME.log"
  else
    tail -1 "results/plot-$STAMP-$NAME.log"
  fi
done

printf '\n\033[1m== flight maps\033[0m\n'
# drawn from the rows' own saved trajectories, so these need no wind record and cannot disagree with the
# sweep: they are the sweep's flights. scripts/map_flight.py re-flies a seed instead, for a cell nobody ran.
for SEED in 9 3 0; do
  $RUN scripts/map_rows.py "$ROWS" --seed "$SEED" --sigma 3.2 --into "figures/$STAMP" 2>/dev/null | head -1 \
    || echo "   no main-arm rows for seed $SEED"
done
# and every fleet.py arm: the new objectives with one balloon, then every objective with each fleet
MAPS=("station 1" "targets 1")
for N in $FLEETS; do for OBJECTIVE in reach escape station targets; do MAPS+=("$OBJECTIVE $N"); done; done
for MAP in "${MAPS[@]}"; do
  read -r OBJECTIVE N <<< "$MAP"
  for SEED in 9 3 0; do
    $RUN scripts/map_fleet.py "$ROWS" --objective "$OBJECTIVE" --n-agents "$N" --seed "$SEED" --sigma 3.2 \
      --into "figures/$STAMP-maps" 2>/dev/null | head -1 || echo "   no $OBJECTIVE rows for $N balloon(s) at seed $SEED"
  done
done

printf '\n\033[1mdone\033[0m\n  summaries: results/summary-%s-*.txt\n  figures:   figures/%s*\n' "$STAMP" "$STAMP"
