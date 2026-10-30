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

# name, then the flags that pick the arm out
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
)

for ARM in "${ARMS[@]}"; do
  NAME=${ARM%%|*}
  FLAGS=${ARM#*|}
  if [ "$NAME" = "main" ]; then INTO="figures/$STAMP"; else INTO="figures/$STAMP-$NAME"; fi
  printf '\n\033[1m== %s\033[0m\n' "$NAME"
  # an arm that was not run is reported and skipped, so one missing arm does not stop the rest
  if ! $RUN scripts/summarise.py "$ROWS" $FLAGS > "results/summary-$STAMP-$NAME.txt" 2>&1; then
    echo "   no rows in this arm; skipped"
    rm -f "results/summary-$STAMP-$NAME.txt"
    continue
  fi
  head -2 "results/summary-$STAMP-$NAME.txt"
  $RUN scripts/plot.py "$ROWS" $FLAGS --into "$INTO" | tail -1
done

printf '\n\033[1m== flight maps\033[0m\n'
# these re-fly a seed rather than reading the rows, so they need the record; without it, say so and carry on
RECORD=$($PYTHON -c "from balloon_sim.config import WindConfig; print(WindConfig().path)")
if [ -f "$RECORD" ]; then
  for SEED in 0 3; do
    $RUN scripts/map_flight.py --seed "$SEED" --sigma 3.2 --into "figures/$STAMP" | head -1
  done
else
  echo "   no record at $RECORD; the maps need it, everything else above does not"
fi

printf '\n\033[1mdone\033[0m\n  summaries: results/summary-%s-*.txt\n  figures:   figures/%s*\n' "$STAMP" "$STAMP"
