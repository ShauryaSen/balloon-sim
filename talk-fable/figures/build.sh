#!/bin/sh
# Rebuild every figure in the talk. Run from anywhere; takes about two minutes, most of it the film.
cd "$(dirname "$0")/../.." || exit 1
for script in realism winds results flight; do
  uv run python talk-fable/figures/$script.py || exit 1
done
