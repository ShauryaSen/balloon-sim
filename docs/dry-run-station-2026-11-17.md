# Station-keeping dry run on Myles's setup, 2026-11-17

`scripts/dry_run_station.py`, Unicorn job 762239, rows in `results/dry-run-station-2026-11-17.jsonl`.
Setup (DESIGN 54): an action every 3 min, 12 h horizon, replan every 72 min, 100 starting plans, 2-day episodes,
TWR50, launches 10S to 10N, on the 6-level July 2021 record. 8 seeds, paired. Intervals are 95% t-intervals
over the 8 paired differences.

| run | TWR50 mean (sd) | cost mean | seconds per flight |
|---|---|---|---|
| coast | 0.134 (0.135) | 243k km | 3 |
| MPC on the truth, L-BFGS | 0.476 (0.285) | 35k km | 85 |
| MPC on the truth, normalised descent | 0.476 (0.283) | 35k km | 16 |
| MPC on the truth 24 h earlier, L-BFGS | 0.339 (0.182) | 55k km | 86 |
| MPC on the truth 24 h earlier, normalised | 0.353 (0.169) | 51k km | 30 |

## What it shows (measured, 8 seeds)

- **The setup leaves room.** MPC on the truth holds the station +0.34 ± 0.17 of the time more than coast.
- **Normalised descent equals L-BFGS and is 5 times faster.** TWR50 difference +0.001 ± 0.014 on the truth,
  +0.014 ± 0.059 on persistence; 16 s against 85 s a flight. Myles's optimizer becomes the default here.
- **A real day-old error costs some control, not yet detected.** Persistence against the truth: +0.12 ± 0.16
  TWR50. Eight seeds cannot resolve a drop this size; the sweep's 32 should.
- **The error is larger in the tropics.** Persistence error RMS 4.89 m/s in 10S to 10N, against 3.16 in 20N to
  50N, so 1x means a different size in each band.
- **Cost-based share of control lost compresses.** Coast drifts far off (243k km summed), so the denominator is
  huge and a large TWR50 drop shows as a small share: seed 5 drops from 0.70 to 0.36 TWR50, a share of 0.06 by
  cost. Per seed, the TWR50 denominator can be near zero (seed 7: 0.078 against 0.068).
- One seed (1) flies better on persistence than on the truth: a 12 h horizon is not optimal, so MPC on the
  truth is a strong reference, not a ceiling.

## Proposed from it

- Score on TWR50, the field's metric, as a ratio of means over seeds:
  (TWR_truth − TWR_copy) / (TWR_truth − TWR_coast), with a paired bootstrap interval over seeds. Stable where a
  per-seed ratio is not, and it reads in the units Myles and BLE report.
- A copy is planned on as a plain field (`belief.Truth(copy)`), not through the GP: readings are off, and the
  GP's conditioning step is what made the persistence runs about twice as slow.
