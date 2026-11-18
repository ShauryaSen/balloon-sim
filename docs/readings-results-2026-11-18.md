# Readings on: how much of the forecast's loss the balloon's own readings win back (2026-11-18)

MPC plans on the real day-old forecast (the July 2021 record's, the truth 24 h earlier) and flies station keeping
in the truth, 32 launches, as in the fidelity sweep. Readings are the true wind where the balloon is, one used
every 18 min, each assumed off by 0.5 m/s (`belief.KernelBelief`). Control lost as in decision 55.

| MPC plans on | time within 50 km | control lost [95%] |
|---|---|---|
| the true wind | 0.438 | 0 |
| the forecast, readings set aside | 0.153 | 0.80 [0.71, 0.88] |
| the forecast corrected with the assumed kernel | 0.267 | 0.48 [0.35, 0.59] |
| the forecast corrected with the measured K | 0.293 | 0.41 [0.27, 0.54] |
| coast | 0.081 | 1 |

- **Readings win back about half of what the forecast's error costs** (0.80 to 0.41-0.48). Measured.
- **The measured K is ahead of the assumed kernel, but not resolved:** +0.026 time within 50 km, 95% interval
  [-0.015, +0.070] over the launches, better on 15 of 32.
- The forecast without readings reproduces the 11-17 sweep's persistence copy exactly (0.80).

Rows: `results/rows-readings-2026-11-18.jsonl`. The measured K: `docs/error-covariance-2026-11-18.md`.
