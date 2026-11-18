# Fidelity sweep, station keeping, 2026-11-17: results

Unicorn job 762483, 1,344 flights, rows in `results/rows-fidelity-2026-11-17.jsonl`, scored by
`scripts/summarise_fidelity.py`. Setup: DESIGN 54-55. Truth: ERA5 2021-07-16, 240 h hourly, 18 levels
(14.3-20.2 km), 30S-30N at 2°; launches 10S-10N; 2-day station keeping, TWR50; MPC with Myles's setup
(3 min actions, 12 h horizon, replan every 72 min, normalised descent), readings off. 32 paired seeds.
1x error = persistence RMS in the launch band, 5.01 m/s.

Score = share of MPC's time within 50 km a copy costs, against coast: 0 nothing lost, 1 no better than never
steering. Ratio of means over seeds, 95% paired bootstrap interval.

References (TWR50, mean and sd over 32 seeds): MPC on the truth 0.438 (0.285); at twice the climb limit 0.546
(0.283); with a 6 h horizon 0.355 (0.272); with 24 h 0.486 (0.296). Coast 0.081 (0.068). Median 15 s a flight.

## What it shows (measured, 32 seeds; intervals in the table)

- **The pre-registered primary test goes against the hypothesis.** Across the 21 base copies, control loss
  follows the plain wind error (rank correlation +0.90), not the level-difference error (+0.39); the gap,
  -0.49 [-1.01, -0.04], excludes zero. For station keeping, a wind model has to get the wind itself right.
- **The sharpest case: a uniform offset leaves the level differences exactly as they are and still costs 0.90
  [0.79, 1.00] of control at 1x.** An offset ramping with altitude, same size, costs less: 0.70 [0.58, 0.79].
  Random error the same at every altitude costs 0.67, varying with altitude 0.79.
- **A real day-old forecast costs most of the control:** persistence 0.80 [0.71, 0.88]. Every error shape at
  1x costs 0.67 to 0.90.
- **Levels:** 10 and 9 levels lose about a tenth (0.08, 0.10); 6 lose a third (0.33); 3 lose three quarters
  (0.75); 2 and 1 lose it all. Twice the climb limit changes none of this (0.05, 0.11, 0.27, 0.79, 0.95,
  0.97), so the loss is what the balloon knows, not how far it can climb. BLE's 10 levels are nearly enough;
  6, about ERA5's pressure levels in the band, are not.
- **Frames:** 3 h costs nothing detectable (0.03 [-0.01, 0.07]), 6 h and 12 h a little (0.07, 0.09), 24 h a lot
  (0.42), frozen most (0.72).
- **Grid:** 4° 0.12, 8° 0.31, 16° 0.60.
- **Horizon × frames, as predicted:** at a 24 h horizon, 12 h frames cost 0.27 [0.17, 0.36], against 0.09 at
  12 h and 0.01 at 6 h. A coarse-in-time model stops a long horizon from paying.

## Reading it (believed, unverified)

- Holding a point means predicting where the balloon will be, and any error in the wind, uniform or not,
  moves every predicted position: 5 m/s over a 12 h horizon is about 200 km, four station radii. Steering
  between directions, as in reach, may still run on level differences; the secondary tasks test that.
- The ramp hurts less than the uniform offset because it averages to zero over the band, so a balloon that
  ranges over the levels sees less net bias (not checked).
- Myles found online GP correction from readings mattered most for station keeping. With every error shape
  costing most of the control here, readings should matter a lot in this setting: the next arm to run.

## Caveats

- One July week, one region (10S-10N), one task. The primary test's 21 points are not independent draws:
  the coarsening ladders move both errors together, so the shapes carry most of the contrast.
- MPC on the truth is a strong reference, not a ceiling (12 h horizon).

## Table

| arm | climb | horizon h | copy | size | score [95%] | TWR50 drop | wind err | level-diff err | n |
|---|---|---|---|---|---|---|---|---|---|
| base | 1 | 12 | frames:12 | 0 | +0.09 [+0.02, +0.16] | +0.033 | 1.24 | 0.61 | 32 |
| base | 1 | 12 | frames:24 | 0 | +0.42 [+0.25, +0.59] | +0.149 | 2.13 | 1.06 | 32 |
| base | 1 | 12 | frames:3 | 0 | +0.03 [-0.01, +0.07] | +0.010 | 0.31 | 0.14 | 32 |
| base | 1 | 12 | frames:6 | 0 | +0.07 [+0.02, +0.11] | +0.024 | 0.58 | 0.28 | 32 |
| base | 1 | 12 | frames:frozen | 0 | +0.72 [+0.60, +0.83] | +0.257 | 6.40 | 2.55 | 32 |
| base | 1 | 12 | grid:2 | 0 | +0.12 [+0.03, +0.23] | +0.042 | 0.76 | 0.46 | 32 |
| base | 1 | 12 | grid:4 | 0 | +0.31 [+0.16, +0.49] | +0.111 | 1.93 | 0.97 | 32 |
| base | 1 | 12 | grid:8 | 0 | +0.60 [+0.42, +0.77] | +0.214 | 4.17 | 1.59 | 32 |
| base | 1 | 12 | levels:1 | 0 | +0.95 [+0.88, +1.00] | +0.338 | 6.62 | 2.24 | 32 |
| base | 1 | 12 | levels:10 | 0 | +0.08 [+0.02, +0.15] | +0.029 | 0.50 | 0.73 | 32 |
| base | 1 | 12 | levels:2 | 0 | +0.92 [+0.80, +1.02] | +0.327 | 6.37 | 2.06 | 32 |
| base | 1 | 12 | levels:3 | 0 | +0.75 [+0.62, +0.88] | +0.269 | 3.45 | 1.73 | 32 |
| base | 1 | 12 | levels:6 | 0 | +0.33 [+0.21, +0.45] | +0.117 | 1.28 | 1.19 | 32 |
| base | 1 | 12 | levels:9 | 0 | +0.10 [+0.01, +0.19] | +0.035 | 0.65 | 0.84 | 32 |
| base | 1 | 12 | offset | 1 | +0.90 [+0.79, +1.00] | +0.321 | 5.01 | 0.00 | 32 |
| base | 1 | 12 | offset | 2 | +0.99 [+0.90, +1.07] | +0.354 | 10.02 | 0.00 | 32 |
| base | 1 | 12 | offset_altitude | 1 | +0.70 [+0.58, +0.79] | +0.249 | 5.01 | 0.97 | 32 |
| base | 1 | 12 | offset_altitude | 2 | +0.87 [+0.77, +0.95] | +0.312 | 10.02 | 1.93 | 32 |
| base | 1 | 12 | persistence | 0 | +0.80 [+0.71, +0.88] | +0.285 | 5.01 | 2.34 | 32 |
| base | 1 | 12 | random | 1 | +0.67 [+0.52, +0.81] | +0.240 | 5.02 | 0.00 | 32 |
| base | 1 | 12 | random_altitude | 1 | +0.79 [+0.68, +0.88] | +0.281 | 5.02 | 1.69 | 32 |
| climb | 2 | 12 | levels:1 | 0 | +0.97 [+0.92, +1.03] | +0.453 | 6.62 | 2.24 | 32 |
| climb | 2 | 12 | levels:10 | 0 | +0.05 [-0.02, +0.11] | +0.021 | 0.50 | 0.73 | 32 |
| climb | 2 | 12 | levels:2 | 0 | +0.95 [+0.87, +1.03] | +0.443 | 6.37 | 2.06 | 32 |
| climb | 2 | 12 | levels:3 | 0 | +0.79 [+0.69, +0.88] | +0.368 | 3.45 | 1.73 | 32 |
| climb | 2 | 12 | levels:6 | 0 | +0.27 [+0.18, +0.37] | +0.124 | 1.28 | 1.19 | 32 |
| climb | 2 | 12 | levels:9 | 0 | +0.11 [+0.04, +0.19] | +0.052 | 0.65 | 0.84 | 32 |
| horizon | 1 | 6 | frames:12 | 0 | +0.01 [-0.14, +0.14] | +0.004 | 1.24 | 0.61 | 32 |
| horizon | 1 | 6 | frames:24 | 0 | +0.28 [+0.07, +0.48] | +0.076 | 2.13 | 1.06 | 32 |
| horizon | 1 | 6 | frames:3 | 0 | +0.01 [-0.04, +0.06] | +0.003 | 0.31 | 0.14 | 32 |
| horizon | 1 | 6 | frames:6 | 0 | -0.04 [-0.16, +0.05] | -0.010 | 0.58 | 0.28 | 32 |
| horizon | 1 | 6 | frames:frozen | 0 | +0.71 [+0.54, +0.84] | +0.193 | 6.40 | 2.55 | 32 |
| horizon | 1 | 24 | frames:12 | 0 | +0.27 [+0.17, +0.36] | +0.108 | 1.24 | 0.61 | 32 |
| horizon | 1 | 24 | frames:24 | 0 | +0.55 [+0.42, +0.66] | +0.222 | 2.13 | 1.06 | 32 |
| horizon | 1 | 24 | frames:3 | 0 | +0.02 [-0.02, +0.06] | +0.010 | 0.31 | 0.14 | 32 |
| horizon | 1 | 24 | frames:6 | 0 | +0.06 [+0.01, +0.12] | +0.026 | 0.58 | 0.28 | 32 |
| horizon | 1 | 24 | frames:frozen | 0 | +0.81 [+0.72, +0.89] | +0.327 | 6.40 | 2.55 | 32 |

**Primary test** (base arm, one point per copy):

- rank correlation of the score with the wind error: +0.90
- rank correlation with the level-difference error: +0.39
- level-difference minus wind, bootstrap over copies: -0.49 [-1.01, -0.04]

coast TWR50 mean 0.081 over 32 seeds
