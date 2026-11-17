# Directions after the lab talk (2026-11-13) and the team meeting (2026-11-14)

Points to come back to. Tags: **(measured)** our data; **(cited)** checked against the source; **(assumed)** a
judgement, unverified.

## Two directions, one codebase

- **Benchmark:** a gymnasium-style environment, modular, well visualised, that others can maintain. Rohan
  researches the style of the RLC benchmark papers this week.
- **Fidelity experiment:** which wind-model properties matter for control. Shaurya runs it this week.
- They are layers, not rivals: the ablation needs a clean configurable environment anyway, so the benchmark is
  the base and the ablation is the first paper on it (assumed). Who builds the environment and API is to settle
  with Vinamr.

## Cheapest generator against best generator

- The fidelity experiment says where fidelity matters: on a limited budget, spend it there.
- The benchmark wants the best generator, not the cheapest that still transfers. The fidelity results still say
  which properties to get right first and check hardest.

## Evaluating the generator is a different experiment

- Fidelity: the controller is built on copies of ERA5, each wrong in one property. Asks which properties matter.
- Generator evaluation: the controller is built on generated wind and flown in held-out ERA5, against the same
  controller built on ERA5 itself and on BLE. Same score, different question: how good is this generator.

## Balloon physics is its own axis

- Ours is kinematic: the balloon moves with the wind, altitude changes within a cap per move.
- Real balloons have limited climb rates, power and a day-night cycle, and do not track the wind perfectly.
- For the benchmark, make physics a separate axis from the wind; look at what BLE models. Kept out of the
  fidelity runs, which vary the wind only.

## Is ERA5 good enough as the gold standard?

- Not as reality: about 3.5 m/s off real balloon winds (Cadiou et al. 2026), effective resolution about 600 km
  (Bolgiani et al. 2022), and it misses opposing-wind pairs that finer data shows (Schuler et al. 2025).
- Yes as the best global wind data there is: complete, consistent, assimilates most observations, and what BLE
  and Loon trained on.
- Framing for the paper: transfer to ERA5 is the measured target. A check against real balloon flight data
  would strengthen it; Loon flight data may be public (unverified).

## The forecast-truth split in the benchmark

- Truth (what the balloon flies in) must be realistic: speeds, structure, change with altitude and time. The
  generator's job.
- Forecast (what the controller sees) must be wrong the way real forecasts are, in size and in structure.
- The guarantee needed is on the pair: a realistic truth with an invented error teaches the wrong lesson about
  how far to trust the forecast. Sarah, 2026-10-09: two random variables, jointly distributed.
- Calibrate on real forecast-analysis pairs (ECMWF forecasts against ERA5; WeatherBench 2 may provide them,
  unverified). The two-stage generator maps onto it: coarse stage as forecast, fine stage as truth.

## Why a generator over ERA5 at all (the paper must argue this)

- ERA5 is one history of the weather: RL on it can memorise days. A generator gives unlimited new episodes.
- A clean test set: train on generated wind, test on held-out ERA5 the controller never saw.
- Control over sampling: a season, a region, rare events.
- Forecast-truth pairs in any number.
- Size: full ERA5 is enormous; the model's weights are a few GB.
- The experiment that would prove it: PPO trained on 1, 2 and 4 years of ERA5 against PPO trained on the
  generator, all tested on held-out ERA5. If the generator matches or beats limited ERA5, the case is made.

## From the talk's questions

- **What a controller observes** should match real sensors. Loon balloons measure the wind only where they are
  and get a forecast for the rest, corrected by a GP from their readings (Bellemare et al. 2020). Ours matches:
  one reading per move, the corrected forecast for planning.
- **What it needs as input is local.** At 60 h of planning and about 10 m/s the balloon can reach about 2,000
  km (assumed). MPC already reads only where its candidate paths go; for PPO, column against patch against
  look-ahead (DESIGN 52) is a nuisance knob, tuned before PPO is compared.
- **Why readings help station keeping (Myles) but not navigation (us).** A reading informs the wind within
  about 544 km and 9 h of where it was taken (the fitted kernel, measured). Holding a point, every reading is
  about the wind flown next. Travelling about 100 km a move, the balloon leaves that zone in a few moves and
  its plan depends on wind 1,000 km ahead. And a reading informs only nearby altitudes (1.3 km), while
  steering means choosing other altitudes.
- **Sarah's point:** our kernel makes correlation depend on straight-line distance only. Real forecast errors
  probably travel with the wind, so a reading now says something about the wind downstream later. A
  flow-aware or learned kernel could make readings useful further away.
- **The catch:** our forecast error is drawn from the same stationary kernel the belief uses, so in our
  simulator the error is local by construction and no better kernel can help. Testing Sarah's idea needs real,
  structured error, such as the persistence forecast (truth 24 h earlier) that the fidelity experiment adds.
- **Follow-ups:** readings on against off on station keeping (not yet run; the 2,000 km results are reach);
  measure whether forecast error correlates further along the wind than across it.
- **Error sizes should be read in real units:** multiples of a day-old forecast's error (1x = 3.2 m/s,
  measured; Cadiou et al. 2026 give 3.2 to 3.8 m/s against balloons). For steering, the anchor is the
  difference between neighbouring altitudes, 4.9 m/s RMS on the 6-level record (measured): one forecast error
  is about two thirds of it.
