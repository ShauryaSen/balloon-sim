# The fidelity-to-transfer experiment: locked design

Supersedes `fidelity-variables-2026-11-06.md` after a literature pass and measurements on the record itself.
Tags: **(measured)** our own data; **(cited)** checked against the paper, with how much of it was read;
**(assumed)** a judgement, unverified.

## The question

Which properties of a wind model must be right for a controller built with it to perform in the real wind, and
which can be made coarse, and so cheap, without losing performance?

**Method.** Each simulator is a copy of the true wind degraded along one property at a time. MPC plans with the
simulator and flies in the truth. The loss is MPC's cost minus hindsight's, paired over 32 seeds, with 95%
intervals, as in every sweep so far. Degrading the truth is what makes the ablation clean: no real generator is
wrong in exactly one property.

## Truth: rebuild the record at 18 levels

The current record has 6 levels, 1.1 km apart. The ERA5 store on Unicorn holds **18 levels (ERA5 model levels
49–66, about 0.3 km apart) at hourly resolution** (measured). A 6-level truth cannot show what is lost against
finer vertical structure, and that is the axis most likely to matter:

- Altitude-only balloons steer on opposing winds at different altitudes (Bellemare et al. 2020, full text).
- ERA5 pressure levels count far fewer opposing-wind cases than a 46-level radiosonde model does: 73% of
  October forecasts had zero opposing pairs on ERA5 levels, against 38% at the finer resolution (Schuler et al.
  2025, full text). That paper confounds data source with resolution, which this experiment untangles.

Keep **3 h** frames: going from 3-hourly to hourly ERA5 changes the stratospheric wind spectrum little, while 6-hourly
changes it substantially (Podglajen et al. 2020, page text). Keep the **2° grid**: the store is 1.8°, and ERA5's
effective resolution is already coarser than that, at about 600 km mid-latitude at 500 hPa (Bolgiani et al. 2022,
full text; extrapolated to the stratosphere, assumed).

**Scope this sets.** "Reality" here is ERA5, which is itself several m/s off real balloon winds, with structured
error: 3.8 and 3.2 m/s standard deviation zonal and meridional against Strateole-2 balloons (Cadiou et al. 2026,
page text). Transfer is to ERA5, and fine scales below about 2° cannot be tested with this truth.

**Consequence.** An 18-level truth may have more steering room, so the task is re-scouted with the existing rule
(`scouting-rule.md`) before the sweep, and the kernel refitted.

## The five properties and their ladders

Every ladder starts at the truth and runs past the wind's own correlation scale along that axis. On the July record
in 20N–50N (measured):

- **Time:** an anomaly e-folds in 31 h.
- **Horizontal:** an anomaly e-folds in about 1,050 km along longitude.
- **Vertical:** the full wind correlates 0.92 between levels 1.1 km apart and 0.31 between levels 5.4 km apart.

1. **Vertical resolution.** Keep a subset of levels and interpolate linearly between them.
   - Ladder: 18 (truth), 9, 6, 3, 2, and 1 as the no-control floor.
   - **Always keep both ends of the band, evenly spaced between.** Which levels are kept matters as much as how
     many: on the 6-level record, 2 levels at the ends give 3.2 m/s of error, and 2 adjacent levels in the middle
     give 5.5 (measured). Dropping the ends would test range, not resolution.
   - 6 levels is the current record and close to ERA5's own pressure levels in the band, so the ladder includes
     what the field uses today.
   - Generator knob: the number of levels generated.
2. **Frame interval.** Keep every k-th frame and interpolate linearly in time.
   - Ladder: 3 h (truth), 6 h, 12 h, 24 h, and a field frozen at launch.
   - Subsampling, not smoothing, because a generator that outputs a frame every k hours is exactly this.
   - **Caveat to report:** the wind correlates slightly more at 24 h than at 12 h (0.58 against 0.56, measured),
     which suggests a daily cycle. A 24 h interval samples it at one phase and makes it look frozen.
   - Generator knob: frames generated per day. cBottle generates independent snapshots (Brenowitz et al. 2025,
     partly read); GenCast steps at 12 h (Price et al. 2025, abstract).
3. **Horizontal resolution.** Keep every k-th grid point and interpolate linearly.
   - Ladder: 2° (truth), 4°, 8°, 16°, and a column: the wind at the balloon's current position, held constant
     in space.
   - The column is the model Pasetsky et al. found beat planning on the full forecast, once corrected online (HTML
     read).
   - Generator knob: the grid size.
4. **Amplitude of variability.** Scale each point's departure from its 10-day mean by α.
   - Ladder: α = 1 (truth), 0.75, 0.5, 0.25, 0 (the 10-day mean flow). This is not a true climatology; it
     covers only 10 days.
   - ERA5 already underestimates stratospheric variability against balloons (Podglajen et al. 2020).
   - Generator knob: calibration of the generated spread.
5. **Error structure at equal size.** Four kinds of error, all at 3.2 m/s RMS per component: the record's fitted
   day-old forecast error, and close to the 3.5 m/s BLE calibrates its noise to (BLE source code) and the 3.2–3.8
   m/s measured against balloons (Cadiou et al.).

   |  | **same at every altitude** | **varies with altitude** |
   |---|---|---|
   | **systematic (bias)** | uniform offset | offset varying linearly from the bottom of the band to the top |
   | **random** | correlated field with infinite vertical lengthscale | correlated field with the fitted 1.3 km vertical lengthscale (the current noise) |

   The left column leaves the differences between levels untouched and the right column corrupts them, at equal
   total error. A uniform bias of 4.5 m/s changes the level differences by exactly 0, against 2.65 m/s for an
   altitude-varying bias of the same size (measured). This is the sharpest test of the central hypothesis.

   Cross this property, and only this one, with MPC learning from its readings on and off. Control theory says
   persistent bias is removable when the planner estimates it online (Pannocchia & Rawlings 2003; Maeder et al.
   2009, abstracts). Our belief's 544 km lengthscale probably cannot learn a uniform bias from point readings
   (assumed), so this is a real question, not a formality.

## The central hypothesis, and the single test that decides it

**An altitude-only balloon steers on the differences between levels, so control loss tracks the error in those
differences, not the error in the wind itself.**

For each condition, measure two errors against the truth, before flying anything:

- the **wind error**: RMS of the wind itself;
- the **level-difference error**: RMS of the change in the differences between adjacent levels.

They rank the conditions differently. A uniform bias has 4.5 m/s wind error and 0 level-difference error. Halving
the variability has 2.5 and 1.8. Coarsening to 8° has 2.7 and 2.7 (measured, 6-level record).

**Pre-registered primary test:** across all conditions, the rank correlation of control loss with each error
measure. If level-difference error wins, it is a fidelity score for balloon wind models that needs no flights.
If wind error wins, plain accuracy is enough. If neither does, the per-property curves rank what matters. Each
outcome is a result, and it is one test, so the many conditions do not multiply false positives.

## Fixed, and why

- **Controller:** MPC at horizon 20, on the model's mean, with no draws. Longer horizons, draws and readings never
  helped MPC (measured, three sweeps). Readings come back only in property 5.
- **Reference:** the fixed plan in the same simulator, as the controller that cannot react.
- **Ceiling:** hindsight in the truth, the best search on the true wind per seed (DESIGN 36).
- **Task:** station keeping (DESIGN 47) as the primary task. It is the task Loon flew and the field scores, as time
  within a radius of a fixed point (Bellemare et al. 2020; Pasetsky et al. 2025), so results compare directly to
  prior work, and holding a point needs opposing winds at different altitudes, which is exactly what the
  vertical properties degrade. Reach (re-scouted on the 18-level truth with the existing rule) and escape
  (DESIGN 45) as secondary tasks, to show the ranking is not one task's.
- **Window:** July 2021, launches 20N to 50N. Repeat on 2021-01-16, a winter window that avoids the store's
  missing days 8–14, at one level per property, to show the ranking is not one week's weather.
- **Seeds:** 32, paired across every condition.
- **One interaction arm:** frame interval × horizon (10, 20, 50), because coarse time should stop long horizons
  from paying.

## Not varied, and why

- **Forecast noise, draws, readings, horizon past 20:** already measured.
- **Detail finer than 2° or 3 h:** cannot be removed from a truth that does not contain it.
- **Direction error versus speed error:** a candidate (Bellemare reports heading errors over 90° are frequent), but
  second-tier. Kept for a follow-up.

## Cost

About 21 conditions × 2 controllers × 32 seeds ≈ 1,350 cells. Add 256 for the interaction arm, 128 for the
readings cross, and about 320 to re-scout the task: roughly 2,000 cells. At the measured 12 to 22 s per cell, about
10 to 15 core-hours, an hour or two on Unicorn. The 18-level record is about three times larger, so check cell
memory on a smoke run first.

## How it informs the generator

Each ladder is a generator cost knob: levels generated, frames per day, grid size, spread calibration. The point
where loss starts on each ladder is how coarse the generator can afford to be on that axis, and the primary test
says whether "good for control" can be scored without flying.

## Novelty, hedged

No study found, in a search of balloon navigation, aviation and ship routing, UAV control and generative weather
models, degrades a wind model one property at a time and measures closed-loop control. The closest are Stohl et al.
1995 (degraded NWP winds, ranked temporal over horizontal over vertical, but measured trajectory error, not control;
abstract) and Xie et al. 2021 (one-parameter-at-a-time simulator ablation, in robotics; abstract). Say "to our
search", not "first".

## Sources

Bellemare et al. 2020, Nature 588:77, doi:10.1038/s41586-020-2939-8 · Schuler et al. 2025, arXiv:2502.05014 ·
Pasetsky et al. 2025, arXiv:2511.07761 · Greaves et al., Balloon Learning Environment, github.com/google/balloon-learning-environment ·
Podglajen et al. 2020, ACP 20:9331 · Cadiou et al. 2026, ACP 26:1665 · Bolgiani et al. 2022, Clim. Dyn. 59:737 ·
Nastrom & Gage 1985, J. Atmos. Sci. 42:950 · Stohl et al. 1995, J. Appl. Meteor. 34:2149 · Xie et al. 2021,
arXiv:2011.02404 · Pannocchia & Rawlings 2003, AIChE J. 49:426 · Maeder, Borrelli & Morari 2009, Automatica 45:2214 ·
Brenowitz et al. 2025 (cBottle), arXiv:2505.06474 · Price et al. 2025 (GenCast), Nature 637 · Lambert et al. 2020
(objective mismatch), PMLR 120:761 · Truong et al. 2022, arXiv:2207.10821
