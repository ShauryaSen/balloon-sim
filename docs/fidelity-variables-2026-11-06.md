# What to vary in the fidelity-to-transfer experiment

For the meeting of 2026-11-06. Sarah asked for a short list of simulator properties to vary, each with a reason
to care and a hypothesis, chosen so the experiment gives a paper whatever it finds.

Every claim below is tagged: **(measured)** is our own sweep, 32 paired seeds; **(assumed)** is a judgement,
unverified; nothing here leans on literature not yet checked.

## The question, made operational

A controller is built with a wind model M and flown in the real wind W. Here W is ERA5 and M is a copy of ERA5
degraded along one property φ at a time. Transfer is how much worse the controller does in W than one built
with W itself, paired seed by seed: the same cost, hindsight ceiling and paired intervals as the noise sweep.

Degrading ERA5 is what makes the ablation possible. No generator is good on one property and bad on the rest,
but a copy of ERA5 coarsened along one axis is exactly that: wrong in φ and true in everything else.

## The central hypothesis

**An altitude-only balloon steers using the differences between levels, so what a simulator must get right is
how the wind changes with height over the next ~60 hours, not the absolute wind or its fine horizontal detail.**

The evidence that points there:

- Released cBottle, with two levels in the band, both at jet speed, left every controller drifting (measured,
  `cbottle-2026-10-21.md`).
- Only downwind goals are reachable, because the six ERA5 levels oppose each other at only 58% of points:
  the balloon can choose among the directions the column offers and nothing else (measured, DESIGN 44).
- Planning past 20 moves (60 h) never helped, at any noise level (measured, two sweeps).
- The balloon's own readings were worth nothing: one point per move says little about a field with a 544 km
  correlation length (measured, `results-2026-10-29.md`).

Each property below tests a piece of this. If it holds, the result is a design rule for wind generators. If it
fails, the experiment says which property matters instead. Either is a paper.

## The five properties

Each ladder is chosen to straddle the wind's own correlation length along that dimension. On the North Pacific
record that is 1,013 km horizontally, 3.5 km vertically and 58 h in time (measured, DESIGN 26), so each ladder
runs from finer than the wind varies to coarser.

| # | property φ | how ERA5 is degraded | levels | hypothesis | why a generator builder cares |
|---|---|---|---|---|---|
| 1 | **Vertical resolution** | keep a subset of the six levels, interpolate linearly between them | 6 (truth), 3, 2, 1 | the steepest loss of the five; 1 level removes all control | levels in the band are a generator design choice: cBottle-video outputs 2, ours 18 |
| 2 | **Time evolution** | keep every k-th frame, interpolate linearly in time; the last rung freezes the first frame | 3 h (truth), 12 h, 24 h, 48 h, frozen | a large loss once frames are coarser than about a day, since planning uses 60 h | generators make snapshots or short blocks; time-stitching is the open problem in ours |
| 3 | **Horizontal resolution** | block-average to a coarser grid, interpolate back | 2° (truth), 4°, 8°, 16° | little loss until the grid passes the 1,000 km wind scale | fine scales are where generators fall furthest short of ERA5 (our poster) |
| 4 | **Amplitude of variability** | shrink each point's departure from its 10-day mean by α | α = 1 (truth), 0.75, 0.5, 0 (climatology) | a graded loss, with climatology near the blind plan | generative models often under-represent variability and extremes (assumed) |
| 5 | **Systematic vs random error** | add one uniform offset, the same at every level, at the RMS of the existing noise ladder | 1.6, 3.2, 6.4 m/s, against the random-error ladder at the same RMS | **a uniform bias hurts less than random error of the same size**: it leaves the differences between levels intact, and replanning from the true position corrects the drift it causes | mean-state bias is the most common generator error; this says whether to fear it |

Property 5 is the sharpest test of the central hypothesis, because the prediction is counter-intuitive and
comes straight from it: by RMS error a bias is as bad as noise, but by the difference between levels it is
harmless.

## The analysis that makes it a paper either way

1. **Per property.** Transfer loss against the degradation level, paired over 32 seeds with 95% intervals.
2. **One curve or many.** For every condition, measure the model's plain RMS error against ERA5, and plot
   transfer loss against it, together with the existing random-noise ladder. If all properties fall on one
   curve, plain RMS error is a sufficient fidelity metric, which is a clean and useful negative. If they
   scatter, the experiment ranks properties by loss per m/s of error.
3. **A control-relevant metric.** Repeat 2 with the RMS error of the *differences between levels* instead of the
   wind itself. If that collapses the scatter where plain RMS does not, it is a fidelity metric for balloon
   control that a generator can be scored on without flying anything.

## Fixed, and why

Following the Tuning Playbook's split of knobs:

- **Scientific:** the property φ and its level.
- **Nuisance:** the seed, paired across every condition (32).
- **Fixed,** each from a measured result:
  - the task: 2,000 km downwind with the arrival rule, the only one of twenty scouted with room in it;
  - the window: ERA5 global, 2021-07-16, launches 20N to 50N;
  - MPC at horizon 20 on the model's mean, with no draws and no readings. Longer horizons never helped, draws
    never helped MPC, and readings were worth nothing, so each would add cost without changing what is
    measured.

**Controllers:** MPC planning in the degraded model, the fixed plan in the same model (a controller that cannot
react, as a reference), and hindsight in the truth (the ceiling, shared across all conditions).

**One interaction arm:** time evolution × horizon (10, 20, 50), because coarsening time is exactly what should
stop long horizons from paying.

**Not varied, and why:** noise magnitude, draws, readings and horizon past 20 are already measured. Detail finer
than the record's 2° and 3 h cannot be removed from a truth that does not contain it. Extremes are folded into
property 4.

## Cost

Five properties at 3 or 4 levels, two controllers, 32 seeds: about 1,000 cells, plus 192 for the interaction
arm. At the 2,000 km task's measured cell times (median 22 s for MPC at horizon 20, 12 s for the fixed plan,
from the 2026-10-24 rows) that is about 6 to 7 core-hours, well under an hour on Unicorn.

## What it sets up

- **A second window** (2021-01-16, winter, avoiding the missing days 8 to 14) at one level per property, to show
  the ranking is not one week's weather.
- **Learned policies:** the same degraded models serve unchanged as training environments for RL, the second
  half of the 09-05 plan. The property list carries over.
- **The generator:** the ranking says which properties it must get right, and the metric from analysis 3 scores
  it. It then joins the experiment as a real rung, with ERA5 and released cBottle, to check that the ablation's
  conclusions hold for actual generators.

## Decisions for Sarah

1. **Is degrading ERA5 an acceptable stand-in for simulator fidelity?** Recommended: yes, as the controlled
   half, with real generators as the check. The alternative, a ladder of real generators only, cannot say which
   property did the work.
2. **MPC only first, RL later?** Recommended: yes. The pipeline exists, and the degraded models carry over.
3. **Five properties, or cut to three?** If cut: keep vertical resolution, time evolution and systematic vs
   random error, the three that test the central hypothesis most directly.
