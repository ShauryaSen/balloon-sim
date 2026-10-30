# Headline figures, for the 2026-10-29 meeting

Copies from `figures/2026-10-24/` (goal 2,000 km downwind, 32 seeds), in the order to present them. Numbers
behind every claim: `docs/results-2026-10-29.md`. Everything here is on `cost_km`, the objective as the last
meeting redefined it: distance summed until the first time the balloon is within 200 km of the goal, nothing
counted after that. The old figures were on `summed_km`, which kept counting forever, so these are not
directly comparable to the 2026-10-23 set.

The task is harder than last time, and it was picked rather than chosen: twenty candidates scored against a
rule written down before the runs, one qualifier. `docs/scouting-rule.md` has the rule, the amendment it
needed, and the table. Crosswind, upwind and random goals are unreachable at every distance we tried: the
wind levels in this record oppose each other at only 58% of points, so an altitude-only balloon goes downwind
or nowhere.

1. **one_flight**: seed 3 at the real forecast error (3.2 m/s). All three ride the same easterly out; the
   difference is whether anything turns them toward the goal in time. Solid is the episode, faint is what was
   flown after the balloon was already inside the 200 km ring, which costs nothing and should not be read as
   part of the flight. One seed, an illustration. (`flight_seed3_sigma3.2.png`)

   The version of this figure made on 2026-10-29 was wrong and is replaced: `map_flight.py` was not passing
   the arrival radius, so its three flights were planned and scored on the plain sum, which is a different
   objective from the one the sweep flew. The sweep's own rows were never affected, only this map. Numbers
   quoted off the old one do not hold; `analyse_all.sh` redraws it.
2. **all_controllers**: the objective against forecast error at MPC horizon 20. Hindsight flat and lowest,
   MPC solid below the fixed plan dashed. The line that matters is the dotted one, MPC that replans but never
   learns: it sits on top of solid MPC the whole way. Bars are each line's own standard error, unpaired, so
   they overstate the uncertainty in the gaps. (`lines_cost_km_h20.png`)
3. **breaking_point**, the headline: MPC's excess over hindsight, a line per horizon, 95% paired intervals.
   Longer horizons help below about 3 m/s; past that the four lines sit on each other and the horizon stops
   buying anything. The breaking point moved in from 3.2-6.4 m/s on the old 1,000 km task to 1.6-3.2 here.
   (`horizons_cost_km_n0.png`)
4. **breaking_point_surface**: the same numbers as one surface over error and horizon; dots are measurements,
   the surface between them is interpolated, axes evenly spaced rather than to scale.
   (`surface_cost_km_n0.png`)
5. **share_of_gap_closed**: (fixed plan - MPC) / (fixed plan - hindsight), paired within each seed. About
   0.8 at 0.8-1.6 m/s, settling to 0.3-0.4 past 6 m/s. (`score_cost_km_h20.png`)
6. **every_seed**: the numbers of figure 2, one dot per seed: why the bars are wide. A handful of launches
   run to 150,000-300,000 km while the bulk sit under 60,000. (`dots_cost_km_h20.png`)
7. **what_replanning_is_worth**, new this round: MPC minus the fixed plan, both planning on the belief's
   mean, so the only difference is replanning. Zero at a perfect forecast and slightly positive at horizons
   5 and 10, which is myopia costing more than a wrong forecast. It falls to -559 km at 3.2 m/s and then
   comes back to about -200 km. Replanning pays most where the forecast is wrong enough to matter but not so
   wrong that what you learn is also wrong. (`replanning_cost_km_n0.png`)

Read 2 and 7 together: replanning is worth hundreds of km, and the balloon's own wind readings are worth
nothing, in 24 of 24 comparisons and again at 1,000 and 3,000 km. At zero noise the readings are worth
exactly nothing on every seed, which is the harness checking its own wiring. The kernel's horizontal
lengthscale is 544 km and a move is one reading every 3 hours along a single line, so there is very little
for a reading to tell the belief that the belief does not already say.

## Known problems with these, if they go on a slide as they are

- Figures 2 and 6 draw all eleven controller variants at once. Legible as a record, not as a slide. For the
  talk they want cutting to four lines: hindsight, fixed plan on the mean, MPC on the mean, MPC not learning.
- Figure 5's y-axis runs to -8 because of one seed at sigma 0, where the fixed plan and hindsight are nearly
  equal and the ratio blows up. Everything of interest is between 0 and 1. The axis wants clipping, or the
  sigma 0 column dropping, and its title is cut off at the right edge.
- Figure 1 is one seed on one day. It is there to show what the three controllers do, not to support a claim.
- Figures 3 and 4 show the horizons separating below 3 m/s, but at 0.8 and 1.6 m/s the paired intervals
  overlap almost completely. The ordering among horizons at low noise is not something 32 seeds can
  establish; only the gap between horizon 5 and the rest at sigma 0 is.
