# Headline figures, for the 2026-10-23 meeting

Copies from `figures/1000km/` (goal 1,000 km downwind, 32 seeds), in the order to present them. Numbers behind
every claim: `docs/results-2026-10-22.md`.

1. **one_flight**: seed 3 at the real forecast error (3.2 m/s). All three pass near the goal; hindsight stays,
   MPC is carried off and partly recovers, the fixed plan is blown thousands of km away. One seed, an
   illustration. (`flight_seed3_sigma3.19.png`)
2. **all_controllers**: summed distance against forecast error at MPC horizon 20. Hindsight flat and lowest,
   MPC (solid) below the fixed plan (dashed); the gap is what replanning buys, about 400 km on average past
   3 m/s. Bars are each line's own standard error, unpaired, so they overstate the uncertainty in the gaps.
   (`lines_summed_km_h20.png`)
3. **breaking_point**, the headline: MPC's excess over hindsight, a line per horizon, 95% paired intervals.
   Longer horizons help up to 60 h with a good forecast; between 3.2 and 6.4 m/s the lines merge.
   (`horizons_summed_km_n0.png`)
4. **breaking_point_surface**: the same numbers as one surface over error and horizon; dots are measurements,
   the surface between them is interpolated, axes evenly spaced rather than to scale.
   (`surface_summed_km_n0.png`)
5. **share_of_gap_closed**: (fixed plan - MPC) / (fixed plan - hindsight), median and quartiles over seeds;
   0.6 to 0.7 up to 3.2 m/s, falling to about 0.2 at 12.8. Ignore the sigma 0 point (17 seeds).
   (`score_summed_km_h20.png`)
6. **every_seed**: the numbers of figure 2, one dot per seed: why the bars are wide (a few bad launches).
   (`dots_summed_km_h20.png`)
