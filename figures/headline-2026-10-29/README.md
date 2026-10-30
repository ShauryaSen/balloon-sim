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

0. **is_it_even_possible**: the twenty candidate tasks, and why the sweep flies the one it does. Top: how
   often hindsight, which knows the true wind, reaches the goal at all; below 24 of 32 nothing can fly the
   task, because the best possible controller cannot. Bottom: how far above hindsight a blind fixed plan
   scores on a typical day; below 25% the wind is doing the work and no controller can show anything.
   Downwind is the only direction above the first line at any distance, and it only clears the second one
   once the goal is 2,000 km away. One of twenty qualifies. Lead with this: it says what an altitude-only
   balloon can be asked to do before any controller is compared. (`scouting.png`)
1. **one_flight**: seed 9 at the real forecast error (3.2 m/s). All three are carried west on the same
   flow; hindsight and MPC both turn down into the ring at 51 h and the fixed plan sails past to the north
   and never arrives, at 1,142 km on average against their 335. Solid is the episode, faint is what was
   flown after the balloon was already inside the 200 km ring, which costs nothing. One seed, an
   illustration, not evidence. (`flight_seed9_sigma3.2.png`; seeds 3 and 0 are drawn too, 0 being a day
   nothing reaches the goal, hindsight included)

   This is drawn from the rows' own saved trajectories by `scripts/map_rows.py`, so it is the sweep's own
   flight and cannot disagree with it. The version shipped on 2026-10-29 could and did: `map_flight.py`
   re-flew the seed and was not passing the arrival radius, so its flights were planned and scored on the
   plain sum over every move, a different objective from the one the sweep flew. The sweep's rows were never
   affected; of the 3,257 flights that arrived, 3,243 have cost_km cut below summed_km and the other 14
   arrived on the last move.

2. **all_controllers**: the objective against forecast error at MPC horizon 20, four lines. Hindsight flat
   and lowest, MPC solid below the fixed plan dashed. The line that matters is the dotted one, MPC that
   replans but never learns. Bars are each line's own standard error, unpaired, so they overstate the
   uncertainty in the gaps, and the apparent gap between solid and dotted around 1.6-3.2 m/s is a handful of
   catastrophic seeds pulling the mean: paired within seeds it is not there, which is what finding 1 rests
   on and what figure 6 shows. (`plain_lines_cost_km_h20.png`; `lines_cost_km_h20.png` has all eleven draw
   counts and is the record rather than a slide)
3. **breaking_point**, the headline: MPC's excess over hindsight, a line per horizon, 95% paired intervals.
   Longer horizons help below about 3 m/s; past that the four lines sit on each other and the horizon stops
   buying anything. The breaking point moved in from 3.2-6.4 m/s on the old 1,000 km task to 1.6-3.2 here.
   (`horizons_cost_km_n0.png`)
4. **breaking_point_surface**: the same numbers as one surface over error and horizon; dots are measurements,
   the surface between them is interpolated, axes evenly spaced rather than to scale.
   (`surface_cost_km_n0.png`)
5. **share_of_gap_closed**: (fixed plan - MPC) / (fixed plan - hindsight), median and quartiles, paired
   within each seed. About 0.8 at 0.8-1.6 m/s, settling to 0.37 past 6 m/s. The score is a ratio and near
   sigma 0 the gap it divides by is nearly zero, so one seed's quartile runs past the bottom of the axis;
   the axis is held to where the medians are and says so in the corner, rather than being stretched to -8 by
   that one seed and flattening everything else onto one line, which is what the first version did.
   (`plain_score_cost_km_h20.png`)
6. **every_seed**: the numbers of figure 2, one dot per seed: why the bars are wide, and why the gap in
   figure 2 is not what it looks like. A handful of launches run to 150,000-300,000 km while the bulk sit
   under 60,000. MPC is a circle, MPC not learning a triangle; they overlap, which is the point.
   (`plain_dots_cost_km_h20.png`)
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

## The rest of the 23rd's list

Figures 0 to 7 are the talk. These are the ones to have open behind it, one per item from the notes of the
23rd, because each is the answer to a question somebody asked rather than something we went looking for.

8. **the_old_objective**, for *"fix the objective?"*: the same cells and the same seeds as figure 2, scored
   the old way, summing distance to the goal over every move with no arrival. Below about 1.6 m/s the two
   figures order the controllers differently, because the old score kept charging a balloon that had already
   arrived and then drifted, so it was partly measuring drift rather than navigation. Both are correct about
   different things, and nothing from before the 23rd is comparable to anything after it.
   (`plain_lines_summed_km_h20.png`)

9. and 10. **distance_1000km** and **distance_3000km**, for *"vary the distance even, maybe we just run this
   over multiple so we also have one where it is barely reachable"*: figure 7 again at the goal either side
   of the one we fly. The same shape all three times, zero at a perfect forecast, a trough near 3 m/s, and
   back up after it, so what replanning is worth is not an artefact of the distance we happened to pick.
   One caveat to say out loud: the goal moved out to 3,000 km but the episode stayed 50 moves, so the far
   task is also the more hurried one, and the two cannot be separated inside this arm. The length arm now
   running is what separates them.
   (`2026-10-24-1000km/replanning_cost_km_n0.png`, `2026-10-24-3000km/...`)

11. to 13. **hedging_every_draw_count**, **belief_half_as_wide**, **belief_twice_as_wide**, for *"find some
    storms and test it so we can figure out if this averaging over belief draws truly has a problem"*: all
    eleven lines, which is the one place the cluttered figure is the right figure, since the question is
    about the draw counts themselves. Planning against 16 possible winds instead of one best guess is worth
    168 to 239 km to the fixed plan and nothing at all to MPC, at every noise level. The obvious objection
    is that our belief is unrealistically well calibrated, sharing a kernel with the noise by construction,
    so 12 and 13 are the same arm with the belief told the error is half and twice its real size. The
    pattern does not move. The storms themselves have not been run; this is the cheaper test of the same
    idea, and it came back negative.
    (`lines_cost_km_h20.png` from `2026-10-24`, `-belief05`, `-belief2`)

14. **nested_draws**, for *"draw the 16, then permute them... so that would slightly reduce variability
    while helping us isolate the effect of the number of samples, versus the actual variability inherent to
    the samples"*: N of 1, 4 and 16 taken as the first n of one shared pool, so the arms differ only in how
    many winds the plan saw and not in which ones it drew. Same pattern again, so the effect in 11 is the
    count and not the luck of the draw. (`2026-10-24-nested/lines_cost_km_h20.png`)

Four items from that list are not here because they have not been run: real GEFS forecasts instead of GP
draws, more days and seasons, actual storms, and a penalty for arriving too early. Two more, the escape
objective and the longer episode, are in the sweep running now.

## Known problems with these, if they go on a slide as they are

- Figure 1 is one seed on one day, chosen because it separates the three controllers cleanly. It is an
  illustration. Seed 0 is the honest counterweight if anyone asks about the spread.
- Figures 3 and 4 show the horizons separating below 3 m/s, but at 0.8 and 1.6 m/s the paired intervals
  overlap almost completely. The ordering among horizons at low noise is not something 32 seeds can
  establish; only the gap between horizon 5 and the rest at sigma 0 is.
