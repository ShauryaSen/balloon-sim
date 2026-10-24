# Scouting rule, 2026-10-24: how the task is chosen, fixed before the results

The 1,000 km downwind task turned out to have no room in it. Under the arrival rule the first seed gave
hindsight 2,625 km, a blind fixed plan 2,630, and MPC 2,636 — four controllers within 10 km of each other,
all arriving at move 5 of 50. The wind carries a balloon 1,000 km downwind whatever it does, so the episode
ends before any controller has had a chance to differ.

This note fixes how the replacement is chosen, and is committed before the scouting arm runs.

## The scout

Hindsight and the fixed plan only, on the setup of the sweep: global ERA5 2021-07-16, launches 20N to 50N,
50 moves of 3 h, the arrival rule at 200 km, the same 32 seeds, the forecast error at the record's own
3.2 m/s. Over every goal direction and distance:

    direction   downwind, crosswind, upwind, random
    distance    300, 600, 1000, 2000, 3000 km

1,280 cells. Hindsight is the ceiling and the fixed plan the floor, which is all that is needed: where the
two score the same there is nothing in between for a controller to win, and where hindsight cannot reach the
goal the task is measuring the wind rather than the control.

Recorded per flight: the objective (distance summed until arrival), closest approach, whether it arrived and
on which move, and the trajectory.

## The rule

A task **qualifies** if both hold over the 32 seeds:

1. **Reachable.** Hindsight arrives — comes within 200 km, ending its episode — on at least **24 of 32** (75%).
2. **Not free.** The fixed plan's objective is at least **25% above** hindsight's on at least **16 of 32**
   seeds, counted seed by seed. (Amended; the original read "as a paired mean over seeds, and the 95% paired
   interval excludes zero". See the amendment below.)

The sweep takes the qualifying task with the largest **median** gap between the floor and the ceiling, since
that is the task with the most room for a controller to show something on a typical seed. Ties are broken by
the longer goal distance, and then by direction in the order written above.

- 24 of 32 is the threshold the goal-distance rule of 2026-10-22 used, kept so the two are comparable.
  (Assumed.)
- 25% is a gap large enough that the controllers can be told apart against the seed-to-seed spread, which on
  the 1,000 km sweep ran to about a fifth of the mean. A 10 km gap on 2,600 km is what this is meant to rule
  out. (Assumed.)

**If nothing qualifies**, the episode length and the wind window are what change next, not the rule: a longer
episode or a season with weaker flow. That is recorded here rather than decided afterwards.

## Amendment, 2026-10-24, before the arm ran

The second condition was a paired mean and is now a count of seeds. The reason, in the open, is that a probe
of 8 seeds at 1,000 km downwind was run between writing this note and running the arm:

| seed | hindsight | fixed plan | arrived, hindsight / fixed | gap |
|---|---|---|---|---|
| 0 | 2,625 | 2,630 | 5 / 5 | +0.2% |
| 4 | 2,413 | 2,416 | 5 / 5 | +0.1% |
| 7 | 4,021 | 3,926 | 8 / 7 | -2% |
| 3 | 4,277 | 4,576 | 8 / 9 | +7% |
| 5 | 3,223 | 3,700 | 6 / 7 | +15% |
| 1 | 9,524 | 55,567 | 18 / never | +483% |
| 2 | 8,283 | 97,979 | 15 / never | +1083% |
| 6 | 9,814 | 119,131 | 25 / never | +1114% |

Five of the eight are free rides and three are catastrophes for the blind plan, with almost nothing between.
The mean gap over them is 556% of hindsight's cost and the median is 7%. So the task this note was written
to reject would have passed the rule as first written, on the strength of three seeds where the fixed plan
never arrived and spent fifty moves being blown away.

That is a heavy tail, and it was known before this probe: the 1,000 km sweep's dot plots show the same shape,
a tight cluster and a handful of bad seeds, which is why they were drawn as dots in the first place. The
amendment makes the rule read the typical seed rather than the mean, which a long tail owns. It is still an
amendment made after seeing data, and a reader should weigh it as one.

The thresholds themselves are unchanged: 24 of 32 to be reachable, 25% to be a material gap.

What is reported alongside, outside the rule: hindsight's median arrival move, since a task whose episodes
end at move 5 says little about control however large the gap; the share of seeds where the fixed plan never
arrives at all; and the mean gap, so the figure the original rule would have used stays visible next to the
one that replaced it.

Arrival rate is worth watching as an outcome in its own right. Under the arrival rule the objective spans a
factor of fifty between a flight that arrives and one that does not, where "did it arrive" is bounded and
reads directly. The probe had hindsight at 8 of 8 and the fixed plan at 5 of 8.

## Result

To be filled in by `scripts/scout.py` when the arm has run.
