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
2. **Not free.** The fixed plan's objective is at least **25% above** hindsight's, as a paired mean over seeds,
   and the 95% paired interval excludes zero.

The sweep takes the qualifying task with the **largest** paired gap between the floor and the ceiling, since
that is the task with the most room for a controller to show something. Ties are broken by the longer goal
distance, and then by direction in the order written above.

- 24 of 32 is the threshold the goal-distance rule of 2026-10-22 used, kept so the two are comparable.
  (Assumed.)
- 25% is a gap large enough that the controllers can be told apart against the seed-to-seed spread, which on
  the 1,000 km sweep ran to about a fifth of the mean. A 10 km gap on 2,600 km is what this is meant to rule
  out. (Assumed.)

**If nothing qualifies**, the episode length and the wind window are what change next, not the rule: a longer
episode or a season with weaker flow. That is recorded here rather than decided afterwards.

What is reported alongside, outside the rule: hindsight's median arrival move, since a task whose episodes
end at move 5 says little about control however large the gap; and the share of seeds where the fixed plan
never arrives at all.

## Result

To be filled in by `scripts/scout.py` when the arm has run.
