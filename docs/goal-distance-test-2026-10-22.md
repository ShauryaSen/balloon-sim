# Goal distance test, 2026-10-22: the rule, fixed before the results

Rohan asked whether the goal can sit 1,000 to 3,000 km away. This note fixes how the answer is decided, written
and committed before the test ran, so the choice cannot bend to the data.

## The test

Hindsight (replanning on the true wind, the ceiling) on the setup of the current sweep: global ERA5
2021-07-16, launches 20N to 50N at the middle altitude, goal downwind, 50 moves of 3 h, the same 32 seeds. Goal
distances 300 (the current one, the baseline), 1,000, 2,000 and 3,000 km. Hindsight is the right controller to
test with: if a controller that knows the wind cannot reach a goal, no controller can, and the task measures the
wind rather than control.

Recorded per flight: closest approach to the goal and the move it happened on, final distance, summed distance.
Closest approach is the reachability measure: final distance misreads far goals, since a balloon still flying
toward a far goal at move 50 can end closer than one that reached a near goal and was blown past it.

## The rule

A distance **qualifies** if hindsight comes within **200 km** of the goal on **at least 24 of 32 seeds (75%)**.

- 200 km: about one move of drift at a typical 10 m/s stratospheric wind (108 km per 3 h) with margin, and
  about two cells of the 2 degree grid. Within it the balloon is at the goal to the resolution of the problem.
  (Assumed: a convention, not from the literature.)
- 75%: a majority large enough that the sweep's 32 paired seeds mostly measure control, not unreachable goals.
  (Assumed.)

The sweep moves to the **longest qualifying distance at or above 1,000 km**. If none qualifies, it stays at 300 km
and this note records why. Ties do not arise: the longest is unique.

What would change the choice regardless of the rule: if the qualifying distance's hindsight reaches the goal only
near the end of the episode (median closest move after 45 of 50), the episode is too short to say anything about
holding it, and that is reported alongside.
