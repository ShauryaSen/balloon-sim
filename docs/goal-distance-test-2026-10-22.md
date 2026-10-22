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

## Result (run on Unicorn, Slurm job 471615, rows in `~/balloon-sim/results-distance/`)

| goal | hindsight within 200 km | within 500 km | closest approach, median [quartiles] | move of closest, median | final, median | average distance, median |
|---|---|---|---|---|---|---|
| 300 km | 30/32 | 32/32 | 47 km [7, 98] | 5 | 480 km | 413 km |
| 1,000 km | 27/32 | 31/32 | 62 km [5, 152] | 26 | 295 km | 332 km |
| 2,000 km | 21/32 | 27/32 | 73 km [9, 383] | 39 | 402 km | 729 km |
| 3,000 km | 14/32 | 25/32 | 269 km [27, 471] | 43 | 579 km | 1,249 km |

By the rule, **1,000 km** is chosen: the longest distance at or above 1,000 km with at least 24 of 32 seeds
within 200 km (27). 2,000 km falls short (21), 3,000 km well short (14). Its closest approach comes at a median
move of 26 of 50, so the episode is long enough to say something about holding the goal afterwards.

Also visible, not part of the rule: the 1,000 km goal ends nearer (median 295 km) and averages nearer (332 km)
than the 300 km goal (480 and 413), because the balloon spends half the episode travelling toward it instead
of being blown off a goal it reached after five moves. The far goal turns the task from holding a point into
steering while drifting, which suits winds no level can cancel.
