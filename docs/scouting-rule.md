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

## Result (Slurm arrays 577661 and 577662, rows in `results/rows-scout/`, all 1,280 cells, no failures)

```
direction    goal   h.arr  move  f.arr   hindsight     median gap  seeds    mean gap  verdict
crosswind     300   20/32    15   7/32       36861     16997  75%  20/32       33440  unreachable
crosswind     600   16/32    20   6/32       45438     17392  61%  22/32       29510  unreachable
crosswind    1000   13/32    25   5/32       52036     14866  43%  21/32       23030  unreachable
crosswind    2000   12/32    36   2/32       76529     17160  22%  15/32       30921  unreachable
crosswind    3000    6/32    37   2/32      114897      9729   9%   6/32       16912  unreachable
downwind      300   32/32     2  31/32         460         0   0%   1/32        2567  too free
downwind      600   31/32     4  26/32        3795        13   1%   7/32       13105  too free
downwind     1000   29/32     8  20/32        9273       845  16%  15/32       28835  too free
downwind     2000   25/32    20  10/32       29916     21016  56%  22/32       38287  QUALIFIES
downwind     3000   17/32    25   4/32       59287     27489  58%  23/32       34615  unreachable
random        300   22/32     9  10/32       35769     23118  55%  19/32       36918  unreachable
random        600   19/32    12   9/32       46028     11260  23%  15/32       27558  unreachable
random       1000   16/32    21   7/32       49539     17623  35%  20/32       28413  unreachable
random       2000   15/32    28   5/32       83732     18581  27%  16/32       27822  unreachable
random       3000   12/32    33   3/32      117363     17667  15%  11/32       27480  unreachable
upwind        300   20/32    19  10/32       41923     20683  55%  24/32       31829  unreachable
upwind        600   16/32    23   6/32       52606     17883  54%  20/32       34134  unreachable
upwind       1000   12/32    21   3/32       62950     17665  36%  18/32       33170  unreachable
upwind       2000    8/32    39   1/32       92307     22239  30%  20/32       26991  unreachable
upwind       3000    4/32    40   1/32      137336     12117   8%   9/32       22472  unreachable
```

By the rule, **downwind at 2,000 km** is chosen: the only one of the twenty that is both reachable and not
free. Hindsight arrives on 25 of 32 seeds at a median move 20 of 50, so there are twenty moves of decisions
in an episode; the fixed plan arrives on 10. The median gap is 21,016 km, 56% above hindsight on a typical
seed, material on 22 of 32. `BASE_GOAL_KM` and `TaskConfig.distance_km` are set to it.

Two things the table says beyond the choice.

**An altitude-only balloon cannot go anywhere but downwind in this record.** Crosswind, upwind and random
fail the reach threshold at every distance tried, 300 km included, where hindsight gets 20, 20 and 22 of 32.
This is the wind report's number in another form: the levels oppose each other at only 58% of points, so a
goal off the flow usually has no level that can reach it. It is a statement about what the vehicle can do,
not a failed experiment, and belongs in the write-up.

**The free ride ends between 1,000 and 2,000 km.** Along the downwind column the arrival move runs 2, 4, 8,
20, and the gap 0%, 1%, 16%, 56%. Downwind was never the wrong choice; 1,000 km was simply short enough that
the wind did the work. At 2,000 km the flow at launch is no longer the flow two and a half days later, so
staying in the right current becomes a decision.

Upwind at 300 km is worth a note: it had the largest material-gap count in the table, 24 of 32, and missed
only on reach at 20 of 32. If a harder steering task is ever wanted, a shorter upwind goal is where to look.
