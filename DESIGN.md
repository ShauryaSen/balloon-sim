# balloon-sim design

Every piece of code here exists because a decision below needs it. Decisions are locked by Shaurya and Rohan.

## The experiment

How well can a balloon controller do as the wind forecast gets worse, compared to what any controller could do?

## Locked decisions

1. **Task: a single leg.** Fixed start, fixed goal, 80 steps of 3 h. Cost $J = \sum_{k=1}^{80} \lVert x_k - \text{goal} \rVert$.
   The same start, goal and true wind for every controller and noise level within a seed.
2. **Outputs.** Every run records both metrics: summed distance and final distance to the goal. For each MPC
   horizon, two figures of the same form (one per metric) against noise σ, with lines for hindsight, best fixed
   plan and MPC at several n_winds, plus the normalised score. Hindsight and fixed plan always plan the full
   80 steps, so their lines are shared across horizon figures. The cost the planners optimise is a parameter
   (summed or final); the bounds are guaranteed only for the metric that was optimised.
3. **Belief.** Behind an interface. Version 1 is fixed: the GP prior with the forecast as mean, never updated.
   An updating belief (readings at the balloon) comes later and must slot in without changing controllers.
4. **Wind.** One record format: true wind and coordinates on a (time, level, lat, lon) grid. Version 1 uses the
   ERA5 wide North Pacific record. cBottle comes later through the same format.
5. **Noise.** Forecast $F = W - e$ with $e$ drawn from the belief's own kernel at scale σ. The truth $W$ never
   changes with σ, and the belief is calibrated by construction.
6. **Position: latitude and longitude in degrees**, longitude 0 to 360. No projection: both the ERA5 and the
   converted cBottle records are regular lat/lon grids. Grid spacing, box and time step are read from the file,
   never hard-coded (ERA5 wide: 2 deg, 3 h; cBottle: 1 deg, 6 h). One step moves
   $\Delta\text{lat} = v\,\Delta t / R$ and $\Delta\text{lon} = u\,\Delta t / (R \cos \text{lat})$, in radians.
   Distance in the cost is great-circle distance in km.
7. **Altitude in km.**
8. **Action: climb rate in km per step**, capped.
9. **Integration: one Euler step per 3 h move**, with the number of substeps as a parameter.
10. **One planner, three controllers.** The planner takes a state, a batch of winds and a horizon, and returns an
    action sequence. Hindsight plans once on the true wind. Fixed plan plans once on belief draws. MPC replans
    every step on belief draws.
11. **Planner: Myles's FOMPC recipe** (random candidates, previous plan, coast; best refined by normalised
    gradient descent), every number a parameter. The refinement optimiser is selectable: L-BFGS (the
    default, after it found much better hindsight plans on ERA5) or Myles's normalised gradient descent.

12. **Belief draws by random Fourier features**, as in zermelo: a product kernel whose factors each draw their
    own frequencies. Horizontal distance is measured on the unit sphere in 3D, so the same code works for a
    global cBottle field. The forecast noise is drawn by the same sampler, so belief and noise share a kernel.
    Chosen because it extends to an updating belief (pathwise conditioning) and is the published method.
13. **Every wind is a grid field.** Truth, forecast and each belief draw are the same type: an array on the
    record's grid plus one interpolation function. The planner never knows which it is flying.
14. **Environment: uv.**
15. **Running experiments: three scripts.** `run_cell.py` runs one (seed, σ, controller, horizon, n_winds) and
    writes one row; `submit.py` lists cells and submits a Slurm array; `plot.py` reads rows and draws figures.
16. **Layout: one file per idea.**

    ```
    balloon_sim/
      wind.py         grid record, loading, interpolation
      dynamics.py     one step, a rollout, the costs
      belief.py       Belief interface, the fixed GP prior, drawing winds
      planner.py      Myles's recipe
      controllers.py  hindsight, fixed plan, MPC
      episode.py      fly one controller in the true wind, return trajectory and metrics
      task.py         where an episode starts and where its goal is
      results.py      the result row: its columns, writing and reading
      cells.py        which flights a sweep is made of, and the keys a flight is drawn from
      config.py       one frozen dataclass per piece
    scripts/          run_cell.py, submit.py, plot.py, fit_kernel.py, convert_cbottle.py
    tests/
    ```

17. **Types and tests.** Plain type hints, shape comments only where shapes are not obvious. Tests check physics
    and math: gradient vs finite differences, still air, belief draws match the kernel's variance, and the three
    controllers ordered correctly on a toy wind.
18. **Start and goal.** Two modes: drawn per seed (start inside the box with a margin, goal at distance D in a
    random direction, redrawn if outside) or fixed coordinates. D defaults to 1,000 km. The noise draw is
    per seed in both modes.
19. **Edges.** Altitude is held inside the record's levels by the step itself, so the planner and the simulator agree
    on it exactly; at the floor or ceiling a climb past it has no gradient. Position is held to the box by the
    simulator only, so the planner's gradient never dies at the box's edge. (Changed after the first ERA5 run,
    where an unclipped planner dived to 11 km under a 14.8 km floor and its plans stopped matching the flights.)
20. **Start altitude** at the middle of the range; **climb cap** 1.1 km per step.
21. **Configuration.** One `config.py` of frozen dataclasses, one per piece (wind, balloon, task, noise and belief,
    planner, experiment), defaults in code, overridden from the command line. Every result row stores the full
    config it ran with. No YAML, no Hydra.
22. **JAX**, limited to `jnp`, `grad`, `vmap`, `jit`, and `lax.scan` in the rollout only. Every function pure, so
    many seeds can later be flown at once on one GPU.

23. **Kernel fitted to data.** `scripts/fit_kernel.py` fits the lengthscales and amplitude to a record's forecast
    error (truth minus forecast). The fitted numbers become the config defaults for that record.
24. **Build order.** wind, dynamics, planner, hindsight end to end, belief with fixed plan and MPC, scripts. Each
    step tested before the next.

25. **The belief learns before the sweep runs.** At each move the balloon reads the wind where it is and the belief
    conditions on it (pathwise update of the random Fourier draws), so MPC plans on what it has learned. Hindsight
    and the fixed plan do not learn: one knows the truth, the other never looks. A reading is the true wind at
    the balloon's own point, exactly, as zermelo's default readout; the belief assumes a 0.01 m/s jitter on it,
    which is what zermelo does and is numerical, not a modelled error.
26. **Kernel fitted to the record's stale forecast** (truth minus the truth 24 h earlier) as the headline: errors
    shaped like a day-old forecast's. On the ERA5 wide record that gives 544 km, 1.18 km in altitude, 9.4 h,
    2.79 m/s, against 1,013 km, 3.54 km, 58 h when fitted to the wind itself. A stale forecast can be built from any
    time series (`fit_kernel.py --stale-hours 24`), and on `cbottle_jul_s100` it gives 578 km, 4.4 km, 9.9 h,
    5.9 m/s: the config's defaults. The sweep's noise levels are multiples of that fitted sigma, so the ladder
    means the same thing on any record.
27. **Both metrics plotted**, summed and final distance; which one leads is Sarah's call.
28. **cBottle is the wind source**, global, so there is no box edge (Sarah's call to use cBottle). The released
    cBottle-video rollouts (four on Unicorn, 14 days at 6 h), converted by `scripts/convert_cbottle.py` to a 2
    degree global grid between 80S and 80N at the 200 and 50 hPa levels (11.8 and 20.6 km), the two levels it
    offers in the balloon band; the balloon is confined between them. Longitude wraps at 360 (the record carries
    its first column again at 360 so the blend crosses the seam); starts and goals stay within 60 degrees of the
    equator. Shaurya's retrained 18-level generator replaces it once time-stitching exists.
29. **Six-hour moves**, the cBottle frame: the climb cap is 2.2 km per move (1.1 km per 3 h) and an episode is
    50 moves, 12.5 days of a 14-day rollout.

30. **The sweep's grid, chosen so it need not be rerun.** Noise: 0, 0.25, 0.5, 1, 2 and 4 times the fitted
    day-old error. MPC horizons 5, 10, 20 and 50 moves: a doubling ladder up to the whole episode, where MPC
    shares the bounds' horizon. MPC draws 0, 1, 4, 16 (the mean, one sample, two doublings to show
    saturation); fixed-plan draws the same plus 64, its best proxy-bound estimate. 32 seeds, which can be
    extended later since seeds are independent and paired; the other axes cannot be changed without rerunning.
31. **Only the moves left in the episode count in a plan.** MPC near the end would otherwise optimise moves
    it never flies, in wind frozen at the record's last frame, more so the longer the horizon.

32. **Wind: a global ERA5 record at the balloon's six levels** (`era5_global_jul2021.npz`: 2021-07-16 for
    240 h at 3 h, 2 degrees, 80S to 80N, 14.8 to 20.2 km), built by `balloon-env-dev/unicorn/make_zermelo_record.py`
    with a global box. Replaces released cBottle after one sweep: with only 11.8 and 20.6 km in the band,
    both at jet speeds, no controller could hold position and the experiment measured drift; every cBottle
    number is in `docs/cbottle-2026-10-21.md`. Global, so there is still no box edge. cBottle returns through
    Shaurya's retrained 18-level generator once time-stitching exists. Moves are back to 3 h with a 1.1 km cap;
    an episode is still 50 moves.
33. **Launches in the summer mid-latitudes, 20N to 50N**, the goal 300 km downwind. Where the balloon
    launches decides whether any controller can hold: a seed in the southern winter jet (35 m/s at every level)
    drifts 7,000 km under hindsight. The band matches the North Pacific record the lab already used, and the
    kernel is fitted within it (3.2 m/s, 544 km, 1.3 km, 9.1 h).
34. **Hindsight replans on the truth** (replacing the one plan flown blind), and **refinement keeps the best
    iterate**: one solve from a random pool left MPC on a perfect forecast beating hindsight by a tenth, and
    L-BFGS's line search could wander uphill and hand back a worse plan than it started from. Both found by the
    first sweep. L-BFGS beat Myles's normalised descent on 7 of 8 ERA5 seeds and random search on all 8, so the
    optimiser is not what limits the controllers.

35. **Goal 1,000 km downwind**, replacing 300 km, chosen by a rule fixed before the test
    (`docs/goal-distance-test-2026-10-22.md`): the longest distance at or above 1,000 km where hindsight comes
    within 200 km on at least 24 of 32 seeds. 1,000 km: 27/32; 2,000 km: 21; 3,000 km: 14. Every row now also
    records closest approach and when it happened, since final distance misreads a far goal.

36. **Hindsight is the best search found on the true wind**, per seed and per metric: the minimum over
    hindsight's own flight and every zero-noise cell, all of which plan on the truth. On the 1,000 km sweep the
    zero-noise MPC at horizon 20 was below hindsight on average (-34 km, within noise) but by up to 1,593 km on
    single seeds, where hindsight's 50-move search went badly. The ceiling should not depend on one search's
    luck. Applied when results are read (`results.best_hindsight`), so no rerun.

## Open

- The normalised score near sigma = 0, where the gap it divides by is close to zero.
- Myles's normalised descent stalls under the 6 h move (about twice L-BFGS's cost on the test wind); it stays
  as the option it is, not the default.
