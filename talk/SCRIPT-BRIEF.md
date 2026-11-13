# Brief for writing my talk script

Attach `talk-2026-11-13.pdf` with this file.

## The setting
- A 30-minute talk to Sarah Dean's lab group at Cornell (ML / control / RL researchers), then 30 minutes of discussion. Several people in the lab already work on balloons (Myles: station keeping with MPC; Vinamr: storm search), so the basics of balloon control are familiar to them.
- Two presenters. I (Shaurya) present **slides 1–13 and 29–35**. Rohan presents slides 14–28.
  - Slide 13 ends with me handing over: "First we need a way to measure it. Rohan."
  - Rohan hands back at the end of slide 28: "That's what the main experiment is for. Back to Shaurya."
- Time budget for my part is about 15 minutes in total: slides 1–13 ≈ 10–11 min, slides 29–34 ≈ 4 min. The whole deck currently runs a bit long, so shorter is better.
- Slide 35 is a backup, only if asked.
- The PDF shows each slide with every build revealed. The "click" markers in the notes show where builds happen.

## How I want the script
- Plain, direct, conversational. Something I can say out loud, not read. No filler, no hype, no "it's worth noting", no rhetorical flourishes.
- Follow the slide: say what's on it, then the one thing it's there for. Don't add claims that aren't on the slide or in the notes below.
- Mark each [click] where a build happens.
- Give a time estimate per slide.
- For each slide, list one or two likely questions and a short answer, drawn only from the facts below.

## Key facts (verified; don't invent others)
- **Balloon control:** balloons can only change altitude; the wind at each altitude moves them sideways. So control means choosing altitudes, and a controller needs to know how wind differs between altitudes.
- **BLE's wind generator (Google's Balloon Learning Environment, Bellemare et al. 2020).** A VAE checked in its code:
  - 21×21 grid covering ±500 km, 10 pressure levels (50–140 hPa), 9 frames over 48 h;
  - a random 64-dimensional latent with no conditioning on place or time;
  - outside the box the wind is clamped to the edge value, and after 48 h time runs backward ("boomerangs").
- **Our generator (the "summer model"):**
  - a conditional diffusion model on 64×64 patches at 0.25° (16° across), 18 levels (about 14–20 km), wind u and v, 4 hourly frames;
  - conditioned on location and time (sin/cos of the annual, semiannual and diurnal cycles);
  - trained on 4 years of ERA5, 250k steps;
  - InfiniteDiffusion tiles patches seamlessly, so it can generate any area and duration.
- **Global generator:** in progress; two stages (coarse, then fine) on a HEALPix equal-area grid. The globe video comes from it.
- **Realism results (summer model, 4yr / 250k).** Lower is better except spread ratio, where ERA5 ≈ 1 is the target.

  | | SR_time | Dispersion | Spread ratio |
  |---|---|---|---|
  | Ours | 2.43 | 0.81 | 0.20 |
  | BLE | 4.19 | 1.79 | 0.11 |
  | ERA5 floor | 0.28 | 0.18 | 1.04 |

  - Simplex noise fitted to ERA5's statistics scores better than both generators on several checks, even though it has no jets or storms and no location or time input. That's the point of slide 13: these checks measure statistics, not usefulness for control.
  - Caveats: the rows use different horizons; ours is 2 runs; the time scores tile our model in time to 24 h.
- **Part 5, fidelity experiment (planned, not yet run):**
  - make copies of ERA5, each with one property degraded;
  - build a controller on the copy: MPC plans on it, or PPO is trained on it;
  - fly the balloon in true ERA5. The drop in score shows how much that property matters for transfer.
  - The truth is ERA5 at 18 levels, 2°, 3-hourly.
  - Ladders: levels 18→9→6→3→2→1; frames 3 h→6→12→24→frozen; grid 2°→4°→8°→16°; locations: one patch or many; kind of error at the same size (e.g. Gaussian process vs simplex).
  - Hypothesis: altitude levels matter most, because the balloon steers on the differences between altitudes.
  - Tasks: station keeping, navigation to waypoints, distance from launch.
  - Payoff (slide 33): each resolution property is a generator setting (today 18 levels, 1 h, 0.25°). The experiment finds the coarsest setting that still transfers, which gives the cheapest generator.
  - Limit: the experiment can only test coarser than ERA5's 2° / 3 h. Training data (locations) is not a resolution setting.
- Rohan's results that I build on (slides 22–28): with realistic random forecast error (3.2 m/s), MPC closes 77% of the gap between a fixed plan and hindsight. MPC settings used from here on: plan 60 h ahead, on one forecast.

## My current speaker notes (draft; improve them, keep the facts)
### Slide 1: Wind models for balloon control

Shaurya. Title.

### Slide 2: Outline

Shaurya, 20 seconds. Walk through the five sections in one sentence each.

### Slide 3: Background and our generator

Shaurya. A few seconds.

### Slide 4: High-altitude balloon control

Shaurya, 45 seconds. Quick framing; the lab has seen this.
This is the picture from Myles's paper, so most of you know the setup. A high-altitude balloon can't push itself anywhere. The wind blows different directions at different altitudes, and the one thing the controller does is go up or down to pick which wind to ride.
The usual benchmark is station keeping: stay above a point on the ground. Next slide, the two kinds of task we care about.

### Slide 5: Balloon tasks

Shaurya, 0.5 min. Myles and Vinamr have presented these, so keep it short.
Station keeping: stay near one point for days. Loon did it with RL; Myles's FOMPC does it with gradient MPC.
Navigation: get to waypoints far away. They can be fixed, or chosen as you go, like Vinamr's storm search, where the balloon decides where to fly next based on what it's learned about the wind.
Both pictures are schematic. The ring is drawn far bigger than real; BLE scores time within 50 km.
Figure: talk/figures/tasks.py

### Slide 6: Prior work: BLE's wind generator

Shaurya, 1.5 min.
The wind most balloon-control work trains on comes from Google's Balloon Learning Environment, BLE. It's a VAE trained on ERA5.
It generates one patch, 1,000 by 1,000 km, 48 hours long. Outside the patch the wind is copied from the edge; after 48 hours time runs backwards.
It isn't given a place or a date, so every patch is just "some wind".
The globe: a real ten-day flight in ERA5 at 19 km. It leaves BLE's patch after 11 hours and ends about 12,000 km away. For station keeping that's fine, which is what BLE was built for. For anything that travels, it isn't.
Sources: BLE code, generative/vae.py (FieldShape: ±500 km, 10 levels, 48 h) and env/grid_based_wind_field.py (edge clipping, time reversal after 48 h). Trained on ERA5: BLE's README asks users of its wind dataset to cite ERA5, and the VAE trains on that dataset.
Figure: talk/figures/globe.py

### Slide 7: Our generator

Shaurya, 1.5 min. ONE CLICK.
So we built our own. It's a diffusion model trained on ERA5: it starts from noise and removes it step by step until it looks like real wind.
What makes it useful is what you give it: a location, a time of year and a time of day. It generates wind for that place and time.
What comes out is a block: 64 by 64 grid points at a quarter degree, so 16 degrees across, 18 altitude levels covering the balloon's range from about 14 to 20 km, east and north wind, over 4 hours.
[click] For comparison, BLE's generator. It takes no inputs, just random numbers, so it doesn't know where or when it is. It gives a 1,000 km box with 10 levels over 48 hours. And you can see it: flat horizontal stripes, where ours has the blobs and gradients real wind has.
Sources: time inputs are sin/cos of the annual, semiannual and daily cycles (balloon-env-dev docs/conditional-base-changes.md); 18 levels = ERA5 model levels 49-66, about 53-134 hPa. BLE: generative/vae.py FieldShape (21 x 21 points over ±500 km, 10 levels 50-140 hPa, 9 frames over 48 h, a random 64-number latent).

### Slide 8: InfiniteDiffusion

Shaurya, 1 min. No clicks.
One block is 16 degrees across. To go further, the naive thing is to generate blocks independently and put them side by side.
That gives seams, left: each tile made its own weather. InfiniteDiffusion covers the region with overlapping windows and denoises them all together, averaging where they overlap at every step, so neighbours have to agree. Right: one continuous field. It works the same way in time.
Two properties that matter for a simulator. Random access: ask for the wind at any place and time and it only generates what that query needs, so a long flight only costs what it passes through. Deterministic: ask again and you get the same wind, so every controller flies the same world.

### Slide 9: Global generator (in progress)

Shaurya, 45 seconds.
That was the summer model. What I'm building now goes further. This is its wind at about 20 km, generated, not ERA5.
It covers the whole globe in two stages: a coarse model for the whole planet, then a second one that fills in fine detail. It tiles in time too, so a flight never runs out of wind. It works on an equal-area grid on the sphere, and it's trained on global data rather than one region.
The realism numbers in the next part are for the summer model.
Video: blog/tools/wind_globe.py, Stage 1 -> Stage 2 output for 2023-07-10, ~53 hPa.

### Slide 10: Evaluating the generator

Shaurya. A few seconds.

### Slide 11: Realism checks

Shaurya, 1 min.
We score generated wind against held-out ERA5 on four properties, one per thing that shapes a flight.
Wind speeds: how hard the balloon is pushed, including extremes.
Structure in space: how the wind varies along the path.
Change with altitude: the options the balloon steers between. The one that matters most for control.
Change over time: how the wind evolves during a flight.
Floor: two halves of real ERA5 still differ, so that's the best possible score.
Reference: held-out ERA5, NE Pacific, days 8-14 of Jan/Apr/Jul/Oct 2023, not used in training.

### Slide 12: Realism results

Shaurya, 2 min. One click adds simplex noise.
Lower is better. The dotted line is the ERA5 floor.
Speeds: distance between wind-speed distributions, 2.98 vs BLE's 4.56. Ours still misses extremes (tail error 13.5 m/s vs a floor of 2.9).
Space: the finest scale it keeps. Ours keeps detail down to about 340 km, BLE about 840. ERA5's floor and simplex both show 56 km, which is the grid limit: they keep detail all the way down, so 56 means 'as fine as the grid can show', not a measured scale.
Altitude: error in the shear between neighbouring levels, mean of east-west and north-south. Ours 1.0, BLE 2.1. Honest note: on north-south alone, BLE is better (0.61 vs 1.01).
Time: dispersion error, how far off the spreading of drifting particles is over time compared with ERA5. Ours 0.81, BLE 1.79, floor 0.18. Better than BLE, but still about 4 times the floor, so time is our weakest check. The summer model makes 4-hour blocks, so we tiled it in time with InfiniteDiffusion to get 24-hour runs. Caveat: BLE's run is 48 hours at 6-hourly, ours 24 hours hourly, two runs of ours, so treat this panel as rough.
Click: simplex noise, the kind BLE uses as forecast error, fitted to ERA5. It beats both generators on all four. That's the next slide.
If asked about the "structure" measure: the spectral residual (SR) is the average gap between our energy spectrum and ERA5's, across all scales. Effective resolution is the smallest scale where we still have half of ERA5's energy. They tell the same story (ours 1.53 vs BLE 2.73); effective resolution is in km, so it's easier to read. Full table in the backup.
Caveats: BLE and simplex are one sample each; ours is 112 blocks. No repeated runs, so no error bars.
Source: balloon-env-dev/docs/benchmark-reports/benchmark_v2_4yr_250000.md, summer model at 4 years / 250k steps. Figure: talk/figures/realism.py Time panel: balloon-env-dev/docs/benchmark-reports/temporal_4yr_250000.md (run 2026-11-12).

### Slide 13: Limits of realism checks

Shaurya, 1 min. The turn in the talk.
The simplex noise was fitted to these statistics: per-level mean and spread, correlations between levels and hours, and its scale tuned to the spectrum. So it scores well by construction.
But it's random texture: no jets or storms that persist and move, and the same wind regardless of place or season. So a good score here doesn't tell us a model is good for control.
[click] That's the question for the rest of the talk: what a wind model needs to get right so a controller built on it, planned or trained, works in real wind. And which properties can be coarse, and so cheap, without hurting that. First we need a way to measure it. Rohan.

### Slide 14: Measuring transfer (Rohan, for the handover)

ROHAN. About 10 seconds.

First up, how we measure whether a wind model is any good for control.

### Slide 28: Preliminary results (Rohan, for the handover)

ROHAN. About 1 minute. THREE CLICKS.

So, preliminary results.

First, control handles realistic random error fine. At a real forecast's error MPC still closes 77 percent of the gap, versus 87 with a perfect forecast.

[click] Second, the method works. The score moves when the wind gets worse, and there's a clear breaking point. So it's sensitive enough for the real question.

[click] Third, it settled the MPC settings for everything after. Plan 60 hours ahead, since going further never helped. Plan on the one forecast, since hedging didn't help MPC. And readings didn't matter either way.

[click] What's left is which properties of a wind model matter. Random error makes everything a bit wrong at once, so it can't pull them apart. That's what the main experiment is for. Back to Shaurya.

### Slide 29: Fidelity experiment (planned)

SHAURYA takes over. About 15 seconds. No clicks.
Rohan showed that control tolerates random error about the size of a real forecast's.
But random error blurs everything a little. It never removes one thing, so it can't say which properties of a wind model matter.
This part is the experiment we plan to run next. It's planned, so there are no results here.
Random error blurs everything. Next: degrade one property at a time.

### Slide 30: Fidelity experiment

Shaurya, about 1 min. THREE CLICKS.
Same method as before: build the controller on a wind model, fly it in the real wind. What changes is the model.
We take ERA5, the true wind, and make copies of it. Each copy has one property degraded and everything else left alone. The example is held-out ERA5 at a quarter degree, the same panel as the poster, and a copy keeping only every 8th grid point, a quarter degree to 2: each block is one kept point, one arrow each. In the experiment the controller sees a straight-line blend between kept points.
[click] MPC plans on the copy, and the balloon flies in real ERA5. It's the setup Rohan showed, with the copy in place of the noisy forecast.
[click] Or we train an RL policy, PPO, on the copy, and then test it in real ERA5. So both kinds of controller: one that plans, one that's trained.
[click] We score it the same way as before. The drop in score from the undegraded case is how much that property matters for transfer.
Why copies of ERA5 and not a real generator: a real generator gets many things wrong at once, so if the score drops you can't say why. A degraded copy differs from the truth in exactly one way, so a drop has one cause.
This is planned. Nothing here has been run yet.
Which properties? Next slide.
Images: talk/public/fig/split-truth-px.png and split-coarse-px.png, drawn by talk/figures/split_pixels.py.

### Slide 31: Candidate properties

Shaurya, about 45 seconds. ONE CLICK.
These are the candidates. None of it is locked, and each is made worse in steps, one at a time.
The first three are resolution: how many altitude levels, how fine the grid, how far apart the frames are in time.
Locations: one patch of ERA5, or winds from many places. If one patch is enough, the generator can output less and train on less.
Kind of error: the error in Rohan's pilot was a Gaussian process draw. Here the size stays the same and only its kind changes, for example a Gaussian process against simplex noise.
[click] Our bet is that altitude levels matter most, because the balloon steers on the differences between altitudes.
Why we expect it: in Rohan's part only downwind goals were reachable, and the balloon's only choice is which altitude to sit at.

### Slide 32: Tasks

Shaurya, about 30 seconds.
Each degraded copy is tested on three tasks, with both controllers. Station keeping and navigation you saw at the start. The third is distance: get as far from launch as you can.
A property might matter for one task and not another. Our guess, not a result: station keeping needs winds that oppose each other; distance mostly needs one fast wind.
None of this is locked. Which task should lead is one of the questions at the end.
The task pictures are schematic. Figures: talk/figures/tasks.py

### Slide 33: Minimal generator

Shaurya, about 45 seconds. ONE CLICK.
Three of the properties are settings of our generator: how many altitude levels it makes, how often it gives a frame, and how fine its grid is. Each one costs compute. Today they're 18 levels, hourly, a quarter degree. We picked those by judgement.
The experiment fills in the last column: for each setting, the coarsest step where the score still holds.
One limit: the experiment's truth is ERA5 at 2 degrees and every 3 hours, so it can say how much coarser than that is fine, not anything about finer detail.
[click] Put the three together and that's the cheapest generator that still transfers.
If asked about locations: it's about training data, not resolution, so it isn't a row here. If one patch is enough, the generator can be smaller and train on less.
If asked about kind of error: it isn't a setting, so it's not here. It tells us which mistakes a generator can afford.

### Slide 34: Questions for you

Shaurya, about 0.5 min. NO CLICKS. Then 30 min discussion.
Three things we'd like your input on before we run this.
One: the properties. We have four so far. What else would you degrade?
Two: the stand-in. A degraded copy of ERA5 is wrong in one way, which is why a drop in score has one cause. A real generator is wrong in several ways at once. Does what we learn from the copies carry over?
Three: the tasks. We have station keeping, navigating to waypoints, and getting as far from launch as possible. Which should we run first and report as the main result?
Leave this slide up for the discussion.
If the discussion goes to training data: the experiment doesn't measure it; it's about resolution.

### Slide 35: Backup: full realism table

Backup. Source: balloon-env-dev/docs/benchmark-reports/benchmark_v2_4yr_250000.md. W1(u|c) and W1(v|c) recomputed from that run's cached samples (the report gives only their mean, 3.52).
Time rows: balloon-env-dev/docs/benchmark-reports/temporal_4yr_250000.md (run 2026-11-12). BLE's 4.19 / 0.11 reproduce the poster. Ours is new; the poster's 5.46 / 0.20 for "ours" were a toy baseline, and our real spread ratio happens to also be 0.20. Simplex time values are from the 250k report.
The poster's BLE W1(u|c) 12.71 and W1(v|c) 5.02 have no traceable source; BLE takes no place or time input, so those cells stay n/a.
