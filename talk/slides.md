---
theme: default
title: Wind models for balloon control
colorSchema: light
aspectRatio: 16/9
canvasWidth: 980
transition: none
---

# Wind models for balloon control

Shaurya Sen and Rohan Shankar

Dean lab group meeting


<!--
Shaurya. Title.
-->

---

<div class="divider">
  <p class="divider-label">Part 1</p>
  <h1>Background and our generator</h1>
  <p class="divider-q">Why do balloons need a new wind generator?</p>
  <Sections :current="1" />
</div>

<!--
Shaurya. A few seconds.
-->

---

# High-altitude balloon control

Balloons can move vertically and are moved horizontally by the wind.

<img src="/figures/hab-control.png" class="fig" style="max-height: 320px" />
<p class="caption" style="font-size: 0.75rem; margin-top: 0.2rem">Figure: Pasetsky et al., ICRA 2026</p>

<!--
Shaurya, 45 seconds. Quick framing; the lab has seen this.
This is the picture from Myles's paper, so most of you know the setup. A high-altitude balloon can't push itself anywhere. The wind blows different directions at different altitudes, and the one thing the controller does is go up or down to pick which wind to ride.
The usual benchmark is station keeping: stay above a point on the ground. Next slide, the two kinds of task we care about.
-->

---

# Balloon tasks

<div class="pair">
  <div>
    <img src="/figures/task-station.png" style="max-height: 260px" />
    <p class="caption"><b>Station keeping</b><br>Bellemare et al. 2020 (Loon); Pasetsky et al. 2026</p>
  </div>
  <div v-click>
    <img src="/figures/task-navigation.png" style="max-height: 260px" />
    <p class="caption"><b>Navigation</b><br>e.g. storm search</p>
  </div>
</div>

<!--
Shaurya, 0.5 min. Myles and Vinamr have presented these, so keep it short.
Station keeping: stay near one point for days. Loon did it with RL; Myles's FOMPC does it with gradient MPC.
Navigation: get to waypoints far away. They can be fixed, or chosen as you go, like Vinamr's storm search, where the balloon decides where to fly next based on what it's learned about the wind.
Both pictures are schematic. The ring is drawn far bigger than real; BLE scores time within 50 km.
Figure: talk/figures/tasks.py
-->

---

# Prior work: BLE's wind generator
Prior works used Google's Balloon Learning Environment to train controllers. Generated wind using a VAE.
<div class="side">
  <img src="/figures/globe-ble.png" />
  <ul>
    <li>A VAE trained on ERA5</li>
    <li v-click>One 1,000 × 1,000 km patch, 48 hours long</li>
    <li v-click>Not conditioned on a specific time or location</li>
    <li v-click>May not actually be realistic. We test this in Part 2.</li>
  </ul>
</div>

<!--
Shaurya, 1.5 min.
The wind most balloon-control work trains on comes from Google's Balloon Learning Environment, BLE. It's a VAE trained on ERA5.
It generates one patch, 1,000 by 1,000 km, 48 hours long. Outside the patch the wind is copied from the edge; after 48 hours time runs backwards.
It isn't given a place or a date, so every patch is just "some wind".
The globe: a real ten-day flight in ERA5 at 19 km. It leaves BLE's patch after 11 hours and ends about 12,000 km away. For station keeping that's fine, which is what BLE was built for. For anything that travels, it isn't.
Sources: BLE code, generative/vae.py (FieldShape: ±500 km, 10 levels, 48 h) and env/grid_based_wind_field.py (edge clipping, time reversal after 48 h). Trained on ERA5: BLE's README asks users of its wind dataset to cite ERA5, and the VAE trains on that dataset.
Figure: talk/figures/globe.py
-->

---

# Our generator

Conditional diffusion, trained on ERA5. Unbounded in space & time.

<div class="ours">
  <div class="ours-in">
    <div>Location</div>
    <div>Time of year</div>
    <div>Time of day</div>
  </div>
  <div class="ours-arrow">→</div>
  <div class="ours-out">
    <p class="ours-h">Ours</p>
    <img src="/figures/intro_pair_diffusion.png" />
    <p class="ours-dim">64 × 64 × 18 levels × (u, v) × 4 hours</p>
    <p class="ours-sub">0.25° grid, 16° across, 14 to 20 km</p>
  </div>
  <div class="ours-ble" v-click>
    <p class="ours-h">BLE's VAE</p>
    <img src="/figures/intro_pair_blevae.png" />
    <p class="ours-dim">21 × 21 × 10 levels × (u, v) × 9 frames</p>
    <p class="ours-sub">1,000 km box, 48 hours, no location or time input</p>
  </div>
</div>

<style>
.ours { display: grid; grid-template-columns: 190px 40px 1fr 1fr; gap: 0 18px; align-items: center; margin-top: 1rem; }
.ours-in { border: 1.5px solid #4f7db3; border-radius: 8px; padding: 12px 14px; font-weight: 600; font-size: 1.05rem; line-height: 2; text-align: center; }
.ours-in-h { font-size: 0.8rem; color: #777; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.05em; }
.ours-in-row { margin-bottom: 8px; line-height: 1.2; }
.ours-in-row b { display: block; font-weight: 600; font-size: 1rem; }
.ours-in-row span { font-size: 0.8rem; color: #666; }
.ours-arrow { font-size: 2.2rem; color: #4f7db3; text-align: center; }
.ours-out img, .ours-ble img { width: 100%; border-radius: 4px; display: block; }
.ours-ble { border-left: 1px solid #e3e3e3; padding-left: 18px; }
.ours-dim { font-size: 0.85rem !important; font-weight: 600; text-align: center; margin: 8px 0 2px !important; color: #1a1a1a !important; }
.ours-h { font-weight: 600; text-align: center; margin: 0 0 6px !important; font-size: 1rem !important; color: #1a1a1a !important; }
.ours-sub { font-size: 0.75rem !important; text-align: center; color: #666 !important; margin: 0 !important; }
</style>

<!--
Shaurya, 1.5 min. ONE CLICK.
So we built our own. It's a diffusion model trained on ERA5: it starts from noise and removes it step by step until it looks like real wind.
What makes it useful is what you give it: a location, a time of year and a time of day. It generates wind for that place and time.
What comes out is a block: 64 by 64 grid points at a quarter degree, so 16 degrees across, 18 altitude levels covering the balloon's range from about 14 to 20 km, east and north wind, over 4 hours.
[click] For comparison, BLE's generator. It takes no inputs, just random numbers, so it doesn't know where or when it is. It gives a 1,000 km box with 10 levels over 48 hours. And you can see it: flat horizontal stripes, where ours has the blobs and gradients real wind has.
Sources: time inputs are sin/cos of the annual, semiannual and daily cycles (balloon-env-dev docs/conditional-base-changes.md); 18 levels = ERA5 model levels 49-66, about 53-134 hPa. BLE: generative/vae.py FieldShape (21 x 21 points over ±500 km, 10 levels 50-140 hPa, 9 frames over 48 h, a random 64-number latent).
-->


---

# InfiniteDiffusion

Generate overlapping windows together, so neighbouring tiles agree.

<div class="gen">
  <div>
    <div class="gen-labels"><span>Independent tiles</span><span>InfiniteDiffusion</span></div>
    <img src="/figures/tiling_comparison.png" />
  </div>
</div>

<style>
.gen { display: flex; justify-content: center; margin-top: 1.6rem; }
.gen img { height: 310px; width: auto; display: block; }
.gen > div { display: flex; flex-direction: column; align-items: center; }
.gen-labels { display: grid; grid-template-columns: 1fr 1fr; width: 100%; text-align: center; font-weight: 600; font-size: 0.95rem; margin-bottom: 4px; }
.gen .caption { font-size: 0.8rem !important; margin-top: 4px; }
</style>

<!--
Shaurya, 1 min. No clicks.
One block is 16 degrees across. To go further, the naive thing is to generate blocks independently and put them side by side.
That gives seams, left: each tile made its own weather. InfiniteDiffusion covers the region with overlapping windows and denoises them all together, averaging where they overlap at every step, so neighbours have to agree. Right: one continuous field. It works the same way in time.
Two properties that matter for a simulator. Random access: ask for the wind at any place and time and it only generates what that query needs, so a long flight only costs what it passes through. Deterministic: ask again and you get the same wind, so every controller flies the same world.
-->

---

# Global generator (in progress)

<div class="row" style="gap: 2rem; align-items: center; margin-top: 0.4rem">
  <div style="flex: none; width: 340px">
    <video src="/figures/globe-wind.mp4" poster="/figures/globe-wind.jpg" autoplay loop muted playsinline style="width: 100%; border-radius: 10px"></video>
    <p class="caption" style="font-size: 0.75rem !important">Generated wind at about 20 km, 10 July 2023</p>
  </div>
  <ul class="glist">
    <li>Two stages: whole globe coarse, then fine detail</li>
    <li>Fine stage tiles in space, InfiniteDiffusion in time</li>
    <li>Equal-area grid on the sphere</li>
    <li>Trained on 2020–2023 ERA5</li>
  </ul>
</div>

<style>
.glist li { font-size: 1.1rem; margin-bottom: 0.9rem; }
</style>

<!--
Shaurya, 45 seconds.
That was the summer model. What I'm building now goes further. This is its wind at about 20 km, generated, not ERA5.
It covers the whole globe in two stages: a coarse model for the whole planet in one pass, then a second one that fills in fine detail by tiling overlapping patches in space. InfiniteDiffusion in time is the plan, so a flight never runs out of wind; the time tiling isn't built yet, and the video is one 13-hour block. Equal-area (HEALPix) grid. Stage 1 trained on ERA5 2020-2023 (~1.8 deg, 6-hourly); Stage 2 on 2022-2023 (~0.23 deg, hourly). Configs: balloon-env-dev/unicorn/configs/stage1_hpx32_ft.yaml, stage2_hpx256_bl.yaml.
The realism numbers in the next part are for the summer model.
Video: blog/tools/wind_globe.py, Stage 1 -> Stage 2 output for 2023-07-10, ~53 hPa.
-->
---

<div class="divider">
  <p class="divider-label">Part 2</p>
  <h1>Evaluating the generator</h1>
  <p class="divider-q">Does our wind look like real wind?</p>
  <Sections :current="2" />
</div>

<!--
Shaurya. A few seconds.
-->

---

# Realism checks
Generated wind against held-out ERA5.

<Checks />

<p v-click class="caption" style="margin-top: 0.8rem">ERA5 reference: one half of held-out ERA5 scored against the other half. Closer to it is better.</p>

<!--
Shaurya, 1 min.
We score generated wind against held-out ERA5 on four properties, one per thing that shapes a flight.
Wind speeds: how hard the balloon is pushed, including extremes.
Structure in space: how the wind varies along the path.
Change with altitude: the options the balloon steers between. The one that matters most for control.
Change over time: how the wind evolves during a flight.
Reference: two halves of real ERA5 still differ, so that's the best possible score. Closer to it is better. (Say "reference", not "floor": "floor" confused people before.)
Reference: held-out ERA5, NE Pacific, days 8-14 of Jan/Apr/Jul/Oct 2023, not used in training.
-->

---

# Realism results

Closer to ERA5 than BLE's generator on all four checks. Change over time is still far off for both.

<img v-if="$clicks < 1" src="/figures/realism-1.png" class="fig" style="max-height: 320px" />
<img v-else src="/figures/realism-2.png" class="fig" style="max-height: 320px" />
<div v-click />

<!--
Shaurya, 2 min. One click adds simplex noise.
Lower is better. The dotted line is the ERA5 reference; closer to it is better.
Speeds: distance between wind-speed distributions, 2.98 vs BLE's 4.56. Ours still misses extremes (tail error 13.5 m/s vs the ERA5 reference of 2.9).
Space: the finest scale it keeps. Ours keeps detail down to about 340 km, BLE about 840. The ERA5 reference and simplex both show 56 km, which is the grid limit: they keep detail all the way down, so 56 means 'as fine as the grid can show', not a measured scale.
Altitude: error in the shear between neighbouring levels, mean of east-west and north-south. Ours 1.0, BLE 2.1. Honest note: on north-south alone, BLE is better (0.61 vs 1.01).
Time: dispersion error, how far off the spreading of drifting particles is over time compared with ERA5. Ours 0.81, BLE 1.79, ERA5 reference 0.18. Better than BLE, but still about 4 times the reference, so time is our weakest check. The summer model makes 4-hour blocks, so we tiled it in time with InfiniteDiffusion to get 24-hour runs. Caveat: BLE's run is 48 hours at 6-hourly, ours 24 hours hourly, two runs of ours, so treat this panel as rough.
Click: simplex noise, the kind BLE uses as forecast error, fitted to ERA5. It beats both generators on all four. That's the next slide.
If asked about the "structure" measure: the spectral residual (SR) is the average gap between our energy spectrum and ERA5's, across all scales. Effective resolution is the smallest scale where we still have half of ERA5's energy. They tell the same story (ours 1.53 vs BLE 2.73); effective resolution is in km, so it's easier to read. Full table in the backup.
Caveats: BLE and simplex are one sample each; ours is 112 blocks. No repeated runs, so no error bars.
Source: balloon-env-dev/docs/benchmark-reports/benchmark_v2_4yr_250000.md, summer model at 4 years / 250k steps. Figure: talk/figures/realism.py Time panel: balloon-env-dev/docs/benchmark-reports/temporal_4yr_250000.md (run 2026-11-12).
-->

---

# Limits of realism checks

- Simplex noise fitted to ERA5's statistics scores better than both generators.
- It doesn't have any weather features like jets or storms and doesn't condition on time/place.
- These checks just measure statistics, not usefulness for control.

<p v-click style="font-size: 1.5rem; font-weight: 600; margin-top: 2.2rem; color: #1a1a1a">What does a wind model need to get right for a controller built on it to transfer to real wind?</p>

<!--
Shaurya, 1 min. The turn in the talk.
The simplex noise was fitted to these statistics: per-level mean and spread, correlations between levels and hours, and its scale tuned to the spectrum. So it scores well by construction.
But it's random texture: no jets or storms that persist and move, and the same wind regardless of place or season. So a good score here doesn't tell us a model is good for control.
[click] That's the question for the rest of the talk: what a wind model needs to get right so a controller built on it, planned or trained, works in real wind. And which properties can be coarse, and so cheap, without hurting that. First we need a way to measure it. Rohan.
-->

---

<div class="divider">
  <p class="divider-label">Part 3</p>
  <h1>Measuring transfer</h1>
  <p class="divider-q">How do we measure whether a controller built on a wind model works in real wind?</p>
  <Sections :current="3" />
</div>

<!--
ROHAN. About 10 seconds.

First up, how we measure whether a wind model is any good for control.
-->

---

# Make the wind worse, measure how performance changes

<div class="pair" style="align-items: center">
  <div>
    <img src="/fig/split-truth.png" class="round" style="max-height: 240px" />
    <p class="caption"><b>ERA5</b><br>The balloon flies in this</p>
  </div>
  <div v-click="1">
    <img src="/fig/split-coarse.png" class="round" style="max-height: 240px" />
    <p class="caption"><b>Degraded ERA5</b><br>The controller plans on this</p>
  </div>
</div>

<p v-click="2" style="text-align: center; font-size: 1.4rem; margin-top: 1.2rem">How much worse does the controller do?</p>

<!--
ROHAN. About 1 minute. TWO CLICKS.

So this is the method for the rest of the talk. ERA5 is the true wind, the one that actually moves the balloon.

[click] We degrade it on purpose, in one way we pick, and hand the degraded version to the controller. The controller plans on that, but the balloon still flies in ERA5.

[click] Then we check how much worse the controller does. If degrading something hurts control a lot, the wind model needs to get that thing right. If it doesn't hurt, then it doesn't matter.

To measure how much worse, we need one number that works for every kind of degrading and every task. For that we need a best case and a worst case.
-->

---

# Controllers and bounds

<div class="bounds">
  <div class="bounds-row">
    <b>Hindsight</b>
    <span>Plans on the true wind.</span>
    <div>

$V_{\text{hind}} = \mathbb{E}_w\big[\min_a J(a,w)\big]$

</div>
  </div>
  <div class="bounds-row" v-click>
    <b class="c-mpc">Best policy</b>
    <span>Reacts to where the balloon actually is, for example MPC.</span>
    <div>

$V_{\text{policy}} = \min_\pi \mathbb{E}_w\big[J(\pi,w)\big]$

</div>
  </div>
  <div class="bounds-row" v-click>
    <b class="c-fixed">Fixed plan</b>
    <span>One plan from the forecast that is made at launch and is never changed.</span>
    <div>

$V_{\text{plan}} = \min_a \mathbb{E}_w\big[J(a,w)\big]$

</div>
  </div>
</div>

<div v-click style="text-align: center; margin-top: 1.2rem; font-size: 1.5rem">

$V_{\text{hind}} \;\le\; V_{\text{policy}} \;\le\; V_{\text{plan}}$

</div>

<style>
.bounds { margin-top: 1rem; border-bottom: 1px solid #e3e3e3; }
.bounds-row { display: grid; grid-template-columns: 150px 1fr 300px; gap: 1rem; align-items: center; padding: 0.5rem 0; border-top: 1px solid #e3e3e3; }
.bounds-row b { font-size: 1.2rem; font-weight: 600; }
.bounds-row span { font-size: 1.05rem; color: #333; }
.bounds-row > div { text-align: right; }
.bounds-row .katex { font-size: 1.35em; }
</style>

<!--
ROHAN. About 1.5 minutes. THREE CLICKS.

To say how good MPC is, we need something to compare it against, so we have two bounds. J is the cost of a flight, roughly how far from the goal you were. w is the wind, which is uncertain given the forecast.

The first bound is hindsight. For each wind, you take the best plan for that wind, then average. Nobody can actually do that, so it's the ceiling.

[click] The best policy reacts to where the balloon actually is. MPC is one policy. It replans every move from where the balloon actually is, so it sits at or above the best policy.

[click] Then the fixed plan. It's one plan, picked at launch from the forecast, and it has to work for every wind. No feedback.

[click] Reacting can only help compared to a fixed plan, and it can't beat knowing the wind. So hindsight ≤ best policy ≤ fixed plan. What we care about is how close MPC gets to hindsight.

All three use the same planner, the one from Myles's paper, so the only differences are what they know and when they plan. Next, one flight with all three.

[If asked: hindsight is really the best plan our search found on the true wind, not a proven optimum. The fixed plan plans on the forecast mean, or on sampled winds when we hedge.]
-->

---

# Choosing the task: only goals downwind are reachable

<div style="width: 760px; margin: 0.6rem auto 0"><Layers name="scouting" :n="3" :shown="1" /></div>

<!--
ROHAN. About 1 minute. TWO CLICKS.

Before the example flight, I'll say how we picked the task. We didn't want one that flattered us, so we wrote down a rule first and then tested twenty candidates, four directions and five distances.

On the left is how often even hindsight reaches the goal. If the best possible controller can't get there, the task is too hard to learn anything from. Upwind, crosswind and random goals all fail. Only downwind goals are reachable.

[click] On the right is whether steering matters at all. If a blind plan does nearly as well as hindsight, the wind is doing all the work. Close downwind goals fail that check.

[click] One task passes both, 2,000 kilometres downwind. That's the task in the example flight and everything after it.

But notice what this says by itself. In this wind, a balloon basically goes downwind or nowhere.
-->

---

# Example flight to a goal 2,000 km downwind

<SlidevVideo autoplay controls style="height: 418px; margin: 0 auto; border-radius: 8px" poster="/fig/flight-s13.png">
  <source src="/fig/flight-s13.mp4" type="video/mp4" />
</SlidevVideo>

<!--
ROHAN. About 1.5 minutes. The film plays by itself, about 10 seconds. Replay it with the controls if you want to talk over it.

One flight with all three, to make it concrete. They all launch from the same spot into the same wind. The controllers' wind here has random error added, about as wrong as using yesterday's wind. More on that in a minute. The goal is the red ring. The bottom panel is altitude over time.

For the first day they all ride the same wind east.

Orange is the fixed plan. It made its whole plan at launch from the forecast, and it stays low the whole flight. The forecast was off, so it drifts past the goal and never comes back. The closest it gets is about 800 km.

Black is hindsight. It turns south at the right time and gets there a little after four days.

Blue is MPC. It starts on a worse route, replans as it goes, loops back, and gets in about nine hours after black.

The bars on the right are each one's cost as it flies, which is distance from the goal added up over the flight. A bar stops growing once its balloon reaches the goal. At the end, hindsight is about 25,000 km, MPC about 32,000, and the fixed plan about 119,000. The next slide turns those three numbers into one score.

[If asked: the bars stop counting when a balloon arrives, which is the cost the controllers plan on. The score slide counts all 50 moves, so hindsight and MPC come out higher there, about 30,000 and 36,000. Both give 92 percent.]

[Launch 13 of the 32, from the 2026-10-24 sweep.]
-->

---

# Scoring MPC

Distance to the goal, summed over the flight

<div class="score-eq">

$\text{score} = \dfrac{\textcolor{#e2852e}{\text{fixed}} - \textcolor{#2f6fde}{\text{MPC}}}{\textcolor{#e2852e}{\text{fixed}} - \text{hindsight}}$

</div>

<div style="border-top: 1px solid #e3e3e3; padding-top: 6px">
  <ScoreRow title="Example flight" :fixed="118639" :mpc="36487" :hindsight="29690" />
</div>

<div style="border-top: 1px solid #e3e3e3; border-bottom: 1px solid #e3e3e3; padding-top: 6px" v-click>
  <ScoreRow title="All 32 launches" :fixed="88911" :mpc="60446" :hindsight="52129" />
</div>

<style>
.score-eq { text-align: center; margin: 0.4rem 0 0.6rem; }
.score-eq .katex { font-size: 1.6em; }
</style>

<!--
ROHAN. About 1.5 minutes. ONE CLICK.

So that gives us a ruler. For every flight we have a worst case, which is the fixed plan, and a best case, which is hindsight. The score is how much of the gap between them MPC closes. Cost here is distance to the goal after every move, added up over the whole flight.

On the example flight we just watched, over all 50 moves, the fixed plan was about 119,000 km, hindsight about 30,000, and MPC about 36,000. So MPC closed 92 percent of the gap.

[click] Over all 32 launches, at 3.2 metres a second, the fixed plan is about 89,000 km, hindsight about 52,000, and MPC about 60,000. That's 77 percent. The rest is what the wrong forecast cost us.

It also isn't 100 percent with a perfect forecast. With no error it's 87 percent, because the planner only looks 60 hours ahead while hindsight gets to plan the whole flight.

This is the number that lets us compare across conditions. If we make the wind model worse in some way and the score drops, that property mattered. If it doesn't drop, it didn't.

[Numbers: these are averages over the same 32 launches, with MPC planning 60 hours ahead on the forecast mean. Scored launch by launch, the median at 3.2 is 80 percent.]
-->

---

# Main experiment

Make ERA5 worse **one property at a time** and observe the change in performance.

<div class="side">
  <div>
    <img src="/fig/split-coarse.png" class="round" />
  </div>
  <ul>
    <v-clicks>
    <li><b>Altitude levels:</b> decreasing the number</li>
    <li><b>Grid:</b> making it coarser (lower resolution)</li>
    <li><b>Frames in time:</b> spacing them further apart</li>
    <li><b>Kind of error:</b> offset or random, same or different by altitude</li>
    </v-clicks>
  </ul>
</div>


<!--
ROHAN. About 45 seconds. FOUR CLICKS.

So, the main experiment, up front. We make ERA5 worse one property at a time and see how the score changes.

[click] Fewer altitude levels. [click] A coarser grid. [click] Frames further apart in time. [click] And the kind of error, at the same size.

Shaurya goes through these in detail at the end. Before that, we ran a pilot using the simplest way to make the wind wrong, which is random error.
-->

---

<div class="divider">
  <p class="divider-label">Part 4</p>
  <h1>Forecast-error experiment</h1>
  <p class="divider-q">How much random error can control tolerate?</p>
  <Sections :current="4" />
</div>

<!--
ROHAN. About 10 seconds.

The pilot is the simplest way to make the wind wrong.
-->

---

# Pilot: random error

<div class="row center" style="gap: 14px">
  <div class="col" style="width: 240px">
    <div class="label">Truth (ERA5)</div>
    <img src="/fig/split-truth.png" class="round" />
  </div>
  <div class="op" v-click="1">−</div>
  <div class="col" style="width: 240px" v-click="1">
    <div class="label">Error</div>
    <img src="/fig/split-error.png" class="round" />
  </div>
  <div class="op" v-click="2">=</div>
  <div class="col" style="width: 240px" v-click="2">
    <div class="label">Forecast</div>
    <img src="/fig/split-forecast.png" class="round" />
  </div>
</div>

<ul v-click="3" style="margin-top: 1rem">
  <li>The balloon flies in the truth, and the controller only sees the forecast.</li>
  <li>Error is a Gaussian process draw (smooth noise), fitted to how much ERA5 wind changes in a day</li>
  <li>Noise levels: 0, 0.8, 1.6, <b>3.2</b>, 6.4, 12.8 m/s. 3.2 ≈ using yesterday's wind as the forecast</li>
  <li>32 launches per level, the same for every controller</li>
</ul>

<style>
ul li { font-size: 1.05rem; margin-bottom: 0.3rem; }
</style>

<!--
ROHAN. About 1.5 minutes. THREE CLICKS.

The truth is ERA5, same as before. That's the wind that actually moves the balloon.

[click] We take the truth and subtract an error. The error is smooth random noise, a draw from a Gaussian process. We fitted it to how much ERA5's wind changes over a day, so at full size it's like using yesterday's wind as the forecast. We pick how big it is.

[click] What's left is the forecast, and the controller only ever sees that.

[click] We test six noise levels, from zero, which is a perfect forecast, up to 12.8 metres a second. 3.2 is the yesterday's-wind level. A real forecast is probably more accurate than that. At each level we fly the same 32 launches with every controller, so when we compare two controllers, it's the same flights.

[If asked: Gaussian process with a Matérn kernel. Patches about 540 km across, 1.3 km in altitude, 9 hours in time.]
-->

---

# Varying the forecast error, and three MPC settings

<ul class="vary">
  <li><b>Forecast error:</b> 0, 0.8, 1.6, 3.2, 6.4, 12.8 m/s (3.2 is about what real forecasts miss by)</li>
</ul>

<div v-click>
<p class="vary-h">Three MPC settings</p>
<ul class="vary">
  <li><b>How far ahead it plans:</b> 15, 30, 60 or 150 hours</li>
  <li><b>What it plans on:</b> one forecast, or many draws from its belief</li>
  <li><b>Its own wind readings:</b> used or not used</li>
</ul>
</div>

<style>
.vary { margin-top: 1rem; }
.vary li { font-size: 1.2rem; margin-bottom: 0.7rem; }
.vary-h { font-size: 1.2rem !important; font-weight: 600; color: #1a1a1a !important; margin-top: 1.6rem !important; }
</style>

<!--
ROHAN. About 1 minute. ONE CLICK.

In the pilot we vary how wrong the forecast is. Six levels, from a perfect forecast up to four times a real one. 3.2 is about what real forecasts miss by when you check them against balloons (3.2 to 3.8 m/s, Cadiou et al.).

[click] We also vary three MPC settings. One is how far ahead it plans, from 15 hours to 150. Another is whether it plans on one forecast or on many draws from its belief about the wind. The last is whether it uses the wind readings the balloon takes as it flies, or not.

Every controller flies the same 32 launches, so a difference is always the same flight with one thing changed. That's about 8,000 flights, run on the cluster.
-->

---

# Replanning is worth a lot

<div style="width: 560px; margin: 0.6rem auto 0"><Layers name="bounds" :n="5" :shown="2" /></div>

<!--
ROHAN. About 1.5 minutes. THREE CLICKS.

Okay, results. Forecast error is on the bottom axis. Up the side is distance to the goal summed over the flight, so lower is better. Hindsight is the flat black line, since it never sees the forecast.

[click] The fixed plan. It gets worse as the forecast gets worse, which is what you'd expect.

[click] And MPC. It's much lower, and it stays close to hindsight until the forecast is pretty bad.

[click] The gap between orange and blue is what replanning is worth, since that's the only difference between them. With a real forecast, it saves about 28,000 km of summed distance. Even with a perfect forecast it's still about 16,000.

[If asked why it helps with a perfect forecast: this is distance over the whole flight. The fixed plan reaches the goal and then drifts away. MPC stays. Intervals: 28,465 plus or minus 10,056, and 16,358 plus or minus 7,904.]
-->

---

# Planning further ahead stops helping

<div style="width: 560px; margin: 0.6rem auto 0"><Layers name="breaking" :n="5" :shown="1" /></div>

<!--
ROHAN. About 1.5 minutes. FOUR CLICKS, one per line.

Next is the horizon result. Same axis across the bottom. Up the side is how far MPC is from hindsight, so zero would be perfect.

[click] That's MPC planning 15 hours ahead.

[click] 30 hours. Better, when the forecast is good.

[click] 60 hours. Better again.

[click] And 150 hours, the whole flight. On the left side the lines are in order. The further ahead you plan, the closer you get to hindsight.

But on the right, by the time the forecast is as wrong as a real one, the lines have basically merged. Planning further ahead on a bad forecast doesn't really help. Going past 60 hours never detectably helped, at any error level.

So there's a breaking point, somewhere between half of a real forecast's error and all of it.

[If asked: paired comparisons between neighbouring horizons are detectable up to 1.6 m/s and not from 3.2 on. Bars are 95 percent intervals over 32 launches.]
-->

---

# Planning on many possible winds

<div style="width: 560px; margin: 0.6rem auto 0"><Layers name="hedging" :n="3" :shown="1" /></div>

<!--
ROHAN. About 1 minute. TWO CLICKS.

Next is what MPC plans on. Normally the planner scores each plan on the one forecast, the mean wind. The alternative is to draw a bunch of possible winds around the forecast and score each plan on its average over all of them. That's the textbook way to plan under uncertainty. Below zero means it helped.

[click] For the fixed plan it helps, by about 10,000 km, up to 3.2 metres a second of error. That makes sense, since it only gets one shot. So a plan that holds up when the forecast is wrong is worth having.

[click] For MPC it does nothing, at any error level.

We checked whether that's just because our uncertainty happens to be exactly right, by making it twice too wide and twice too narrow. Same answer. MPC fixes its mistakes as they happen, so preparing for them ahead of time doesn't help.

[If asked: 64 sampled winds for the fixed plan, 16 for MPC. Bars are 95 percent intervals over 32 launches.]
-->

---

# Using the balloon's own wind readings

<div style="width: 560px; margin: 0.6rem auto 0"><Layers name="readings" :n="3" :shown="1" /></div>

<!--
ROHAN. About 1 minute. TWO CLICKS.

The last result from the sweep surprised us a bit. As the balloon flies it measures the wind where it is, and MPC can use that to correct its forecast. Loon did this. So we ran MPC with that on and with it off.

[click] Each dot is one launch, showing the difference between using the readings and ignoring them. Below zero means the readings helped.

[click] The black diamonds are the averages, and they sit on zero the whole way across. We made 24 of these comparisons, across planning horizons and error levels. Readings didn't detectably help in any of them.

We think it's a scale thing. The balloon gets one reading every three hours along a single line, and the forecast error comes in patches about 500 km wide. One reading doesn't tell you much about where you're headed next.

So the gain over the fixed plan comes from replanning from where you actually are. It doesn't come from learning about the wind.

[If asked: on the cost that stops counting at arrival, readings help detectably in 1 of the 24 comparisons (30 hours ahead, 1.6 m/s, by about 300 km), which is about what you'd expect by chance.]
-->

---

# Preliminary results

<ul class="prelim">
  <li>Control tolerates realistic random error: MPC closes 77% of the gap at 3.2 m/s, 87% with a perfect forecast</li>
  <li v-click="1">The score is sensitive enough to show a drop, with a breaking point around a real forecast's error</li>
  <li v-click="2">MPC settings for what comes next: plan 60 h ahead, on one forecast</li>
</ul>

<p v-click="3" style="margin-top: 2rem; font-size: 1.3rem; color: #1a1a1a"><b>Open question:</b> which specific properties matter? Random error changes everything a little at once, so it can't separate them.</p>

<style>
.prelim { margin-top: 1.2rem; }
.prelim li { font-size: 1.2rem; margin-bottom: 0.9rem; }
</style>

<!--
ROHAN. About 1 minute. THREE CLICKS.

So, preliminary results.

First, control handles realistic random error fine. At a real forecast's error MPC still closes 77 percent of the gap, versus 87 with a perfect forecast.

[click] Second, the method works. The score moves when the wind gets worse, and there's a clear breaking point. So it's sensitive enough for the real question.

[click] Third, it settled the MPC settings for everything after. Plan 60 hours ahead, since going further never helped. Plan on the one forecast, since hedging didn't help MPC. And readings didn't matter either way.

[click] What's left is which properties of a wind model matter. Random error makes everything a bit wrong at once, so it can't pull them apart. That's what the main experiment is for. Back to Shaurya.
-->

---

<div class="divider">
  <p class="divider-label">Part 5</p>
  <h1>Fidelity experiment (planned)</h1>
  <p class="divider-q">Which properties of a wind model matter for transfer?</p>
  <Sections :current="5" />
</div>

<!--
SHAURYA takes over. About 15 seconds. No clicks.
Rohan showed that control tolerates random error about the size of a real forecast's.
But random error blurs everything a little. It never removes one thing, so it can't say which properties of a wind model matter.
This part is the experiment we plan to run next. It's planned, so there are no results here.
Random error blurs everything. Next: degrade one property at a time.
-->

---

# Fidelity experiment

Copies of ERA5, each with one property degraded.

<svg viewBox="0 0 884 262" style="width: 100%; display: block; margin-top: 30px">
  <defs>
    <clipPath id="fx-r1"><rect x="0" y="24" width="230" height="159" rx="8" /></clipPath>
    <clipPath id="fx-r2"><rect x="330" y="24" width="230" height="159" rx="8" /></clipPath>
  </defs>
  <text x="115" y="14" text-anchor="middle" style="font-size: 15px; font-weight: 600" fill="#1a1a1a">True wind (ERA5)</text>
  <image href="/fig/split-truth-px.png" x="0" y="24" width="230" height="159" preserveAspectRatio="xMidYMid slice" clip-path="url(#fx-r1)" />
  <text x="280" y="78" text-anchor="middle" style="font-size: 13px" fill="#888">degrade</text>
  <text x="280" y="94" text-anchor="middle" style="font-size: 13px" fill="#888">one property</text>
  <path d="M 242 104 H 316" stroke="#b7bcc6" stroke-width="2.5" />
  <path d="M 308 97 L 318 104 L 308 111" fill="none" stroke="#b7bcc6" stroke-width="2.5" stroke-linejoin="round" />
  <text x="445" y="14" text-anchor="middle" style="font-size: 15px; font-weight: 600" fill="#1a1a1a">Degraded copy</text>
  <image href="/fig/split-coarse-px.png" x="330" y="24" width="230" height="159" preserveAspectRatio="xMidYMid slice" clip-path="url(#fx-r2)" />
  <text x="445" y="202" text-anchor="middle" style="font-size: 13px" fill="#888">Example: grid 8× coarser, 0.25° → 2°</text>
  <g v-click="1">
    <path d="M 572 104 H 646" stroke="#b7bcc6" stroke-width="2.5" />
    <path d="M 638 97 L 648 104 L 638 111" fill="none" stroke="#b7bcc6" stroke-width="2.5" stroke-linejoin="round" />
    <text x="772" y="14" text-anchor="middle" style="font-size: 15px; font-weight: 600" fill="#1a1a1a">Controller</text>
    <rect x="660.75" y="24.75" width="222.5" height="157.5" rx="8" fill="#fff" stroke="#b7bcc6" stroke-width="1.5" />
    <text x="680" y="60" style="font-size: 18px; font-weight: 600" fill="#1a1a1a">MPC</text>
    <text x="680" y="82" style="font-size: 14px" fill="#555">plans on the copy</text>
    <path d="M 772 190 V 238 H 115 V 196" fill="none" stroke="#4f7db3" stroke-width="2.5" stroke-linejoin="round" />
    <path d="M 108 204 L 115 194 L 122 204" fill="none" stroke="#4f7db3" stroke-width="2.5" stroke-linejoin="round" />
    <rect x="345" y="227" width="200" height="22" fill="#fff" />
    <text x="445" y="243" text-anchor="middle" style="font-size: 14px; font-weight: 600" fill="#4f7db3">then flown in the true wind</text>
  </g>
  <g v-click="2">
    <line x1="676" y1="104" x2="868" y2="104" stroke="#e3e3e3" stroke-width="1" />
    <text x="680" y="139" style="font-size: 18px; font-weight: 600" fill="#1a1a1a">PPO</text>
    <text x="680" y="161" style="font-size: 14px" fill="#555">trained on the copy</text>
  </g>
</svg>

<p v-click="3" style="text-align: center; margin-top: 26px">The drop in score is how much that property matters for transfer.</p>

<!--
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
-->

---

# Candidate properties

<ul class="cp">
  <li><b>Altitude levels</b></li>
  <li><b>Grid spacing</b></li>
  <li><b>Time between frames</b></li>
  <li><b>Locations:</b> one patch of ERA5, or many</li>
  <li><b>Kind of error, same size:</b> e.g. Gaussian process or simplex</li>
</ul>

<p v-click class="pt-bet"><b>Expected:</b> altitude levels matter most. The balloon steers on differences between altitudes.</p>

<style>
.cp { margin-top: 1.6rem !important; font-size: 1.3rem; padding-left: 1.4rem !important; }
.cp li { margin: 0.7rem 0 !important; }
.cp b { font-weight: 600; color: #1a1a1a; }
.pt-bet { margin-top: 2rem !important; color: #1a1a1a !important; }
.pt-bet b { font-weight: 600; color: #4f7db3; }
</style>

<!--
Shaurya, about 45 seconds. ONE CLICK.
These are the candidates. None of it is locked, and each is made worse in steps, one at a time.
The first three are resolution: how many altitude levels, how fine the grid, how far apart the frames are in time.
Locations: one patch of ERA5, or winds from many places. If one patch is enough, the generator can output less and train on less.
Kind of error: the error in Rohan's pilot was a Gaussian process draw. Here the size stays the same and only its kind changes, for example a Gaussian process against simplex noise.
[click] Our bet is that altitude levels matter most, because the balloon steers on the differences between altitudes.
Why we expect it: in Rohan's part only downwind goals were reachable, and the balloon's only choice is which altitude to sit at.
-->

---

# Tasks

Each degraded copy is tested on three tasks.

<div class="tk">
  <div><img src="/figures/task-station.png" /><p><b>Station keeping</b><br>stay near one point</p></div>
  <div><img src="/figures/task-navigation.png" /><p><b>Navigation</b><br>reach waypoints</p></div>
  <div><img src="/figures/task-distance.png" /><p><b>Distance</b><br>get as far from launch as possible</p></div>
</div>

<style>
.tk { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin-top: 1.6rem; text-align: center; }
.tk img { height: 200px; width: auto; margin: 0 auto; display: block; }
.tk p { font-size: 0.95rem !important; line-height: 1.35 !important; margin: 8px 0 0 !important; color: #555 !important; }
.tk b { color: #1a1a1a; font-size: 1.05rem; }
</style>

<!--
Shaurya, about 30 seconds.
Each degraded copy is tested on three tasks, with both controllers. Station keeping and navigation you saw at the start. The third is distance: get as far from launch as you can.
A property might matter for one task and not another. Our guess, not a result: station keeping needs winds that oppose each other; distance mostly needs one fast wind.
None of this is locked. Which task should lead is one of the questions at the end.
The task pictures are schematic. Figures: talk/figures/tasks.py
-->

---

# Minimal generator

<table class="pt mg">
  <tr><th>Generator setting</th><th>Today</th><th>Coarsest that still transfers</th></tr>
  <tr><td>Altitude levels</td><td>18</td><td class="q">?</td></tr>
  <tr><td>Time between frames</td><td>1 h</td><td class="q">?</td></tr>
  <tr><td>Grid spacing</td><td>0.25°</td><td class="q">?</td></tr>
</table>

<p v-click class="pt-bet"><b>Result:</b> the cheapest generator that still transfers.</p>

<style>
.pt { width: 100%; margin-top: 1.4rem; border-collapse: collapse; font-size: 1.15rem; }
.pt td, .pt th { padding: 0.75rem 0.4rem; border-bottom: 1px solid #e6e8ec; border-top: none; text-align: left; }
.pt th { font-size: 0.8rem; font-weight: 500; color: #888; text-transform: uppercase; letter-spacing: 0.05em; background: none; }
.pt td:first-child { font-weight: 600; width: 260px; color: #1a1a1a; }
.mg td:nth-child(2) { width: 160px; color: #333; }
.mg td.q { color: #4f7db3; font-weight: 600; }
.mg-note { font-size: 0.8rem !important; color: #888 !important; margin-top: 0.6rem !important; }
.pt-bet { margin-top: 1.4rem !important; color: #1a1a1a !important; }
.pt-bet b { font-weight: 600; color: #4f7db3; }
</style>

<!--
Shaurya, about 45 seconds. ONE CLICK.
Three of the properties are settings of our generator: how many altitude levels it makes, how often it gives a frame, and how fine its grid is. Each one costs compute. Today they're 18 levels, hourly, a quarter degree. We picked those by judgement.
The experiment fills in the last column: for each setting, the coarsest step where the score still holds.
One limit: the experiment's truth is ERA5 at 2 degrees and every 3 hours, so it can say how much coarser than that is fine, not anything about finer detail.
[click] Put the three together and that's the cheapest generator that still transfers.
If asked about locations: it's about training data, not resolution, so it isn't a row here. If one patch is enough, the generator can be smaller and train on less.
If asked about kind of error: it isn't a setting, so it's not here. It tells us which mistakes a generator can afford.
-->

---

# Questions for you

<div class="qs">
  <div class="q">
    <div class="q-n">1</div>
    <div>
      <p class="q-t">Which other properties would you test?</p>
      <p class="q-c">So far: altitude levels, grid spacing, time between frames, locations, kind of error.</p>
    </div>
  </div>
  <div class="q">
    <div class="q-n">2</div>
    <div>
      <p class="q-t">Is degraded ERA5 a fair stand-in for a real generator's errors?</p>
      <p class="q-c">A degraded copy is wrong in one way. A generator is wrong in several ways at once.</p>
    </div>
  </div>
  <div class="q">
    <div class="q-n">3</div>
    <div>
      <p class="q-t">Which task should lead?</p>
      <p class="q-c">Station keeping, navigating to waypoints, or distance from launch.</p>
    </div>
  </div>
</div>

<style>
.qs { margin-top: 2rem; }
.q { display: grid; grid-template-columns: 44px 1fr; align-items: start; margin-bottom: 2.2rem; }
.q-n { font-size: 1.6rem; font-weight: 600; color: #4f7db3; line-height: 1.25; }
.q-t { font-size: 1.6rem !important; font-weight: 500; color: #1a1a1a !important; line-height: 1.25 !important; margin: 0 !important; }
.q-c { font-size: 1rem !important; color: #888 !important; margin: 6px 0 0 !important; line-height: 1.4 !important; }
</style>

<!--
Shaurya, about 0.5 min. NO CLICKS. Then 30 min discussion.
Three things we'd like your input on before we run this.
One: the properties. We have four so far. What else would you degrade?
Two: the stand-in. A degraded copy of ERA5 is wrong in one way, which is why a drop in score has one cause. A real generator is wrong in several ways at once. Does what we learn from the copies carry over?
Three: the tasks. We have station keeping, navigating to waypoints, and getting as far from launch as possible. Which should we run first and report as the main result?
Leave this slide up for the discussion.
If the discussion goes to training data: the experiment doesn't measure it; it's about resolution.
-->

---

# Backup: full realism table

<div class="compact">

| Check | Metric (lower is better unless noted) | ERA5 reference | Simplex | BLE's VAE | Ours (summer) |
|---|---|---|---|---|---|
| Speeds | W1(u \| place, time) (m/s) | 2.08 | 4.69 | n/a | 3.00 |
| | W1(v \| place, time) (m/s) | 2.67 | 2.08 | n/a | 4.04 |
| | W1 wind speed (m/s) | 1.11 | 1.69 | 4.56 | 2.98 |
| | Tail error, top 0.1% (m/s) | 2.92 | 2.34 | 18.94 | 13.54 |
| Space | SR_E / SR_div / SR_vort | 0.25 / 0.25 / 0.25 | 0.75 / 0.87 / 0.72 | 2.73 / 5.23 / 2.16 | 1.53 / 1.56 / 1.53 |
| | Effective resolution (km) | 56† | 56† | 842 | 337 |
| Altitude | W1 shear u / v ((m/s)/km) | 0.20 / 0.10 | 0.31 / 0.97 | 3.62 / 0.61 | 1.00 / 1.01 |
| | Opposing-wind fraction (closer to ERA5's 0.20) | 0.20 | 0.23 | 0.58 | 0.04 |
| Time | SR_time | 0.28 | 1.19 | 4.19 | 2.43 |
| | Dispersion error | 0.18 | 0.70 | 1.79 | 0.81 |
| | Final spread ratio (closer to ERA5's 1.04) | 1.04 | 0.53 | 0.11 | 0.20 |

</div>

<p class="caption" style="text-align: left; font-size: 0.8rem">† resolved down to the grid (56 km). n/a: BLE takes no place or time input. Time: ours tiled in time to 24 h (2 runs), BLE 48 h at 6-hourly, so rough. Summer model at 4 years / 250k steps. BLE and simplex are one sample each.</p>

<!--
Backup. Source: balloon-env-dev/docs/benchmark-reports/benchmark_v2_4yr_250000.md. W1(u|c) and W1(v|c) recomputed from that run's cached samples (the report gives only their mean, 3.52).
Time rows: balloon-env-dev/docs/benchmark-reports/temporal_4yr_250000.md (run 2026-11-12). BLE's 4.19 / 0.11 reproduce the poster. Ours is new; the poster's 5.46 / 0.20 for "ours" were a toy baseline, and our real spread ratio happens to also be 0.20. Simplex time values are from the 250k report.
The poster's BLE W1(u|c) 12.71 and W1(v|c) 5.02 have no traceable source; BLE takes no place or time input, so those cells stay n/a.
-->
