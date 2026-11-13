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

<img src="/figures/cover.png" class="cover-strip" />
<div class="cover-credit">ERA5 wind at 19 km, 16 July 2021</div>

<!--
Shaurya. Title.
-->

---

# Outline

<Sections />

<!--
Shaurya, 20 seconds. Walk through the five sections in one sentence each.
-->

---

<div class="divider">
  <p class="divider-label">Part 1 · Shaurya</p>
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

Conditional diffusion, trained on ERA5.

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
    <li>Trained on global data, two layers (coarse and fine)</li>
    <li>InfiniteDiffusion in time as well as space</li>
    <li>An equal-area grid on the sphere</li>
    <li>Trained on global ERA5</li>
  </ul>
</div>

<style>
.glist li { font-size: 1.1rem; margin-bottom: 0.9rem; }
</style>

<!--
Shaurya, 45 seconds.
That was the summer model. What I'm building now goes further. This is its wind at about 20 km, generated, not ERA5.
It covers the whole globe in two stages: a coarse model for the whole planet, then a second one that fills in fine detail. It tiles in time too, so a flight never runs out of wind. It works on an equal-area grid on the sphere, and it's trained on global data rather than one region.
The realism numbers in the next part are for the summer model.
Video: blog/tools/wind_globe.py, Stage 1 -> Stage 2 output for 2023-07-10, ~53 hPa.
-->
---

<div class="divider">
  <p class="divider-label">Part 2 · Shaurya</p>
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

<p v-click class="caption" style="margin-top: 0.8rem">ERA5 floor: one half of held-out ERA5 scored against the other half.</p>

<!--
Shaurya, 1 min.
We score generated wind against held-out ERA5 on four properties, one per thing that shapes a flight.
Wind speeds: how hard the balloon is pushed, including extremes.
Structure in space: how the wind varies along the path.
Change with altitude: the options the balloon steers between. The one that matters most for control.
Change over time: how the wind evolves during a flight.
Floor: two halves of real ERA5 still differ, so that's the best possible score.
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
Lower is better. The dotted line is the ERA5 floor.
Speeds: distance between wind-speed distributions, 2.98 vs BLE's 4.56. Ours still misses extremes (tail error 13.5 m/s vs a floor of 2.9).
Space: the finest scale it keeps. Ours keeps detail down to about 340 km, BLE about 840. ERA5's floor and simplex both show 56 km, which is the grid limit: they keep detail all the way down, so 56 means 'as fine as the grid can show', not a measured scale.
Altitude: error in the shear between neighbouring levels, mean of east-west and north-south. Ours 1.0, BLE 2.1. Honest note: on north-south alone, BLE is better (0.61 vs 1.01).
Time: dispersion error, how far off the spreading of drifting particles is over time compared with ERA5. Ours 0.81, BLE 1.79, floor 0.18. Better than BLE, but still about 4 times the floor, so time is our weakest check. The summer model makes 4-hour blocks, so we tiled it in time with InfiniteDiffusion to get 24-hour runs. Caveat: BLE's run is 48 hours at 6-hourly, ours 24 hours hourly, two runs of ours, so treat this panel as rough.
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
  <p class="divider-label">Part 3 · Rohan</p>
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
  <p class="divider-label">Part 4 · Rohan</p>
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
  <p class="divider-label">Part 5 · Shaurya</p>
  <h1>Fidelity experiment (planned)</h1>
  <p class="divider-q">Which properties of a wind model matter for transfer?</p>
  <Sections :current="5" />
</div>

<!--
SHAURYA takes over. About 15 seconds.

So: which properties of a wind model matter for transfer? This is the fidelity experiment, and it's what we're running next.
-->

---

# Our hypothesis: the balloon steers on the differences between altitudes

<div class="row" style="align-items: stretch; gap: 20px">
  <div class="card line grow">
    <Column :winds="[-0.55, -0.8]" colour="#e2852e" />
    <h3>Two altitudes</h3>
    <p>cBottle. Nothing could steer.</p>
  </div>
  <div class="card line grow" v-click>
    <Column :winds="[-0.5, -0.38, -0.2, 0.12, -0.3, -0.6]" colour="#0f8a7a" />
    <h3>Six altitudes</h3>
    <p>ERA5. Goals downwind are reachable, and nothing else.</p>
  </div>
  <div class="col" style="width: 250px; justify-content: center" v-click>
    <p style="font-size: 21px; font-weight: 650; line-height: 1.3; letter-spacing: -0.01em">So control should suffer from errors in how the wind changes with altitude, not from errors in the wind itself.</p>
  </div>
</div>

<!--
SHAURYA. About 1 minute. TWO CLICKS.

Here's what we expect, and why. Each picture is one column of air, one arrow per altitude.

On the left is cBottle, NVIDIA's global generator. We tried flying on it in October. Near the balloon it has only two altitudes, both blowing hard the same way, and nothing could steer.

[click] ERA5 has six, and a balloon can reach goals downwind but not across the wind or against it. In this record the winds at different altitudes oppose each other at only 58 percent of places.

[click] So here's our hypothesis. A balloon can only use the directions its altitudes offer, so it steers on the differences between altitudes. Control should suffer when a wind model gets those differences wrong, and much less when it gets the wind wrong in a way that leaves them alone.
Our noise experiment can't test that: its error was always one shape. So next we make the wind model wrong in different ways, one at a time.
-->

---

# Four properties, each made coarser step by step

<div class="ladders">
  <div class="name"><strong>Altitude levels</strong><span>18 down to 1</span></div>
  <img src="/fig/ladder-levels.png" />
  <div class="name" v-click="1"><strong>Frames in time</strong><span>every 3 h up to frozen</span></div>
  <img src="/fig/ladder-frames.png" v-click="1" />
  <div class="name" v-click="2"><strong>Grid</strong><span>2° up to 16°</span></div>
  <img src="/fig/ladder-grid.png" v-click="2" />
  <div class="name" v-click="3"><strong>Kind of error</strong><span>at the same size</span></div>
  <div class="row" style="gap: 8px" v-click="3">
    <span class="chip">a constant offset</span><span class="muted small">or</span><span class="chip">random</span>
    <span class="muted small" style="margin: 0 6px">×</span>
    <span class="chip">the same at every altitude</span><span class="muted small">or</span><span class="chip">different at each</span>
  </div>
</div>

<div class="cap" style="margin-top: 10px"><span class="dot" style="background: #dfe2e7"></span>true wind&emsp;<span class="dot" style="background: var(--accent)"></span>what the controller is given&emsp;·&emsp;most detailed on the left, coarsest on the right</div>

<style>
.ladders { display: grid; grid-template-columns: 170px 1fr; gap: 10px 18px; align-items: center; }
.ladders img { width: 100%; height: 84px; object-fit: fill; }
.ladders .name { display: flex; flex-direction: column; font-size: 17px; line-height: 1.25; }
.ladders .name span { font-size: 13.5px; color: var(--muted); }
</style>

<!--
SHAURYA. About 1.5 minutes. THREE CLICKS, one per row.

These are the properties. In each row, the left is the most detailed and the right is the coarsest. Grey is the true wind, green is what the controller gets.

First, altitude levels. This is the wind up one column of air. We keep fewer and fewer levels and draw straight lines between them. With one level, every altitude has the same wind and there's nothing to steer with. We expect this to matter most.

[click] Second, time. The wind at one place over ten days. We keep a frame every 3 hours, then 12, then 24, then freeze it.

[click] Third, the grid. From 2 degrees up to 16.

[click] And fourth, the kind of error, holding its size fixed. Is the wind off by a constant amount, or randomly? And is the error the same at every altitude, or different at each one?

Each ladder goes from finer than the wind really varies to much coarser. So somewhere along each one, control should start to suffer. Where that happens is the result.

[Task: station keeping, staying near one point, because it's what Loon flew and what the field scores, and Part 4 showed a balloon only reaches goals downwind.]
[The picture uses our current 6-level record. The real experiment rebuilds the truth at 18 levels first, so the first row will start from 18.]
-->

---

# The sharpest test: two errors of the same size

<div class="pair" style="align-items: stretch">
  <div class="card">
    <h3>The same offset at every altitude</h3>
    <p>Wind error: 4.5 m/s<br>Change in the differences between altitudes: <b>0</b></p>
    <p class="c-mpc">Hypothesis: barely hurts</p>
  </div>
  <div class="card" v-click>
    <h3>An offset that changes with altitude</h3>
    <p>Wind error: 4.5 m/s<br>Change in the differences between altitudes: <b>2.65 m/s</b></p>
    <p class="c-fixed">Hypothesis: hurts</p>
  </div>
</div>

<p v-click style="margin-top: 1.2rem">Across every condition, we'll check which tracks control loss better: the error in the wind, or the error in the differences between altitudes. We wrote this test down before running anything.</p>

<!--
SHAURYA. About 1 minute. TWO CLICKS.

This is the cleanest test of the hypothesis. Two errors, exactly the same size, 4.5 metres a second.
On the left, the whole wind is shifted by the same amount at every altitude. The differences between altitudes don't change at all.
[click] On the right, the shift grows from the bottom of the band to the top. Same size overall, but now the differences between altitudes are off by 2.65.
If we're right, the left one barely hurts and the right one does.
[click] And the general version, across every condition in the experiment: does control loss track plain wind error, or the error in the differences between altitudes? It's one pre-registered test, so the many conditions don't multiply false positives. If the second wins, that's a way to score a wind model for control without flying anything.
Numbers: measured on our 6-level record (docs/fidelity-design-2026-11-06.md). Uniform bias: 4.5 wind error, 0 difference error. Altitude-varying bias of the same size: 2.65.
-->

---

# Learned controller (PPO)

<Policy />

<div class="row" style="gap: 10px; margin-top: 14px; justify-content: center" v-click="3">
  <span class="chip on">64,000 training flights</span>
  <span class="chip">tested on the same 32 launches as MPC</span>
  <span class="chip">all 4 tasks</span>
  <span class="chip">1 balloon and fleets of 4</span>
</div>

<!--
SHAURYA. About 1 minute. THREE CLICKS.

Everything so far has been MPC. But we also care about learned controllers: if you train a policy in a simulator, does it work in real wind? So Rohan added PPO to balloon-sim.

On the left is what each balloon sees: the forecast wind at every altitude where it is, plus how far and which way the goal is, how much time is left, what its own wind sensor reads compared to the forecast, and in a fleet, where the other balloons are.

[click] That goes into one policy network, and every balloon in a fleet shares the same network. That's the standard multi-agent setup, MAPPO.

[click] And it outputs climb or sink, every three hours. Same choice MPC makes.

[click] The plan is to train it on 64,000 flights across all the noise levels, on launches separate from the ones we test on, then test it on the same 32 launches as MPC so it goes on the same plots. It's set up on all four tasks: reaching a goal, getting as far from the start as possible, station-keeping, and visiting several targets, with one balloon and with fleets of four.

That's what we're running next.
-->

---

# The answers tell us how to build the next generator

<div style="margin: 0 auto"><Arc :current="6" loop /></div>

<div class="row" style="gap: 16px; margin-top: 6px; align-items: stretch">
  <div class="card grow" v-click><h3 style="font-size: 17px">Levels to generate</h3><p class="small">cBottle: 2. Ours: 18. How many are enough?</p></div>
  <div class="card grow" v-click><h3 style="font-size: 17px">Frames per day</h3><p class="small">Every hour is expensive. Is every 12 fine?</p></div>
  <div class="card grow" v-click><h3 style="font-size: 17px">Grid size</h3><p class="small">Fine detail was our biggest realism gap. Does control care?</p></div>
</div>

<div class="cap" style="margin-top: 14px" v-click><strong style="color: var(--ink)">Wherever control doesn't care, the generator can be coarse, and cheap.</strong> And if the hypothesis holds, "error in the differences between altitudes" becomes a fifth realism check, one that scores a generator for control without flying it.</div>

<!--
SHAURYA. About 1 minute. FOUR CLICKS.

So this is how it comes back around. We started by building a generator, and the experiment tells us how to build the next one.

Every row on that last slide is something a generator spends compute on.

[click] How many altitude levels to generate. cBottle has two near the balloon, ours has 18. Is the right number 6? 9?

[click] How many frames per day.

[click] And how fine a grid. Fine detail was the biggest gap in our realism scores. If control doesn't care about it, that's a gap we can stop worrying about.

[click] Wherever control doesn't care, the generator can be coarse, and that makes it cheap. And if the hypothesis holds, we'd add a fifth check to the four from Part 2, one that scores a generator on what it's actually for, without flying anything. That closes the loop back to where we started.
-->

---

<div class="muted" style="font-size: 16px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 22px">Where we'd like your help</div>

<div class="col" style="gap: 22px; max-width: 820px">
  <div class="row" style="align-items: baseline; gap: 18px"><div class="big muted" style="font-size: 32px; width: 28px">1</div><p style="font-size: 24px; font-weight: 600; line-height: 1.25; letter-spacing: -0.01em">Is a coarsened copy of ERA5 a fair stand-in for a worse simulator?</p></div>
  <div class="row" style="align-items: baseline; gap: 18px" v-click><div class="big muted" style="font-size: 32px; width: 28px">2</div><p style="font-size: 24px; font-weight: 600; line-height: 1.25; letter-spacing: -0.01em">Which task should lead: holding a position, or reaching a goal?</p></div>
  <div class="row" style="align-items: baseline; gap: 18px" v-click><div class="big muted" style="font-size: 32px; width: 28px">3</div><p style="font-size: 24px; font-weight: 600; line-height: 1.25; letter-spacing: -0.01em">What would make a learned controller a fair comparison to MPC here?</p></div>
</div>

<!--
BOTH. About 30 seconds, then discussion. TWO CLICKS.

That's where we are. Three things we'd really like your thoughts on.

One. In the next experiment our "worse simulators" are coarsened copies of ERA5. That's what lets us change one thing at a time. Is that convincing, or would you want to see real generators in there?

[click] Two. Everything so far is reaching a goal. The balloon literature mostly does station keeping: stay within 50 kilometres of a point. Which should lead?

[click] Three. For the learned controller: what would you need to see to believe a comparison with MPC is fair?

Thanks. Happy to go back to anything.
-->

---

# Backup: why not noise, or just replay ERA5?

<div class="pair" style="text-align: left">
  <div>
    <img src="/figures/noise.png" style="max-height: 180px" />
    <p><b>Noise</b> (simplex, GP, …). You choose its smoothness, but it has no jets, fronts or extremes, and no turning of the wind with altitude.</p>
  </div>
  <div>
    <img src="/figures/intro_pair_era5.png" style="max-height: 180px" />
    <p><b>ERA5 replay</b> is real, but each day happened once. There's one true wind per forecast and no new weather. It's also what we score on.</p>
  </div>
</div>

<!--
Backup, only if asked.
Noise: a GP can even match the real spectrum if you pick its covariance well, but nothing beyond second-order statistics. In BLE, simplex noise is the forecast error and the GP is the controller's wind estimate; neither is the wind itself.
ERA5: to train or test under uncertainty you want many plausible true winds for one forecast; replay gives exactly one. Replay is fine for scoring a planner, which is what balloon-sim does. Whether extra fidelity matters for control is Part 5.
-->

---

# Backup: full realism table

<div class="compact">

| Check | Metric (lower is better unless noted) | ERA5 floor | Simplex | BLE's VAE | Ours (summer) |
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

---

# Backup: cBottle as the true wind

<img src="/fig/cbottle-globe.png" class="round" style="width: 100%; height: 330px; object-fit: cover" />
<div class="row" style="justify-content: flex-end; margin-top: 14px">
  <div class="small"><span class="chip">coarse global model</span> → <span class="chip">fine patches</span></div>
</div>

<!--
ROHAN. About 1 minute. No clicks.

In September we found cBottle, from NVIDIA. It's a generative model of the whole atmosphere, and it's built the way we'd planned to build ours: a coarse global model first, then fine patches on top of it.

This is its wind at about 20 km, from the released model. It looks really good.

So the first question was what to use as the true wind. We wanted a generator there, so the wind never runs out, and cBottle was already global. In October we took the released model and used its wind as the true wind for our balloon.

[The picture: cBottle's wind at 20.6 km, released model, July.]
-->

---

# Backup: cBottle gave the balloon nothing to choose between

<div class="row">
  <div class="grow"><Layers name="levels" :n="2" :shown="1" /></div>
  <div class="note-side">
    <p>Slowest wind on offer, at a typical place</p>
    <div class="big c-fixed" style="margin-top: 14px">8.5 <span style="font-size: 20px">m/s</span></div>
    <p>cBottle</p>
    <div v-click="1">
      <div class="big c-accent" style="margin-top: 18px">3.5 <span style="font-size: 20px">m/s</span></div>
      <p>ERA5</p>
    </div>
  </div>
</div>

<!--
ROHAN. About 1 minute. ONE CLICK brings in ERA5.

And it didn't work. Every controller we tried just drifted.

The reason's pretty simple once you see it. cBottle outputs eight pressure levels for the whole atmosphere, and only two of them are anywhere near where a balloon flies, one around 12 km and one around 21. So the balloon has two winds to pick from, and both are fast. At a typical place the slower one is still 8.5 metres a second.

[click] ERA5 has six levels in the same range. With six to pick from there's usually a slow one, around 3.5.

So we switched to ERA5 as the true wind, and everything since uses it.

And notice what happened. cBottle is a very good climate model by general measures, and it was useless for steering, because of how few altitudes it has near the balloon. Keep that in mind; it comes back at the end.

[If asked: 8.5 and 3.5 are the median of the calmest level, 20 to 50 N. Bars are average speeds from our two records.]
-->
