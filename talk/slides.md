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

<img src="/figures/intro_pair_era5.png" class="fig" style="max-height: 220px" />

<!--
Shaurya. Title.
-->

---

# A balloon steers by picking a wind

It can only go up or down. Each altitude has a different wind, so that's the choice.

<Todo>Cartoon: one balloon in a column of arrows, each altitude pointing a different way. Click 2: the path it takes.</Todo>

<!--
Shaurya, 1 min.
The balloon has no propeller. All it can do is change altitude. The wind is different at each altitude, so steering means picking which wind to ride.
-->

---

# Long flights need wind everywhere

A simulator has to supply wind along the whole route, and it has to be realistic.

<Todo>Map with a long flight path. A small square tile covers only part of it.</Todo>

<!--
Shaurya, 1 min.
We wanted a simulator for long flights. That means wind everywhere along the route. And it has to be realistic, or what we learn in the simulator won't hold in real wind.
-->

---

# What was out there

Generators either didn't look like real wind, or came in small tiles with seams.

<div class="pair">
  <div>
    <img src="/figures/intro_pair_era5.png" />
    <p class="caption">ERA5 (real)</p>
  </div>
  <div v-click>
    <img src="/figures/intro_pair_blevae.png" />
    <p class="caption">BLE generator</p>
  </div>
</div>

<!--
Shaurya, 1 min.
Left is real wind from ERA5. Right is the generator the Balloon Learning Environment uses. There also wasn't a good benchmark for what "realistic" even means here.
-->

---

# Our generator

A diffusion model trained on ERA5, made seamless and as large as you want.

<div class="pair">
  <div>
    <img src="/figures/intro_pair_diffusion.png" />
    <p class="caption">Our model</p>
  </div>
  <div v-click>
    <img src="/figures/tiling_comparison.png" />
    <p class="caption">Independent tiles (left) vs InfiniteDiffusion (right)</p>
  </div>
</div>

<!--
Shaurya, 1 min.
First click: our model's output, conditioned on place and time.
Second: InfiniteDiffusion. Left is generating tiles independently, with seams. Right is ours, one field.
In September we scaled this to the whole globe.
-->

---

# How we checked it

We compared it to ERA5 on four things.

<div class="pair">
  <div><img src="/figures/marginal_u.png" /><p class="caption">Wind speeds</p></div>
  <div v-click><img src="/figures/psd_final_methods.png" /><p class="caption">Spatial structure</p></div>
</div>

<div v-click class="caption">Also: how wind changes with altitude, and over time.</div>

<!--
Shaurya, 2 min.
The spread of wind speeds, including extremes. How structure is spread across scales. How the wind changes with altitude. How it evolves over time.
TODO: decide whether to show all four panels or these two.
-->

---

# How it did

Better than earlier generators on most checks. Fine detail and extremes are the biggest gaps.

<img src="/figures/realism_gap.png" class="fig" />

<!--
Shaurya, 1 min.
Each row is a check. Closer to the dashed line is closer to real wind. Orange is the BLE generator, blue is ours.
-->

---

# Looks right. Does it fly right?

The checks tell us the wind looks like ERA5. They don't tell us if a controller built on it works in real wind.

<Todo>The four check panels, greyed out, with a balloon and a question mark.</Todo>

<!--
Shaurya, 1 min.
This is the turn in the talk. A simulator is for building controllers. None of our checks measured that.
-->

---

# cBottle

NVIDIA had already built a global generator the way we planned to. So we tried flying on it.

<Todo>cBottle's design next to ours: coarse global model, then fine patches.</Todo>

<!--
Rohan, 1 min.
cBottle is NVIDIA's climate generator. It already does coarse global plus fine patches, which was our plan. So we tried using its released winds directly as the balloon's wind.
-->

---

# Nothing could steer on it

cBottle has two levels in the balloon's range, both at jet speeds. ERA5 has six.

<Todo>Altitude bars, 14–21 km. cBottle: 2 levels (11.8 and 20.6 km). Click: ERA5's 6 levels.</Todo>

<!--
Rohan, 1 min.
With two levels, both fast, there's nothing to choose between. Every flight just drifted. So we switched the truth to ERA5.
Numbers are in docs/cbottle-2026-10-21.md.
-->

---

# What do we actually need to get right?

A generator can pass general checks and still be useless for control. We needed a way to measure that.

<!--
Rohan, 0.5 min. Text-only slide on purpose.
This is the question the rest of the talk is about, and we'd need the answer before building the next generator.
-->

---

# balloon-sim

The balloon flies in the true wind. The controller plans on a forecast we make by adding error to it.

<Todo>Diagram. Truth (ERA5) → balloon flies. Click: error. Click: forecast = truth − error → controller plans.</Todo>

<!--
Rohan, 1.5 min.
The main design decision. Because we make the forecast, we choose exactly how wrong it is. The error is drawn to look like a real day-old forecast's error.
-->

---

# One flight

Same launch, same wind, three controllers.

<img src="/figures/1_one_flight.png" class="fig" />

<!--
Rohan, 1.5 min.
TODO: replace with an animated version from the saved trajectories.
Seed 9, real forecast error. Hindsight and MPC turn into the goal ring; the fixed plan sails past.
-->

---

# Three controllers

<v-clicks>

- **Hindsight** plans on the true wind. The best we could hope for.
- **Fixed plan** decides everything at launch and never adjusts.
- **MPC** replans every move using the forecast.

</v-clicks>

<Todo>Small icon or mini-path next to each line.</Todo>

<!--
Rohan, 2 min.
Hindsight and the fixed plan are the top and bottom of the range. MPC is what we actually care about.
-->

---

# How we score MPC

MPC lands between the fixed plan and hindsight. The score is how much of that gap it closes.

<Todo>A bar from fixed plan to hindsight. Click: MPC's marker. Click: the % closed.</Todo>

<!--
Rohan, 1.5 min.
100% means the forecast error cost nothing. 0% means replanning didn't help at all.
-->

---

# The forecast-error experiment

Make the forecast worse and worse. 32 launches, every controller flies the same ones.

<Todo>A dial: forecast error from 0 to 4× a real day-old forecast. Click: three MPC switches (how far ahead, several possible winds, uses its own readings).</Todo>

<!--
Shaurya, 1.5 min.
Task: reach a goal 2,000 km downwind. Error at 0, ¼, ½, 1, 2, 4 times 3.2 m/s.
-->

---

# First, what's even possible

Of twenty candidate tasks, only downwind goals are reachable at all.

<img src="/figures/0_is_it_even_possible.png" class="fig" />

<!--
Shaurya, 1 min.
We wrote the rule for picking a task before running anything. Top: how often hindsight even gets there. Bottom: whether a blind plan already does fine.
-->

---

# Replanning is worth a lot

It saves thousands of kilometres over a flight.

<img src="/figures/7_what_replanning_is_worth.png" class="fig" />

<!--
Shaurya, 1.5 min.
MPC minus the fixed plan, same forecast, so the only difference is replanning.
-->

---

# The balloon's own readings added nothing

One reading every three hours says little about a wind pattern hundreds of kilometres wide.

<img src="/figures/6_every_seed.png" class="fig" />

<!--
Rohan, 1 min.
Circles: MPC that learns from readings. Triangles: MPC that ignores them. They overlap.
-->

---

# Looking further ahead stops helping

Past about half a real forecast's error, planning further ahead buys nothing.

<img src="/figures/3_breaking_point.png" class="fig" />

<!--
Shaurya, 1.5 min.
One line per horizon. They separate at low error and collapse together past about 1.6–3.2 m/s.
-->

---

# Planning over several possible winds

Helped the fixed plan. Didn't help MPC.

<img src="/figures/11_hedging_every_draw_count.png" class="fig" />

<!--
Rohan, 1 min.
A controller that can't react benefits from hedging. One that replans doesn't need it.
-->

---

# Altitude is what the balloon has to work with

It can only use the directions its altitudes offer. Get those wrong and control breaks.

<Todo>cBottle's 2 levels next to ERA5's 6. Below: the downwind-only result.</Todo>

<!--
Shaurya, 1.5 min.
Our experiment only made the wind wrong in one way. Downwind-only and cBottle suggest which way matters most. So next we test that directly.
-->

---

# Next: the fidelity experiment

Take the true wind and make it worse one property at a time. Score it the same way.

<Todo>Ladders, one per property, from truth to coarse: altitude levels, time frames, grid size, kinds of error. One ladder per click.</Todo>

<!--
Shaurya, 1.5 min.
These are close to the same things we checked for realism in the summer. Now we find out which of them a controller cares about.
-->

---

# The same question for learned controllers

PPO, with one policy shared across a fleet.

<Todo>Rohan: diagram of the PPO and fleet setup.</Todo>

<!--
Rohan, 1.5 min. Status only, no results yet.
-->

---

# Back to the generator

Wherever control doesn't care, the generator can be coarse and cheap.

<Todo>Each ladder mapped to a generator setting: number of levels, frames per day, grid size.</Todo>

<!--
Shaurya, 0.5 min.
That closes the loop with where we started.
-->

---

# Questions for you

<v-clicks>

1. TODO
2. TODO
3. TODO

</v-clicks>

<!--
Shaurya, 0.5 min. Then 30 min discussion.
-->
