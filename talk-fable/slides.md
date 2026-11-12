---
theme: default
title: What does a wind simulator need to get right?
subtitle: From generating realistic wind for balloons to measuring what a controller needs from it
authors: Shaurya Sen · Rohan Shankar
venue: Dean lab group meeting · November 2026
layout: cover
fonts:
  provider: none
  sans: Inter
canvasWidth: 980
transition: fade
routerMode: hash
---

<!--
SHAURYA. About 30 seconds.

Hi everyone. Rohan and I are going to walk through what we've been doing since the summer.

It's really one question the whole way through: what does a wind simulator need to get right, if you want to use it to control a balloon?

We started by building the simulator. Then we realised we couldn't tell if it was any good for control. So we built a way to measure that. And that's now shaping what we build next.
-->

---
layout: part
part: 1
who: Shaurya
---

<!--
SHAURYA. About 20 seconds.

Here's the whole talk as five steps. We built a wind generator. We checked it looked real. Then we actually flew a controller in wind and measured how it did. We made the forecast wrong on purpose. And next we're going to make the wind model itself coarser on purpose.

I'll start at step one, which was the summer.
-->

---
part: 1
claim: A balloon steers by choosing which wind to ride
clicks: 3
---

<div class="row" style="gap: 36px">
  <div class="grow"><Balloon /></div>
  <div class="note-side">
    <p><strong>Up or down is the only control.</strong></p>
    <p class="muted" style="margin-top: 10px">The wind blows a different way at each altitude. Everything else is the wind's decision.</p>
  </div>
</div>

<!--
SHAURYA. About 1 minute. THREE CLICKS, each one moves the balloon.

These are stratospheric balloons, flying between about 15 and 20 kilometres up. They have no engine. The only thing they can do is go up or down.

[click] Left alone, the balloon just goes where the wind at its altitude takes it.

[click] But the wind is different at each altitude. So if it wants to go the other way, it climbs to a level where the wind blows that way.

[click] And if it wants to slow down, it finds a level where the wind is weak.

So controlling a balloon is choosing winds. That means the wind model is basically the whole simulator.
-->

---
part: 1
claim: A long flight outruns any fixed patch of wind
---

<div class="row">
  <div class="grow round"><Layers name="route" :n="2" :shown="1" /></div>
  <div class="note-side">
    <p>We wanted a simulator for <strong>long flights</strong>, so it needs wind that is:</p>
    <p style="margin-top: 14px"><span class="chip on">realistic</span></p>
    <p style="margin-top: 8px"><span class="chip on">unbounded</span></p>
  </div>
</div>

<!--
SHAURYA. About 1 minute. ONE CLICK draws the flight.

Our goal in the summer was a simulator for long-distance tasks.

The box is one tile from a wind generator, about 1,300 kilometres across. That's the size our summer model makes in one go.

[click] And this is one real flight from our experiments, six days long. It ends up 4,300 kilometres away. It's out of the box in about two days.

So we needed two things. The wind has to be realistic, or what you learn in simulation won't hold in real wind. And it can't run out.
-->

---
part: 1
claim: Existing generators were unrealistic, or had seams
---

<div class="row" style="align-items: flex-start; gap: 22px">
  <div class="col grow">
    <div class="label">Real wind (ERA5)</div>
    <img src="/poster/era5.png" class="round" />
  </div>
  <div class="col grow" v-click>
    <div class="label">The standard generator (BLE)</div>
    <img src="/poster/blevae.png" class="round" />
    <div class="cap">Bands, not weather.</div>
  </div>
  <div class="col" style="width: 262px" v-click>
    <div class="label">Tiles generated one by one</div>
    <img src="/poster/tiles-independent.png" class="round" />
    <div class="cap">Nothing lines up at the edges.</div>
  </div>
</div>

<!--
SHAURYA. About 1 minute. TWO CLICKS.

On the left is real wind from ERA5, which is the reanalysis dataset everyone treats as ground truth. Colour is speed, arrows are direction.

[click] This is what the generator in the Balloon Learning Environment produces. That's the standard simulator, from the Loon team. It's a VAE, and you get these horizontal bands. It doesn't look like weather.

[click] And if you take a better generator and just make more tiles to cover more ground, each tile is fine on its own, but the edges don't match. A balloon crossing that line sees the wind jump.

There also wasn't an agreed way to say how realistic a generator was. So we had to build that too.
-->

---
part: 1
claim: We built a diffusion model that makes seamless wind, as far as you want to fly
---

<div class="row" style="align-items: flex-start; gap: 26px">
  <div class="col" style="width: 300px">
    <div class="label">Ours, for a given place and date</div>
    <img src="/poster/diffusion.png" class="round" />
    <div class="cap">A diffusion model trained on ERA5.</div>
  </div>
  <div class="col grow" v-click>
    <div class="label">InfiniteDiffusion: ask for any region, get the same wind every time</div>
    <img src="/poster/zoom.png" class="round" />
    <div class="cap">It generates only the windows a query needs and blends them into one field.</div>
    <div style="margin-top: 16px" v-click><span class="chip teal">September</span>&ensp;<span class="small">scaled up to the whole globe</span></div>
  </div>
</div>

<!--
SHAURYA. About 1 minute. TWO CLICKS.

So this is what we built. It's a diffusion model trained on ERA5. You give it a place and a date and it generates a block of wind: 18 altitude levels, over a few hours.

[click] To make it unbounded we used a method called InfiniteDiffusion. When you ask for a region, it works out which windows it needs, generates them, and blends the overlaps, so there are no seams. And it's deterministic: ask for the same place twice and you get the same wind, in any order. That's what a simulator needs.

[click] In September we rebuilt it to cover the whole globe, in two stages: a coarse model for the whole sphere, then a second model that fills in fine detail in patches.
-->

---
layout: part
part: 2
who: Shaurya
---

<!--
SHAURYA. About 10 seconds.

So we had wind that never runs out. The next question was: is it realistic? And how would we know?
-->

---
part: 2
claim: We compared generated wind to ERA5 in four ways
---

<Checks />

<div class="cap" style="margin-top: 16px"><span class="dot" style="background: #b7bcc6"></span>ERA5&emsp;<span class="dot" style="background: var(--accent)"></span>generated</div>

<!--
SHAURYA. About 1.5 minutes. THREE CLICKS, one per card.

There was no standard benchmark, so we put one together. Four kinds of check, each comparing generated wind against held-out ERA5.

First, wind speeds. Do you get the right mix of calm and fast winds, including the rare extremes?

[click] Second, structure in space. Real wind has detail at every scale. Generators tend to be too smooth, so they lose the small scales.

[click] Third, how the wind changes with altitude. Remember this one. It's the one that matters for a balloon, and it comes back later.

[click] Fourth, time. We drop particles in the wind and check they drift apart at the right rate.
-->

---
part: 2
claim: By these checks our generator was close to ERA5, and far ahead of the standard one
---

<Layers name="realism" :n="1" :shown="1" />

<div class="row" style="justify-content: space-between; margin-top: 10px">
  <div class="small"><span class="dot" style="background: #b7bcc6"></span>standard generator (BLE)&emsp;<span class="dot" style="background: #8fd1c8"></span>ours, summer&emsp;<span class="dot" style="background: var(--accent)"></span>ours, global</div>
  <div class="small muted">lower is better · dotted line: ERA5 against itself</div>
</div>

<!--
SHAURYA. About 1 minute. No clicks.

Here's one number from each kind of check. Lower is better. The dotted line is what you get comparing one half of ERA5 against the other half, so that's as good as it gets.

Grey is the standard generator. Light green is our summer model, dark green is the global one from September.

On wind speeds and on change with altitude, the global model is basically at the ERA5 line. Fine detail and time are where we still had a gap.

So at this point we were happy. By these scores it beat the balloon wind generators we compared it to.

[Numbers are from the generator repo's benchmark report, not rerun for this talk. The two time bars for BLE and our summer model are from the poster.]
-->

---
part: 2
claim: But noise tuned to ERA5's statistics scores just as well
---

<Layers name="realism" :n="2" :shown="1" />

<div class="small" style="margin-top: 10px"><span class="dot" style="background: #b7bcc6"></span>standard generator (BLE)&emsp;<span class="dot" style="background: #8fd1c8"></span>ours, summer&emsp;<span class="dot" style="background: var(--accent)"></span>ours, global&emsp;<span v-click="1"><span class="dot" style="background: var(--noise)"></span><strong>random noise, fitted to ERA5</strong></span></div>

<div style="margin-top: 14px; font-size: 21px; font-weight: 650; letter-spacing: -0.01em" v-click="2">These checks say the wind looks right. Not that you can fly in it.</div>

<!--
SHAURYA. About 1 minute. TWO CLICKS. This is the turn of the talk, slow down here.

Then we added one more baseline, and it was uncomfortable.

[click] Purple is random noise. No learning at all. We just tuned its statistics to match ERA5. And it scores as well as our trained model, or better, on almost everything.

That's not because noise is a good wind model. It's because every one of these checks is a statistic, and you can match statistics without having any weather in there.

[click] So these checks tell us the wind looks right on average. They can't tell us the thing we actually care about: if I build a controller in this wind, will it work in the real wind? That's the whole reason to have a simulator.

[Numbers from the generator repo's benchmark, July. Worth knowing if asked: the noise is fractal simplex noise, with its speed, vertical and time correlations fitted to ERA5.]
-->

---
layout: part
part: 3
who: Rohan
---

<!--
ROHAN takes over. About 15 seconds.

So that's the question for the rest of the talk. Not "does it look real", but "is it good for control". I'll explain how we set up to measure that, starting with the thing that pushed us there.
-->

---
part: 3
claim: Trying cBottle
---

<img src="/fig/cbottle-globe.png" class="round" style="width: 100%; height: 330px; object-fit: cover" />
<div class="row" style="justify-content: flex-end; margin-top: 14px">
  <div class="small"><span class="chip">coarse global model</span> → <span class="chip">fine patches</span></div>
</div>

<!--
ROHAN. About 1 minute. No clicks.

In September we found cBottle, from NVIDIA. It's a generative model of the whole atmosphere, and it's built the way we'd planned to build ours: a coarse global model first, then fine patches on top of it.

This is its wind at about 20 km, from the released model. It looks really good.

So the obvious thing was to just use it. If someone's already built the generator, we skip ahead and start flying. In October we took the released model and used its wind as the world for our balloon.

[The picture: cBottle's wind at 20.6 km, released model, July.]
-->

---
part: 3
claim: "Wind by altitude: cBottle vs ERA5"
---

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

[If asked: 8.5 and 3.5 are the median of the calmest level, 20 to 50 N. Bars are average speeds from our two records.]
-->

---
layout: statement
part: 3
---

<div style="font-size: 42px; font-weight: 650; letter-spacing: -0.025em; line-height: 1.15; max-width: 820px">
  What does a wind model need to get right for a balloon to <span class="c-accent">steer</span> in it?
</div>

<!--
ROHAN. About 30 seconds. No clicks.

So cBottle is an excellent wind model by every general measure, and it was useless for us, because of one property nobody was checking.

That's really the question now. What does a wind model actually have to get right for control? We need that answer anyway before we build our next generator.

To answer it we needed a way to measure "good for control". That's balloon-sim.
-->

---
part: 3
claim: balloon-sim setup
---

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

<div class="col" style="gap: 6px; margin-top: 20px; font-size: 17px" v-click="3">
  <p>• The balloon flies in the truth. The controller only sees the forecast.</p>
  <p>• Error is a Gaussian process draw (smooth noise), fitted to how much ERA5 wind changes in a day</p>
  <p>• Noise levels: 0, 0.8, 1.6, <strong>3.2</strong>, 6.4, 12.8 m/s. 3.2 ≈ using yesterday's wind as the forecast</p>
  <p>• 32 launches per level, the same for every controller</p>
</div>

<!--
ROHAN. About 1.5 minutes. THREE CLICKS.

We wrote balloon-sim from scratch. Here's the setup. The truth is ERA5. That's the wind that actually moves the balloon.

[click] We take the truth and subtract an error. The error is smooth random noise, a draw from a Gaussian process. We fitted it to how much ERA5's wind changes over a day, so at full size it's like using yesterday's wind as the forecast. And we pick how big it is.

[click] What's left is the forecast. The controller only ever sees the forecast.

[click] We test six noise levels, from zero, which is a perfect forecast, up to 12.8 metres a second. 3.2 is the yesterday's-wind level. A real weather forecast is probably more accurate than that. At each level we fly the same 32 launches with every controller, so when we compare two controllers it's the same flights.

[If asked: Gaussian process with a Matérn kernel. Patches about 540 km across, 1.3 km in altitude, 9 hours in time.]
-->

---
part: 3
claim: Controllers and bounds
---

<div class="row" style="justify-content: space-between; padding: 16px 0; border-top: 1.5px solid var(--faint);">
  <div class="col" style="width: 440px; gap: 4px">
    <p style="font-size: 24px; font-weight: 650">Hindsight</p>
    <p style="font-size: 18px">Plans on the true wind.</p>
  </div>
  <div style="font-size: 24px">

$V_{\text{hind}} = \mathbb{E}_w\big[\min_a J(a,w)\big]$

</div>
</div>
<div class="row" style="justify-content: space-between; padding: 16px 0; border-top: 1.5px solid var(--faint);" v-click>
  <div class="col" style="width: 440px; gap: 4px">
    <p style="font-size: 24px; font-weight: 650" class="c-mpc">Best policy</p>
    <p style="font-size: 18px">Reacts to where the balloon actually is, for example MPC.</p>
  </div>
  <div style="font-size: 24px">

$V_{\text{policy}} = \min_\pi \mathbb{E}_w\big[J(\pi,w)\big]$

</div>
</div>
<div class="row" style="justify-content: space-between; padding: 16px 0; border-top: 1.5px solid var(--faint); border-bottom: 1.5px solid var(--faint);" v-click>
  <div class="col" style="width: 440px; gap: 4px">
    <p style="font-size: 24px; font-weight: 650" class="c-fixed">Fixed plan</p>
    <p style="font-size: 18px">One plan from the forecast that is made at launch and is never changed.</p>
  </div>
  <div style="font-size: 24px">

$V_{\text{plan}} = \min_a \mathbb{E}_w\big[J(a,w)\big]$

</div>
</div>

<div style="text-align: center; margin-top: 22px; font-size: 32px" v-click>

$V_{\text{hind}} \;\le\; V_{\text{policy}} \;\le\; V_{\text{plan}}$

</div>

<!--
ROHAN. About 1.5 minutes. THREE CLICKS.

To say how good MPC is, we need something to compare it to. So we have two bounds. J is the cost of a flight, roughly how far from the goal you were. w is the wind, which is uncertain given the forecast.

Hindsight: for each wind, take the best plan for that wind, then average. That's knowing the true wind ahead of time. Nobody can do that, so nothing beats it.

[click] The best policy reacts to where the balloon actually is. MPC is one policy. It replans every move from where the balloon actually is, so it sits at or above the best policy.

[click] Fixed plan: one plan, picked at launch from the forecast, and it has to work for every wind. No feedback.

[click] Reacting can only help compared to a fixed plan, and it can't beat knowing the wind. So hindsight ≤ best policy ≤ fixed plan. What we care about is how close to hindsight MPC gets.

All three use the same planner, the one from Myles's paper. So the only differences are what they know and when they plan. Here's what that looks like on one flight.

[If asked: in practice hindsight is the best plan our search found on the true wind, not a proven optimum. The fixed plan plans on the forecast's mean, or on sampled winds when we hedge.]
-->

---
part: 3
claim: Example flight to a goal 2,000 km downwind
---

<SlidevVideo autoplay controls style="height: 418px; margin: 0 auto; border-radius: 8px" poster="/fig/flight-s13.png">
  <source src="/fig/flight-s13.mp4" type="video/mp4" />
</SlidevVideo>

<!--
ROHAN. About 1.5 minutes. The film plays by itself, about 10 seconds. Replay it with the controls if you want to talk over it.

Here's one flight with all three, to make it concrete. They launch from the same spot into the same wind, with the forecast off by 3.2 metres a second. The goal is the red ring. The bottom panel is altitude over time.

For the first day they all ride the same wind east.

Orange, the fixed plan, made its whole plan at launch from the forecast. It stays low the whole flight. The forecast was off, so it drifts past and never comes back. Closest it gets is about 800 km.

Black, hindsight, turns south at the right time and gets there in about four and a half days.

Blue, MPC, starts on a worse route, replans as it goes, loops back, and gets in about six hours after black.

The bars on the right are each one's cost as it flies: distance from the goal, added up over the flight. A bar stops growing when its balloon gets to the goal. At the end: hindsight about 500 km, MPC about 640, the fixed plan about 2,400. Next slide turns those three numbers into one score.

[If asked: these bars use the cost the controllers are planned on, which stops counting at arrival. The averages on the score slide count all 50 moves. On this flight both give a score of 92 percent.]

[Launch 13 of the 32, from the 2026-10-24 sweep.]
-->

---
part: 3
claim: Scoring MPC
---

<div style="font-size: 24px; text-align: center; margin-bottom: 18px">

$\text{score} = \dfrac{\textcolor{#e2852e}{\text{fixed}} - \textcolor{#2f6fde}{\text{MPC}}}{\textcolor{#e2852e}{\text{fixed}} - \text{hindsight}}$

</div>

<p style="font-size: 16px; margin: 0 0 4px">Average distance from the goal over a flight</p>

<div style="border-top: 1.5px solid var(--faint); padding-top: 6px">
  <ScoreRow title="Example flight" :fixed="2373" :mpc="644" :hindsight="503" />
</div>

<div style="border-top: 1.5px solid var(--faint); border-bottom: 1.5px solid var(--faint); padding-top: 6px" v-click>
  <ScoreRow title="All 32 launches" :fixed="1778" :mpc="1209" :hindsight="1043" />
</div>

<!--
ROHAN. About 1.5 minutes. ONE CLICK.

So that gives us a ruler. For every flight we have a worst case, the fixed plan, and a best case, hindsight. The score is how much of the gap between them MPC closes. Distance here is average distance from the goal over the flight.

On the example flight we just watched: fixed plan about 2,400 km, hindsight about 500, MPC about 640. So MPC closed 92 percent of the gap.

[click] Over all 32 launches, at 3.2 metres a second: fixed plan about 1,800 km, hindsight about 1,000, MPC about 1,200. That's 77 percent. The rest is what the wrong forecast cost us.

Note it isn't 100 percent even with a perfect forecast. With no error it's 87 percent, because the planner only looks 60 hours ahead and hindsight gets to plan the whole flight.

This is the number that lets us compare across conditions. If we make the wind model worse in some way and the score drops, that property mattered. If it doesn't drop, it didn't.

Back to Shaurya.

[Numbers: averages over the same 32 launches, MPC planning 60 hours ahead on the forecast's mean. Scored launch by launch, the median at 3.2 is 80 percent.]
-->

---
layout: part
part: 4
who: Shaurya
---

<!--
SHAURYA takes over. About 10 seconds.

So that's the ruler. The first thing we did with it was the simplest kind of wrong: we made the forecast worse and worse, and watched what happened.
-->

---
part: 4
claim: The experiment turns one dial and flips three switches
---

<div class="row" style="align-items: stretch; gap: 22px">
  <div class="card grow">
    <h3>The dial: how wrong the forecast is</h3>
    <div class="row" style="gap: 8px; margin-top: 16px; align-items: flex-end">
      <div v-for="(s, i) in ['0', '0.8', '1.6', '3.2', '6.4', '12.8']" :key="s" class="col center" style="flex: 1">
        <div :style="{ height: 8 + i * 15 + 'px', width: '100%', borderRadius: '5px', background: i === 3 ? 'var(--ink)' : '#c9ced8' }"></div>
        <div class="small" :style="{ marginTop: '6px', fontWeight: i === 3 ? 700 : 400 }">{{ s }}</div>
      </div>
    </div>
    <p class="tiny muted" style="margin-top: 6px">m/s. <strong style="color: var(--ink)">3.2 is a real day-old forecast.</strong></p>
  </div>
  <div class="card grow" v-click>
    <h3>Three switches on MPC</h3>
    <p style="margin-top: 12px"><strong>How far ahead it plans</strong><br>15, 30, 60 or 150 hours</p>
    <p style="margin-top: 10px"><strong>What it plans on</strong><br>one forecast, or many possible winds</p>
    <p style="margin-top: 10px"><strong>Its own wind readings</strong><br>used, or ignored</p>
  </div>
</div>

<div class="row" style="margin-top: 16px; gap: 10px" v-click>
  <span class="chip on">32 launches</span><span class="small">Every controller flies the same 32, so we compare them launch by launch.</span>
</div>

<!--
SHAURYA. About 1.5 minutes. TWO CLICKS.

Here's the experiment. The dial is forecast error. Six settings, from a perfect forecast up to four times a real one. The dark bar, 3.2, is a real day-old forecast.

[click] Then three switches on MPC. How far ahead it plans, from 15 hours to 150. Whether it plans on the single forecast, or on many possible winds drawn around it. And whether it uses the wind the balloon measures as it flies, or ignores it.

[click] Everything is flown on the same 32 launches. Same start, same goal, same true wind for every controller. So when I show a difference, it's the same flight with one thing changed.

That's about 8,000 flights. Rohan built the pipeline that runs them on the cluster.
-->

---
part: 4
claim: "First: only goals downwind are reachable at all"
lede: We scored twenty candidate tasks against a rule written down beforehand. One passed.
---

<Layers name="scouting" :n="3" :shown="1" />

<!--
SHAURYA. About 1 minute. TWO CLICKS.

Before comparing controllers we had to pick a task, and we didn't want to pick one that flattered us. So we wrote down a rule first, then tested twenty candidates: four directions, five distances.

On the left: how often does even hindsight reach the goal? If the best possible controller can't get there, the task is too hard to learn anything from. Upwind, crosswind and random goals all fail. Only downwind is reachable.

[click] On the right: does steering matter? If a blind plan does nearly as well as hindsight, the wind is doing all the work. Close downwind goals fail that.

[click] One task passes both. 2,000 kilometres downwind. That's what everything else here uses.

But notice what this says by itself: in this wind, a balloon goes downwind or nowhere.
-->

---
part: 4
claim: Replanning is worth a lot
---

<div class="row">
  <div class="grow"><Layers name="bounds" :n="5" :shown="2" /></div>
  <div class="note-side" v-click="3">
    <div class="big c-mpc">570 <span style="font-size: 20px">km</span></div>
    <p style="margin-top: 6px">closer to the goal on average, with a real forecast.</p>
    <p class="small muted" style="margin-top: 10px">330 km even with a perfect one.</p>
  </div>
</div>

<!--
SHAURYA. About 1.5 minutes. THREE CLICKS.

Now results. Across the bottom is forecast error. Up the side is average distance from the goal, so lower is better. Hindsight is the flat black line, because it never sees the forecast.

[click] The fixed plan. It gets worse as the forecast gets worse, as you'd expect.

[click] And MPC. Much lower, and it stays close to hindsight until the forecast is quite bad.

[click] The gap between orange and blue is what replanning is worth, since that's the only difference between them. With a real forecast it's about 570 kilometres closer to the goal, on average over the flight. And it's 330 even with a perfect forecast.

[If asked why it helps with a perfect forecast: this is distance over the whole flight. The fixed plan gets to the goal and then drifts away. MPC stays. Intervals: 569 plus or minus 201, 327 plus or minus 158.]
-->

---
part: 4
claim: Using the balloon's own wind readings
---

<div class="row">
  <div class="grow"><Layers name="readings" :n="3" :shown="1" /></div>
  <div class="note-side" v-click="2">
    <div class="big">0 <span style="font-size: 20px">of 24</span></div>
    <p style="margin-top: 6px">comparisons where readings detectably helped</p>
    <p style="margin-top: 10px">(4 horizons × 6 noise levels)</p>
  </div>
</div>

<!--
ROHAN. About 1 minute. TWO CLICKS.

This one surprised us. As the balloon flies it measures the wind where it is, and MPC can use that to correct its forecast. Loon did this. So we ran MPC with that switched on and switched off.

[click] Each dot is one launch: the difference between using the readings and ignoring them. Below zero means the readings helped.

[click] The black diamonds are the averages, and they sit on zero the whole way across. We made 24 of these comparisons, across planning horizons and error levels, and readings didn't detectably help in any of them.

We think it's scale. The balloon gets one reading every three hours along a single line, and the forecast error comes in patches about 500 km wide. One reading just doesn't tell you much about where you're headed next.

So what MPC gains over the fixed plan comes from replanning from where you actually are, not from learning about the wind.

[If asked: same answer on the cost that stops counting once the balloon arrives. The averages are a bit bigger there but still inside the noise.]
-->

---
part: 4
claim: Once the forecast is as bad as a real one, planning further ahead stops helping
---

<div class="row">
  <div class="grow"><Layers name="breaking" :n="5" :shown="1" /></div>
  <div class="note-side" v-click="5">
    <p><strong>Good forecast:</strong> looking further ahead gets you closer to hindsight.</p>
    <p style="margin-top: 12px"><strong>Real forecast or worse:</strong> the lines meet.</p>
    <p class="small muted" style="margin-top: 12px">Past 60 hours never detectably helped.</p>
  </div>
</div>

<!--
SHAURYA. About 1.5 minutes. FIVE CLICKS: four lines, then the takeaway.

This is the result we set out to find. Same axis across the bottom. Up the side is how far MPC is from hindsight, so zero would be perfect.

[click] This is MPC planning 15 hours ahead.

[click] 30 hours. Better, when the forecast is good.

[click] 60 hours. Better again.

[click] And 150 hours, the whole flight. On the left side the lines are in order: the further ahead you plan, the closer you get to hindsight.

[click] But look at the right. By the time the forecast is as wrong as a real one, the lines have come together. Planning further ahead on a wrong forecast doesn't buy you anything. And going past 60 hours never detectably helped, at any error level.

So there's a breaking point, and it sits somewhere between half of a real forecast's error and all of it.

[If asked: paired comparisons between neighbouring horizons are detectable up to 1.6 m/s and not from 3.2 on. Bars are 95 percent intervals over 32 launches.]
-->

---
part: 4
claim: Planning on many possible winds
---

<div class="row">
  <div class="grow"><Layers name="hedging" :n="3" :shown="1" /></div>
  <div class="note-side" v-click="3">
    <p><strong class="c-fixed">Fixed plan:</strong> ~200 km better, up to 3.2 m/s error</p>
    <p style="margin-top: 12px"><strong class="c-mpc">MPC:</strong> no difference at any noise level</p>
    <p style="margin-top: 12px">Same with the belief 2× too wide or 2× too narrow</p>
  </div>
</div>

<!--
ROHAN. About 1 minute. THREE CLICKS.

Last result from the sweep. Normally the planner scores each plan on the one forecast, the mean wind. The alternative is to draw a bunch of possible winds around the forecast and score each plan on its average over all of them. That's the textbook way to plan under uncertainty. Below zero means it helped.

[click] For the fixed plan it helps, by about 200 km, up to 3.2 metres a second of error. That makes sense. It gets one shot, so a plan that holds up when the forecast is wrong is worth having.

[click] For MPC it does nothing, at any error level.

[click] We checked whether that's just because our uncertainty happens to be exactly right, by making it twice too wide and twice too narrow. Same answer. MPC fixes its mistakes as they happen, so preparing for them in advance doesn't buy anything.

[If asked: 64 sampled winds for the fixed plan, 16 for MPC. Bars are 95 percent intervals over 32 launches.]
-->

---
layout: part
part: 5
who: Shaurya
---

<!--
SHAURYA takes over. About 15 seconds.

So that's what we know when the forecast is wrong. But a forecast with noise on it is only one way for a wind model to be wrong. This last part is what we're doing next.
-->

---
part: 5
claim: A balloon can only use the directions its altitudes offer
---

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
    <p style="font-size: 21px; font-weight: 650; line-height: 1.3; letter-spacing: -0.01em">What matters is how the wind changes with altitude.</p>
    <p class="small muted" style="margin-top: 12px">Adding noise to a forecast doesn't test that. It makes the wind wrong in only one way.</p>
  </div>
</div>

<!--
SHAURYA. About 1 minute. TWO CLICKS.

Two things you've already seen point the same way. Each picture is one column of air, one arrow per altitude.

cBottle had two altitudes, both blowing hard the same way, and nothing could steer.

[click] ERA5 has six, and a balloon can reach goals downwind but not across the wind or against it. In this record the winds at different altitudes oppose each other at only 58 percent of places.

[click] So what matters is how the wind changes with altitude. A balloon can only use the directions its altitudes offer. If a wind model gets that wrong, control breaks, however good it looks otherwise.

And our noise experiment doesn't really test that. It makes the forecast wrong in one particular way. So next we want to make the wind model wrong in different ways, one at a time, and see which ones hurt.
-->

---
part: 5
claim: "Next: make the wind model coarser, one property at a time"
---

<div class="row center" style="gap: 14px">
  <div class="col" style="width: 250px">
    <div class="label">True wind</div>
    <img src="/fig/split-truth.png" class="round" />
    <div class="cap">The balloon still flies here.</div>
  </div>
  <div class="col center" style="width: 200px" v-click="1">
    <div class="chip teal">coarsen one property</div>
    <div class="op" style="line-height: 1">→</div>
  </div>
  <div class="col" style="width: 250px" v-click="1">
    <div class="label">A cheaper wind model</div>
    <img src="/fig/split-coarse.png" class="round" />
    <div class="cap">MPC plans here.</div>
  </div>
</div>

<div class="center" style="margin-top: 22px; text-align: center" v-click="2">
  <p style="font-size: 20px"><strong>Same score as before:</strong> how much of the gap to hindsight does MPC close?</p>
  <p class="small muted" style="margin-top: 6px">If it holds up, that property can be coarse. If it drops, it matters.</p>
</div>

<!--
SHAURYA. About 1 minute. TWO CLICKS.

Here's the experiment. Same setup as before. The balloon flies in the true wind.

[click] But instead of adding noise, we give the controller a coarser copy of the truth. On the right is the same wind on a grid eight times coarser. We only change one property at a time, which you can't do with real generators, because a real generator is wrong in lots of ways at once.

[click] And we score it exactly the same way: how much of the gap to hindsight does MPC close? If the score holds up when a property is coarse, a simulator doesn't need to get it right. If it drops, it matters.

The design is locked. We haven't run it yet.
-->

---
part: 5
claim: Four properties, each made coarser step by step
---

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

[The picture uses our current 6-level record. The real experiment rebuilds the truth at 18 levels first, so the first row will start from 18.]
-->

---
part: 5
claim: Learned controller (PPO)
---

<Policy />

<div class="row" style="gap: 10px; margin-top: 14px; justify-content: center" v-click="3">
  <span class="chip on">64,000 training flights</span>
  <span class="chip">tested on the same 32 launches as MPC</span>
  <span class="chip">all 4 tasks</span>
  <span class="chip">1 balloon and fleets of 4</span>
</div>

<!--
ROHAN. About 1 minute. THREE CLICKS.

Everything so far has been MPC. But we also care about learned controllers: if you train a policy in a simulator, does it work in real wind? So I added PPO to balloon-sim.

On the left is what each balloon sees: the forecast wind at every altitude where it is, plus how far and which way the goal is, how much time is left, what its own wind sensor reads compared to the forecast, and in a fleet, where the other balloons are.

[click] That goes into one policy network, and every balloon in a fleet shares the same network. That's the standard multi-agent setup, MAPPO.

[click] And it outputs climb or sink, every three hours. Same choice MPC makes.

[click] The plan is to train it on 64,000 flights across all the noise levels, on launches separate from the ones we test on, then test it on the same 32 launches as MPC so it goes on the same plots. It's set up on all four tasks: reaching a goal, getting as far from the start as possible, station-keeping, and visiting several targets, with one balloon and with fleets of four.

That's what we're running next. Back to Shaurya.
-->

---
part: 5
claim: The answers tell us how to build the next generator
---

<div style="margin: 0 auto"><Arc :current="6" loop /></div>

<div class="row" style="gap: 16px; margin-top: 6px; align-items: stretch">
  <div class="card grow" v-click><h3 style="font-size: 17px">Levels to generate</h3><p class="small">cBottle: 2. Ours: 18. How many are enough?</p></div>
  <div class="card grow" v-click><h3 style="font-size: 17px">Frames per day</h3><p class="small">Every hour is expensive. Is every 12 fine?</p></div>
  <div class="card grow" v-click><h3 style="font-size: 17px">Grid size</h3><p class="small">Fine detail was our biggest realism gap. Does control care?</p></div>
</div>

<div class="cap" style="margin-top: 14px" v-click><strong style="color: var(--ink)">Wherever control doesn't care, the generator can be coarse, and cheap.</strong></div>

<!--
SHAURYA. About 1 minute. FOUR CLICKS.

So this is how it comes back around. We started by building a generator, and the experiment tells us how to build the next one.

Every row on that last slide is something a generator spends compute on.

[click] How many altitude levels to generate. cBottle has two near the balloon, ours has 18. Is the right number 6? 9?

[click] How many frames per day.

[click] And how fine a grid. Fine detail was the biggest gap in our realism scores. If control doesn't care about it, that's a gap we can stop worrying about.

[click] Wherever control doesn't care, the generator can be coarse, and that makes it cheap. And we'd have a way to score a generator on what it's actually for.
-->

---
layout: statement
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
