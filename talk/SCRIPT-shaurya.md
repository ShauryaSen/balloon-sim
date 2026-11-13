# Script: Shaurya's slides

Parts 1–2 (slides 1–13), then Part 5 to the end (slides 29–34). [click] marks a click.
Slides 4–7 are Shaurya's own wording; the rest continues in that voice.

---

## 1 · Title
Hi everyone, I'm Shaurya and this is Rohan, and today we're talking about wind models for balloon control.

## 2 · Outline
So quickly, the plan: I'll start with some background and the wind generator we built, and how we checked whether it's realistic. Then Rohan's going to talk about how we actually measure whether a controller works in real wind, and an experiment he ran on that. And I'll finish with the experiment we're planning next.

## 3 · Part 1 divider
So first, why do we even need a new wind generator?

## 4 · High-altitude balloon control
So most of you are probably already familiar with this setting. We've had a lot of work in the area of high-altitude balloon control, where we have balloons that can control going up and down, and are moved horizontally by the wind, and so rely on the wind forecast to plan and navigate.

## 5 · Balloon tasks
So one task we've seen this used in is station keeping, the best-known example being Loon, which was Google's balloon-internet project, and Myles's recent paper.
[click] And there's also navigation, reaching waypoints that could be further away, like Vinamr's storm search.

## 6 · Prior work: BLE's wind generator
After Loon, Google released their simulator as the Balloon Learning Environment, which they and many following papers used to train controllers. It generates wind using a VAE trained on ERA5, which is a record of past weather.
[click] A potential issue with it is that it generates a patch about 1,000 by 1,000 km. That can be fine for station keeping, but it doesn't really support long-range navigation tasks.
[click] It also doesn't condition on a specific time or location.
[click] And it might not be realistic. They haven't shown that it's realistic, so that's something we built an evaluation for.

## 7 · Our generator
Our solution to this was creating our own generator, which is a diffusion model trained on ERA5, and it conditions on time and location. This is what a sample output looks like,
[click] compared to BLE's VAE, which is a lot coarser, both horizontally and vertically.

## 8 · InfiniteDiffusion
Now one sample only covers a patch about 16 degrees across, and a balloon flight goes a lot further than that. If you just generate patches separately and stick them together, you get these seams on the left, because each patch makes up its own weather. So we use InfiniteDiffusion, which generates overlapping patches together so they have to agree with each other, and you get one smooth field like on the right.
And it's nice for a simulator, because you only generate the wind where the balloon actually goes, and if you ask for the same place again you get the same wind, so every controller flies in the same world.

## 9 · Global generator (in progress)
That was the model from this summer. We've also been working on a global version, and this is its generated wind, so not ERA5. It does a coarse pass over the whole globe and then fills in the detail with overlapping patches, and the idea is to use InfiniteDiffusion in time so a flight never runs out of wind. It's trained on 2020 to 2023 ERA5. But everything I show next is from the summer model.

## 10 · Part 2 divider
So next, does our wind actually look like real wind?

## 11 · Realism checks
So we compare our generated wind against ERA5 the model hasn't seen, on four things that matter for a flight: how fast the wind is, how it changes across space, how it changes with altitude, which is what the balloon is actually choosing between, and how it changes over time.
[click] And to know what a good score even is, we score real ERA5 against other real ERA5. That's our ERA5 reference, the best you could hope to do, so the closer to it the better.

## 12 · Realism results
The dotted line is that ERA5 reference, and closer to it is better. We're closer to real wind than BLE on all four, and time is still the weakest for both of us.
[click] But then if you add simplex noise, just random noise fitted to ERA5's statistics, it actually beats both of us on all four.

## 13 · Limits of realism checks
And that's kind of the problem. Simplex was fitted to these exact statistics, so of course it scores well, but it has no jet streams or storms, and it doesn't know where or when it is. So scoring well here doesn't really tell you whether a model is good for control.
[click] So the question we actually care about is: what does a wind model need to get right so that a controller trained on it works in real wind? And to answer that we need a way to measure it, so I'll hand it over to Rohan.

---

*Rohan: Parts 3–4.*

---

## 29 · Part 5 divider
Thanks Rohan. So Rohan showed that control can handle random error about the size of a real forecast's error. But random error makes everything a little wrong at once, so it can't tell us which parts of the wind actually matter. So this last part is the experiment we're planning to run next.

## 30 · Fidelity experiment
The idea is we take ERA5 and make copies of it where we make just one thing worse, like here, where the grid is a lot coarser.
[click] Then we can plan on that copy with MPC, like Rohan's setup, and fly the balloon in the real wind.
[click] Or we can train a policy with RL, PPO, on the copy, and then test it in the real wind.
[click] And the drop in score tells us how much that one thing matters.
The reason we use copies of ERA5 and not our actual generator is that a generator gets a bunch of things wrong at once, so if the score drops you don't know why.

## 31 · Candidate properties
So these are the things we're thinking of making worse. The number of altitude levels, how coarse the grid is, and how far apart the frames are in time. Whether we use one patch of ERA5 or lots of locations, because if one patch works, the generator could output less and train on less. And the kind of error, so keeping the size the same but changing the type, like the Gaussian process error Rohan used versus something like simplex.

## 32 · Tasks
And we'd test each of these on three tasks: station keeping and navigation from before, and distance, which is just getting as far from where you started as you can. Something might matter for one task and not for another.

## 33 · Minimal generator
And the reason this matters for us is that a few of these are just settings in our generator. Right now we use 18 altitude levels, a frame every hour, and a quarter-degree grid, and honestly we picked those by judgement. This experiment tells us how coarse each one can go before it starts to hurt.
So in the end you get the cheapest generator that still transfers.

## 34 · Questions for you
So that's it, and there are a few things we'd love your input on. What other properties would you test? Do you think degrading ERA5 is a fair stand-in for a real generator's errors? And which task should we focus on first? Thanks.
