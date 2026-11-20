# A balloon control benchmark: what the paper needs (2026-11-19)

Target: RL Journal (anytime submission). A paper accepted before May 1 is presented at RLC that August
(Providence). Tags as in `docs/directions-2026-11-17.md`: **(measured)** our data; **(cited)** checked against
the source; **(assumed)** a judgement, unverified.

## What the two example papers do

Both are RLJ 2026 papers that introduce a benchmark (read 2026-11-19).

**Building2Building (B2B)**, Taboga et al. HVAC control in EnergyPlus, Gymnasium API. 23 pages with appendix.
(cited)
- The argument: generalization in RL needs many environments that differ in dynamics, observation, action and
  goal. Existing suites are mostly robotics and vary little beyond friction or target position. HVAC is varied,
  shares one physics, and matters (buildings use about a third of global energy).
- A generator that makes buildings, then a fixed dataset of 6,000 environments split into train and test.
- Four named benchmark settings, each isolating one source of variation: goal adaptation, dynamics
  adaptation, action-space shift, cross-domain transfer. Each is a table of train and test conditions.
- A domain-standard baseline (ASHRAE Guideline 36 reactive control) and a `compute_normalized_score` that
  divides return by that baseline's, so 1.0 means as good as what buildings run today.
- PPO baselines, then two experiments that show the benchmark is not solved: specialists beat generalists,
  with a long tail of buildings where transfer fails; a transformer over a structured representation handles
  mismatched observation and action spaces.

**Assistax**, Hinckeldey et al. Assistive robotics in JAX and MuJoCo-MJX, multi-agent. 38 pages with appendix.
(cited)
- A comparison table against eight related benchmarks, one checkmark column per property (continuous, 3D,
  hardware acceleration, multi-agent, ad-hoc teamwork). Theirs is the only row with every box ticked.
- Speed as a headline: up to 412x faster simulation and 25x faster training than CPU environments, one GPU.
- Tuned MARL baselines, a population of pretrained partner policies released on Hugging Face, and a held-out
  evaluation that exposes a "coordination gap" against unseen partners.

**What both have** (assumed, from the two):
1. A real application, and a specific reason the current benchmarks can't study it.
2. A comparison table against related benchmarks.
3. Structured variation with train and test splits, organised into a few named settings.
4. A standard API (Gymnasium) and a throughput number.
5. A domain-standard non-learned baseline, and a normalized score built on it.
6. RL baselines run carefully, and at least one experiment that shows a gap the field should work on.
7. Released artifacts: code, data, trained policies, docs.

## The case for ours

The prior benchmark is BLE (Greaves et al. 2021; Gym; station keeping; one balloon; wind from a VAE plus simplex
noise; QR-DQN, the Loon recipe). What ours adds (assumed until each is checked against BLE's code and paper):

| | BLE | ours |
|---|---|---|
| truth wind | VAE samples | real ERA5, held-out dates; our generator as a second source |
| forecast error | simplex noise, one size | four error models at swept sizes, including the real day-old error |
| tasks | station keeping | station keeping, reach a goal, max distance, waypoint chains |
| balloons | one | one or a fleet |
| reference controllers | none beyond the agent | coast, greedy, MPC, hindsight |
| speed | CPU, Python | JAX, thousands of balloons a step on one GPU |

The reason it's needed, in one line: **policies trained in a balloon simulator are scored in that simulator, so
nobody can tell whether they would work in real wind.** Ours scores them in real ERA5 that training never saw,
with the forecast as wrong as real forecasts are. Shaurya's fidelity results are the evidence that this matters
(`docs/fidelity-results-2026-11-18.md`, measured): the error's size sets how much control MPC loses, and BLE's
noise at BLE's own size gives a forecast about five times better than a day-old one. MPC loses 0.09 to 0.20 of
its control at BLE's size and 0.73 to 0.79 at a day-old forecast's size. The shape of the error made no
difference that 32 launches could detect.

## Benchmark settings (proposed)

Named like B2B's, each holding everything fixed but one thing:

1. **Control.** Each task in ERA5, train dates against held-out test dates. The baseline table.
2. **Forecast shift.** Train with one error model or size, test with another. Does a policy trained on BLE's
   simplex noise at BLE's size hold up against the real day-old error, which is about five times larger?
3. **Wind transfer.** Train on generated wind (ours, BLE's VAE, a degraded copy), test on held-out ERA5. This is
   where the fidelity work and the generator plug in.
4. **Region and season.** Train in the tropics in July, test at mid-latitudes or in January.
5. **Fleet.** N balloons, one shared reward: coverage or targets.

## Score

One normalized score per task, B2B-style, using controllers we already have:

`score = (agent − coast) / (MPC − coast)`

0 is never changing altitude, 1 is MPC with the real error-corrected forecast. Above 1 beats MPC. Hindsight is
reported as the ceiling. This is the fidelity score's denominator turned around (decision 55). Coast rather than
the fixed plan, because the fixed-plan-to-hindsight gap is about zero on station keeping (decision 53, measured).

Report the IQM with bootstrap intervals over seeds (Agarwal et al. 2021, rliable). That is the field's default
now (assumed).

## API (proposed, to settle with Vinamr)

Two layers:

- **A JAX core**, gymnax-style: `reset(key, params) -> (obs, state)` and `step(key, state, action, params)`, pure
  and vmappable. `station_env.py` already has this shape; the other tasks would get the same.
- **A Gymnasium wrapper** for everyone else: `gymnasium.make("BalloonSim/StationKeeping-v0", wind="era5",
  forecast="real", sigma=3.5)`, registered through `gymnasium.register`, with `render_mode="rgb_array"`.

Every axis is a config field, not a code path: truth source, forecast error model and size, physics, task,
number of balloons, observation (column, patch, ahead, or BLE's features), action (up/stay/down or continuous
climb).

IDs: `StationKeeping-v0`, `Reach-v0`, `MaxDistance-v0`, `Waypoints-v0`, and a `Fleet` variant of each.

Data: a small bundled record so tests and the quickstart run with no download, and a script or a Hugging Face
dataset for the full ERA5 records.

Docs on Read the Docs: install, quickstart, one page per task, how to add a wind source.

## What's missing

- **RL that works.** Our PPO ignores the forecast; QR-DQN is in but its go/no-go hasn't been run (decision 57).
  A benchmark paper can report that RL loses to MPC (that's Assistax's gap). It can't report RL that was
  never made to work. Needs honest tuning and 5+ seeds.
- **One env API across tasks.** Station keeping is in BLE's form at 3 min steps; reach and the others are in
  the 3 h sweep form. They need one interface and one time step per task.
- **Held-out data.** Train and test date ranges, and more than July 2021.
- **Throughput numbers** against BLE on the same hardware.
- **A physics axis.** Our balloon is kinematic. BLE models power and a day-night cycle. Either add it or say
  plainly that it's out of scope.
- **Visualisation.** A render and a replay page. The directions doc asks for "well visualised".

## Timeline (assumed)

RLJ's review time isn't known here; check it. If review takes about two months, submit by early March.

| when | what |
|---|---|
| late Nov | agree the API and who owns it; one env interface for station keeping and reach |
| Dec | the other tasks, the Gymnasium wrapper, bundled data, tests |
| Jan | baselines on every task: coast, greedy, MPC, hindsight, PPO, QR-DQN, 5 seeds |
| Feb | the five settings; throughput; docs site; write |
| early Mar | submit |

## Open questions

- One paper or two: the benchmark with the fidelity results as its first study, or the fidelity paper on its own.
  (The directions doc leans towards one: "the benchmark is the base and the ablation is the first paper on it".)
- Does our generator ship in the benchmark? It strengthens setting 3, but the weights and the code have to be
  releasable.
- ERA5 licence terms for redistributing records (unverified).
