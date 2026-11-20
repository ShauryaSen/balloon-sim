# balloongym: design decisions

What was decided in building `v2/`, and why. The simulator underneath has its own decisions in the repo's
`DESIGN.md`, cited here as "DESIGN n". Tags: **(measured)** checked by running it; **(assumed)** a judgement,
unverified.

## 1. Structure

**1.1 A separate package in `v2/`.** `balloongym` has its own `pyproject.toml`, venv and tests, on branch
`benchmark-plan`.
- Why: the benchmark should be one installable thing others can use. Shaurya's experiments keep running in
  `balloon_sim/` without anything here getting in the way.

**1.2 The simulator is copied in, not imported.** `balloongym/sim/` is `balloon_sim/` at b50295b (2026-11-18).
Only the import path is changed (`balloon_sim` → `balloongym.sim`). His tests are copied into `tests/sim/`
the same way.
- Why: we reuse what is built and vetted (the dynamics, the forecast and belief, the fleet tasks, Myles's
  planner, MPC, PPO, QR-DQN), while later changes to `balloon_sim/` can't change our results.
- To take a later change: copy the module over, rerun both test suites.
- One edit in the copied tests: `tests/sim/test_results.py` points one level further up for `scripts/`.

**1.3 balloongym's own code only wires the simulator together.** It has no physics, forecast, task or planner
code of its own:

| module | what it does |
|---|---|
| `tasks.py` | named settings |
| `env.py` | reset and step from the simulator's functions |
| `gym_env.py` | the Gymnasium wrapper |
| `benchmark.py` | scoring |
| `train.py` | calls the trainers |
| `wind.py` | records and the synthetic field |
| `render.py` | pictures |

- v0.1 (b59cca1) had its own implementations of all of these. It was replaced so there is one simulator, not two.

**1.4 Ruff and formatting skip `sim/` and `tests/sim/`**, so the copy stays byte-identical to the source
apart from imports.

## 2. Tasks

**2.1 Two families, because the simulator has two control setups.**

| ID | simulator code | step | action | episode | launches |
|---|---|---|---|---|---|
| StationKeeping-v0 | `station_env` (DESIGN 54, 57) | 3 min | up / stay / down | 960 steps, 2 days | 10S–10N |
| Reach-v0 | `fleet` + `rl`, objective `reach` | 3 h | climb in [-1, 1] × 1.1 km | 50 moves, 6.25 days | 20N–50N |
| MaxDistance-v0 | objective `escape` | 3 h | same | same | same |
| Waypoints-v0 | objective `targets` | 3 h | same | same | same |

- Settings are the sweeps' own:
  - **Station keeping:** `fidelity.configs` for the base cell. That is Pasetsky et al.: a 12 h horizon,
    replanning every 72 min, the normalised optimizer, a 50 km radius.
  - **Fleet tasks:** `TaskConfig` defaults: 2,000 km downwind, a 200 km arrival radius, 3 waypoints, and the
    MPC horizon of 20 (ExperimentConfig.new_objective_horizon).
- Why not one time step for everything: station keeping at 3 h steps can't be scored the field's way. A move
  carries the balloon about 100 km, twice the radius (DESIGN 54). The fleet tasks at 3 min would be 3,000-step
  episodes for no gain.

**2.2 No fleet station keeping.** It would mean either the 3 h form (unscorable, 2.1) or a fleet version of
`station_env` (not built). One balloon only for now.

**2.3 The headline value per task.**
- Station keeping: time within 50 km (TWR50, DESIGN 55).
- Fleet tasks: the objective's distance summed over the moves (`fleet_cost`, km, lower better). This is what
  the planner minimises and what the sweeps report.
- `info["score"]` also gives a plain per-task number: the fraction of balloons arrived (reach), the fraction of
  waypoints visited, or km from launch (max distance).

**2.4 Task settings can be overridden** (`make_task(name, **TaskConfig fields)`, or `task_kwargs` in
`gym.make`). The registered IDs are the defaults; any change is visible in the call.

## 3. Environment semantics

**3.1 The fleet env's step is `rl.episode`'s move, reordered for reset/step.**
- `reset` gives the first observation.
- Each `step`:
  1. moves the balloons in the true wind;
  2. charges `rl.move_cost` with the flags from before the move;
  3. updates the flags;
  4. reads the wind at every balloon;
  5. adds the readings to the belief;
  6. builds the next observation with `rl.observe`.
- Tested (measured):
  - The return of a coast episode equals −`fleet.fly`'s cost / 1,000 for the same launch and forecast.
  - The first observation equals `rl.episode`'s.

**3.2 Reward = −(move cost) / 1,000 km**, `RLConfig.reward_km`, so the return is −cost / 1,000. Unchanged from
PPO's training, so a policy trained either way sees the same reward.

**3.3 Reach and waypoints terminate when done**: every balloon has arrived, or every point is visited.
- `rl.episode` never terminates. It runs all 50 moves, and the cost is zero after arrival.
- Terminating loses no reward, and gives Gymnasium users a real `terminated` signal.
- Station keeping and max distance only truncate, at the time limit.

**3.4 After the episode ends, `step` changes nothing and returns reward 0.** The state is frozen with
`jnp.where`, so vmapped batches of episodes can run a fixed number of steps.

**3.5 Station keeping's step is `station_env.env_step` exactly** (tested, measured). The env adds a count of
steps within the radius, for the score.

## 4. Observation, forecast and belief

**4.1 The fleet observation is `rl.observe`, unchanged:** 11 + 2 × levels numbers per balloon (DESIGN 50, 52).
- `view` ("column", "patch", "ahead") is exposed.
- The Gymnasium box is ±50 and observations are clipped to it. Every feature is scaled to about ±1. The bound
  exists because the env checker objects to infinite bounds; nothing is ever near it (assumed).

**4.2 The fleet forecast is the GP error, read at points** (`rl.forecast_at`).
- Kernel: `BeliefConfig`: 544 km, 1.3 km, 9.1 h, fitted to a day-old forecast's error.
- σ is fixed per environment at the task's own (3.2 m/s by default; `sigma_ms` changes it).
- `rl.train` samples σ from a ladder; `train_ppo` defaults to the one σ and takes `sigmas` for a ladder.
- Why fixed: a benchmark task should be one problem. Robustness across σ is a benchmark setting to define
  later, not the default.

**4.3 `learns=True` by default.** The belief is the GP posterior mean given every reading the fleet has taken
(see the README). `learns=False` gives the raw forecast, as `rl_blind`.
- Why on: real balloons correct their forecast with what they measure (Bellemare et al. 2020).
- Found while testing (measured, 8 launches on synthetic wind): one reading cuts the belief's error at the
  levels beside the balloon by about a third on average. Between launches it varies from 2.2 → 0.6 m/s to
  almost nothing. A reading says little about levels a few km away (the 1.3 km altitude length), and altitude
  is the only control. This is the simulator's behaviour, not a bug: the observation matched `rl.episode`'s
  exactly.

**4.4 Station keeping's forecast is persistence; its belief uses the measured kernel** (DESIGN 57).
- The belief uses the latest 48 readings, one every 18 min.
- σ is `degrade.error_size` of the record's own persistence error over the launch band, as QR-DQN's script
  sets it.
- `MeasuredKernel` is loaded from `sim/data/measured-kernel-2026-11-18.json`, copied from `results/`.

**4.5 Readings are capped at what the simulator holds:**
- fleet: every move of every balloon (`n_moves × N`);
- station keeping: a 48-slot ring.

## 5. Actions

**5.1 Fleet tasks are continuous by default:** a climb in [-1, 1] of the 1.1 km cap per move.
- PPO's policy outputs `cap × tanh(u)`, which is the same space.
- `action="discrete"` maps 0, 1, 2 to up, stay, down at the full cap.

**5.2 Discrete actions are 0 up, 1 stay, 2 down everywhere**, which is `station_env`'s convention (UP=0).
v0.1 used 0 = down; changed so both families agree.

**5.3 One balloon gets a flat observation and a scalar action; a fleet gets (N, obs) and (N,).** The policy
is meant to be shared across balloons, as MAPPO with parameter sharing is.

## 6. Launches and seeds

**6.1 Three disjoint seed ranges, the sweeps' own:**

| use | seeds |
|---|---|
| test | 0–31 |
| training | from 100,000 (`RLConfig.seed_offset`) |
| QR-DQN validation | from 200,000 |

Training never sees a test launch. Test results compare with every sweep result on the same seeds.

**6.2 Launches are drawn once, outside jit, and `reset` picks from the pool.**
- `task.draw_task` uses Python control flow (it tries 1,000 starts and keeps the first whose goal fits), so it
  can't run inside jit. `rl.training_missions` and `qrdqn.missions` pool launches for the same reason.
- `gym.make(..., launches=256)` sets the pool size; `test=True` uses the test seeds.

**6.3 Every episode starts at hour 0 of the record** (`TaskConfig.start_hours`). Episodes differ by place,
not date. This is a limit of the simulator's task drawing (assumed fine for now; noted in 10).

## 7. Benchmark scoring

**7.1 Every controller is flown by `fleet.fly`** with `cells.keys(seed)` for launch, forecast error and flight.
Learned policies are flown the same way, PPO through `rl.controller` and QR-DQN through `qrdqn.evaluate`.
- Why: one simulator scores everything, and the numbers line up with Shaurya's sweeps.

**7.2 References and their beliefs:**

| controller | fleet tasks | station keeping |
|---|---|---|
| coast | never climbs | same |
| fixed_plan | plans once on the forecast mean | not run (DESIGN 55 dropped it) |
| mpc | `fleet.mpc` on the GP belief, readings on | `fleet.mpc` on the measured-kernel belief (the readings arm) |
| mpc_blind | readings set aside | plans on the persistence forecast as a plain field |
| mpc_truth | the planner on the true wind | same (the fidelity sweep's top) |
| hindsight | whole remaining flight on the truth every move | not run (960-step plans) |

**7.3 `value`, higher better:** TWR50, or −summed cost.

**7.4 Normalized score = (agent − coast) / (MPC − coast), as a ratio of means over launches.**
- Why coast: on station keeping the fixed-plan-to-hindsight gap is about zero (DESIGN 53). A per-launch ratio
  has near-zero denominators (DESIGN 55).
- Why MPC as 1: it is the strongest controller every task has. Hindsight is reported as the ceiling where it
  runs.

**7.5 `summary` gives the mean, IQM and a 95% bootstrap interval.** IQM follows Agarwal et al. 2021, the
field's default for RL results (assumed).

**7.6 Reference results on synthetic wind, 4 launches (measured):** coast < MPC < hindsight on every task.
These aren't benchmark numbers (synthetic wind, 4 launches). They show the scoring orders controllers
correctly. About 10–20 s per MPC or hindsight flight on a laptop CPU.

| task | coast | MPC | hindsight |
|---|---|---|---|
| station keeping, TWR50 | 0.05 | 0.17 | n/a |
| reach, summed km | 126,702 | 45,270 | 33,866 |
| max distance, final km from launch | 2,297 | 4,857 | 5,468 |
| waypoints, summed km | 267,796 | 63,864 | 47,271 |

## 8. Training

**8.1 PPO is `rl.train`, unchanged.** `train_ppo` draws the training launches and passes the task's settings.
`RLConfig` defaults are PPO's usual and untuned (the simulator's own note).

**8.2 QR-DQN is `qrdqn.make`, run in the loop `scripts/train_qrdqn.py` uses:**
- validate on 64 launches every 5,000 vector steps;
- keep the best-on-validation weights;
- the test seeds choose nothing.

**8.3 Saved formats:**
- **PPO:** a pickle with params, `rl.Spec`, `RLConfig`, the curve and the task.
- **QR-DQN:** `params_best.npz` (layer i, part j as "i_j"), `config.json` and `curve.jsonl`.
- `evaluate.py` reads either. It rebuilds QR-DQN's network from `config.json`, so non-default sizes score
  correctly.

## 9. Wind

**9.1 `load(path)` reads the project's ERA5 records with `sim.wind.load_record`.** It takes the record's own
persistence forecast (`forecast_u/v`) when there is one; otherwise it builds one by shifting frames 24 h.

**9.2 `synthetic()` is for tests and getting started only.** It is a global field:
- 3 h frames over 10 days, 10 levels from 14.3 to 20.2 km (the records' band), 2.5°;
- a mean wind that turns from westerly at the bottom to easterly at the top;
- smooth random structure (about 1,000 km, 1.5 km, 24 h).

It isn't a model of real weather and no result should be reported on it.

**9.3 The synthetic persistence forecast holds the first frame for the first 24 h.** There is no history
before hour 0, and episodes start at hour 0 (6.3). So the first day of a synthetic episode has a frozen
forecast. Real records carry a true earlier field.

## 10. Known limits and open questions

- **Nothing has run on ERA5 yet.** No record is on the laptop. The first runs go on Unicorn.
- **No real training runs yet.** PPO and QR-DQN are only tested for a few steps.
- **Episodes all start at record hour 0** (6.3). For a benchmark, start times should vary across the record,
  and train and test should use different dates. This needs a jittable or pooled draw over start times.
- **The measured kernel was fitted in the tropics** and is used wherever station keeping launches (it launches
  in the tropics, so fine for now).
- **The physics is kinematic only:** no power, battery or day-night cycle (DESIGN 57 dropped BLE's power
  features).
- **Station keeping has no fleet version** (2.2).
- **The benchmark settings from the plan** (forecast shift, wind transfer, region and season) aren't defined as
  code yet. Each needs held-out ERA5 dates or regions.
- **Speed:** the reference MPC takes 10–20 s a flight on CPU. A full 32-launch reference table per task wants
  the cluster.

## 11. Testing

- `tests/test_balloongym.py` (18 tests):
  - equivalence with the simulator: fleet return against `fleet.fly`, the first observation against
    `rl.episode`, station steps against `station_env`;
  - readings improve the belief on average;
  - Gymnasium's `check_env` on every registered ID;
  - a rendered episode;
  - MPC beating coast through the benchmark;
  - a short PPO and QR-DQN train-then-score each.
- `tests/sim/`: the simulator's own 127 tests, run against the copy.

See also `README.md` (how to use it) and `../docs/benchmark-paper-plan-2026-11-19.md` (what the paper needs).
