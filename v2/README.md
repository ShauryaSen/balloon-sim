# balloongym

Stratospheric balloon control as reinforcement-learning environments. A balloon can't steer. It can only change
altitude, and the wind at each altitude blows a different way. The controller gets a forecast of that wind,
which is wrong the way real forecasts are, and corrects it with what the balloon measures. The balloon flies in
the true wind.

balloongym is the project's simulator (`balloongym/sim`, a snapshot of `balloon_sim`) with a Gymnasium API, a
JAX API and a benchmark on top. Every environment step, controller, belief and trainer is the simulator's own
code.

## Tasks

| ID | task | step | score |
|---|---|---|---|
| `BalloonGym/StationKeeping-v0` | stay within 50 km of a point for 2 days | 3 min, up/stay/down | time within 50 km (TWR50) |
| `BalloonGym/Reach-v0` | get within 200 km of a goal 2,000 km downwind, and soon | 3 h, climb in [-1, 1] | summed distance to the goal (km) |
| `BalloonGym/MaxDistance-v0` | get as far from launch as possible | 3 h | summed distance from launch (km) |
| `BalloonGym/Waypoints-v0` | visit three points in a 2,000 km downwind chain | 3 h | summed distance to unvisited points (km) |

`ReachFleet-v0`, `MaxDistanceFleet-v0` and `WaypointsFleet-v0` fly four balloons that share one belief and one
reward.

Station keeping is in BLE's form: Pasetsky et al.'s control setup, and BLE's observation, reward and agent
(DESIGN 54, 57). The other three are the sweep's fleet objectives: 50 moves of 3 h, launches between 20N and
50N (DESIGN 45, 47, 48).

## Install

```bash
pip install -e ".[dev]"      # from this folder
pytest                       # balloongym's tests, then the simulator's own (tests/sim)
```

## Gymnasium

```python
import gymnasium as gym
import balloongym  # registers the environments

env = gym.make("BalloonGym/Reach-v0")
obs, info = env.reset(seed=0)
done = False
while not done:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
print(info)  # {"cost_km": ..., "score": ...}
```

`examples/quickstart.py` flies each task and saves a picture of each.

| argument | default | |
|---|---|---|
| `wind` | `"synthetic"` | or a path to an ERA5 `.npz` record, or a `balloongym.Wind` |
| `n_balloons` | `1` | fleet tasks only |
| `sigma_ms` | `3.2` | the forecast error's size on the fleet tasks, m/s per component (3.2 is a day-old forecast) |
| `learns` | `True` | whether the belief takes the balloons' readings |
| `view` | `"column"` | what of the field the fleet observation shows: `"column"`, `"patch"` or `"ahead"` (DESIGN 52) |
| `action` | `"continuous"` | fleet tasks: a climb in [-1, 1] of the cap, or `"discrete"` (0 up, 1 stay, 2 down) |
| `launches` | `256` | training launches `reset()` draws from (seeds 100,000 up) |
| `test` | `False` | draw from the 32 test launches instead |
| `task_kwargs` | | any simulator `TaskConfig` field, e.g. `{"distance_km": 1000, "n_moves": 30}` |

## The forecast and the belief

The forecast is what the controller is told the wind is. The belief is the forecast corrected by what the
balloons measure.

**Fleet tasks.**
- The forecast is the truth minus a smooth random error. The error is a Gaussian process with the kernel
  fitted to a day-old forecast's error on ERA5: 544 km across, 1.3 km in altitude, 9.1 h in time.
- Every move, each balloon reads the true wind where it is. The belief is the GP posterior mean given every
  reading so far:

  `belief(x) = forecast(x) + k(x, X) K^-1 (reading(X) - forecast(X))`

  Near a reading, in place, altitude and time, the belief is close to the truth. The correction fades over the
  kernel's lengths.
- The observation gives the belief's wind at every level of the balloon's column. A reading says little about
  levels more than a kilometre or two from where it was taken, so a correction mostly helps the levels close
  to the balloon.
- `learns=False` sets the readings aside, and the belief is the raw forecast.

**Station keeping.**
- The forecast is persistence: the wind 24 h earlier (the record's own, or built by `wind.persistence`).
- The correction uses the kernel measured on eight weeks of real forecast error (DESIGN 57). It reaches about
  twice as far sideways as the fitted one and has its own altitude structure.
- The belief uses the latest 48 readings, one every 18 min.
- The observation gives each level's corrected mean, as BLE's features, and its uncertainty.

## Controllers and scoring

```python
from balloongym import benchmark, make_task, synthetic

task, wind = make_task("reach"), synthetic()
coast = benchmark.evaluate(task, wind, "coast")       # the 32 test launches
mpc = benchmark.evaluate(task, wind, "mpc")
benchmark.normalized(mpc, coast, mpc)                 # 0 = coast, 1 = MPC
benchmark.summary(mpc["value"])                       # mean, IQM, 95% bootstrap interval
```

Every controller is flown by the simulator's `fleet.fly` on the sweeps' test seeds and keys. Launch, forecast
error and flight are the same for every controller, and results compare with the sweep results.

| controller | |
|---|---|
| `coast` | never changes altitude (the floor) |
| `fixed_plan` | plans the whole flight once on the forecast, then flies it blind |
| `mpc` | the sweep's MPC (Myles's planner): replans on the belief, the forecast corrected by readings |
| `mpc_blind` | the same, readings set aside |
| `mpc_truth` | the same planner on the true wind |
| `hindsight` | knows the true wind, plans the whole remaining flight every move (the ceiling; fleet tasks) |
| `ppo` | a trained PPO policy (`train_ppo`) |
| `qrdqn` | a trained QR-DQN policy (`train_qrdqn`, station keeping) |

`value` is what the benchmark ranks on, higher better: TWR50 for station keeping, minus the summed cost (km)
for the fleet tasks.

## Training

```bash
python scripts/train_ppo.py --task reach --out policies/reach.pkl
python scripts/evaluate.py --task reach --controller ppo --params policies/reach.pkl

python scripts/train_qrdqn.py --out runs/qrdqn/seed0
python scripts/evaluate.py --task station_keeping --controller qrdqn --params runs/qrdqn/seed0/params_best.npz
```

- **PPO** is `sim/rl.py`: one policy shared by every balloon, PPO's usual settings (`RLConfig`), untuned.
- **QR-DQN** is `sim/qrdqn.py`, BLE's published recipe (`QRConfig`).
- Both train on launches from seed 100,000 up, never the test seeds. QR-DQN keeps the weights that do best on
  validation launches (seed 200,000 up).

## JAX

```python
import jax
from balloongym import FleetEnv, make_task, synthetic, missions_for

wind, task = synthetic(), make_task("reach", n_balloons=4)
env = FleetEnv(task)
launches = missions_for(task, wind, range(100_000, 100_256))
obs, state = env.reset(wind.truth, launches, jax.random.key(0))
obs, state, reward, terminated, truncated, info = env.step(wind.truth, state, action)  # action (4,) in [-1, 1]
```

`reset` and `step` are pure and jitted, so `jax.vmap` runs many episodes at once. `StationEnv` is the same for
station keeping.

## Wind

`balloongym.load(path)` reads the project's ERA5 records: an `.npz` with `u`, `v` of shape
(time, altitude, lat, lon) and `hours`, `altitude_km`, `latitude`, `longitude`, plus `forecast_u` and
`forecast_v` if the record carries its persistence forecast.

`balloongym.synthetic()` is a global field for tests and getting started. It has winds that turn with
altitude, but it is not a model of real weather.

## Layout

```
balloongym/
  sim/          the simulator, copied unchanged from balloon_sim at b50295b (see sim/__init__.py)
  tasks.py      the four tasks, each a fixed set of the simulator's settings
  wind.py       records, persistence forecasts, the synthetic field
  env.py        FleetEnv and StationEnv: reset and step from the simulator's functions
  gym_env.py    the Gymnasium wrapper
  benchmark.py  every controller on the test launches, the normalized score, summaries
  train.py      PPO and QR-DQN
  render.py     a picture of an episode
scripts/        evaluate.py, train_ppo.py, train_qrdqn.py
tests/          balloongym's tests; tests/sim is the simulator's own suite
```

## Status

Version 0.2.

- The simulator, controllers and RL are the project's code.
- What's left:
  - The ERA5 records aren't bundled.
  - There are no baseline results on ERA5 yet.
  - The physics is kinematic only (no power or day-night cycle).
  - Station keeping is one balloon only.
