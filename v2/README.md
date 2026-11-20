# balloongym

Stratospheric balloon control as reinforcement-learning environments. A balloon can't steer; it can only change
altitude, and the wind at each altitude blows a different way. The controller gets a forecast of that wind, which
is wrong the way real forecasts are, and the balloon flies in the true wind.

- Four tasks: station keeping, reach a goal, max distance, waypoints, each for one balloon or a fleet
- The truth is an ERA5 record, or a synthetic field that needs no download
- Forecasts: a perfect one, a Gaussian-process error of any size, or persistence (the real wind a day earlier)
- A Gymnasium API, and the same environment as pure JAX functions for thousands of episodes at once
- Reference controllers (coast, greedy, MPC) and a normalized score against them

## Install

```bash
pip install -e ".[render]"     # from this folder; add ",dev" for the tests
```

## Gymnasium

```python
import gymnasium as gym
import balloongym  # registers the environments

env = gym.make("BalloonGym/Reach-v0", forecast="gp", sigma_ms=3.2)
obs, info = env.reset(seed=0)
done = False
while not done:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
print(info["score"])
```

`examples/quickstart.py` flies each task with MPC and saves a picture of the episode.

| ID | task | length | score |
|---|---|---|---|
| `BalloonGym/StationKeeping-v0` | stay within 50 km of the launch point | 2 days | fraction of steps inside |
| `BalloonGym/Reach-v0` | get within 100 km of a goal 1,500 km away | 6 days | fraction of balloons that arrive |
| `BalloonGym/MaxDistance-v0` | get as far from launch as possible | 3 days | final distance, km |
| `BalloonGym/Waypoints-v0` | visit 3 points in order, 1,800 km in all | 6 days | fraction of points reached |

Each has a `...Fleet-v0` version with 4 balloons sharing one reward. Goals are placed downwind of where a random
altitude would carry the balloon, so some altitude leads toward them.

Arguments to `gym.make`:

| argument | default | |
|---|---|---|
| `wind` | `"synthetic"` | or a path to an ERA5 `.npz` record, or a `WindField` |
| `forecast` | `"gp"` | `"perfect"`, `"gp"` or `"persistence"` |
| `sigma_ms` | `3.2` | GP error size, m/s per component (3.2 is about a day-old forecast) |
| `n_balloons` | `1` | |
| `action` | `"discrete"` | down/stay/up, or `"continuous"` climb command in [-1, 1] |
| `dt_minutes` | `30` | time between decisions |
| `task_kwargs` | | e.g. `{"radius_km": 100, "hours": 72, "distance_km": 2000}` |

## JAX

The same environment as pure functions. The wind field is an argument, so one compiled environment runs on any
field of the same shape.

```python
import jax
from balloongym import BalloonEnv, EnvConfig, task_config, synthetic_wind, rollout
from balloongym import controllers

field = synthetic_wind()
env = BalloonEnv(EnvConfig(task=task_config("station_keeping"), n_balloons=1))

obs, state = env.reset(field, jax.random.key(0))
obs, state, reward, terminated, truncated, info = env.step(field, state, action)

# 1,000 episodes at once
keys = jax.random.split(jax.random.key(0), 1000)
episodes = jax.jit(jax.vmap(lambda k: rollout(env, field, k, controllers.mpc(env, field))))(keys)
episodes.score  # (1000,)
```

## Observation and action

One row per balloon, scaled to about [-1, 1]:
- altitude in the band
- fraction of the episode left
- offset and squashed distance to the task's point
- task progress
- the wind measured where the balloon is
- the forecast at the balloon's position at every level
- for fleets, the mean offset to the other balloons

A real balloon measures the wind only where it is and gets a forecast for the rest; the observation gives the
same.

The action is down, stay or up (or a continuous command), at up to 1.1 km per 3 h. The balloon is kinematic: it
moves at the wind's speed, with no power, battery or day-night cycle.

## Scoring

```python
from balloongym.benchmark import evaluate, reference_scores, normalized_score, summary

scores = evaluate(env, field, my_policy)       # 64 fixed test launches
ref = reference_scores(env, field)             # coast and MPC on the same launches
normalized_score(scores, ref)                  # 0 = coast, 1 = MPC, above 1 beats MPC
summary(scores)                                # mean, IQM and a 95% bootstrap interval
```

Train on `benchmark.train_keys`, which never overlap the test launches. With `forecast="perfect"`, MPC plans on
the truth, which gives a ceiling.

| controller | what it does |
|---|---|
| coast | never changes altitude |
| greedy | tries down, stay and up, each held for 3 h on the forecast; takes the best |
| mpc | plans 64 altitude sequences 12 h ahead on the forecast; flies the first step; replans every step |

## Wind records

`load_record(path)` reads an `.npz` with `u`, `v` of shape (time, altitude, lat, lon) and the axes `hours`,
`altitude_km`, `latitude`, `longitude`. That is the format of the ERA5 records this project builds.

The synthetic field is for tests and getting started. It has winds that turn with altitude, but it is not a model
of real weather.

## Layout

```
balloongym/
  wind.py         WindField, ERA5 records, the synthetic field, reading the wind at a point
  forecast.py     the forecast: perfect, GP error, persistence
  physics.py      the balloon: drift and climb
  tasks.py        the four tasks: placement, reward, score
  env.py          the environment as pure JAX functions, and whole-episode rollouts
  gym_env.py      the Gymnasium wrapper
  controllers.py  coast, greedy, MPC
  benchmark.py    test launches, normalized score, summaries
  render.py       a picture of an episode
tests/            pytest; the Gymnasium env checker runs on every registered ID
examples/
```

## Status

Version 0.1, a first cut.

- Synthetic wind and the API are in. ERA5 records are not bundled yet.
- Baseline RL results are not in yet.
- The physics is kinematic only.

```bash
pytest   # 29 tests, about 40 s on a laptop CPU
```
