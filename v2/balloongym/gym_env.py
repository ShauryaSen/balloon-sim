"""The Gymnasium interface.

    import gymnasium as gym
    import balloongym  # registers the environments

    env = gym.make("BalloonGym/Reach-v0")
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

Keyword arguments to gym.make:

    wind         "synthetic" (default, no download), a path to an ERA5 .npz record, or a balloongym.wind.Wind
    wind_seed    which synthetic field
    n_balloons   a fleet sharing one belief and one reward (not station keeping)
    sigma_ms     the forecast error's size on the fleet tasks, m/s per component (default 3.2: a day-old forecast)
    learns       whether the belief takes the balloons' readings (default True)
    view         what of the field a fleet task's observation shows: "column", "patch" or "ahead"
    action       fleet tasks: "continuous" (default; a climb in [-1, 1] of the cap per move) or "discrete"
                 (0 up, 1 stay, 2 down, as station keeping)
    launches     how many training launches reset() draws from (default 256)
    test         reset() draws from the 32 test launches instead (for evaluating, not training)
    task_kwargs  overrides of the task's settings, e.g. {"distance_km": 1000, "n_moves": 30}

One balloon gets a flat observation; a fleet gets (n_balloons, obs_size), one row per balloon.
info: "score" (the task's score so far) and, on the fleet tasks, "cost_km" (the summed objective so far).

Each step is one compiled JAX call. For thousands of episodes at once, use balloongym.env directly.
"""

from __future__ import annotations

import functools
from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from balloongym import wind as winds
from balloongym.env import TEST_SEEDS, TRAIN_SEED_OFFSET, FleetEnv, StationEnv, make_env, missions_for
from balloongym.tasks import make_task

OBS_BOUND = 50.0  # every feature is scaled to about [-1, 1]; offsets per 1,000 km can reach a few, never this


@functools.lru_cache(maxsize=4)
def _cached_wind(wind: str, seed: int) -> winds.Wind:
    return winds.synthetic(seed) if wind == "synthetic" else winds.load(wind)


@functools.lru_cache(maxsize=16)
def _cached_missions(task, wind_name: str, wind_seed: int, seeds: tuple[int, ...]):
    return missions_for(task, _cached_wind(wind_name, wind_seed), seeds)


class BalloonGymEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 8}

    def __init__(
        self,
        task: str = "reach",
        wind: str | winds.Wind = "synthetic",
        wind_seed: int = 0,
        n_balloons: int = 1,
        sigma_ms: float | None = None,
        learns: bool = True,
        view: str = "column",
        action: str = "continuous",
        launches: int = 256,
        test: bool = False,
        task_kwargs: dict[str, Any] | None = None,
        render_mode: str | None = None,
    ):
        self.task = make_task(task, n_balloons=n_balloons, sigma_ms=sigma_ms, **(task_kwargs or {}))
        self.wind = wind if isinstance(wind, winds.Wind) else _cached_wind(wind, wind_seed)
        seeds = TEST_SEEDS if test else tuple(range(TRAIN_SEED_OFFSET, TRAIN_SEED_OFFSET + launches))
        if isinstance(wind, winds.Wind):
            self.missions = missions_for(self.task, self.wind, seeds)
        else:
            self.missions = _cached_missions(self.task, wind, wind_seed, seeds)
        self.core = make_env(self.task, self.wind, learns, view)
        self.station = isinstance(self.core, StationEnv)
        self.render_mode = render_mode
        if action not in ("continuous", "discrete"):
            raise ValueError("action is 'continuous' or 'discrete'")
        self.discrete = self.station or action == "discrete"

        n = self.task.n_balloons
        size = self.core.obs_size(self.wind.truth.altitude_km.shape[0])
        self.observation_space = spaces.Box(-OBS_BOUND, OBS_BOUND, (size,) if n == 1 else (n, size), np.float32)
        if self.discrete:
            self.action_space = spaces.Discrete(3) if n == 1 else spaces.MultiDiscrete([3] * n)
        else:
            self.action_space = spaces.Box(-1.0, 1.0, (n,), np.float32)
        self._state = None
        self._path: list[np.ndarray] = []

    # --- the loop ---------------------------------------------------------------------------------------------------

    def _obs(self, obs) -> np.ndarray:
        obs = np.clip(np.asarray(obs, np.float32), -OBS_BOUND, OBS_BOUND)
        return obs.reshape(-1) if self.task.n_balloons == 1 else obs

    def _arg(self):
        return self.wind if self.station else self.wind.truth

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        key = jax.random.key(int(self.np_random.integers(2**31 - 1)))
        obs, self._state = self.core.reset(self._arg(), self.missions, key)
        self._path = [self._position()]
        return self._obs(obs), {"score": 0.0}

    def _to_core(self, action) -> jax.Array:
        a = jnp.asarray(np.asarray(action))
        if self.station:
            return a.astype(jnp.int32).reshape(())
        if self.discrete:  # 0 up, 1 stay, 2 down -> +1, 0, -1 of the cap
            return 1.0 - a.astype(jnp.float32)
        return a.astype(jnp.float32)

    def step(self, action):
        if self._state is None:
            raise RuntimeError("call reset() before step()")
        obs, self._state, reward, terminated, truncated, info = self.core.step(self._arg(), self._state, self._to_core(action))
        self._path.append(self._position())
        return self._obs(obs), float(reward), bool(terminated), bool(truncated), {k: float(v) for k, v in info.items()}

    # --- pictures ---------------------------------------------------------------------------------------------------

    def _position(self) -> np.ndarray:
        b = self._state.env.balloon if self.station else self._state.balloons
        return np.stack([np.atleast_1d(np.asarray(x)) for x in (b.lat, b.lon, b.altitude_km)])  # (3, N)

    def _points(self) -> tuple[np.ndarray, np.ndarray]:
        if self.station:
            e = self._state.env
            return np.atleast_1d(np.asarray(e.station_lat)), np.atleast_1d(np.asarray(e.station_lon))
        m = self._state.mission
        if self.task.objective == "escape":
            return np.zeros(0), np.zeros(0)
        return np.asarray(m.targets_lat), np.asarray(m.targets_lon)

    def render(self):
        if self.render_mode != "rgb_array" or self._state is None:
            return None
        from balloongym.render import draw

        lat, lon = self._points()
        score = float(self.core.score(self._state)) if isinstance(self.core, FleetEnv) else float(self._state.inside / max(int(self._state.env.move), 1))
        return draw(self.task, self.wind.truth, np.stack(self._path), lat, lon, score)
