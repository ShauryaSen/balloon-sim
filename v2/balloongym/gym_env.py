"""The Gymnasium interface.

    import gymnasium as gym
    import balloongym  # registers the environments

    env = gym.make("BalloonGym/StationKeeping-v0")
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

Keyword arguments to gym.make:

    wind         "synthetic" (default, no download), a path to an ERA5 .npz record, or a WindField
    wind_seed    which synthetic field
    forecast     "gp" (default), "persistence" or "perfect"
    sigma_ms     the GP forecast error's size, m/s per component (default 3.2, about a day-old forecast)
    n_balloons   a fleet sharing one reward (default 1)
    action       "discrete" (down, stay, up; default) or "continuous" (a climb command in [-1, 1])
    dt_minutes   time between decisions (default 30)
    task_kwargs  overrides for the task, e.g. {"radius_km": 100, "hours": 72}

One balloon gets a flat observation (obs_size,); a fleet gets (n_balloons, obs_size), one row per balloon.
info["score"] is the task's benchmark score so far.

Each step is one compiled JAX call. For thousands of episodes at once, use balloongym.env directly.
"""

from __future__ import annotations

import functools
from typing import Any

import gymnasium as gym
import jax
import numpy as np
from gymnasium import spaces

from balloongym import controllers, tasks
from balloongym.env import BalloonEnv, EnvConfig, EnvState, obs_size
from balloongym.forecast import ForecastConfig
from balloongym.physics import BalloonConfig
from balloongym.wind import WindField, load_record, synthetic_wind

OBS_BOUND = 20.0  # every feature is scaled to about [-1, 1]; offsets per 1,000 km can reach a few, never 20


@functools.lru_cache(maxsize=4)
def _cached_wind(wind: str, seed: int) -> WindField:
    return synthetic_wind(seed) if wind == "synthetic" else load_record(wind)


class BalloonGymEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 8}

    def __init__(
        self,
        task: str = "station_keeping",
        wind: str | WindField = "synthetic",
        wind_seed: int = 0,
        forecast: str = "gp",
        sigma_ms: float = 3.2,
        n_balloons: int = 1,
        action: str = "discrete",
        dt_minutes: float = 30.0,
        task_kwargs: dict[str, Any] | None = None,
        render_mode: str | None = None,
    ):
        self.field = wind if isinstance(wind, WindField) else _cached_wind(wind, wind_seed)
        config = EnvConfig(
            task=tasks.task_config(task, **(task_kwargs or {})),
            balloon=BalloonConfig(dt_minutes=dt_minutes),
            forecast=ForecastConfig(kind=forecast, sigma_ms=sigma_ms),
            n_balloons=n_balloons,
            action=action,
        )
        self.core = BalloonEnv(config)
        self.render_mode = render_mode
        n, size = n_balloons, obs_size(config, self.field.altitude_km.shape[0])
        shape = (size,) if n == 1 else (n, size)
        self.observation_space = spaces.Box(-OBS_BOUND, OBS_BOUND, shape, np.float32)
        if action == "discrete":
            self.action_space = spaces.Discrete(3) if n == 1 else spaces.MultiDiscrete([3] * n)
        else:
            self.action_space = spaces.Box(-1.0, 1.0, (n,), np.float32)
        self._state: EnvState | None = None
        self._key = jax.random.key(0)
        self._path: list[np.ndarray] = []
        self._policies: dict = {}

    def _obs(self, obs) -> np.ndarray:
        obs = np.clip(np.asarray(obs), -OBS_BOUND, OBS_BOUND)
        return obs[0] if self.core.config.n_balloons == 1 else obs

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._key = jax.random.key(int(self.np_random.integers(2**31 - 1)))
        self._key, k = jax.random.split(self._key)
        obs, self._state = self.core.reset(self.field, k)
        self._path = [self._position()]
        return self._obs(obs), {"score": 0.0}

    def step(self, action):
        if self._state is None:
            raise RuntimeError("call reset() before step()")
        action = np.atleast_1d(np.asarray(action))
        obs, self._state, reward, terminated, truncated, info = self.core.step(self.field, self._state, action)
        self._path.append(self._position())
        return self._obs(obs), float(reward), bool(terminated), bool(truncated), {"score": float(info["score"])}

    def reference_action(self, name: str = "mpc", **kwargs) -> np.ndarray:
        """What a reference controller ("coast", "greedy" or "mpc") would do now, in this env's action format"""
        cache_key = (name, tuple(sorted(kwargs.items())))
        if cache_key not in self._policies:
            if name == "coast":
                policy = controllers.coast(self.core)
            else:
                policy = getattr(controllers, name)(self.core, self.field, **kwargs)
            self._policies[cache_key] = jax.jit(lambda state, k: policy(None, state, k))
        self._key, k = jax.random.split(self._key)
        action = np.asarray(self._policies[cache_key](self._state, k))
        return action[0] if self.core.config.n_balloons == 1 and self.core.config.action == "discrete" else action

    def _position(self) -> np.ndarray:
        b = self._state.balloons
        return np.stack([np.asarray(b.lat), np.asarray(b.lon), np.asarray(b.altitude_km)])  # (3, N)

    def render(self):
        if self.render_mode != "rgb_array" or self._state is None:
            return None
        from balloongym.render import draw

        return draw(self.core, self.field, self._state, np.stack(self._path))
