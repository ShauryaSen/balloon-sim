"""The environment as pure JAX functions, so episodes can be vmapped, jitted and scanned.

    env = BalloonEnv(EnvConfig(task=task_config("reach")))
    field = synthetic_wind()
    obs, state = env.reset(field, key)
    obs, state, reward, terminated, truncated, info = env.step(field, state, action)

The wind field is an argument, not part of the environment, so one compiled environment runs on any field of
the same shape and a large record is never baked into the compiled program.

Observation, one row per balloon (shape (N, obs_size)), everything scaled to about [-1, 1]:

    0      altitude, 0 at the bottom of the band, 1 at the top
    1      fraction of the episode left
    2, 3   east and north offset to the task's point, per 1,000 km (away from the launch for max distance)
    4      distance to that point, squashed: d / (d + 250 km)
    5      task progress: inside the radius (station keeping), arrived (reach), fraction of points reached
           (waypoints), 0 for max distance
    6, 7   the wind the balloon measures where it is (the truth), per 30 m/s
    then   the forecast's (u, v) at the balloon's position at every level of the field, per 30 m/s
    last 2 (only with more than one balloon) mean offset to the other balloons, per 1,000 km

A real balloon measures the wind only where it is and gets a forecast for the rest, which is what the
observation gives.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from balloongym import tasks
from balloongym.forecast import (
    ErrorDraw,
    ForecastConfig,
    draw_error,
    earliest_start_hours,
    forecast_at,
)
from balloongym.physics import (
    BalloonConfig,
    Balloons,
    destination,
    distance_km,
    offset_km,
)
from balloongym.physics import step as physics_step
from balloongym.wind import WindField, wind_at

DOWN, STAY, UP = 0, 1, 2
ACTIONS = ("discrete", "continuous")


@dataclass(frozen=True)
class EnvConfig:
    task: tasks.TaskConfig = dataclass_field(default_factory=lambda: tasks.task_config("station_keeping"))
    balloon: BalloonConfig = dataclass_field(default_factory=BalloonConfig)
    forecast: ForecastConfig = dataclass_field(default_factory=ForecastConfig)
    n_balloons: int = 1
    spread_km: float = 150.0  # fleet launch ring
    action: str = "discrete"

    def __post_init__(self):
        if self.action not in ACTIONS:
            raise ValueError(f"action {self.action!r} is not one of {ACTIONS}")
        if self.n_balloons < 1:
            raise ValueError("n_balloons must be at least 1")

    @property
    def n_steps(self) -> int:
        return int(round(self.task.hours * 60.0 / self.balloon.dt_minutes))


class EnvState(NamedTuple):
    balloons: Balloons
    task: tasks.TaskState
    error: ErrorDraw
    t: jax.Array  # steps taken
    done: jax.Array  # the task has ended (terminated) or time ran out (truncated)


def obs_size(config: EnvConfig, n_levels: int) -> int:
    return 8 + 2 * n_levels + (2 if config.n_balloons > 1 else 0)


@dataclass(frozen=True)
class BalloonEnv:
    config: EnvConfig = dataclass_field(default_factory=EnvConfig)

    # --- reset ---------------------------------------------------------------------------------------------------

    @functools.partial(jax.jit, static_argnums=0)
    def reset(self, field: WindField, key: jax.Array) -> tuple[jax.Array, EnvState]:
        c = self.config
        k_lat, k_lon, k_t, k_alt, k_task, k_err = jax.random.split(key, 6)
        margin = 2.0
        lat_low = jnp.maximum(c.task.lat_band[0], field.lat[0] + margin)
        lat_high = jnp.minimum(c.task.lat_band[1], field.lat[-1] - margin)
        lat = jax.random.uniform(k_lat, (), minval=lat_low, maxval=lat_high)
        lon_margin = jnp.where(field.is_global, 0.0, margin)
        lon = jax.random.uniform(k_lon, (), minval=field.lon[0] + lon_margin, maxval=field.lon[-1] - lon_margin) % 360.0
        first = field.hours[0] + earliest_start_hours(c.forecast)
        last = field.hours[-1] - c.task.hours
        hours = jax.random.uniform(k_t, (), minval=first, maxval=jnp.maximum(last, first))
        altitude = jax.random.uniform(k_alt, (), minval=field.altitude_km[0], maxval=field.altitude_km[-1])

        n = c.n_balloons
        if n == 1:
            lats, lons = lat[None], lon[None]
        else:
            bearing = 2.0 * jnp.pi * jnp.arange(n - 1) / (n - 1)
            ring_lat, ring_lon = destination(lat, lon, bearing, c.spread_km)
            lats, lons = jnp.concatenate([lat[None], ring_lat]), jnp.concatenate([lon[None], ring_lon])
        balloons = Balloons(lats, lons, jnp.full(n, altitude), jnp.full(n, hours))

        targets = tasks.place(c.task, k_task, field, lat, lon, hours)
        state = EnvState(balloons, tasks.start(c.task, balloons, *targets), draw_error(c.forecast, k_err),
                         jnp.array(0), jnp.array(False))
        return self.observe(field, state), state

    # --- step ----------------------------------------------------------------------------------------------------

    def command(self, action: jax.Array) -> jax.Array:
        """An action (N,) as a climb command in [-1, 1] per balloon"""
        if self.config.action == "discrete":
            return jnp.asarray(action).astype(jnp.float32) - 1.0
        return jnp.clip(jnp.asarray(action, jnp.float32), -1.0, 1.0)

    @functools.partial(jax.jit, static_argnums=0)
    def step(self, field: WindField, state: EnvState, action: jax.Array):
        """(obs, state, reward, terminated, truncated, info). After the episode ends, steps change nothing."""
        c = self.config
        command = jnp.broadcast_to(self.command(action), (c.n_balloons,))
        moved = physics_step(field, c.balloon, state.balloons, command)
        r, task_state, finished = tasks.reward(c.task, state.task, moved)
        t = state.t + 1
        terminated = finished
        truncated = (t >= c.n_steps) & ~finished
        new = EnvState(moved, task_state, state.error, t, terminated | truncated)
        new = jax.tree.map(lambda a, b: jnp.where(state.done, a, b), state, new)  # frozen once done
        r = jnp.where(state.done, 0.0, r)
        info = {"score": tasks.score(c.task, new.task, new.balloons, jnp.maximum(new.t, 1))}
        return self.observe(field, new), new, r, terminated & ~state.done, truncated & ~state.done, info

    # --- observation ---------------------------------------------------------------------------------------------

    def forecast_column(self, field: WindField, state: EnvState) -> jax.Array:
        """The forecast (u, v) at every level at each balloon. Shape (N, levels, 2)"""
        b = state.balloons
        f = lambda lat, lon, alt, h: forecast_at(field, self.config.forecast, state.error, lat, lon, alt, h)
        per_level = jax.vmap(f, in_axes=(None, None, 0, None))
        return jax.vmap(lambda la, lo, h: per_level(la, lo, field.altitude_km, h))(b.lat, b.lon, b.hours)

    def observe(self, field: WindField, state: EnvState) -> jax.Array:
        c = self.config
        b, ts = state.balloons, state.task
        n = c.n_balloons
        low, high = field.altitude_km[0], field.altitude_km[-1]
        altitude = (b.altitude_km - low) / (high - low)
        left = jnp.full(n, 1.0 - state.t / c.n_steps)

        if c.task.name == "max_distance":
            vector = offset_km(ts.start_lat, ts.start_lon, b.lat, b.lon)
            progress = jnp.zeros(n)
        else:
            a_lat, a_lon = tasks.aim(c.task, ts)
            vector = offset_km(b.lat, b.lon, a_lat, a_lon)
            d = distance_km(b.lat, b.lon, a_lat, a_lon)
            if c.task.name == "station_keeping":
                progress = (d <= c.task.radius_km).astype(jnp.float32)
            elif c.task.name == "reach":
                progress = ts.arrived.astype(jnp.float32)
            else:
                progress = jnp.full(n, jnp.mean(ts.visited.astype(jnp.float32)))
        length = jnp.linalg.norm(vector, axis=-1)
        measured = jax.vmap(lambda *p: wind_at(field, *p))(b.lat, b.lon, b.altitude_km, b.hours)
        column = self.forecast_column(field, state).reshape(n, -1)

        parts = [altitude[:, None], left[:, None], vector / 1000.0, (length / (length + 250.0))[:, None],
                 progress[:, None], measured / 30.0, column / 30.0]
        if n > 1:
            pairwise = jax.vmap(lambda la, lo: offset_km(la, lo, b.lat, b.lon))(b.lat, b.lon)  # (N, N, 2)
            parts.append(jnp.sum(pairwise, axis=1) / (n - 1) / 1000.0)
        return jnp.concatenate(parts, axis=1).astype(jnp.float32)


# --- whole episodes ---------------------------------------------------------------------------------------------

Policy = Callable[[jax.Array, EnvState, jax.Array], jax.Array]
"""(obs, state, key) -> action. Learned policies should read only obs; the reference controllers also read the
state, to plan on the forecast."""


class Episode(NamedTuple):
    score: jax.Array  # the task's score
    episode_return: jax.Array
    lat: jax.Array  # (T + 1, N) the path, for plotting
    lon: jax.Array
    altitude_km: jax.Array


def rollout(env: BalloonEnv, field: WindField, key: jax.Array, policy: Policy) -> Episode:
    """One whole episode under `policy`. Jit it, or vmap it over keys for many at once."""
    k_reset, k_act = jax.random.split(key)
    obs, state = env.reset(field, k_reset)

    def one(carry, k):
        obs, state = carry
        obs, new, r, _, _, info = env.step(field, state, policy(obs, state, k))
        return (obs, new), (r, new.balloons, info["score"])

    (_, final), (rewards, path, scores) = jax.lax.scan(one, (obs, state), jax.random.split(k_act, env.config.n_steps))
    start = state.balloons
    stack = lambda first, rest: jnp.concatenate([first[None], rest])
    return Episode(scores[-1], jnp.sum(rewards), stack(start.lat, path.lat), stack(start.lon, path.lon),
                   stack(start.altitude_km, path.altitude_km))
