"""The four tasks. Each says where the balloons start, what they aim for, the reward, and the score.

    station_keeping  stay within `radius_km` of the launch point. Reward is BLE's: 1 inside, else
                     0.4 * 2^(-(d - radius) / 100 km). Score: fraction of steps some balloon is inside
                     (TWR50 at the default 50 km). Never ends early.
    reach            get within `radius_km` of a goal `distance_km` away. Reward: progress toward the goal per
                     100 km, plus 1 on arrival. Score: fraction of balloons that arrive. Ends when all have.
    max_distance     get as far from the launch point as possible. Reward: progress away per 100 km. Score:
                     mean final distance from launch, km.
    waypoints        visit `n_targets` points in order, each `distance_km / n_targets` beyond the last.
                     Reward: progress toward the next point per 100 km, plus 1 per point reached by any balloon.
                     Score: fraction of points reached. Ends when all are.

Goals are placed along the wind: `direction="level"` picks one of the field's levels at random and puts the goal
downwind of where that level would carry the balloon, so some altitude leads toward it. `"random"` uses any
bearing, which can make a task impossible; reach and waypoints then measure how close a balloon gets.

With several balloons, they launch on a ring `spread_km` round the launch point and share one reward:
station keeping counts the balloon nearest the station, reach and max distance average over balloons,
waypoints follow the balloon nearest the next point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from balloongym.physics import Balloons, destination, distance_km
from balloongym.wind import WindField, wind_at

TASKS = ("station_keeping", "reach", "max_distance", "waypoints")
PROGRESS_KM = 100.0  # progress rewards are in units of this


@dataclass(frozen=True)
class TaskConfig:
    name: str = "station_keeping"
    hours: float = 48.0  # episode length
    radius_km: float = 50.0
    distance_km: float = 0.0  # how far the goal (or the last waypoint) is
    n_targets: int = 1
    direction: str = "level"
    lat_band: tuple[float, float] = (-30.0, 30.0)  # where launches are drawn

    def __post_init__(self):
        if self.name not in TASKS:
            raise ValueError(f"task {self.name!r} is not one of {TASKS}")
        if self.direction not in ("level", "random"):
            raise ValueError("direction is 'level' or 'random'")


DEFAULTS = {
    "station_keeping": dict(hours=48.0, radius_km=50.0),
    "reach": dict(hours=144.0, radius_km=100.0, distance_km=1500.0),
    "max_distance": dict(hours=72.0, radius_km=0.0),
    "waypoints": dict(hours=144.0, radius_km=100.0, distance_km=1800.0, n_targets=3),
}


def task_config(name: str, **overrides) -> TaskConfig:
    """A task's defaults, with any field overridden"""
    if name not in TASKS:
        raise ValueError(f"task {name!r} is not one of {TASKS}")
    return TaskConfig(name=name, **{**DEFAULTS[name], **overrides})


class TaskState(NamedTuple):
    start_lat: jax.Array  # (N,) each balloon's own launch point
    start_lon: jax.Array
    targets_lat: jax.Array  # (K,) the station, the goal, or the waypoints in order
    targets_lon: jax.Array
    visited: jax.Array  # (K,) waypoints reached
    arrived: jax.Array  # (N,) reach: this balloon has arrived
    previous_km: jax.Array  # (N,) the distance each balloon's progress is measured from
    inside_steps: jax.Array  # () station keeping: steps with some balloon inside


def _bearing_of(field: WindField, lat, lon, altitude_km, hours) -> jax.Array:
    u, v = wind_at(field, lat, lon, altitude_km, hours)
    return jnp.arctan2(u, v)


def place(config: TaskConfig, key: jax.Array, field: WindField, lat, lon, hours) -> tuple[jax.Array, jax.Array]:
    """The task's points (K,) for a launch at (lat, lon) at `hours`"""
    if config.name in ("station_keeping", "max_distance"):
        return jnp.atleast_1d(lat), jnp.atleast_1d(lon)
    k_level, k_bearing = jax.random.split(key)
    if config.direction == "level":
        level = field.altitude_km[jax.random.randint(k_level, (), 0, field.altitude_km.shape[0])]
    leg = config.distance_km / config.n_targets
    out_lat, out_lon = [], []
    for _ in range(config.n_targets):
        if config.direction == "level":
            bearing = _bearing_of(field, lat, lon, level, hours)
        else:
            bearing = jax.random.uniform(k_bearing, (), maxval=2 * jnp.pi)
            k_bearing, _ = jax.random.split(k_bearing)
        lat, lon = destination(lat, lon, bearing, leg)
        out_lat.append(lat)
        out_lon.append(lon)
    return jnp.stack(out_lat), jnp.stack(out_lon)


def start(config: TaskConfig, balloons: Balloons, targets_lat, targets_lon) -> TaskState:
    n, k = balloons.lat.shape[0], targets_lat.shape[0]
    state = TaskState(balloons.lat, balloons.lon, targets_lat, targets_lon, jnp.zeros(k, bool), jnp.zeros(n, bool),
                      jnp.zeros(n), jnp.array(0))
    return state._replace(previous_km=_measure(config, state, balloons.lat, balloons.lon))


def aim(config: TaskConfig, state: TaskState) -> tuple[jax.Array, jax.Array]:
    """The point the task steers toward now (scalar lat, lon): the station, the goal, or the next waypoint.
    For max distance it is the launch point, to steer away from."""
    if config.name == "waypoints":
        i = jnp.minimum(jnp.argmin(state.visited), state.targets_lat.shape[0] - 1)
        return state.targets_lat[i], state.targets_lon[i]
    return state.targets_lat[0], state.targets_lon[0]


def _measure(config: TaskConfig, state: TaskState, lat, lon) -> jax.Array:
    """Each balloon's distance (N,) to the point its progress counts against"""
    if config.name == "max_distance":
        return distance_km(lat, lon, state.start_lat, state.start_lon)
    a_lat, a_lon = aim(config, state)
    return distance_km(lat, lon, a_lat, a_lon)


def cost_to_go(config: TaskConfig, state: TaskState, lat, lon) -> jax.Array:
    """A heuristic of how bad a position is, lower is better, for the reference controllers (km)"""
    d = _measure(config, state, lat, lon)
    if config.name == "max_distance":
        return -d
    if config.name == "station_keeping":
        return jnp.maximum(d - config.radius_km, 0.0)
    return d


def ble_reward(d_km: jax.Array, radius_km: float) -> jax.Array:
    return jnp.where(d_km <= radius_km, 1.0, 0.4 * 2.0 ** (-(d_km - radius_km) / 100.0))


def reward(config: TaskConfig, state: TaskState, balloons: Balloons) -> tuple[jax.Array, TaskState, jax.Array]:
    """(reward, new task state, done) after a step that put the balloons at `balloons`"""
    d = _measure(config, state, balloons.lat, balloons.lon)
    if config.name == "station_keeping":
        nearest = jnp.min(d)
        inside = nearest <= config.radius_km
        return ble_reward(nearest, config.radius_km), state._replace(inside_steps=state.inside_steps + inside), jnp.array(False)
    if config.name == "max_distance":
        r = jnp.mean(d - state.previous_km) / PROGRESS_KM
        return r, state._replace(previous_km=d), jnp.array(False)
    if config.name == "reach":
        progress = jnp.where(state.arrived, 0.0, state.previous_km - d) / PROGRESS_KM
        now = state.arrived | (d <= config.radius_km)
        r = jnp.mean(progress + (now & ~state.arrived))
        return r, state._replace(arrived=now, previous_km=d), jnp.all(now)
    # waypoints: the balloon nearest the next point carries the team
    best = jnp.min(d)
    hit = best <= config.radius_km
    i = jnp.argmin(state.visited)
    visited = state.visited.at[i].set(state.visited[i] | hit)
    r = (jnp.min(state.previous_km) - best) / PROGRESS_KM + hit
    new = state._replace(visited=visited)
    new = new._replace(previous_km=_measure(config, new, balloons.lat, balloons.lon))  # toward the next point
    return r, new, jnp.all(visited)


def score(config: TaskConfig, state: TaskState, balloons: Balloons, steps: jax.Array) -> jax.Array:
    """The task's score for a finished episode (see the module docstring)"""
    if config.name == "station_keeping":
        return state.inside_steps / steps
    if config.name == "reach":
        return jnp.mean(state.arrived.astype(jnp.float32))
    if config.name == "max_distance":
        return jnp.mean(distance_km(balloons.lat, balloons.lon, state.start_lat, state.start_lon))
    return jnp.mean(state.visited.astype(jnp.float32))


SCORE_NAMES = {
    "station_keeping": "time within radius (fraction of steps)",
    "reach": "balloons that arrive (fraction)",
    "max_distance": "final distance from launch (km)",
    "waypoints": "waypoints reached (fraction)",
}
