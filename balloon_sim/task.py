"""Where an episode starts and where its goal is"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from balloon_sim.config import TaskConfig
from balloon_sim.dynamics import EARTH_RADIUS_M, State
from balloon_sim.wind import WindField


class Task(NamedTuple):
    start: State
    goal_lat: jax.Array
    goal_lon: jax.Array


def _destination(lat: jax.Array, lon: jax.Array, bearing: jax.Array, distance_km: float) -> tuple[jax.Array, jax.Array]:
    """The point distance_km away from (lat, lon) along the great circle heading bearing (radians, 0 = north)"""
    phi, lam = jnp.radians(lat), jnp.radians(lon)
    delta = distance_km * 1000.0 / EARTH_RADIUS_M  # the distance as an angle at the Earth's centre
    phi2 = jnp.arcsin(jnp.sin(phi) * jnp.cos(delta) + jnp.cos(phi) * jnp.sin(delta) * jnp.cos(bearing))
    lam2 = lam + jnp.arctan2(jnp.sin(bearing) * jnp.sin(delta) * jnp.cos(phi), jnp.cos(delta) - jnp.sin(phi) * jnp.sin(phi2))
    return jnp.degrees(phi2), jnp.degrees(lam2)


def draw_task(config: TaskConfig, key: jax.Array, truth: WindField) -> Task:
    """The start (at the middle altitude, at start_hours) and the goal, by the config's mode"""
    middle_km = (truth.altitude_km[0] + truth.altitude_km[-1]) / 2
    if config.mode == "fixed":
        if None in (config.start_lat, config.start_lon, config.goal_lat, config.goal_lon):
            raise ValueError("fixed mode needs start_lat, start_lon, goal_lat and goal_lon")
        start = State(jnp.array(config.start_lat), jnp.array(config.start_lon), middle_km, jnp.array(config.start_hours))
        return Task(start, jnp.array(config.goal_lat), jnp.array(config.goal_lon))

    # random mode: draw many tries at once, keep the first whose goal also lands inside the box
    m = config.margin_deg
    k_lat, k_lon, k_bearing = jax.random.split(key, 3)
    tries = 1000
    lat = jax.random.uniform(k_lat, (tries,), minval=truth.lat[0] + m, maxval=truth.lat[-1] - m)
    lon = jax.random.uniform(k_lon, (tries,), minval=truth.lon[0] + m, maxval=truth.lon[-1] - m)
    goal_lat, goal_lon = _destination(lat, lon, jax.random.uniform(k_bearing, (tries,), maxval=2 * jnp.pi), config.distance_km)
    inside = (goal_lat >= truth.lat[0] + m) & (goal_lat <= truth.lat[-1] - m) & (goal_lon >= truth.lon[0] + m) & (goal_lon <= truth.lon[-1] - m)
    if not bool(inside.any()):
        raise ValueError(f"no goal {config.distance_km} km away fits inside the box with a {m} degree margin")
    i = jnp.argmax(inside)  # the first try that fits
    start = State(lat[i], lon[i], middle_km, jnp.array(config.start_hours))
    return Task(start, goal_lat[i], goal_lon[i])
