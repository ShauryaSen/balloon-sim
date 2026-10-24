"""Where an episode starts and where its goal is"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from balloon_sim.config import TaskConfig
from balloon_sim.dynamics import EARTH_RADIUS_M, State
from balloon_sim.wind import WindField, is_global, wind_at


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

    # random mode: draw many tries at once, keep the first whose goal also lands inside the allowed area.
    # Latitude is limited to the band and to the record's own latitudes less the margin; longitude is limited
    # to the record's less the margin, or not at all when the record goes round the globe
    m = config.margin_deg
    lat_low, lat_high = max(float(truth.lat[0]) + m, config.lat_band[0]), min(float(truth.lat[-1]) - m, config.lat_band[1])
    lon_low, lon_high = (0.0, 360.0) if is_global(truth.lon) else (float(truth.lon[0]) + m, float(truth.lon[-1]) - m)
    k_lat, k_lon, k_bearing = jax.random.split(key, 3)
    tries = 1000
    lat = jax.random.uniform(k_lat, (tries,), minval=lat_low, maxval=lat_high)
    lon = jax.random.uniform(k_lon, (tries,), minval=lon_low, maxval=lon_high)
    if config.direction == "random":
        bearing = jax.random.uniform(k_bearing, (tries,), maxval=2 * jnp.pi)
    else:
        # the flow at the start, averaged over the levels: the direction the balloon goes if it does nothing
        at_levels = jax.vmap(lambda la, lo: jax.vmap(lambda al: wind_at(truth, la, lo, al, jnp.array(config.start_hours)))(truth.altitude_km))
        u, v = jnp.moveaxis(jnp.mean(at_levels(lat, lon), axis=1), -1, 0)  # (tries,) each
        downwind = jnp.arctan2(u, v)  # 0 is north, pi/2 east
        if config.direction == "downwind":
            bearing = downwind  # the wind carries the balloon there: no steering needed, and none measured
        elif config.direction == "upwind":
            bearing = downwind + jnp.pi  # only reachable on a level that opposes the mean flow
        else:  # crosswind, a quarter turn off the flow, the side drawn per try so neither one is systematic
            side = jnp.where(jax.random.bernoulli(k_bearing, shape=(tries,)), 1.0, -1.0)
            bearing = downwind + side * jnp.pi / 2
    goal_lat, goal_lon = _destination(lat, lon, bearing, config.distance_km)
    goal_lon = goal_lon % 360.0
    inside = (goal_lat >= lat_low) & (goal_lat <= lat_high) & (goal_lon >= lon_low) & (goal_lon <= lon_high)
    if not bool(inside.any()):
        raise ValueError(f"no goal {config.distance_km} km away fits inside the allowed area")
    i = jnp.argmax(inside)  # the first try that fits
    start = State(lat[i], lon[i], middle_km, jnp.array(config.start_hours))
    return Task(start, goal_lat[i], goal_lon[i])
