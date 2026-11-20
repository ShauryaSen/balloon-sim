"""Station keeping as an RL environment, BLE's way (decision 57), in JAX so thousands of balloons step at once

One balloon holds a station for 960 steps of 3 min. Each step it picks up, stay or down; it moves in the true
wind; it reads the true wind where it is. The observation is BLE's Perciatelli features (`env/features.py`),
cut to what our simulator has:

    0       altitude, rescaled to [0, 1] over the band (BLE: pressure over its range)
    1, 2    sin and cos of the heading to the station, from north toward east
    3       distance to the station, squashed: d / (d + 250 km)
    4-6     the last command, one-hot: up, stay, down
    then, for 2L - 1 slots centred on the balloon's level (L levels; slots past the band are (0, 1, 1), BLE's
    "certain, wrong way, full speed"), three numbers per level of the belief's wind at the balloon:
            its uncertainty, sd / sigma in [0, 1]
            the angle between it and the direction to the station, / pi
            its speed, squashed: s / (s + 30 m/s)

Dropped, because our balloon has no power model: battery, solar angle and cycle, navigation paused, excess
energy, power to use, pressure ratio.

The belief is the forecast corrected by the readings with the measured K (balloon_sim.belief): the posterior
mean and standard deviation at every level of the balloon's column. It conditions on the latest `memory`
readings of one taken every `thin` steps (48 of one every 18 min, the last 14.4 h): older ones are past the
measured 10.8 h time scale, and a fixed small set keeps each step's solve cheap (assumed adequate).

Reward (BLE's `perciatelli_reward_function`, without its power penalty): 1 inside 50 km, else
0.4 * 2^(-(d - 50) / 100) with d in km.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from balloongym.sim.belief import MeasuredKernel, _level_weights, _matern32, _measured_sideways
from balloongym.sim.config import BalloonConfig
from balloongym.sim.dynamics import EARTH_RADIUS_M, State, clip, distance_km, step
from balloongym.sim.wind import WindField, wind_at

UP, STAY, DOWN = 0, 1, 2
N_AMBIENT = 7


class EnvConfig(NamedTuple):
    balloon: BalloonConfig
    kernel: MeasuredKernel
    sigma: float  # the forecast error's size, m/s per component
    radius_km: float = 50.0
    n_moves: int = 960
    thin: int = 6
    memory: int = 48
    reading_noise: float = 0.5


class EnvState(NamedTuple):
    balloon: State
    station_lat: jax.Array
    station_lon: jax.Array
    move: jax.Array
    last: jax.Array  # last command
    lat: jax.Array  # (memory,) the reading buffer, oldest overwritten first
    lon: jax.Array
    altitude_km: jax.Array
    hours: jax.Array
    residual: jax.Array  # (memory, 2) reading minus forecast
    count: jax.Array  # readings taken so far


def obs_size(n_levels: int) -> int:
    return N_AMBIENT + 3 * (2 * n_levels - 1)


def reset(config: EnvConfig, start: State, station_lat: jax.Array, station_lon: jax.Array) -> EnvState:
    z = jnp.zeros(config.memory)
    return EnvState(start, station_lat, station_lon, jnp.array(0), jnp.array(STAY), z, z, z, z, jnp.zeros((config.memory, 2)), jnp.array(0))


def _column(config: EnvConfig, forecast: WindField, s: EnvState) -> tuple[jax.Array, jax.Array]:
    """The belief's mean (L, 2) and sd / sigma (L,) at every level at the balloon"""
    levels = forecast.altitude_km
    n = levels.shape[0]
    b = s.balloon
    prior = jax.vmap(lambda z: wind_at(forecast, b.lat, b.lon, z, b.hours))(levels)  # (L, 2)
    k = config.kernel
    real = jnp.arange(config.memory) < jnp.minimum(s.count, config.memory)
    wr = _level_weights(k.levels_km, s.altitude_km)  # (M, levels)
    side = _measured_sideways(k, s.lat[:, None], s.lon[:, None], s.lat[None], s.lon[None])
    time = _matern32(jnp.abs(s.hours[:, None] - s.hours[None]) / k.hours)
    K = config.sigma**2 * side * (wr @ k.level_matrix @ wr.T) * time
    K = jnp.where(real[:, None] & real[None], K, 0.0) + jnp.diag(jnp.where(real, config.reading_noise**2, 1.0))
    wq = _level_weights(k.levels_km, levels)  # (L, levels)
    kq = (config.sigma**2 * (wq @ k.level_matrix @ wr.T)
          * _measured_sideways(k, b.lat, b.lon, s.lat, s.lon)[None]
          * _matern32(jnp.abs(b.hours - s.hours) / k.hours)[None])  # (L, M)
    kq = jnp.where(real[None], kq, 0.0)
    factor = jnp.linalg.cholesky(K)
    alpha = jax.scipy.linalg.cho_solve((factor, True), jnp.where(real[:, None], s.residual, 0.0))  # (M, 2)
    mean = prior + kq @ alpha
    v = jax.scipy.linalg.solve_triangular(factor, kq.T, lower=True)  # (M, L)
    variance = jnp.clip(config.sigma**2 - jnp.sum(v * v, axis=0), 0.0, None)
    return mean, jnp.sqrt(variance) / config.sigma


def observe(config: EnvConfig, forecast: WindField, s: EnvState) -> jax.Array:
    levels = forecast.altitude_km
    n = levels.shape[0]
    b = s.balloon
    radius_km = EARTH_RADIUS_M / 1000.0
    east = radius_km * jnp.radians((s.station_lon - b.lon + 180.0) % 360.0 - 180.0) * jnp.cos(jnp.radians(b.lat))
    north = radius_km * jnp.radians(s.station_lat - b.lat)
    d = distance_km(b.lat, b.lon, s.station_lat, s.station_lon)
    heading = jnp.arctan2(east, north)
    ambient = jnp.array([(b.altitude_km - levels[0]) / (levels[-1] - levels[0]), jnp.sin(heading), jnp.cos(heading), d / (d + 250.0),
                         s.last == UP, s.last == STAY, s.last == DOWN], dtype=jnp.float32)

    mean, sd = _column(config, forecast, s)
    to_station = jnp.stack([east, north]) / (d + 1e-6)
    speed = jnp.linalg.norm(mean, axis=1)
    cos_error = jnp.clip(mean @ to_station / (speed + 1e-6), -1.0, 1.0)
    angle = jnp.where(d < 1e-3, 0.0, jnp.arccos(cos_error)) / jnp.pi
    per_level = jnp.stack([sd, angle, speed / (speed + 30.0)], axis=1)  # (L, 3), bottom to top

    here = jnp.argmin(jnp.abs(levels - b.altitude_km))
    padded = jnp.concatenate([jnp.tile(jnp.array([0.0, 1.0, 1.0]), (n - 1, 1)), per_level, jnp.tile(jnp.array([0.0, 1.0, 1.0]), (n - 1, 1))])
    window = jax.lax.dynamic_slice(padded, (here, 0), (2 * n - 1, 3))  # the balloon's level at the centre slot
    return jnp.concatenate([ambient, window.reshape(-1)]).astype(jnp.float32)


def reward(config: EnvConfig, s: EnvState) -> jax.Array:
    d = distance_km(s.balloon.lat, s.balloon.lon, s.station_lat, s.station_lon)
    return jnp.where(d <= config.radius_km, 1.0, 0.4 * jnp.exp2(-(d - config.radius_km) / 100.0))


def env_step(config: EnvConfig, truth: WindField, forecast: WindField, s: EnvState, action: jax.Array) -> tuple[EnvState, jax.Array, jax.Array, jax.Array]:
    """(next state, its observation, the reward for arriving there, whether the episode has ended)"""
    cap = config.balloon.climb_cap_km
    climb = jnp.where(action == UP, cap, jnp.where(action == DOWN, -cap, 0.0))
    b = clip(truth, step(truth, s.balloon, climb, config.balloon.step_hours, config.balloon.substeps))
    reading = wind_at(truth, b.lat, b.lon, b.altitude_km, b.hours) - wind_at(forecast, b.lat, b.lon, b.altitude_km, b.hours)
    take = (s.move % config.thin) == 0
    i = jnp.where(take, s.count % config.memory, 0)
    put = lambda buffer, value: jnp.where(take, buffer.at[i].set(value), buffer)
    s = EnvState(b, s.station_lat, s.station_lon, s.move + 1, action, put(s.lat, b.lat), put(s.lon, b.lon), put(s.altitude_km, b.altitude_km),
                 put(s.hours, b.hours), jnp.where(take, s.residual.at[i].set(reading), s.residual), s.count + take)
    return s, observe(config, forecast, s), reward(config, s), s.move >= config.n_moves
