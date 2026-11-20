"""The forecast: what a controller is told the wind is. The balloon always flies in the truth.

Three kinds, picked by `ForecastConfig.kind`:

    "perfect"      the forecast is the truth
    "gp"           the truth minus a smooth random error, drawn once per episode from a Gaussian process
                   (Matern 5/2 in space and altitude, Matern 3/2 in time) by random Fourier features.
                   Defaults are the error of a day-old forecast fitted on ERA5: 544 km, 1.3 km, 9.1 h.
    "persistence"  the truth `lag_hours` earlier at the same place: a real forecast error with real structure,
                   and the one an operator gets for free. Needs `lag_hours` of record before the episode.

`sigma_ms` sets the GP error's size (m/s per component). A persistence forecast's size is whatever the weather
gives; `sigma_ms` is not used for it.

The GP error is read at a point directly, so no forecast grid is ever built: a forecast costs nothing to hold
per episode, which is what lets thousands of episodes run at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from balloongym.wind import WindField, wind_at

KINDS = ("perfect", "gp", "persistence")
EARTH_RADIUS_KM = 6371.0


@dataclass(frozen=True)
class ForecastConfig:
    kind: str = "gp"
    sigma_ms: float = 3.2
    length_km: float = 544.0
    height_km: float = 1.3
    time_hours: float = 9.1
    n_features: int = 128
    lag_hours: float = 24.0

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"forecast kind {self.kind!r} is not one of {KINDS}")


class ErrorDraw(NamedTuple):
    """One episode's GP error: random Fourier features and their weights"""

    omega_place: jax.Array  # (D, 4)
    omega_time: jax.Array  # (D,)
    phase: jax.Array  # (D,)
    weights: jax.Array  # (2, D)


def _student_t(key: jax.Array, shape: tuple[int, ...], dof: float) -> jax.Array:
    """Spectral draws for a Matern kernel of smoothness dof / 2: a normal over a shared chi-square"""
    k_normal, k_chi = jax.random.split(key)
    chi = jax.random.chisquare(k_chi, dof, shape[:1] + (1,) * (len(shape) - 1))
    return jax.random.normal(k_normal, shape) * jnp.sqrt(dof / chi)


def draw_error(config: ForecastConfig, key: jax.Array) -> ErrorDraw:
    k1, k2, k3, k4 = jax.random.split(key, 4)
    d = config.n_features
    return ErrorDraw(
        omega_place=_student_t(k1, (d, 4), 5.0),
        omega_time=_student_t(k2, (d,), 3.0),
        phase=jax.random.uniform(k3, (d,), maxval=2 * jnp.pi),
        weights=jax.random.normal(k4, (2, d)),
    )


def _error_at(config: ForecastConfig, draw: ErrorDraw, lat, lon, altitude_km, hours) -> jax.Array:
    phi, lam = jnp.radians(lat), jnp.radians(lon)
    xyz = EARTH_RADIUS_KM * jnp.stack([jnp.cos(phi) * jnp.cos(lam), jnp.cos(phi) * jnp.sin(lam), jnp.sin(phi)])
    point = jnp.concatenate([xyz / config.length_km, jnp.atleast_1d(altitude_km / config.height_km)])
    angle = draw.omega_place @ point + draw.omega_time * hours / config.time_hours + draw.phase
    return config.sigma_ms * jnp.sqrt(2.0 / config.n_features) * (draw.weights @ jnp.cos(angle))


def forecast_at(field: WindField, config: ForecastConfig, draw: ErrorDraw, lat, lon, altitude_km, hours) -> jax.Array:
    """(u, v) the forecast gives at one point"""
    if config.kind == "persistence":
        return wind_at(field, lat, lon, altitude_km, hours - config.lag_hours)
    truth = wind_at(field, lat, lon, altitude_km, hours)
    if config.kind == "perfect":
        return truth
    return truth - _error_at(config, draw, lat, lon, altitude_km, hours)


def earliest_start_hours(config: ForecastConfig) -> float:
    """How far into a record an episode must start for its forecast to exist"""
    return config.lag_hours if config.kind == "persistence" else 0.0
