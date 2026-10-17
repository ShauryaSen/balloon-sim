"""The belief over the wind: interface, the fixed GP prior, and drawing winds by random Fourier features

One kernel serves both the forecast's error and the belief, so the belief is calibrated by construction:
    k(x, x') = sigma^2 * Matern52(place and altitude) * Matern32(time)
Place is the point on the globe in 3D, in km, so distances never break at the date line or the poles.

Random Fourier features (Rahimi and Recht): a random field with that kernel is approximately
    f(x) = sigma * sqrt(2 / D) * sum over j of  w_j cos(omega_j . x + b_j)
with D features, w_j ~ N(0, 1), b_j ~ Uniform(0, 2 pi), and each frequency omega_j drawn from the kernel's spectrum.
A Matern-nu kernel's spectrum is a Student-t with 2 nu degrees of freedom, over inputs divided by their lengthscales.
A product of kernels over separate inputs draws each part of omega from its own factor.
"""

from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp

from balloon_sim.config import BeliefConfig
from balloon_sim.dynamics import EARTH_RADIUS_M
from balloon_sim.wind import WindField


class Belief(Protocol):
    def winds(self, key: jax.Array, n: int) -> WindField:
        """n winds drawn from the belief, as a batch (uv gains a leading axis of n); n = 0 gives its mean alone"""
        ...


def _student_t(key: jax.Array, n: int, dim: int, dof: float) -> jax.Array:
    """n draws of a dim-dimensional Student-t: a normal divided by the square root of a shared chi-square"""
    k_normal, k_chi = jax.random.split(key)
    return jax.random.normal(k_normal, (n, dim)) * jnp.sqrt(dof / jax.random.chisquare(k_chi, dof, (n, 1)))


def _places(config: BeliefConfig, grid: WindField) -> jax.Array:
    """Every (altitude, lat, lon) of the grid as a kernel input: 3D position and altitude, each over its lengthscale.
    Shape (n_alt * n_lat * n_lon, 4), in the order of the grid"""
    alt, lat, lon = jnp.meshgrid(grid.altitude_km, jnp.radians(grid.lat), jnp.radians(grid.lon), indexing="ij")
    radius_km = EARTH_RADIUS_M / 1000.0
    x, y, z = radius_km * jnp.cos(lat) * jnp.cos(lon), radius_km * jnp.cos(lat) * jnp.sin(lon), radius_km * jnp.sin(lat)
    inputs = jnp.stack([x / config.horizontal_km, y / config.horizontal_km, z / config.horizontal_km, alt / config.altitude_km], -1)
    return inputs.reshape(-1, 4)


def error_draws(config: BeliefConfig, key: jax.Array, grid: WindField, n: int) -> jax.Array:
    """n random fields with the kernel, zero on average, on the grid. Shape (n, n_t, n_alt, n_lat, n_lon, 2).
    The n draws share one set of features and differ in their weights w"""
    k_place, k_time, k_phase, k_weights = jax.random.split(key, 4)
    D = config.n_features
    omega_place = _student_t(k_place, D, 4, dof=5.0)  # (D, 4), Matern 5/2
    omega_time = _student_t(k_time, D, 1, dof=3.0)[:, 0]  # (D,), Matern 3/2
    phase = jax.random.uniform(k_phase, (D,), maxval=2 * jnp.pi)

    # cos(a + b) = cos a cos b - sin a sin b, so places and times are worked out separately, then combined
    a = _places(config, grid) @ omega_place.T  # (places, D)
    b = (grid.hours / config.hours)[:, None] * omega_time[None, :] + phase  # (times, D)
    w = jax.random.normal(k_weights, (n, 2, D))  # one set per draw and per wind component
    scale = config.sigma * jnp.sqrt(2.0 / D)
    fields = scale * (
        jnp.einsum("ncd,td,pd->nctp", w, jnp.cos(b), jnp.cos(a)) - jnp.einsum("ncd,td,pd->nctp", w, jnp.sin(b), jnp.sin(a))
    )  # (n, 2, times, places)
    n_t, n_alt, n_lat, n_lon = grid.uv.shape[:4]
    return jnp.moveaxis(fields.reshape(n, 2, n_t, n_alt, n_lat, n_lon), 1, -1)


def forecast(config: BeliefConfig, key: jax.Array, truth: WindField) -> WindField:
    """F = W - e: the truth with one error field taken away. The truth itself never changes with sigma"""
    return truth._replace(uv=truth.uv - error_draws(config, key, truth, 1)[0])


@dataclass(frozen=True)
class FixedBelief:
    """The GP prior: the forecast as its mean and the shared kernel around it. Readings never change it"""

    config: BeliefConfig
    mean: WindField  # the forecast

    def winds(self, key: jax.Array, n: int) -> WindField:
        if n == 0:
            return self.mean._replace(uv=self.mean.uv[None])
        return self.mean._replace(uv=self.mean.uv[None] + error_draws(self.config, key, self.mean, n))
