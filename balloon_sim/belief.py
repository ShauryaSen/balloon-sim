"""The belief over the wind: the GP prior around the forecast, readings that condition it, and drawing winds

One kernel serves both the forecast's error and the belief, so the belief is calibrated by construction:
    k(x, x') = sigma^2 * Matern52(place and altitude) * Matern32(time)
Place is the point on the globe in 3D, in km, so distances never break at the date line or the poles.

Random Fourier features (Rahimi and Recht): a random field with that kernel is approximately
    f(x) = sigma * sqrt(2 / D) * sum over j of  w_j cos(omega_j . x + b_j)
with D features, w_j ~ N(0, 1), b_j ~ Uniform(0, 2 pi), and each frequency omega_j drawn from the kernel's spectrum.
A Matern-nu kernel's spectrum is a Student-t with 2 nu degrees of freedom, over inputs divided by their lengthscales.
A product of kernels over separate inputs draws each part of omega from its own factor.

Readings condition the belief pathwise (Wilson et al. 2020): with readings y at points X and prior mean F,
    mean(x)  = F(x) + k(x, X) K^-1 (y - F(X))
    draw(x)  = F(x) + f(x) + k(x, X) K^-1 (y - F(X) - f(X))        K = k(X, X) + jitter
where f is a prior draw. Each draw is pulled through the readings exactly, and the mean is the usual GP posterior.
"""

from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp

from balloon_sim.config import BeliefConfig
from balloon_sim.dynamics import EARTH_RADIUS_M, State
from balloon_sim.wind import WindField, wind_at


class Readings(NamedTuple):
    """Up to `capacity` point readings of the wind, in fixed-size arrays so they can ride through a scan"""

    lat: jax.Array  # (capacity,)
    lon: jax.Array
    altitude_km: jax.Array
    hours: jax.Array
    uv: jax.Array  # (capacity, 2)
    count: jax.Array  # how many of the rows are real

    @staticmethod
    def empty(capacity: int) -> "Readings":
        zeros = jnp.zeros(capacity)
        return Readings(zeros, zeros, zeros, zeros, jnp.zeros((capacity, 2)), jnp.array(0))

    def add(self, state: State, uv: jax.Array) -> "Readings":
        """The readings with one more: the wind uv read at state"""
        i = self.count
        return Readings(
            self.lat.at[i].set(state.lat),
            self.lon.at[i].set(state.lon),
            self.altitude_km.at[i].set(state.altitude_km),
            self.hours.at[i].set(state.hours),
            self.uv.at[i].set(uv),
            i + 1,
        )


def read(truth: WindField, state: State) -> jax.Array:
    """What a balloon carrying one anemometer reads: the true wind at its own point, exactly"""
    return wind_at(truth, state.lat, state.lon, state.altitude_km, state.hours)


class Belief(Protocol):
    def winds(self, key: jax.Array, n: int, readings: Readings) -> WindField:
        """n winds drawn from the belief given the readings, as a batch (uv gains a leading axis of n);
        n = 0 gives the belief's mean alone"""
        ...


# --- the kernel's inputs and features -------------------------------------------------------------------------


def _inputs(config: BeliefConfig, lat: jax.Array, lon: jax.Array, altitude_km: jax.Array) -> jax.Array:
    """Points as kernel inputs: 3D position and altitude, each over its lengthscale. Shape (..., 4)"""
    phi, lam = jnp.radians(lat), jnp.radians(lon)
    radius_km = EARTH_RADIUS_M / 1000.0
    x, y, z = radius_km * jnp.cos(phi) * jnp.cos(lam), radius_km * jnp.cos(phi) * jnp.sin(lam), radius_km * jnp.sin(phi)
    return jnp.stack([x / config.horizontal_km, y / config.horizontal_km, z / config.horizontal_km, altitude_km / config.altitude_km], -1)


def _grid_inputs(config: BeliefConfig, grid: WindField) -> jax.Array:
    """Every (altitude, lat, lon) of the grid as a kernel input, in the grid's order. Shape (n_alt * n_lat * n_lon, 4)"""
    alt, lat, lon = jnp.meshgrid(grid.altitude_km, grid.lat, grid.lon, indexing="ij")
    return _inputs(config, lat, lon, alt).reshape(-1, 4)


def _student_t(key: jax.Array, n: int, dim: int, dof: float) -> jax.Array:
    """n draws of a dim-dimensional Student-t: a normal divided by the square root of a shared chi-square"""
    k_normal, k_chi = jax.random.split(key)
    return jax.random.normal(k_normal, (n, dim)) * jnp.sqrt(dof / jax.random.chisquare(k_chi, dof, (n, 1)))


class Features(NamedTuple):
    """One set of random Fourier features: frequencies over place and time, and phases. Shape (D, ...)"""

    omega_place: jax.Array  # (D, 4), Matern 5/2
    omega_time: jax.Array  # (D,), Matern 3/2
    phase: jax.Array  # (D,)

    @staticmethod
    def draw(config: BeliefConfig, key: jax.Array) -> "Features":
        k_place, k_time, k_phase = jax.random.split(key, 3)
        D = config.n_features
        return Features(_student_t(k_place, D, 4, dof=5.0), _student_t(k_time, D, 1, dof=3.0)[:, 0], jax.random.uniform(k_phase, (D,), maxval=2 * jnp.pi))


def _prior_on_grid(config: BeliefConfig, features: Features, w: jax.Array, grid: WindField) -> jax.Array:
    """n prior draws f on the grid, from weights w of shape (n, 2, D). Shape (n, n_t, n_alt, n_lat, n_lon, 2).
    cos(a + b) = cos a cos b - sin a sin b, so places and times are worked out separately, then combined"""
    a = _grid_inputs(config, grid) @ features.omega_place.T  # (places, D)
    b = (grid.hours / config.hours)[:, None] * features.omega_time[None, :] + features.phase  # (times, D)
    scale = config.sigma * jnp.sqrt(2.0 / config.n_features)
    fields = scale * (jnp.einsum("ncd,td,pd->nctp", w, jnp.cos(b), jnp.cos(a)) - jnp.einsum("ncd,td,pd->nctp", w, jnp.sin(b), jnp.sin(a)))
    n_t, n_alt, n_lat, n_lon = grid.uv.shape[:4]
    return jnp.moveaxis(fields.reshape(w.shape[0], 2, n_t, n_alt, n_lat, n_lon), 1, -1)


def _prior_at(config: BeliefConfig, features: Features, w: jax.Array, readings: Readings) -> jax.Array:
    """The same prior draws at the reading points. Shape (n, capacity, 2)"""
    inputs = _inputs(config, readings.lat, readings.lon, readings.altitude_km)  # (capacity, 4)
    angle = inputs @ features.omega_place.T + (readings.hours / config.hours)[:, None] * features.omega_time[None, :] + features.phase
    scale = config.sigma * jnp.sqrt(2.0 / config.n_features)
    return scale * jnp.einsum("ncd,rd->nrc", w, jnp.cos(angle))


def error_draws(config: BeliefConfig, key: jax.Array, grid: WindField, n: int) -> jax.Array:
    """n random fields with the kernel, zero on average, on the grid. Shape (n, n_t, n_alt, n_lat, n_lon, 2)"""
    k_features, k_weights = jax.random.split(key)
    w = jax.random.normal(k_weights, (n, 2, config.n_features))
    return _prior_on_grid(config, Features.draw(config, k_features), w, grid)


def forecast(config: BeliefConfig, key: jax.Array, truth: WindField) -> WindField:
    """F = W - e: the truth with one error field taken away. The truth itself never changes with sigma"""
    return truth._replace(uv=truth.uv - error_draws(config, key, truth, 1)[0])


# --- the kernel itself, for conditioning -----------------------------------------------------------------------


def _matern52(r: jax.Array) -> jax.Array:
    return (1 + jnp.sqrt(5.0) * r + 5 * r**2 / 3) * jnp.exp(-jnp.sqrt(5.0) * r)


def _matern32(r: jax.Array) -> jax.Array:
    return (1 + jnp.sqrt(3.0) * r) * jnp.exp(-jnp.sqrt(3.0) * r)


def _kernel_grid_to_readings(config: BeliefConfig, grid: WindField, readings: Readings) -> tuple[jax.Array, jax.Array]:
    """k between the grid and the readings, split by the product: a place part (places, R) and a time part (times, R)"""
    place = jnp.sqrt(jnp.sum((_grid_inputs(config, grid)[:, None, :] - _inputs(config, readings.lat, readings.lon, readings.altitude_km)[None]) ** 2, -1) + 1e-12)
    time = jnp.abs(grid.hours[:, None] - readings.hours[None, :]) / config.hours
    return config.sigma**2 * _matern52(place), _matern32(time)


def _kernel_readings(config: BeliefConfig, readings: Readings) -> jax.Array:
    """K = k(X, X) + jitter, with the unused rows made harmless (identity rows, zero elsewhere). Shape (R, R)"""
    inputs = _inputs(config, readings.lat, readings.lon, readings.altitude_km)
    place = jnp.sqrt(jnp.sum((inputs[:, None, :] - inputs[None]) ** 2, -1) + 1e-12)
    time = jnp.abs(readings.hours[:, None] - readings.hours[None, :]) / config.hours
    k = config.sigma**2 * _matern52(place) * _matern32(time)
    real = jnp.arange(readings.lat.shape[0]) < readings.count
    both = real[:, None] & real[None, :]
    return jnp.where(both, k, 0.0) + jnp.diag(jnp.where(real, config.reading_noise**2, 1.0))


def _conditioned(config: BeliefConfig, fields: jax.Array, at_readings: jax.Array, grid: WindField, readings: Readings) -> jax.Array:
    """fields (n, grid, 2) corrected so each passes through the readings: + k(grid, X) K^-1 (y - field(X))"""
    real = (jnp.arange(readings.lat.shape[0]) < readings.count)[None, :, None]
    residual = jnp.where(real, readings.uv[None] - at_readings, 0.0)  # (n, R, 2)
    factor = jnp.linalg.cholesky(_kernel_readings(config, readings))
    alpha = jax.scipy.linalg.cho_solve((factor, True), jnp.moveaxis(residual, 0, 1).reshape(readings.lat.shape[0], -1))  # (R, n*2)
    alpha = jnp.moveaxis(alpha.reshape(readings.lat.shape[0], -1, 2), 1, 0)  # (n, R, 2)
    k_place, k_time = _kernel_grid_to_readings(config, grid, readings)
    correction = jnp.einsum("pr,tr,nrc->ntpc", k_place, k_time, alpha)  # (n, times, places, 2)
    return fields + correction.reshape(fields.shape)


def _mean_at(mean: WindField, readings: Readings) -> jax.Array:
    """The forecast read at the reading points. Shape (capacity, 2)"""
    return jax.vmap(lambda la, lo, al, h: wind_at(mean, la, lo, al, h))(readings.lat, readings.lon, readings.altitude_km, readings.hours)


class Truth(NamedTuple):
    """The true wind as a belief: what hindsight plans on. Draws and readings mean nothing to it"""

    field: WindField

    def winds(self, key: jax.Array, n: int, readings: Readings) -> WindField:
        return self.field._replace(uv=self.field.uv[None])


class GPBelief(NamedTuple):
    """The GP: the forecast as its mean, the shared kernel around it, conditioned on whatever readings it is given"""

    config: BeliefConfig
    mean: WindField  # the forecast

    def winds(self, key: jax.Array, n: int, readings: Readings) -> WindField:
        grid = self.mean
        if n == 0:
            fields, at_readings = grid.uv[None], _mean_at(grid, readings)[None]
        else:
            k_features, k_weights = jax.random.split(key)
            features = Features.draw(self.config, k_features)
            w = jax.random.normal(k_weights, (n, 2, self.config.n_features))
            fields = grid.uv[None] + _prior_on_grid(self.config, features, w, grid)
            at_readings = _mean_at(grid, readings)[None] + _prior_at(self.config, features, w, readings)
        return grid._replace(uv=_conditioned(self.config, fields, at_readings, grid, readings))
