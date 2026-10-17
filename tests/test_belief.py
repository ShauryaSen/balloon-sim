import jax
import jax.numpy as jnp
import numpy as np

from balloon_sim.belief import FixedBelief, error_draws, forecast
from balloon_sim.config import BeliefConfig
from balloon_sim.dynamics import distance_km
from balloon_sim.wind import WindField

CONFIG = BeliefConfig(sigma=2.8)
HOURS = jnp.array([0.0, 12.0, 24.0, 48.0])
ALTITUDE = jnp.array([15.0, 16.5, 18.0, 20.0])
LAT = jnp.array([20.0, 25.0, 30.0, 35.0, 40.0])
LON = jnp.array([150.0, 155.0, 160.0, 165.0, 170.0])


def grid(uv=None) -> WindField:
    shape = (len(HOURS), len(ALTITUDE), len(LAT), len(LON), 2)
    return WindField(HOURS, ALTITUDE, LAT, LON, jnp.zeros(shape) if uv is None else uv)


def matern52(r):
    return (1 + np.sqrt(5) * r + 5 * r**2 / 3) * np.exp(-np.sqrt(5) * r)


def matern32(r):
    return (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)


def many_draws(n_feature_sets: int = 3000) -> jax.Array:
    """One draw from each of many independent feature sets, so the average is over the features too.
    Shape (n_feature_sets, 4, 4, 5, 5, 2)"""
    keys = jax.random.split(jax.random.key(0), n_feature_sets)
    return jax.vmap(lambda k: error_draws(CONFIG, k, grid(), 1)[0])(keys)


def correlation(draws, a, b):
    """Correlation between two grid points (t, alt, lat, lon) over every draw and both components"""
    x, y = draws[(slice(None), *a)].ravel(), draws[(slice(None), *b)].ravel()
    return float(jnp.mean(x * y) / jnp.sqrt(jnp.mean(x**2) * jnp.mean(y**2)))


def test_draws_have_the_kernel_variance_and_correlations():
    draws = many_draws()
    assert abs(float(jnp.std(draws)) - CONFIG.sigma) < 0.05 * CONFIG.sigma
    # 24 h apart, same place
    assert abs(correlation(draws, (0, 1, 2, 2), (2, 1, 2, 2)) - matern32(24.0 / CONFIG.hours)) < 0.04
    # 5 degrees of latitude apart, same altitude and time (the kernel uses straight-line distance through the Earth,
    # which over 556 km is within 0.1% of the distance along the surface)
    r = float(distance_km(25.0, 160.0, 30.0, 160.0)) / CONFIG.horizontal_km
    assert abs(correlation(draws, (1, 1, 1, 2), (1, 1, 2, 2)) - matern52(r)) < 0.04
    # 3 km apart in altitude, same place and time
    assert abs(correlation(draws, (1, 0, 2, 2), (1, 2, 2, 2)) - matern52(3.0 / CONFIG.altitude_km)) < 0.04


def test_forecast_is_the_truth_less_an_error_of_size_sigma():
    truth = grid(jax.random.normal(jax.random.key(1), (4, 4, 5, 5, 2)) * 10.0)
    assert jnp.allclose(forecast(BeliefConfig(sigma=0.0), jax.random.key(2), truth).uv, truth.uv)
    keys = jax.random.split(jax.random.key(3), 500)
    errors = jax.vmap(lambda k: truth.uv - forecast(CONFIG, k, truth).uv)(keys)
    assert abs(float(jnp.std(errors)) - CONFIG.sigma) < 0.1 * CONFIG.sigma


def test_fixed_belief_gives_its_mean_or_draws_around_it():
    mean = grid(jnp.full((4, 4, 5, 5, 2), 7.0))
    belief = FixedBelief(CONFIG, mean)
    alone = belief.winds(jax.random.key(0), 0)
    assert alone.uv.shape == (1, 4, 4, 5, 5, 2) and jnp.allclose(alone.uv, 7.0)
    drawn = belief.winds(jax.random.key(0), 6)
    assert drawn.uv.shape == (6, 4, 4, 5, 5, 2)
    assert not jnp.allclose(drawn.uv[0], drawn.uv[1])  # different draws differ
    assert jnp.allclose(drawn.hours, mean.hours)  # all on the mean's grid
