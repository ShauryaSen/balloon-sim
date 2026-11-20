import jax
import jax.numpy as jnp
import numpy as np

from balloongym.sim.belief import GPBelief, Readings, error_draws, forecast
from balloongym.sim.config import BeliefConfig
from balloongym.sim.dynamics import State, distance_km
from balloongym.sim.wind import WindField, wind_at

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


def test_unread_belief_gives_its_mean_or_draws_around_it():
    mean = grid(jnp.full((4, 4, 5, 5, 2), 7.0))
    belief, nothing = GPBelief(CONFIG, mean), Readings.empty(3)
    alone = belief.winds(jax.random.key(0), 0, nothing)
    assert alone.uv.shape == (1, 4, 4, 5, 5, 2) and jnp.allclose(alone.uv, 7.0, atol=1e-4)
    drawn = belief.winds(jax.random.key(0), 6, nothing)
    assert drawn.uv.shape == (6, 4, 4, 5, 5, 2)
    assert not jnp.allclose(drawn.uv[0], drawn.uv[1])  # different draws differ
    assert jnp.allclose(drawn.hours, mean.hours)  # all on the mean's grid
    assert abs(float(jnp.std(drawn.uv - 7.0)) - CONFIG.sigma) < 0.3  # still the prior's spread


def test_a_reading_pins_the_mean_and_every_draw_there_and_fades_with_distance():
    """At a read grid point the belief is the reading, exactly; far away it is still the forecast"""
    mean = grid(jnp.full((4, 4, 5, 5, 2), 7.0))
    here = State(LAT[2], LON[2], ALTITUDE[1], HOURS[1])  # a grid point, so the field there is one value
    readings = Readings.empty(5).add(here, jnp.array([12.0, -3.0]))
    belief = GPBelief(CONFIG, mean)
    posterior_mean = belief.winds(jax.random.key(0), 0, readings).uv[0]
    draws = belief.winds(jax.random.key(1), 4, readings).uv
    at = (1, 1, 2, 2)  # (t, alt, lat, lon) of the reading
    assert jnp.allclose(posterior_mean[at], jnp.array([12.0, -3.0]), atol=0.05)
    assert jnp.allclose(draws[(slice(None), *at)], jnp.array([12.0, -3.0]), atol=0.05)
    far = (3, 3, 0, 0)  # 48 h later, 5 km higher, 1,100 km away
    assert jnp.allclose(posterior_mean[far], 7.0, atol=0.3)
    # the pull is in the kernel's direction: a point 5 degrees north, same time and altitude, moves part of the way
    near = (1, 1, 3, 2)
    assert 7.5 < float(posterior_mean[near][0]) < 12.0


def _kernel_beliefs():
    from balloongym.sim.belief import KernelBelief, MeasuredKernel
    from balloongym.sim.config import BeliefConfig
    hours, alt = jnp.arange(0, 31.0), jnp.linspace(14.3, 20.2, 18)
    lat, lon = jnp.arange(-12, 13, 2.0), jnp.arange(80, 131, 2.0)
    mean = WindField(hours, alt, lat, lon, jnp.zeros((31, 18, 13, 26, 2)))
    levels = np.linspace(14.3, 20.2, 18)
    measured = MeasuredKernel(1.5, 1055.0, 715.0, 10.8, jnp.array(levels),
                              jnp.array(np.cos(np.subtract.outer(levels, levels) * 1.2) * np.exp(-np.abs(np.subtract.outer(levels, levels)) / 3)))
    return mean, {kind: KernelBelief(mean, kind, 5.0, BeliefConfig(), measured) for kind in ("assumed", "measured")}


def _with_readings(n):
    from balloongym.sim.dynamics import State
    r = Readings.empty(40)
    for i in range(n):
        r = r.add(State(jnp.array(0.0), jnp.array(100.0 + 0.1 * i), jnp.array(17.2), jnp.array(10.0 + 0.05 * i)), jnp.array([3.0, 0.0]))
    return r


def test_a_kernel_belief_with_no_readings_is_the_forecast():
    mean, beliefs = _kernel_beliefs()
    for b in beliefs.values():
        assert jnp.allclose(b.winds(jax.random.PRNGKey(0), 0, _with_readings(0)).uv[0], mean.uv)


def test_a_kernel_belief_passes_near_its_readings_and_only_corrects_the_window():
    mean, beliefs = _kernel_beliefs()
    for kind, b in beliefs.items():
        uv = b.winds(jax.random.PRNGKey(0), 0, _with_readings(13)).uv[0]
        at = wind_at(mean._replace(uv=uv), jnp.array(0.0), jnp.array(100.6), jnp.array(17.2), jnp.array(10.6))
        assert 2.5 < float(at[0]) < 3.0, kind  # pulled to the 3 m/s readings, short of it by the reading noise
        assert jnp.allclose(uv[:10], 0.0) and jnp.allclose(uv[26:], 0.0), kind  # outside 10 h to 24 h, untouched


def test_the_measured_kernel_can_correct_the_other_way_at_another_altitude():
    mean, beliefs = _kernel_beliefs()
    uv = beliefs["measured"].winds(jax.random.PRNGKey(0), 0, _with_readings(13)).uv[0]
    far = wind_at(mean._replace(uv=uv), jnp.array(0.0), jnp.array(100.6), jnp.array(19.9), jnp.array(10.6))
    assert float(far[0]) < 0.0  # the test matrix turns negative about 2.6 km away, as the measured one does
