import jax
import jax.numpy as jnp
import numpy as np

from balloongym.sim import station_env as env
from balloongym.sim.belief import MeasuredKernel
from balloongym.sim.config import BalloonConfig
from balloongym.sim.dynamics import State, clip, step
from balloongym.sim.wind import WindField

HOURS, ALT = jnp.arange(0, 49.0), jnp.linspace(14.3, 20.2, 18)
LAT, LON = jnp.arange(-12, 13, 2.0), jnp.arange(80, 131, 2.0)
BALLOON = BalloonConfig(step_hours=0.05, substeps=1, climb_cap_km=1.1 * 0.05 / 3)


def fields():
    key = jax.random.PRNGKey(0)
    shape = (len(HOURS), len(ALT), len(LAT), len(LON), 2)
    truth = WindField(HOURS, ALT, LAT, LON, 8.0 * jax.random.normal(key, shape))
    forecast = truth._replace(uv=truth.uv + 3.0)
    return truth, forecast


def config():
    levels = np.asarray(ALT)
    gap = np.subtract.outer(levels, levels)
    kernel = MeasuredKernel(1.5, 1055.0, 715.0, 10.8, jnp.array(levels), jnp.array(np.cos(gap * 1.2) * np.exp(-np.abs(gap) / 3)))
    return env.EnvConfig(BALLOON, kernel, sigma=5.0, n_moves=60)


def start(altitude=17.2):
    return env.reset(config(), State(jnp.array(0.0), jnp.array(100.0), jnp.array(altitude), jnp.array(2.0)), jnp.array(0.3), jnp.array(100.4))


def test_the_observation_has_bles_layout_and_ranges():
    truth, forecast = fields()
    obs = env.observe(config(), forecast, start())
    assert obs.shape == (env.obs_size(len(ALT)),) == (7 + 3 * 35,)
    assert jnp.all(obs[3:] >= 0.0) and jnp.all(obs[3:] <= 1.0) and jnp.all(jnp.abs(obs[1:3]) <= 1.0)


def test_the_column_is_centred_on_the_balloons_level():
    truth, forecast = fields()
    obs = env.observe(config(), forecast, start(altitude=float(ALT[0])))
    slots = obs[7:].reshape(35, 3)
    assert jnp.allclose(slots[:17], jnp.array([0.0, 1.0, 1.0]))  # below the floor: unreachable
    assert not jnp.allclose(slots[17], jnp.array([0.0, 1.0, 1.0]))  # the balloon's own level, in the centre slot


def test_the_reward_is_bles():
    s = start()
    near = env.reward(config(), s._replace(station_lat=jnp.array(0.1), station_lon=jnp.array(100.1)))  # about 16 km
    far = env.reward(config(), s._replace(station_lat=jnp.array(0.0), station_lon=jnp.array(100.0 + 150.0 / 111.32)))
    assert float(near) == 1.0
    assert np.isclose(float(far), 0.4 * 2 ** (-1.0), atol=0.01)  # 150 km out: 100 km past the radius, half of 0.4


def test_with_no_readings_the_belief_is_the_forecast_and_fully_uncertain():
    truth, forecast = fields()
    mean, sd = env._column(config(), forecast, start())
    assert jnp.allclose(sd, 1.0)
    assert jnp.allclose(mean[:, 0], jax.vmap(lambda z: forecast.uv[0, 0, 0, 0, 0] * 0 + env.wind_at(forecast, jnp.array(0.0), jnp.array(100.0), z, jnp.array(2.0))[0])(ALT))


def test_readings_make_the_belief_surer_and_closer_to_the_truth_where_they_were_taken():
    truth, forecast = fields()
    c = config()
    s = start()
    for _ in range(13):
        s, obs, r, done = env.env_step(c, truth, forecast, s, jnp.array(env.STAY))
    mean, sd = env._column(c, forecast, s)
    here = int(jnp.argmin(jnp.abs(ALT - s.balloon.altitude_km)))
    true_here = env.wind_at(truth, s.balloon.lat, s.balloon.lon, ALT[here], s.balloon.hours)
    fc_here = env.wind_at(forecast, s.balloon.lat, s.balloon.lon, ALT[here], s.balloon.hours)
    assert float(sd[here]) < 0.5
    assert float(jnp.linalg.norm(mean[here] - true_here)) < float(jnp.linalg.norm(fc_here - true_here))


def test_the_environment_moves_the_balloon_exactly_as_the_simulator_does():
    truth, forecast = fields()
    c = config()
    actions = jnp.array([env.UP, env.UP, env.STAY, env.DOWN, env.UP, env.STAY] * 5)
    s, b = start(), start().balloon
    for a in actions:
        s, *_ = env.env_step(c, truth, forecast, s, a)
        climb = {env.UP: BALLOON.climb_cap_km, env.STAY: 0.0, env.DOWN: -BALLOON.climb_cap_km}[int(a)]
        b = clip(truth, step(truth, b, jnp.array(climb), BALLOON.step_hours, BALLOON.substeps))
    for x, y in zip(s.balloon, b):
        assert jnp.allclose(x, y)
