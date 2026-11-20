"""The environment's contract: physics, forecasts, tasks, the JAX core and the Gymnasium wrapper"""

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import balloongym
from balloongym import controllers, tasks
from balloongym.benchmark import normalized_score, reference_scores, summary
from balloongym.env import BalloonEnv, EnvConfig, rollout
from balloongym.forecast import ForecastConfig, draw_error, forecast_at
from balloongym.physics import BalloonConfig, Balloons, distance_km, step
from balloongym.wind import WindField, synthetic_wind, wind_at


@pytest.fixture(scope="module")
def field() -> WindField:
    return synthetic_wind(seed=1, hours=120.0, levels=5, resolution_deg=5.0)


def uniform(u: float, v: float) -> WindField:
    """The same wind everywhere"""
    hours, alt = jnp.array([0.0, 500.0]), jnp.array([15.0, 20.0])
    lat, lon = jnp.arange(-60.0, 61.0, 10.0), jnp.arange(0.0, 361.0, 10.0)
    uv = jnp.zeros((2, 2, lat.size, lon.size, 2)).at[..., 0].set(u).at[..., 1].set(v)
    return WindField(hours, alt, lat, lon, uv)


# --- wind and physics ---------------------------------------------------------------------------------------------

def test_wind_reads_grid_values_exactly(field):
    t, a, y, x = 3, 2, 4, 7
    got = wind_at(field, field.lat[y], field.lon[x], field.altitude_km[a], field.hours[t])
    assert np.allclose(got, field.uv[t, a, y, x], atol=1e-4)


def test_balloon_drifts_with_the_wind():
    """10 m/s north for 30 min is 18 km"""
    b = Balloons(jnp.array([0.0]), jnp.array([100.0]), jnp.array([17.0]), jnp.array([0.0]))
    moved = step(uniform(0.0, 10.0), BalloonConfig(), b, jnp.array([0.0]))
    assert np.isclose(float(distance_km(b.lat, b.lon, moved.lat, moved.lon)[0]), 18.0, rtol=1e-3)
    assert np.isclose(float(moved.hours[0]), 0.5)


def test_climb_is_limited_and_held_in_band():
    b = Balloons(jnp.array([0.0]), jnp.array([100.0]), jnp.array([17.0]), jnp.array([0.0]))
    config = BalloonConfig()
    up = step(uniform(0.0, 0.0), config, b, jnp.array([1.0]))
    assert np.isclose(float(up.altitude_km[0]) - 17.0, config.climb_km_per_hour * 0.5, rtol=1e-4)
    top = b._replace(altitude_km=jnp.array([19.99]))
    assert float(step(uniform(0.0, 0.0), config, top, jnp.array([1.0])).altitude_km[0]) == pytest.approx(20.0)


def test_global_field_wraps_longitude():
    b = Balloons(jnp.array([0.0]), jnp.array([359.9]), jnp.array([17.0]), jnp.array([0.0]))
    moved = step(uniform(20.0, 0.0), BalloonConfig(), b, jnp.array([0.0]))
    assert 0.0 <= float(moved.lon[0]) < 1.0


# --- forecasts ----------------------------------------------------------------------------------------------------

def test_perfect_forecast_is_the_truth(field):
    config = ForecastConfig(kind="perfect")
    draw = draw_error(config, jax.random.key(0))
    assert np.allclose(forecast_at(field, config, draw, 5.0, 40.0, 17.0, 30.0), wind_at(field, 5.0, 40.0, 17.0, 30.0))


def test_gp_error_has_the_asked_size(field):
    config = ForecastConfig(kind="gp", sigma_ms=3.0)
    lat = jnp.linspace(-40, 40, 40)
    errors = []
    for i in range(40):
        draw = draw_error(config, jax.random.key(i))
        f = jax.vmap(lambda la: forecast_at(field, config, draw, la, 100.0, 17.0, 30.0) - wind_at(field, la, 100.0, 17.0, 30.0))
        errors.append(f(lat))
    assert np.std(np.concatenate(errors)) == pytest.approx(3.0, rel=0.15)


def test_persistence_is_the_truth_earlier(field):
    config = ForecastConfig(kind="persistence", lag_hours=24.0)
    draw = draw_error(config, jax.random.key(0))
    assert np.allclose(forecast_at(field, config, draw, 5.0, 40.0, 17.0, 60.0), wind_at(field, 5.0, 40.0, 17.0, 36.0))


# --- the JAX core -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("task", tasks.TASKS)
@pytest.mark.parametrize("n_balloons", [1, 3])
def test_episode_runs_and_scores_in_range(field, task, n_balloons):
    env = BalloonEnv(EnvConfig(task=tasks.task_config(task, hours=24.0), n_balloons=n_balloons))
    obs, state = env.reset(field, jax.random.key(0))
    assert obs.shape == (n_balloons, balloongym.env.obs_size(env.config, field.altitude_km.size))
    assert np.all(np.isfinite(obs))
    ep = jax.jit(lambda k: rollout(env, field, k, controllers.coast(env)))(jax.random.key(1))
    assert np.isfinite(float(ep.episode_return))
    if task != "max_distance":
        assert 0.0 <= float(ep.score) <= 1.0
    assert ep.lat.shape == (env.config.n_steps + 1, n_balloons)


def test_reset_is_deterministic_and_launches_differ(field):
    env = BalloonEnv(EnvConfig(task=tasks.task_config("reach")))
    a, _ = env.reset(field, jax.random.key(3))
    b, _ = env.reset(field, jax.random.key(3))
    c, _ = env.reset(field, jax.random.key(4))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_reach_goal_is_placed_at_the_distance(field):
    env = BalloonEnv(EnvConfig(task=tasks.task_config("reach", distance_km=1200.0)))
    _, state = env.reset(field, jax.random.key(0))
    d = distance_km(state.balloons.lat[0], state.balloons.lon[0], state.task.targets_lat[0], state.task.targets_lon[0])
    assert float(d) == pytest.approx(1200.0, rel=1e-3)


def test_reach_ends_on_arrival_and_steps_after_change_nothing():
    """Wind blowing straight at a goal 50 km away: arrival in the first step, then the episode is frozen"""
    wind = uniform(0.0, 20.0)
    env = BalloonEnv(EnvConfig(task=tasks.task_config("reach", distance_km=60.0, radius_km=30.0, direction="random")))
    _, state = env.reset(wind, jax.random.key(0))
    # put the goal 60 km due north
    goal = balloongym.physics.destination(state.balloons.lat[0], state.balloons.lon[0], 0.0, 60.0)
    state = state._replace(task=tasks.start(env.config.task, state.balloons, jnp.array([goal[0]]), jnp.array([goal[1]])))
    _, state, r, terminated, truncated, info = env.step(wind, state, jnp.array([1]))
    assert bool(terminated) and not bool(truncated) and float(info["score"]) == 1.0
    assert float(r) > 1.0  # progress plus the arrival bonus
    _, after, r2, terminated2, _, _ = env.step(wind, state, jnp.array([1]))
    assert float(r2) == 0.0 and not bool(terminated2)
    assert np.array_equal(after.balloons.lat, state.balloons.lat)


def test_station_reward_is_bles():
    assert float(tasks.ble_reward(jnp.array(10.0), 50.0)) == 1.0
    assert float(tasks.ble_reward(jnp.array(150.0), 50.0)) == pytest.approx(0.2)


def test_episodes_vmap_over_keys(field):
    env = BalloonEnv(EnvConfig(task=tasks.task_config("max_distance", hours=12.0)))
    keys = jax.random.split(jax.random.key(0), 8)
    eps = jax.jit(jax.vmap(lambda k: rollout(env, field, k, controllers.coast(env))))(keys)
    assert eps.score.shape == (8,) and len(set(np.round(np.asarray(eps.score), 3))) > 1


def test_planning_beats_coasting_on_reach(field):
    """With a perfect forecast, MPC should get closer than drifting at the launch altitude"""
    env = BalloonEnv(EnvConfig(task=tasks.task_config("reach", hours=72.0, distance_km=800.0),
                               forecast=ForecastConfig(kind="perfect")))
    ref = reference_scores(env, field, n=16, horizon=12, samples=32)
    assert np.mean(ref["mpc"]) > np.mean(ref["coast"])
    assert normalized_score(ref["mpc"], ref) == pytest.approx(1.0)


def test_summary_brackets_the_mean():
    s = summary(np.arange(20.0))
    assert s["low"] < s["mean"] < s["high"] and s["iqm"] == pytest.approx(9.5)


# --- Gymnasium ----------------------------------------------------------------------------------------------------

SMALL = dict(wind_seed=1)


@pytest.mark.parametrize("env_id", ["BalloonGym/StationKeeping-v0", "BalloonGym/Reach-v0",
                                    "BalloonGym/MaxDistance-v0", "BalloonGym/Waypoints-v0",
                                    "BalloonGym/ReachFleet-v0"])
def test_gymnasium_env_checker(field, env_id):
    env = gym.make(env_id, wind=field, task_kwargs={"hours": 12.0}).unwrapped
    check_env(env, skip_render_check=True)


def test_gymnasium_episode_and_render(field):
    env = gym.make("BalloonGym/Reach-v0", wind=field, task_kwargs={"hours": 6.0}, render_mode="rgb_array")
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    done, steps = False, 0
    while not done:
        obs, reward, terminated, truncated, info = env.step(env.unwrapped.reference_action("greedy"))
        done, steps = terminated or truncated, steps + 1
    assert steps <= 12 and 0.0 <= info["score"] <= 1.0
    image = env.render()
    assert image.ndim == 3 and image.shape[2] == 3


def test_continuous_actions(field):
    env = gym.make("BalloonGym/StationKeeping-v0", wind=field, action="continuous", task_kwargs={"hours": 6.0})
    env.reset(seed=0)
    env.step(np.array([0.5], np.float32))
    assert isinstance(env.action_space, gym.spaces.Box)
