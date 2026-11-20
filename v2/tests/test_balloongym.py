"""balloongym's own layer: the environments are the simulator's, step for step, and the Gymnasium wrappers obey
the API. The simulator's own tests are in tests/sim."""

import dataclasses

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import balloongym  # noqa: F401
from balloongym import benchmark
from balloongym import wind as winds
from balloongym.env import FleetEnv, StationEnv, missions_for
from balloongym.sim import fleet, rl, station_env
from balloongym.sim.cells import keys
from balloongym.sim.config import RLConfig
from balloongym.sim.qrdqn import QRConfig
from balloongym.tasks import make_task
from balloongym.train import train_ppo, train_qrdqn


@pytest.fixture(scope="module")
def wind():
    return winds.synthetic(seed=1, resolution_deg=5.0, levels=6)


SHORT = dict(n_moves=12)


@pytest.mark.parametrize("name,n", [("reach", 1), ("max_distance", 1), ("waypoints", 1), ("reach", 3), ("waypoints", 2)])
def test_fleet_env_is_the_simulators_episode(wind, name, n):
    """Coast through the env: the return is minus fleet.fly's cost of the same launch, per 1,000 km"""
    task = make_task(name, n_balloons=n, distance_km=1000.0, **SHORT)
    env = FleetEnv(task)
    k_task, k_noise, k_fly = keys(3)
    mission, _ = fleet.draw_mission(task.task, k_task, wind.truth)
    obs, s = env.reset_to(wind.truth, mission, rl.noise_from_key(task.belief, k_noise))
    total = 0.0
    for _ in range(task.n_moves):
        obs, s, r, term, trunc, info = env.step(wind.truth, s, jnp.zeros(n))
        total += float(r)
    flown = fleet.fly(task.balloon, fleet.coast(mission), wind.truth, mission, task.objective, task.n_moves, k_fly, task.radius_km)
    assert total == pytest.approx(-float(flown.cost_km) / 1000.0, rel=1e-4, abs=1e-3)
    assert float(info["cost_km"]) == pytest.approx(float(flown.cost_km), rel=1e-4, abs=1.0)


def test_fleet_observation_is_rl_observe(wind):
    """The first observation is what rl.episode's policy sees at move 0"""
    task = make_task("reach", distance_km=1000.0, **SHORT)
    env = FleetEnv(task)
    k_task, k_noise, _ = keys(5)
    mission, _ = fleet.draw_mission(task.task, k_task, wind.truth)
    noise = rl.noise_from_key(task.belief, k_noise)
    obs, _ = env.reset_to(wind.truth, mission, noise)
    params = rl.init_params(jax.random.key(0), env.obs_size(wind.truth.altitude_km.size), RLConfig())
    steps = rl.episode(env.spec, wind.truth, params, mission, noise, jnp.asarray(task.belief.sigma), jax.random.key(0), 1000.0)
    assert np.allclose(obs, steps.obs[0], atol=1e-5)


def test_readings_correct_the_belief(wind):
    """With learns the belief moves toward the truth near the reading: at the levels beside the balloon (a reading says
    little about levels several kernel lengths, 1.3 km each, away)"""
    task = make_task("reach", distance_km=1000.0, **SHORT)
    gaps = {True: [], False: []}
    for seed in range(8):  # one reading is weak evidence on any one launch; on average it helps
        k_task, k_noise, _ = keys(seed)
        mission, _ = fleet.draw_mission(task.task, k_task, wind.truth)
        noise = rl.noise_from_key(task.belief, k_noise)
        for learns in (True, False):
            obs, s = FleetEnv(task, learns=learns).reset_to(wind.truth, mission, noise)
            levels = wind.truth.altitude_km.size
            believed = obs[0, 11:].reshape(levels, 2) * 20.0  # rl.observe scales the belief by 1/20
            b = s.balloons
            true = jax.vmap(lambda al: balloongym.wind.wind_at(wind.truth, b.lat[0], b.lon[0], al, b.hours[0]))(wind.truth.altitude_km)
            near = jnp.argsort(jnp.abs(wind.truth.altitude_km - b.altitude_km[0]))[:2]  # the levels beside the balloon
            gaps[learns].append(float(jnp.mean(jnp.abs(believed - true)[near])))
    assert np.mean(gaps[True]) < 0.8 * np.mean(gaps[False])


def test_station_env_is_station_env(wind):
    task = make_task("station_keeping")
    env = StationEnv.build(task, wind)
    m = missions_for(task, wind, [0])
    obs, s = env.reset(wind, m, jax.random.key(0))
    start = jax.tree.map(lambda a: a[0], m.start)
    raw = station_env.reset(env.config, start, m.station_lat[0], m.station_lon[0])
    assert np.allclose(obs, station_env.observe(env.config, wind.earlier, raw))
    for action in (station_env.UP, station_env.STAY, station_env.DOWN):
        obs, s, r, _, _, info = env.step(wind, s, jnp.array(action))
        raw, raw_obs, raw_r, _ = station_env.env_step(env.config, wind.truth, wind.earlier, raw, jnp.array(action))
        assert np.allclose(obs, raw_obs) and float(r) == pytest.approx(float(raw_r))


@pytest.mark.parametrize("env_id", ["BalloonGym/StationKeeping-v0", "BalloonGym/Reach-v0", "BalloonGym/MaxDistance-v0",
                                    "BalloonGym/Waypoints-v0", "BalloonGym/WaypointsFleet-v0"])
def test_gymnasium_checker(wind, env_id):
    env = gym.make(env_id, wind=wind, launches=4).unwrapped
    check_env(env, skip_render_check=True)


def test_gymnasium_episode_render_and_discrete(wind):
    env = gym.make("BalloonGym/Reach-v0", wind=wind, launches=4, action="discrete", render_mode="rgb_array",
                   task_kwargs={"distance_km": 1000.0, "n_moves": 8})
    obs, _ = env.reset(seed=0)
    done, n = False, 0
    while not done:
        obs, r, term, trunc, info = env.step(1)  # stay
        done, n = term or trunc, n + 1
    assert n <= 8 and "cost_km" in info
    assert env.render().shape[2] == 3


def test_benchmark_runs_and_mpc_beats_coast_on_max_distance(wind):
    task = make_task("max_distance", n_moves=10)
    task = dataclasses.replace(task, planner=dataclasses.replace(task.planner, horizon=5, candidates=16, iterations=10))
    coast = benchmark.evaluate(task, wind, "coast", seeds=range(3))
    mpc = benchmark.evaluate(task, wind, "mpc_truth", seeds=range(3))
    assert np.mean(mpc["value"]) > np.mean(coast["value"])
    assert benchmark.normalized(mpc, coast, mpc) == pytest.approx(1.0)


def test_ppo_trains_and_is_scored(wind):
    task = make_task("reach", distance_km=1000.0, n_moves=6)
    config = dataclasses.replace(RLConfig(), iterations=2, envs=4, train_seeds=4)
    params, spec, curve = train_ppo(task, wind, config, log=lambda *_: None)
    out = benchmark.evaluate(task, wind, "ppo", seeds=range(2), params=params, policy_spec=spec)
    assert len(curve) >= 1 and np.all(np.isfinite(out["cost_km"]))


def test_qrdqn_trains_and_is_scored(wind):
    task = make_task("station_keeping", n_moves=40)
    c = QRConfig()._replace(layers=2, hidden=16, envs=4, warmup=8, replay=1000, batch=8)
    params, curve = train_qrdqn(task, wind, c, steps=20, chunk=10, eval_every=20, train_missions=4, log=lambda *_: None)
    out = benchmark.evaluate(task, wind, "qrdqn", seeds=range(2), params=params, qr_config=c)
    assert "validation_twr50" in curve[-1] and np.all((out["score"] >= 0) & (out["score"] <= 1))


def test_summary():
    s = benchmark.summary(np.arange(20.0))
    assert s["low"] < s["mean"] < s["high"] and s["iqm"] == pytest.approx(9.5)
