"""The learned controller (decision 50): the parts that must agree exactly with the rest of the sweep

The forecast it reads must be the forecast every other controller of the seed plans on, and the reward it is
trained on must add up to the cost it is scored by. If either drifted, the policy would be learning a
different problem from the one its row reports.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from balloon_sim import fleet, rl
from balloon_sim.belief import GPBelief, Readings, forecast
from balloon_sim.config import BalloonConfig, BeliefConfig, RLConfig
from balloon_sim.wind import wind_at
from tests.test_fleet import a_record, a_task

BELIEF = BeliefConfig(sigma=2.0, n_features=32)


def test_the_point_forecast_is_the_grid_forecast():
    truth, key = a_record(), jax.random.key(7)
    grid = forecast(BELIEF, key, truth)
    noise = rl.noise_from_key(BELIEF, key)
    rng = np.random.default_rng(0)
    for _ in range(20):
        lat, lon = rng.uniform(15, 55), rng.uniform(125, 215)
        alt, hours = rng.uniform(15, 20), rng.uniform(0, 150)
        want = wind_at(grid, lat, lon, alt, hours)
        got = rl.forecast_at(truth, BELIEF, noise, jnp.array(BELIEF.sigma), lat, lon, alt, hours)
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), atol=2e-4)


def test_the_point_belief_mean_is_the_grid_belief_mean_at_grid_points():
    """With readings, the GP's mean at a grid point is what GPBelief puts there"""
    truth, key = a_record(), jax.random.key(3)
    grid = forecast(BELIEF, key, truth)
    readings = Readings.empty(3)
    from balloon_sim.dynamics import State
    for lat, lon in ((34.0, 150.0), (38.0, 160.0), (30.0, 170.0)):
        s = State(jnp.array(lat), jnp.array(lon), jnp.array(17.5), jnp.array(12.0))
        readings = readings.add(s, wind_at(truth, s.lat, s.lon, s.altitude_km, s.hours))
    mean = GPBelief(BELIEF, grid).winds(key, 0, readings)
    residual = readings._replace(uv=readings.uv - jax.vmap(lambda la, lo, al, h: wind_at(grid, la, lo, al, h))(readings.lat, readings.lon, readings.altitude_km, readings.hours))
    t, a, y, x = 2, 1, 6, 8  # hours 12, 17.5 km, lat 34, lon 152
    point = [jnp.array([v]) for v in (truth.lat[y], truth.lon[x], truth.altitude_km[a], truth.hours[t])]
    got = rl.belief_mean_at(BELIEF, jnp.array(BELIEF.sigma), grid.uv[t, a, y, x][None], *point, residual)
    np.testing.assert_allclose(np.asarray(got[0]), np.asarray(mean.uv[0, t, a, y, x]), atol=2e-3)


def _spec(objective: str, n: int, learns: bool = True, n_moves: int = 6) -> rl.Spec:
    return rl.Spec(objective, n, learns, n_moves, 200.0, BalloonConfig(), BELIEF)


def test_the_return_is_minus_the_cost_the_sweep_scores():
    truth = a_record()
    for objective, n in (("reach", 1), ("escape", 3), ("station", 3), ("targets", 2)):
        spec = _spec(objective, n)
        mission, _ = fleet.draw_mission(a_task(objective, n), jax.random.key(0), truth)
        params = rl.init_params(jax.random.key(1), rl.obs_size(truth.altitude_km.shape[0]), RLConfig())
        noise = rl.noise_from_key(BELIEF, jax.random.key(2))
        steps = rl.episode(spec, truth, params, mission, noise, jnp.array(2.0), jax.random.key(3), 1.0)
        # fly the same climbs through fleet.fly and score them the sweep's way
        climbs = BalloonConfig().climb_cap_km * jnp.tanh(steps.u)  # (T, N)
        replay = fleet.Controller(lambda k, s: 0, lambda m, k, s, move, r: (climbs[move], m))
        flight = fleet.fly(BalloonConfig(), replay, truth, mission, objective, spec.n_moves, jax.random.key(4), 200.0)
        np.testing.assert_allclose(-float(jnp.sum(steps.reward)), float(flight.cost_km), rtol=1e-4, err_msg=objective)


def test_the_policy_flies_through_the_sweeps_simulator_and_the_blind_one_ignores_readings():
    truth = a_record()
    mission, _ = fleet.draw_mission(a_task("targets", 3), jax.random.key(0), truth)
    params = rl.init_params(jax.random.key(1), rl.obs_size(truth.altitude_km.shape[0]), RLConfig())
    noise = rl.noise_from_key(BELIEF, jax.random.key(2))
    for learns in (True, False):
        spec = _spec("targets", 3, learns)
        flight = fleet.fly(BalloonConfig(), rl.controller(spec, truth, params, mission, noise, jnp.array(2.0)), truth, mission,
                           "targets", spec.n_moves, jax.random.key(4), 200.0)
        assert flight.visited.lat.shape == (3, spec.n_moves) and np.isfinite(float(flight.cost_km))
    # the blind policy's observation does not move when the readings do
    spec = _spec("reach", 1, learns=False)
    obs = lambda residual: rl.observe(spec, truth, noise, jnp.array(2.0), mission._replace(starts=jax.tree.map(lambda a: a[:1], mission.starts)),
                                      jax.tree.map(lambda a: a[:1], mission.starts), Readings.empty(1),
                                      rl.Flags(jnp.zeros(1, bool), jnp.zeros(3, bool)), residual, jnp.array(0))
    np.testing.assert_array_equal(np.asarray(obs(jnp.zeros((1, 2)))), np.asarray(obs(jnp.ones((1, 2)) * 5)))


def test_training_runs_and_saves(tmp_path):
    truth = a_record()
    spec = _spec("escape", 2, n_moves=5)
    config = RLConfig(iterations=3, envs=4, train_seeds=3, minibatches=2, epochs=1)
    missions = rl.training_missions(dataclasses.replace(a_task("escape", 2), n_moves=5), truth, config)
    params, curve = rl.train(spec, truth, missions, (0.0, 2.0), config, jax.random.key(0), log=lambda _: None, every=1)
    assert len(curve) == 3 and all(np.isfinite(c["cost_km"]) for c in curve)
    rl.save(tmp_path / "p.pkl", params, spec, config, curve, {})
    loaded, spec_back, blob = rl.load(tmp_path / "p.pkl")
    assert spec_back == spec and blob["curve"] == curve
