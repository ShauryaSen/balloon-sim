"""The arms added for the 2026-10-23 sweep: MPC that does not learn, nested draws, and the sweep's grid"""

import jax
import jax.numpy as jnp
import pytest

from balloon_sim.belief import GPBelief, Nested, Readings, forecast
from balloon_sim.cells import ARMS, BASE_GOAL_KM, cells
from balloon_sim.config import BeliefConfig, ExperimentConfig, TaskConfig
from balloon_sim.wind import WindField

HOURS, ALTITUDE = jnp.array([0.0, 12.0, 24.0]), jnp.array([15.0, 17.0, 19.0])
LAT, LON = jnp.array([20.0, 30.0, 40.0]), jnp.array([150.0, 160.0, 170.0])


def a_record() -> WindField:
    """A small still record: the belief's draws are what these tests look at, not the wind under them"""
    return WindField(HOURS, ALTITUDE, LAT, LON, jnp.zeros((3, 3, 3, 3, 2)))


def test_nested_draws_are_the_first_of_the_pool():
    """What the arm is for: the four winds a plan sees must be four of the sixteen, not four fresh ones"""
    truth = a_record()
    config = BeliefConfig(sigma=2.0, n_features=32)
    belief = GPBelief(config, forecast(config, jax.random.key(0), truth))
    nothing = Readings.empty(1)
    key = jax.random.key(1)
    pool = Nested(belief, 16).winds(key, 16, nothing)
    few = Nested(belief, 16).winds(key, 4, nothing)
    assert jnp.allclose(few.uv, pool.uv[:4])


def test_nested_does_not_change_the_mean():
    truth = a_record()
    config = BeliefConfig(sigma=2.0, n_features=32)
    belief = GPBelief(config, forecast(config, jax.random.key(0), truth))
    nothing, key = Readings.empty(1), jax.random.key(1)
    assert jnp.allclose(Nested(belief, 16).winds(key, 0, nothing).uv, belief.winds(key, 0, nothing).uv)


def test_the_grid_has_no_repeated_flight():
    grid = cells(ExperimentConfig(), TaskConfig().n_moves, BeliefConfig().sigma)
    assert len(set(grid)) == len(grid)


def test_every_arm_is_reachable_on_its_own():
    experiment, n_moves, sigma = ExperimentConfig(), TaskConfig().n_moves, BeliefConfig().sigma
    for name in ARMS:
        assert cells(experiment, n_moves, sigma, (name,)), f"{name} is empty"


def test_each_goal_distance_carries_its_own_hindsight():
    """A different goal is a different task, so its ceiling cannot be borrowed from another distance"""
    experiment = ExperimentConfig()
    grid = cells(experiment, TaskConfig().n_moves, BeliefConfig().sigma)
    hind = {(c.seed, c.goal_km) for c in grid if c.controller == "hindsight"}
    flown = {(c.seed, c.goal_km) for c in grid}
    assert hind == flown or all(pair in hind for pair in flown)


def test_the_calibration_arm_leaves_the_noise_alone():
    """Only what the belief assumes changes; a scale is never applied to the error actually flown"""
    grid = cells(ExperimentConfig(), TaskConfig().n_moves, BeliefConfig().sigma, ("calibration",))
    assert grid and all(c.belief_scale != 1.0 and c.sigma > 0 for c in grid)


def test_the_main_arm_is_calibrated_and_at_the_base_goal():
    grid = cells(ExperimentConfig(), TaskConfig().n_moves, BeliefConfig().sigma, ("main",))
    assert all(c.belief_scale == 1.0 and c.goal_km == BASE_GOAL_KM and not c.nested for c in grid)
