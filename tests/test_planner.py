import dataclasses

import jax
import jax.numpy as jnp
import pytest

from balloon_sim.config import BalloonConfig, PlannerConfig
from balloon_sim.dynamics import State, rollout, summed_cost
from balloon_sim.planner import plan
from balloon_sim.wind import WindField

BALLOON = BalloonConfig()
SMALL = PlannerConfig(horizon=8, candidates=30, iterations=50)  # small enough to run fast
START = State(jnp.array(30.0), jnp.array(150.0), jnp.array(17.5), jnp.array(0.0))
GOAL = (jnp.array(30.0), jnp.array(156.0))  # about 580 km east


def east_low_west_high() -> WindField:
    """15 m/s east at 15 km, 15 m/s west at 20 km: to go east, the balloon has to sink"""
    hours, altitude, lat, lon = jnp.array([0.0, 100.0]), jnp.array([15.0, 20.0]), jnp.array([0.0, 70.0]), jnp.array([100.0, 200.0])
    uv = jnp.zeros((2, 2, 2, 2, 2)).at[:, 0, ..., 0].set(15.0).at[:, 1, ..., 0].set(-15.0)
    return WindField(hours, altitude, lat, lon, uv)


def batch_of(field: WindField, n: int) -> WindField:
    """n copies of one wind, stacked the way the planner expects a batch"""
    return field._replace(uv=jnp.stack([field.uv] * n))


def cost_of(field: WindField, climbs_km: jax.Array) -> jax.Array:
    return summed_cost(rollout(field, START, climbs_km, BALLOON.step_hours), *GOAL)


@pytest.mark.parametrize("optimizer", ["normalised", "lbfgs"])
def test_sinks_to_ride_the_eastward_wind(optimizer):
    planner = dataclasses.replace(SMALL, optimizer=optimizer)
    climbs = plan(BALLOON, planner, jax.random.key(0), batch_of(east_low_west_high(), 1), START, *GOAL, jnp.zeros(8))
    assert jnp.sum(climbs[:3]) < -2.0  # heads down early
    assert jnp.all(jnp.abs(climbs) <= BALLOON.climb_cap_km + 1e-6)


@pytest.mark.parametrize("optimizer", ["normalised", "lbfgs"])
def test_beats_every_random_plan(optimizer):
    field = east_low_west_high()
    planner = dataclasses.replace(SMALL, optimizer=optimizer)
    climbs = plan(BALLOON, planner, jax.random.key(0), batch_of(field, 1), START, *GOAL, jnp.zeros(8))
    random_plans = jax.random.uniform(jax.random.key(1), (500, 8), minval=-1.1, maxval=1.1)
    best_random = jnp.min(jax.vmap(lambda c: cost_of(field, c))(random_plans))
    assert cost_of(field, climbs) <= best_random


def test_a_batch_of_identical_winds_plans_like_one_wind():
    field = east_low_west_high()
    one = plan(BALLOON, SMALL, jax.random.key(0), batch_of(field, 1), START, *GOAL, jnp.zeros(8))
    four = plan(BALLOON, SMALL, jax.random.key(0), batch_of(field, 4), START, *GOAL, jnp.zeros(8))
    assert jnp.allclose(one, four, atol=1e-4)
