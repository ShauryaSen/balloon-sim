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
    climbs = plan(BALLOON, planner, jax.random.key(0), batch_of(east_low_west_high(), 1), START, *GOAL, jnp.zeros(8), jnp.array(8))
    assert climbs[0] < -0.5 * BALLOON.climb_cap_km  # heads down at once
    assert jnp.all(jnp.abs(climbs) <= BALLOON.climb_cap_km + 1e-6)


def test_lbfgs_beats_every_random_plan():
    """The default refinement finds a better plan than the best of many random ones"""
    field = east_low_west_high()
    climbs = plan(BALLOON, SMALL, jax.random.key(0), batch_of(field, 1), START, *GOAL, jnp.zeros(8), jnp.array(8))
    cap = BALLOON.climb_cap_km
    random_plans = jax.random.uniform(jax.random.key(1), (2000, 8), minval=-cap, maxval=cap)
    best_random = jnp.min(jax.vmap(lambda c: cost_of(field, c))(random_plans))
    assert cost_of(field, climbs) <= best_random


def test_normalised_descent_beats_coasting():
    """Myles's refinement, the option: much better than not steering, though under a 6 h step it stalls well
    above L-BFGS (about twice its cost on this wind)"""
    field = east_low_west_high()
    planner = dataclasses.replace(SMALL, optimizer="normalised")
    climbs = plan(BALLOON, planner, jax.random.key(0), batch_of(field, 1), START, *GOAL, jnp.zeros(8), jnp.array(8))
    assert cost_of(field, climbs) < 0.6 * cost_of(field, jnp.zeros(8))


def test_a_batch_of_identical_winds_plans_like_one_wind():
    field = east_low_west_high()
    one = plan(BALLOON, SMALL, jax.random.key(0), batch_of(field, 1), START, *GOAL, jnp.zeros(8), jnp.array(8))
    four = plan(BALLOON, SMALL, jax.random.key(0), batch_of(field, 4), START, *GOAL, jnp.zeros(8), jnp.array(8))
    # L-BFGS's line search turns last-digit rounding differences into slightly different plans of nearly the
    # same cost, so compare what the plans score, not the climbs themselves
    assert jnp.allclose(cost_of(field, one), cost_of(field, four), rtol=0.05)


def test_moves_after_the_episode_ends_do_not_count():
    """With 3 moves left, climbs beyond the third change nothing, and the plan ignores a wind it would only
    meet after the episode: the cost of the first three moves alone is what it minimises"""
    from balloon_sim.planner import expected_cost

    field = east_low_west_high()
    a = jnp.array([-1.0, 0.5, 0.2, 0.9, -0.9, 0.9, -0.9, 0.9])
    changed_tail = a.at[3:].set(0.0)
    cost = lambda plan_a: expected_cost(BALLOON, SMALL, plan_a, batch_of(field, 1), START, *GOAL, jnp.array(3))
    assert jnp.allclose(cost(a), cost(changed_tail))
    first_three = rollout(field, START, BALLOON.climb_cap_km * a[:3], BALLOON.step_hours)
    assert jnp.allclose(cost(a), summed_cost(first_three, *GOAL), rtol=1e-5)


def test_refinement_never_returns_a_worse_plan_and_more_iterations_never_hurt():
    """The refined plan costs no more than the best candidate it started from, and 500 iterations cost no more
    than 100: a line search that wanders uphill is not allowed to hand back where it ended"""
    from balloon_sim.planner import candidates, expected_cost

    field = east_low_west_high()
    winds = batch_of(field, 1)
    key = jax.random.key(0)
    pool = candidates(BALLOON, SMALL, key, START, jnp.zeros(8), winds.altitude_km[0], winds.altitude_km[-1])
    best_candidate = jnp.min(jax.vmap(lambda a: expected_cost(BALLOON, SMALL, a, winds, START, *GOAL, jnp.array(8)))(pool))
    few = plan(BALLOON, SMALL, key, winds, START, *GOAL, jnp.zeros(8), jnp.array(8))
    many = plan(BALLOON, dataclasses.replace(SMALL, iterations=500), key, winds, START, *GOAL, jnp.zeros(8), jnp.array(8))
    assert cost_of(field, few) <= best_candidate + 1e-3
    assert cost_of(field, many) <= cost_of(field, few) + 1e-3
