"""The arrival rule: the episode ends when the balloon gets there, and the moves after it cost nothing"""

import jax
import jax.numpy as jnp
import pytest

from balloon_sim.config import BalloonConfig, BeliefConfig, PlannerConfig, TaskConfig
from balloon_sim.dynamics import State, arrival, counted, episode_cost, summed_cost
from balloon_sim.episode import fly
from balloon_sim.controllers import hindsight, mpc
from balloon_sim.belief import GPBelief, Nested, forecast
from balloon_sim.task import draw_task


def a_track(distances_km):
    """A trajectory placed due north of a goal at (0, 0), at the given distances"""
    degrees = jnp.array(distances_km) / 111.195  # km per degree of latitude
    zeros = jnp.zeros_like(degrees)
    return State(degrees, zeros, zeros, zeros)


def test_with_no_radius_the_cost_is_the_plain_sum():
    """arrival_km = 0 is the rule the earlier sweeps scored, so the two must agree exactly"""
    visited = a_track([900.0, 600.0, 300.0, 150.0, 400.0])
    assert episode_cost(visited, jnp.array(0.0), jnp.array(0.0), 0.0) == pytest.approx(
        float(summed_cost(visited, jnp.array(0.0), jnp.array(0.0))), rel=1e-6
    )


def test_the_moves_after_arrival_are_free():
    """Once inside the radius, what happens next must not change the cost"""
    goal = (jnp.array(0.0), jnp.array(0.0))
    stays = a_track([900.0, 600.0, 150.0, 140.0, 130.0])
    leaves = a_track([900.0, 600.0, 150.0, 5000.0, 9000.0])
    assert episode_cost(stays, *goal, 200.0) == pytest.approx(float(episode_cost(leaves, *goal, 200.0)), rel=1e-6)


def test_arriving_sooner_costs_less():
    """The rule's whole point: the same arrival, reached earlier, is worth more"""
    goal = (jnp.array(0.0), jnp.array(0.0))
    soon = a_track([150.0, 150.0, 150.0, 150.0, 150.0])
    late = a_track([900.0, 900.0, 900.0, 900.0, 150.0])
    assert float(episode_cost(soon, *goal, 200.0)) < float(episode_cost(late, *goal, 200.0))


def test_the_arrival_move_is_reported_from_one():
    visited = a_track([900.0, 600.0, 150.0, 140.0])
    ever, move = arrival(visited, jnp.array(0.0), jnp.array(0.0), 200.0)
    assert bool(ever) and int(move) == 3


def test_never_arriving_is_reported_as_zero():
    visited = a_track([900.0, 600.0, 400.0])
    ever, move = arrival(visited, jnp.array(0.0), jnp.array(0.0), 200.0)
    assert not bool(ever) and int(move) == 0


def test_the_mask_also_stops_at_the_end_of_the_episode():
    """A horizon reaching past the last move must not count the moves beyond it, arrival or no arrival"""
    distance = jnp.array([900.0, 600.0, 150.0, 140.0])
    keep = counted(distance, 200.0, jnp.array(2))
    assert list(map(bool, keep)) == [True, True, False, False]
