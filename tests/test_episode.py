import dataclasses

import jax
import jax.numpy as jnp

from balloon_sim.config import BalloonConfig, PlannerConfig
from balloon_sim.controllers import Controller, hindsight
from balloon_sim.dynamics import State, distance_km, rollout, summed_cost
from balloon_sim.episode import fly
from balloon_sim.wind import WindField

BALLOON = BalloonConfig()
SMALL = PlannerConfig(candidates=30, iterations=50)
MOVES = 8
START = State(jnp.array(30.0), jnp.array(150.0), jnp.array(17.5), jnp.array(0.0))


def field(u_low: float, u_high: float) -> WindField:
    """Eastward wind u_low at 15 km and u_high at 20 km, on a box from 0 to 70N and 100 to 200E"""
    hours, altitude, lat, lon = jnp.array([0.0, 100.0]), jnp.array([15.0, 20.0]), jnp.array([0.0, 70.0]), jnp.array([100.0, 200.0])
    uv = jnp.zeros((2, 2, 2, 2, 2)).at[:, 0, ..., 0].set(u_low).at[:, 1, ..., 0].set(u_high)
    return WindField(hours, altitude, lat, lon, uv)


def coast() -> Controller:
    """Never climbs: the baseline every planner should beat"""
    return Controller(lambda key, state: jnp.zeros(()), lambda memory, key, state, move, reading: (jnp.zeros(()), memory))


def test_still_air_goes_nowhere():
    goal = (jnp.array(31.0), jnp.array(150.0))
    flight = fly(BALLOON, coast(), field(0.0, 0.0), START, *goal, MOVES, jax.random.key(0))
    assert jnp.allclose(flight.visited.lat, 30.0) and jnp.allclose(flight.visited.lon, 150.0)
    assert jnp.allclose(flight.summed_km, MOVES * distance_km(30.0, 150.0, 31.0, 150.0), atol=0.1)


def test_hindsight_beats_coast_and_flies_its_plan():
    truth, goal = field(15.0, -15.0), (jnp.array(30.0), jnp.array(156.0))
    clairvoyant = hindsight(BALLOON, SMALL, truth, *goal, MOVES)
    flight = fly(BALLOON, clairvoyant, truth, START, *goal, MOVES, jax.random.key(0))
    baseline = fly(BALLOON, coast(), truth, START, *goal, MOVES, jax.random.key(0))
    assert flight.summed_km < 0.5 * baseline.summed_km


def test_the_simulator_holds_the_balloon_inside_the_box():
    """A 60 m/s westerly blows the balloon past the box's eastern edge at 200E within a few moves"""
    flight = fly(BALLOON, coast(), field(60.0, 60.0), START, jnp.array(30.0), jnp.array(150.0), 20, jax.random.key(0))
    assert jnp.max(flight.visited.lon) <= 200.0
    assert jnp.isclose(flight.visited.lon[-1], 200.0)


def test_hindsight_flies_the_cost_it_planned():
    """With the true wind known, the flown cost equals what its own climbs cost in that wind: the planner holds
    altitude where the simulator does, and replanning on the truth never disagrees with the flight"""
    truth, goal = field(15.0, -15.0), (jnp.array(30.0), jnp.array(156.0))
    flight = fly(BALLOON, hindsight(BALLOON, SMALL, truth, *goal, MOVES), truth, START, *goal, MOVES, jax.random.key(0))
    imagined = summed_cost(rollout(truth, START, flight.climbs_km, BALLOON.step_hours), *goal)
    assert jnp.allclose(flight.summed_km, imagined, rtol=0.01)
