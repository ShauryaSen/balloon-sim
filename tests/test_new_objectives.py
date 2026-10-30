"""The two arms added for the 2026-10-29 sweep: the escape objective, and the longer episode

The escape task has no goal. The planner is pointed at the start and the sign of the cost is flipped, so it
runs from the start instead of to it. These tests are mostly about that sign being the right way round, since
getting it backwards would still run, still produce figures, and quietly measure the opposite of the claim.
"""

import jax
import jax.numpy as jnp

from balloon_sim.cells import ARMS, BASE_GOAL_KM, BASE_MOVES
from balloon_sim.config import BalloonConfig, BeliefConfig, ExperimentConfig, PlannerConfig, TaskConfig
from balloon_sim.controllers import hindsight
from balloon_sim.dynamics import State, escape_cost, farthest
from balloon_sim.episode import fly
from balloon_sim.results import one_arm
from balloon_sim.wind import WindField

HOURS = jnp.array([0.0, 60.0, 120.0, 240.0])
ALTITUDE = jnp.array([15.0, 20.0])
LAT, LON = jnp.array([20.0, 35.0, 50.0]), jnp.array([150.0, 165.0, 180.0])


def a_sheared_record(upper_east_ms: float = 30.0) -> WindField:
    """Still at the lower level, blowing hard east at the upper one.

    A balloon that stays low goes nowhere; one that climbs is carried away. So under the escape objective the
    planner must climb, and under the reach objective with the goal at the start it must not.
    """
    uv = jnp.zeros((len(HOURS), len(ALTITUDE), len(LAT), len(LON), 2))
    return WindField(HOURS, ALTITUDE, LAT, LON, uv.at[:, 1, :, :, 0].set(upper_east_ms))


def a_start() -> State:
    return State(jnp.array(35.0), jnp.array(165.0), jnp.array(15.0), jnp.array(0.0))


def test_escape_cost_is_negative_and_falls_as_the_balloon_gets_farther():
    """The sign, on its own: going farther must cost less, or every figure reads backwards"""
    start = a_start()
    near = State(jnp.array([35.0, 35.1]), jnp.array([165.0, 165.1]), jnp.zeros(2), jnp.zeros(2))
    far = State(jnp.array([35.0, 40.0]), jnp.array([165.0, 175.0]), jnp.zeros(2), jnp.zeros(2))
    assert escape_cost(far, start.lat, start.lon) < escape_cost(near, start.lat, start.lon) < 0.0
    assert farthest(far, start.lat, start.lon) > farthest(near, start.lat, start.lon) > 0.0


def _flight(away: bool, n_moves: int = 6):
    """Hindsight flown on a sheared record, pointed at its own start: away runs from it, otherwise it stays"""
    truth = a_sheared_record()
    balloon, planner = BalloonConfig(), PlannerConfig(horizon=n_moves, candidates=24, iterations=12)
    start = a_start()
    controller = hindsight(balloon, planner, truth, start.lat, start.lon, n_moves, 0.0, away)
    return fly(balloon, controller, truth, start, start.lat, start.lon, n_moves, jax.random.key(0), 0.0, away)


def test_escaping_climbs_into_the_wind_and_staying_does_not():
    """End to end: the same controller and record, the sign the only difference between them"""
    escaped, stayed = _flight(away=True), _flight(away=False)
    assert float(escaped.farthest_km) > 1000.0  # 30 m/s for 18 h is about 1,900 km
    assert float(stayed.farthest_km) < float(escaped.farthest_km) / 4.0
    assert float(escaped.cost_km) < float(stayed.cost_km)  # lower is better under both


def test_the_escape_arm_flies_the_escape_objective_and_nothing_else_does():
    experiment = ExperimentConfig()
    sigma = BeliefConfig().sigma
    escape = ARMS["escape"](experiment, BASE_MOVES, sigma)
    assert escape and all(cell.objective == "escape" for cell in escape)
    for name in ("main", "distance", "calibration", "nested", "length"):
        assert all(cell.objective == "reach" for cell in ARMS[name](experiment, BASE_MOVES, sigma))


def test_the_length_arm_is_the_longer_episode_at_two_distances():
    experiment = ExperimentConfig()
    longer = max(experiment.episode_lengths)
    cells = ARMS["length"](experiment, BASE_MOVES, BeliefConfig().sigma)
    assert all(cell.n_moves == longer for cell in cells)
    assert {cell.goal_km for cell in cells} == {BASE_GOAL_KM, max(experiment.goal_distances)}
    assert longer > BASE_MOVES  # the whole point: more time for the same task


def test_one_arm_keeps_the_arms_apart():
    """The reason this function exists: without it the new arms overwrite the main one seed by seed"""
    def made(**over):
        row = {"controller": "mpc", "seed": 0, "goal_km": BASE_GOAL_KM, "belief_scale": 1.0,
               "nested": False, "task.n_moves": BASE_MOVES, "objective": "reach"}
        return {**row, **over}

    rows = [made(), made(**{"task.n_moves": 80}), made(objective="escape")]
    assert one_arm(rows) == [rows[0]]
    assert one_arm(rows, n_moves=80) == [rows[1]]
    assert one_arm(rows, objective="escape") == [rows[2]]


def test_a_row_without_an_objective_column_is_a_reach_row():
    """Every row written before the escape arm existed flew the reach task, so it must not answer to escape"""
    old = {"controller": "mpc", "seed": 0, "goal_km": BASE_GOAL_KM, "belief_scale": 1.0, "nested": False,
           "task.n_moves": BASE_MOVES}
    assert one_arm([old]) == [old]
    assert one_arm([old], objective="escape") == []
