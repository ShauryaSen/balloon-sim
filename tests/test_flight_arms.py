"""The greedy and flight arms (decision 51): a one-move horizon, and long flights with long horizons"""

import jax
import jax.numpy as jnp
import numpy as np

from balloon_sim import fleet
from balloon_sim.belief import GPBelief, forecast
from balloon_sim.cells import SWEEP, cells
from balloon_sim.config import BalloonConfig, BeliefConfig, ExperimentConfig, PlannerConfig, TaskConfig
from tests.test_fleet import a_record, a_task

EXPERIMENT, N_MOVES, SIGMA = ExperimentConfig(), TaskConfig().n_moves, BeliefConfig().sigma


def test_the_greedy_arm_is_mpc_one_move_ahead_in_the_arms_it_copies():
    grid = cells(EXPERIMENT, N_MOVES, SIGMA, ("greedy",))
    assert {c.horizon for c in grid} == {1}
    assert {c.controller for c in grid} == {"mpc", "mpc_blind"}
    assert {c.n_moves for c in grid} == {N_MOVES} and {c.n_winds for c in grid} == {0}
    shapes = {(c.objective, c.n_agents) for c in grid}
    assert shapes == {(o, 1) for o in ("reach", "escape", "station", "targets")} | {
        (o, n) for n in EXPERIMENT.fleet_sizes for o in ("reach", "escape", "station", "targets")}


def test_the_flight_arm_flies_every_horizon_that_fits_and_its_own_hindsight():
    grid = cells(EXPERIMENT, N_MOVES, SIGMA, ("flight",))
    assert len(set(grid)) == len(grid)
    for length in EXPERIMENT.flight_lengths:
        at = [c for c in grid if c.n_moves == length]
        flown = {c.horizon for c in at if c.controller.startswith("mpc")}
        assert flown == {h for h in EXPERIMENT.flight_horizons if h <= length}
        for objective in EXPERIMENT.flight_objectives:
            hind = {c.seed for c in at if c.controller == "hindsight" and c.objective == objective}
            assert hind == set(EXPERIMENT.seeds)
    assert max(c.n_moves for c in grid) == max(EXPERIMENT.flight_lengths)


def test_neither_new_arm_is_in_the_default_sweep():
    """Both are run on purpose: the flight arm needs its own record"""
    assert "greedy" not in SWEEP and "flight" not in SWEEP


def test_a_one_move_horizon_flies_and_picks_the_better_level_over_the_next_move():
    """Greedy, on a perfect forecast: its first climb leads to the lowest one-move cost of any climb"""
    truth, balloon = a_record(), BalloonConfig()
    mission, _ = fleet.draw_mission(a_task("reach"), jax.random.key(0), truth)
    exact = BeliefConfig(sigma=0.0, n_features=32)
    believed = GPBelief(exact, forecast(exact, jax.random.key(1), truth))
    greedy = fleet.mpc(balloon, PlannerConfig(horizon=1), believed, "reach", mission, 0, 6, 0.0)
    flight = fleet.fly(balloon, greedy, truth, mission, "reach", 6, jax.random.key(2), 0.0)
    assert np.isfinite(float(flight.cost_km))
    first = flight.climbs_km[:, 0]
    one_move = lambda c: fleet.fleet_cost("reach", *(lambda s: (s.lat, s.lon))(fleet._fly_plan(balloon, truth, mission.starts, c[:, None])),
                                          mission, 0.0, jnp.array(1), jnp.zeros(1, bool))
    tried = [float(one_move(jnp.full((1,), c))) for c in np.linspace(-balloon.climb_cap_km, balloon.climb_cap_km, 9)]
    assert float(one_move(first)) <= min(tried) + 1.0
