"""The fleet and the two new objectives (decisions 47 and 48)

The first tests pin a fleet of one to the single-balloon code on the two objectives that code already flies:
if they agree, a fleet row can be read beside the rows of every sweep before it. The rest check that each new
objective rewards what it says it does, on records simple enough that the right answer is known.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from balloon_sim import fleet
from balloon_sim.belief import GPBelief, forecast
from balloon_sim.cells import ARMS, cells
from balloon_sim.config import BalloonConfig, BeliefConfig, ExperimentConfig, PlannerConfig, TaskConfig
from balloon_sim.controllers import fixed_plan, hindsight, mpc
from balloon_sim.dynamics import State
from balloon_sim.episode import fly
from balloon_sim.results import one_arm
from balloon_sim.wind import WindField

HOURS = jnp.arange(0.0, 160.0, 6.0)
ALTITUDE = jnp.array([15.0, 17.5, 20.0])
LAT, LON = jnp.arange(10.0, 62.0, 4.0), jnp.arange(120.0, 220.0, 4.0)
SMALL = PlannerConfig(horizon=6, candidates=16, iterations=8)
BELIEF = BeliefConfig(sigma=2.0, n_features=32)


def a_record(seed: int = 0) -> WindField:
    """Calm at the bottom, east at the top, and a smooth random wind on top of that, so the levels differ
    and a plan's choice matters"""
    shape = (len(HOURS), len(ALTITUDE), len(LAT), len(LON), 2)
    rough = np.random.default_rng(seed).normal(0.0, 3.0, shape)
    uv = np.zeros(shape)
    uv[:, 2, :, :, 0] = 25.0
    uv[:, 1, :, :, 0] = 8.0
    uv[:, 0, :, :, 0] = -6.0
    return WindField(HOURS, ALTITUDE, LAT, LON, jnp.asarray(uv + rough, dtype=jnp.float32))


def a_task(objective: str = "reach", n_agents: int = 1, **more) -> TaskConfig:
    return TaskConfig(mode="fixed", start_lat=35.0, start_lon=150.0, goal_lat=35.0, goal_lon=165.0,
                      objective=objective, n_agents=n_agents, distance_km=1400.0, **more)


def _both(controller_name: str, objective: str):
    """One flight by the single-balloon code and the same flight by a fleet of one"""
    truth, balloon, n_moves = a_record(), BalloonConfig(), 8
    config = a_task(objective)
    mission, task = fleet.draw_mission(config, jax.random.key(1), truth)
    believed = GPBelief(BELIEF, forecast(BELIEF, jax.random.key(2), truth))
    away = objective == "escape"
    radius = 0.0 if away else config.arrival_km
    aim = (task.start.lat, task.start.lon) if away else (task.goal_lat, task.goal_lon)
    if controller_name == "hindsight":
        one = hindsight(balloon, SMALL, truth, *aim, n_moves, radius, away)
        many = fleet.hindsight(balloon, SMALL, truth, objective, mission, n_moves, config.arrival_km)
    elif controller_name == "fixed_plan":
        one = fixed_plan(balloon, SMALL, believed, *aim, n_moves, 0, radius, away)
        many = fleet.fixed_plan(balloon, SMALL, believed, objective, mission, n_moves, 0, config.arrival_km)
    else:
        one = mpc(balloon, SMALL, believed, *aim, 0, n_moves, radius, away=away)
        many = fleet.mpc(balloon, SMALL, believed, objective, mission, 0, n_moves, config.arrival_km)
    key = jax.random.key(3)
    single = fly(balloon, one, truth, task.start, task.goal_lat, task.goal_lon, n_moves, key, radius, away)
    together = fleet.fly(balloon, many, truth, mission, objective, n_moves, key, config.arrival_km)
    return single, together


def test_a_fleet_of_one_is_the_single_balloon_on_reach():
    for name in ("hindsight", "fixed_plan", "mpc"):
        single, together = _both(name, "reach")
        np.testing.assert_allclose(together.climbs_km[0], single.climbs_km, atol=1e-4, err_msg=name)
        np.testing.assert_allclose(float(together.cost_km), float(single.cost_km), rtol=1e-5, err_msg=name)
        assert bool(together.complete) == bool(single.arrived), name


def test_a_fleet_of_one_is_the_single_balloon_on_escape():
    for name in ("hindsight", "fixed_plan", "mpc"):
        single, together = _both(name, "escape")
        np.testing.assert_allclose(together.climbs_km[0], single.climbs_km, atol=1e-4, err_msg=name)
        np.testing.assert_allclose(float(together.cost_km), float(single.cost_km), rtol=1e-5, err_msg=name)
        np.testing.assert_allclose(float(together.farthest_km), float(single.farthest_km), rtol=1e-5, err_msg=name)


def test_the_ring_and_the_chain():
    """Balloon 0 on the start, the others spread_km out; stops spaced along the flow, which here is east"""
    truth = a_record()
    mission, task = fleet.draw_mission(a_task("targets", 4), jax.random.key(0), truth)
    from balloon_sim.dynamics import distance_km
    out = distance_km(mission.starts.lat, mission.starts.lon, task.start.lat, task.start.lon)
    np.testing.assert_allclose(np.asarray(out), [0.0, 150.0, 150.0, 150.0], atol=0.5)
    assert mission.targets_lat.shape == (3,)
    legs = distance_km(jnp.concatenate([task.start.lat[None], mission.targets_lat[:-1]]),
                       jnp.concatenate([task.start.lon[None], mission.targets_lon[:-1]]),
                       mission.targets_lat, mission.targets_lon)
    np.testing.assert_allclose(np.asarray(legs), 1400.0 / 3, rtol=1e-3)
    assert bool(jnp.all(jnp.diff(mission.targets_lon) > 0))  # downwind is east on this record


def _hindsight_flight(objective: str, n_agents: int = 1, n_moves: int = 10, truth=None):
    truth = a_record() if truth is None else truth
    config = a_task(objective, n_agents)
    mission, _ = fleet.draw_mission(config, jax.random.key(0), truth)
    planner = dataclasses.replace(SMALL, candidates=24, iterations=12)
    controller = fleet.hindsight(BalloonConfig(), planner, truth, objective, mission, n_moves, config.arrival_km)
    return fleet.fly(BalloonConfig(), controller, truth, mission, objective, n_moves, jax.random.key(5), config.arrival_km), mission


def test_station_keeping_rides_the_calm_and_escape_rides_the_jet():
    """With a calm-ish level and a jet, holding the station means staying out of the jet"""
    held, _ = _hindsight_flight("station")
    run, _ = _hindsight_flight("escape")
    assert float(held.covered_fraction) > 0.5
    assert float(held.farthest_km) < float(run.farthest_km) / 3
    assert float(held.cost_km) >= 0.0


def test_the_station_cost_is_zero_inside_and_counts_only_the_nearest_balloon():
    mission = fleet.Mission(State(jnp.zeros(2), jnp.zeros(2), jnp.zeros(2), jnp.zeros(2)), jnp.array([0.0]), jnp.array([0.0]))
    one_deg = 111.195  # km per degree of latitude
    lat = jnp.array([[0.5, 5.0], [9.0, 9.0]])  # balloon 0 inside then out; balloon 1 always far
    cost = fleet.fleet_cost("station", lat, jnp.zeros((2, 2)), mission, 200.0, jnp.array(2), jnp.zeros(1, bool))
    np.testing.assert_allclose(float(cost), 5 * one_deg - 200.0, rtol=1e-3)


def test_targets_cost_stops_once_a_target_is_visited_by_anyone():
    mission = fleet.Mission(State(jnp.zeros(2), jnp.zeros(2), jnp.zeros(2), jnp.zeros(2)),
                            jnp.array([0.0, 10.0]), jnp.array([0.0, 0.0]))
    lat = jnp.array([[0.0, 0.0, 0.0], [5.0, 9.0, 10.0]])  # balloon 0 sits on target 0, balloon 1 walks to target 1
    cost = fleet.fleet_cost("targets", lat, jnp.zeros((2, 3)), mission, 200.0, jnp.array(3), jnp.zeros(2, bool))
    km = 111.195
    np.testing.assert_allclose(float(cost), 0.0 + 5 * km + 1 * km + 0.0, rtol=1e-3)  # target 0 counted on its arrival move only
    gone = fleet.fleet_cost("targets", lat, jnp.zeros((2, 3)), mission, 200.0, jnp.array(3), jnp.array([False, True]))
    np.testing.assert_allclose(float(gone), 0.0, atol=0.05)  # haversine adds ~13 m at zero distance


def test_hindsight_visits_the_chain():
    flight, mission = _hindsight_flight("targets", n_moves=16)
    assert int(flight.visited_count) == 3
    assert bool(flight.complete) and int(flight.complete_move) > 0


def test_a_fleet_shares_its_readings_and_flies_every_balloon():
    truth = a_record()
    config = a_task("reach", 3)
    mission, _ = fleet.draw_mission(config, jax.random.key(0), truth)
    believed = GPBelief(BELIEF, forecast(BELIEF, jax.random.key(2), truth))
    controller = fleet.mpc(BalloonConfig(), SMALL, believed, "reach", mission, 0, 4, config.arrival_km)
    memory = controller.begin(jax.random.key(0), mission.starts)
    reading = jnp.ones((3, 2))
    climbs, memory = controller.act(memory, jax.random.key(1), mission.starts, jnp.array(0), reading)
    assert climbs.shape == (3,) and int(memory.readings.count) == 3
    flight = fleet.fly(BalloonConfig(), controller, truth, mission, "reach", 4, jax.random.key(1), config.arrival_km)
    assert flight.visited.lat.shape == (3, 4)


def test_the_arms_and_the_reader_keep_fleets_apart():
    experiment = ExperimentConfig(seeds=(0, 1))
    fleet_cells = ARMS["fleet"](experiment, 50, 3.2)
    assert {c.objective for c in fleet_cells} == {"reach", "escape", "station", "targets"}
    assert {c.n_agents for c in fleet_cells} == set(experiment.fleet_sizes)
    assert len(fleet_cells) == len(set(fleet_cells)) == 4 * len(experiment.fleet_sizes) * 2 * (1 + 6 * 3)
    assert all(c.n_agents == 1 for c in ARMS["station"](experiment, 50, 3.2) + ARMS["main"](experiment, 50, 3.2))
    everything = cells(experiment, 50, 3.2)
    assert len(everything) == len(set(everything))
    rows = [{"controller": "mpc", "objective": "reach", "goal_km": 2000.0, "task.n_moves": 50},
            {"controller": "mpc", "objective": "reach", "goal_km": 2000.0, "task.n_moves": 50, "n_agents": 4}]
    assert one_arm(rows) == rows[:1] and one_arm(rows, n_agents=4) == rows[1:]
