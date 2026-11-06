"""Several balloons at once, and the objectives beyond reaching one goal (decisions 47 and 48)

A fleet is N balloons flown together in one true wind. They share one belief: every balloon's reading goes
into it every move, so what one balloon learns about the wind every other balloon plans with. They are
planned jointly: the plan is N rows of H climbs, one row per balloon, refined together against one cost, so
an objective that couples the balloons (the station is held if any of them is over it; a target is visited
if any of them reaches it) is planned for as one thing rather than N separate things.

N = 1 is a fleet of one, and is how the two new single-balloon objectives are flown. For the two old ones it
reproduces balloon_sim.planner, controllers and episode exactly (tests/test_fleet.py checks it), which is
what lets a fleet row be read beside a single-balloon row of the same seed. Those modules are not changed.

The four objectives, each written as a cost so lower is better and hindsight is the floor:

    reach     every balloon to the one goal: each balloon's distance to it summed until that balloon
              arrives (the arrival rule, per balloon), averaged over the balloons. N = 1 is the old task.
    escape    every balloon as far from its own start as it can: minus each balloon's distance from its
              start summed over every move, averaged over the balloons. N = 1 is decision 45.
    station   hold a point: at every move the distance the nearest balloon is outside a circle of
              task.arrival_km round the station, summed over the moves. Zero while any balloon is inside.
              The station is the drawn start; balloon 0 launches on it.
    targets   visit every one of K targets, in any order, by any balloon: at every move, for each target
              not yet visited, the distance from it to the nearest balloon, summed until it is visited. A
              target is visited the first time any balloon comes within task.arrival_km of it. The episode
              is over for a target once it is visited, and over for the fleet once all of them are.

Where the balloons start: balloon 0 on the drawn start, the rest evenly round a ring of task.spread_km about
it. The drawn start and goal are draw_task's, from the same seed key, so a fleet cell and a single-balloon
cell of the same seed fly the same weather toward the same goal.

Where the targets are: a chain from the start. Target 1 is task.distance_km / K downwind of the start, along
the flow there at the episode's start (averaged over the levels, the same rule draw_task uses for the goal);
target 2 the same distance downwind of target 1, along the flow at target 1; and so on. Total distance along
the chain is the base task's goal distance, so the multi-target task is the base task broken into stops that
bend with the flow.
"""

import dataclasses
import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from balloon_sim.belief import Readings, Truth, read
from balloon_sim.config import BalloonConfig, PlannerConfig, TaskConfig
from balloon_sim.controllers import Controller
from balloon_sim.dynamics import State, clip, counted, distance_km, rollout, step
from balloon_sim.planner import BATCHED, _lbfgs, _normalised_descent, candidates
from balloon_sim.task import Task, _destination, draw_task
from balloon_sim.wind import WindField, wind_at

OBJECTIVES = ("reach", "escape", "station", "targets")


class Mission(NamedTuple):
    """What a fleet flies: where each balloon starts and the points the objective is about"""

    starts: State  # each field (N,)
    targets_lat: jax.Array  # (K,): the goal for reach, the station for station, the stops for targets
    targets_lon: jax.Array  # escape has no point; it carries the drawn goal so the row still records it


# --- the mission ------------------------------------------------------------------------------------------------


def _ring(lat: jax.Array, lon: jax.Array, n: int, spread_km: float) -> tuple[jax.Array, jax.Array]:
    """Balloon 0 on (lat, lon), the other n - 1 evenly round a ring of spread_km about it. Shape (n,) each"""
    if n == 1:
        return lat[None], lon[None]
    bearing = 2.0 * jnp.pi * jnp.arange(n - 1) / (n - 1)
    ring_lat, ring_lon = _destination(lat, lon, bearing, spread_km)
    return jnp.concatenate([lat[None], ring_lat]), jnp.concatenate([lon[None], ring_lon % 360.0])


def _downwind(truth: WindField, lat: jax.Array, lon: jax.Array, hours: jax.Array) -> jax.Array:
    """The bearing (radians, 0 north) the flow at a point carries a balloon, averaged over the levels"""
    u, v = jnp.mean(jax.vmap(lambda al: wind_at(truth, lat, lon, al, hours))(truth.altitude_km), axis=0)
    return jnp.arctan2(u, v)


def _chain(truth: WindField, config: TaskConfig, lat: jax.Array, lon: jax.Array) -> tuple[jax.Array, jax.Array]:
    """K stops, each distance_km / K downwind of the one before, the first downwind of (lat, lon)"""
    leg = config.distance_km / config.n_targets
    hours = jnp.array(config.start_hours)
    stops_lat, stops_lon = [], []
    for _ in range(config.n_targets):
        lat, lon = _destination(lat, lon, _downwind(truth, lat, lon, hours), leg)
        lon = lon % 360.0
        stops_lat.append(lat)
        stops_lon.append(lon)
    return jnp.stack(stops_lat), jnp.stack(stops_lon)


def draw_mission(config: TaskConfig, key: jax.Array, truth: WindField) -> tuple[Mission, Task]:
    """The fleet's starts and the objective's points, built on draw_task's start and goal for the same key.

    Returns the Task too, so the row records the same start and goal columns a single-balloon row does.
    """
    task = draw_task(config, key, truth)
    n = config.n_agents
    lat, lon = _ring(task.start.lat, task.start.lon, n, config.spread_km)
    starts = State(lat, lon, jnp.full(n, task.start.altitude_km), jnp.full(n, task.start.hours))
    if config.objective == "station":
        points = (task.start.lat[None], task.start.lon[None])
    elif config.objective == "targets":
        points = _chain(truth, config, task.start.lat, task.start.lon)
    else:  # reach: the goal; escape: nothing reads it, but the row carries it
        points = (task.goal_lat[None], task.goal_lon[None])
    return Mission(starts, *points), task


# --- the cost ---------------------------------------------------------------------------------------------------


def fleet_cost(
    objective: str,
    lat: jax.Array,
    lon: jax.Array,
    mission: Mission,
    radius_km: float,
    remaining: jax.Array,
    visited: jax.Array,
) -> jax.Array:
    """The objective of a run of moves. lat and lon are (N, T), each balloon's position after each move.

    `remaining` cuts the moves past the end of the episode, as in planner.expected_cost. `visited` (K,) marks
    targets already visited before this run of moves began, which then cost nothing; only the targets
    objective reads it, and the final score passes none.
    """
    live = jnp.arange(lat.shape[1]) < remaining
    if objective == "reach":
        d = distance_km(lat, lon, mission.targets_lat[0], mission.targets_lon[0])  # (N, T)
        keep = jax.vmap(lambda one: counted(one, radius_km, remaining))(d)
        return jnp.mean(jnp.sum(jnp.where(keep, d, 0.0), axis=1))
    if objective == "escape":
        d = distance_km(lat, lon, mission.starts.lat[:, None], mission.starts.lon[:, None])
        return -jnp.mean(jnp.sum(jnp.where(live, d, 0.0), axis=1))
    if objective == "station":
        d = distance_km(lat, lon, mission.targets_lat[0], mission.targets_lon[0])
        outside = jnp.min(jnp.maximum(d - radius_km, 0.0), axis=0)  # (T,) the nearest balloon's
        return jnp.sum(jnp.where(live, outside, 0.0))
    if objective == "targets":
        d = jnp.min(distance_km(lat[None], lon[None], mission.targets_lat[:, None, None], mission.targets_lon[:, None, None]), axis=1)  # (K, T)
        keep = jax.vmap(lambda one: counted(one, radius_km, remaining))(d) & ~visited[:, None]
        return jnp.sum(jnp.where(keep, d, 0.0))
    raise ValueError(f"no such objective: {objective}")


def reached(state: State, mission: Mission, radius_km: float) -> jax.Array:
    """(K,) which targets some balloon is within radius_km of right now"""
    d = distance_km(state.lat[None], state.lon[None], mission.targets_lat[:, None], mission.targets_lon[:, None])  # (K, N)
    return jnp.any(d < radius_km, axis=1)


# --- the planner ------------------------------------------------------------------------------------------------


def _fly_plan(balloon: BalloonConfig, wind: WindField, state: State, climbs_km: jax.Array) -> State:
    """Each balloon's own plan rolled out in one wind. climbs_km (N, H); each field of the result (N, H)"""
    return jax.vmap(lambda s, c: rollout(wind, s, c, balloon.step_hours, balloon.substeps))(state, climbs_km)


def expected_cost(
    balloon: BalloonConfig,
    planner: PlannerConfig,
    objective: str,
    a: jax.Array,
    winds: WindField,
    state: State,
    mission: Mission,
    remaining: jax.Array,
    radius_km: float,
    visited: jax.Array,
) -> jax.Array:
    """The fleet's plan a (N, H) scored in each wind of the batch, averaged: planner.expected_cost for N"""

    def in_one_wind(w: WindField) -> jax.Array:
        flown = _fly_plan(balloon, w, state, balloon.climb_cap_km * a)
        return fleet_cost(objective, flown.lat, flown.lon, mission, radius_km, remaining, visited)

    return jnp.mean(jax.vmap(in_one_wind, in_axes=(BATCHED,))(winds))


@functools.partial(jax.jit, static_argnames=("balloon", "planner", "objective"))
def plan(
    balloon: BalloonConfig,
    planner: PlannerConfig,
    objective: str,
    key: jax.Array,
    winds: WindField,
    state: State,
    mission: Mission,
    previous_km: jax.Array,
    remaining: jax.Array,
    radius_km: float,
    visited: jax.Array,
) -> jax.Array:
    """The (N, H) climbs in km with the lowest cost averaged over the winds: planner.plan, for a fleet.

    The pool is each balloon's own pool of M random plans, previous plan and coast, taken row by row: pool
    plan j is every balloon's j-th. The best few are refined jointly, all N * H climbs at once, so a coupled
    objective moves the balloons together. With N = 1 the balloon's pool is drawn from the key itself, as
    planner.plan draws it, so a fleet of one is the single-balloon planner.
    """
    if planner.cost != "summed":
        raise ValueError("the fleet scores the summed objective only")
    n = state.lat.shape[0]
    J = lambda a: expected_cost(balloon, planner, objective, a, winds, state, mission, remaining, radius_km, visited)

    keys = key[None] if n == 1 else jax.random.split(key, n)
    lowest, highest = winds.altitude_km[0], winds.altitude_km[-1]
    each = jax.vmap(lambda k, s, p: candidates(balloon, planner, k, s, p, lowest, highest))(keys, state, previous_km)  # (N, M+2, H)
    pool = jnp.swapaxes(each, 0, 1)  # (M+2, N, H)
    best = pool[jnp.argsort(jax.vmap(J)(pool))[: planner.descend_top]]

    J_of_z = lambda z: J(jnp.tanh(z / 2.0))
    refine = _lbfgs if planner.optimizer == "lbfgs" else _normalised_descent
    start = 2.0 * jnp.arctanh(jnp.clip(best, -0.999, 0.999))
    refined = jnp.tanh(jax.vmap(lambda z: refine(planner, J_of_z, z))(start) / 2.0)
    return balloon.climb_cap_km * refined[jnp.argmin(jax.vmap(J)(refined))]


# --- the controllers --------------------------------------------------------------------------------------------


def _n_k(mission: Mission) -> tuple[int, int]:
    return mission.starts.lat.shape[0], mission.targets_lat.shape[0]


def fixed_plan(
    balloon: BalloonConfig,
    planner: PlannerConfig,
    belief: Any,
    objective: str,
    mission: Mission,
    n_moves: int,
    n_winds: int,
    radius_km: float,
) -> Controller:
    """Plans every balloon's whole episode once, on n_winds draws from the belief, before any reading. The floor"""
    n, k = _n_k(mission)
    whole_episode = dataclasses.replace(planner, horizon=n_moves)

    def begin(key: jax.Array, state: State) -> jax.Array:
        k_winds, k_plan = jax.random.split(key)
        winds = belief.winds(k_winds, n_winds, Readings.empty(1))
        return plan(balloon, whole_episode, objective, k_plan, winds, state, mission, jnp.zeros((n, n_moves)), jnp.array(n_moves), radius_km, jnp.zeros(k, bool))

    def act(memory: jax.Array, key: jax.Array, state: State, move: jax.Array, reading: jax.Array):
        return memory[:, move], memory

    return Controller(begin, act)


class FleetMemory(NamedTuple):
    previous: jax.Array  # (N, horizon) the last plan shifted by one move
    readings: Readings  # every balloon's readings so far, shared
    visited: jax.Array  # (K,) targets some balloon has already reached


def mpc(
    balloon: BalloonConfig,
    planner: PlannerConfig,
    belief: Any,
    objective: str,
    mission: Mission,
    n_winds: int,
    n_moves: int,
    radius_km: float,
    learns: bool = True,
) -> Controller:
    """controllers.mpc for a fleet: every move each balloon reads the wind where it is and the readings all go
    into one shared belief; the fleet replans jointly from where every balloon really is; each balloon flies
    the first climb of its row. With `learns` false the readings are taken and set aside, as in mpc_blind."""
    n, k = _n_k(mission)

    def begin(key: jax.Array, state: State) -> FleetMemory:
        return FleetMemory(jnp.zeros((n, planner.horizon)), Readings.empty(n_moves * n), jnp.zeros(k, bool))

    def act(memory: FleetMemory, key: jax.Array, state: State, move: jax.Array, reading: jax.Array):
        readings = memory.readings
        for i in range(n):  # every balloon's reading, into the one belief
            readings = readings.add(State(state.lat[i], state.lon[i], state.altitude_km[i], state.hours[i]), reading[i])
        given = readings if learns else memory.readings
        visited = memory.visited | reached(state, mission, radius_km)
        k_winds, k_plan = jax.random.split(key)
        climbs = plan(balloon, planner, objective, k_plan, belief.winds(k_winds, n_winds, given), state, mission,
                      memory.previous, n_moves - move, radius_km, visited)
        shifted = jnp.concatenate([climbs[:, 1:], jnp.zeros((n, 1))], axis=1)
        return climbs[:, 0], FleetMemory(shifted, readings, visited)

    return Controller(begin, act)


def hindsight(balloon: BalloonConfig, planner: PlannerConfig, truth: WindField, objective: str, mission: Mission, n_moves: int, radius_km: float) -> Controller:
    """Knows the true wind and replans the whole fleet every move over the whole episode. The ceiling"""
    return mpc(balloon, dataclasses.replace(planner, horizon=n_moves), Truth(truth), objective, mission, 0, n_moves, radius_km)


# --- the flight -------------------------------------------------------------------------------------------------


class FleetFlight(NamedTuple):
    climbs_km: jax.Array  # (N, T)
    visited: State  # each field (N, T): where each balloon was after each move
    cost_km: jax.Array  # the objective, as fleet_cost scores it over the whole episode
    complete: jax.Array  # the task done: reach every balloon arrived; targets every one visited; station held every move
    complete_move: jax.Array  # the move it was done on, from 1 (station: the last move); 0 if never, and for escape
    arrived_count: jax.Array  # reach: balloons that came within the radius of the goal
    visited_count: jax.Array  # targets: targets some balloon came within the radius of
    covered_fraction: jax.Array  # station: the fraction of moves some balloon was within the radius
    summed_km: jax.Array  # distance to the objective's point (the first target; escape: each start), summed over every move, mean over balloons
    final_km: jax.Array  # the same after the last move, mean over balloons
    closest_km: jax.Array  # the nearest any balloon came to it
    closest_move: jax.Array  # after which move, from 1
    farthest_km: jax.Array  # the farthest a balloon got from its own start, mean over balloons
    escape_final_km: jax.Array  # how far from its own start a balloon finished, mean over balloons


def _first(hit: jax.Array) -> jax.Array:
    """The first move (from 1) a (T,) mask is true, 0 if never"""
    return jnp.where(jnp.any(hit), jnp.argmax(hit) + 1, 0)


def fly(
    balloon: BalloonConfig,
    controller: Controller,
    truth: WindField,
    mission: Mission,
    objective: str,
    n_moves: int,
    key: jax.Array,
    radius_km: float,
) -> FleetFlight:
    """episode.fly for a fleet: each move every balloon reads the true wind where it is, the controller picks
    every balloon's climb, and each balloon moves in the true wind, held inside the record's latitudes"""
    k_begin, k_moves = jax.random.split(key)
    move_one = jax.vmap(lambda s, c: clip(truth, step(truth, s, c, balloon.step_hours, balloon.substeps)))

    def one_move(carry, inputs):
        state, memory = carry
        move, k = inputs
        climb, memory = controller.act(memory, k, state, move, jax.vmap(lambda s: read(truth, s))(state))
        state = move_one(state, climb)
        return (state, memory), (climb, state)

    memory = controller.begin(k_begin, mission.starts)
    moves = (jnp.arange(n_moves), jax.random.split(k_moves, n_moves))
    _, (climbs, flown) = jax.lax.scan(one_move, (mission.starts, memory), moves)
    flown = State(*(field.T for field in flown))  # (T, N) -> (N, T)
    lat, lon = flown.lat, flown.lon
    n, k = _n_k(mission)

    cost = fleet_cost(objective, lat, lon, mission, radius_km, jnp.array(n_moves), jnp.zeros(k, bool))
    to_point = distance_km(lat, lon, mission.targets_lat[0], mission.targets_lon[0])  # (N, T)
    from_start = distance_km(lat, lon, mission.starts.lat[:, None], mission.starts.lon[:, None])
    reference = from_start if objective == "escape" else to_point

    each_arrival = jax.vmap(lambda d: _first(d < radius_km))(to_point)  # (N,)
    to_targets = jnp.min(distance_km(lat[None], lon[None], mission.targets_lat[:, None, None], mission.targets_lon[:, None, None]), axis=1)  # (K, T)
    each_visit = jax.vmap(lambda d: _first(d < radius_km))(to_targets)  # (K,)
    covered = jnp.min(to_point, axis=0) < radius_km  # (T,)

    if objective == "reach":
        complete, move = jnp.all(each_arrival > 0), jnp.max(each_arrival)
    elif objective == "targets":
        complete, move = jnp.all(each_visit > 0), jnp.max(each_visit)
    elif objective == "station":
        complete, move = jnp.all(covered), jnp.array(n_moves)  # held to the last move
    else:
        complete, move = jnp.array(False), jnp.array(0)
    move = jnp.where(complete, move, 0)

    nearest = jnp.min(reference, axis=0)  # (T,)
    return FleetFlight(
        climbs.T,
        flown,
        cost,
        complete,
        move,
        jnp.sum(each_arrival > 0),
        jnp.sum(each_visit > 0),
        jnp.mean(covered),
        jnp.mean(jnp.sum(reference, axis=1)),
        jnp.mean(reference[:, -1]),
        jnp.min(nearest),
        jnp.argmin(nearest) + 1,
        jnp.mean(jnp.max(from_start, axis=1)),
        jnp.mean(from_start[:, -1]),
    )


def fleet_row(flight: FleetFlight, mission: Mission, task: Task, *, controller: str, n_winds: int, seed: int,
              seconds: float, settings: dict, extra: dict | None = None, keep_path: bool = False) -> dict:
    """One row, with the columns results.row writes (so every reader works on it) and the fleet's own.

    The goal-distance columns mean the distance to the objective's point here (the goal, the station, the
    first target; for escape each balloon's own start), averaged over the balloons; see FleetFlight. The
    path columns are one list per balloon when the fleet has more than one.
    """
    columns: dict[str, Any] = {"controller": controller, "n_winds": n_winds, "seed": seed}
    for piece, config in settings.items():
        for field, value in dataclasses.asdict(config).items():
            columns[f"{piece}.{field}"] = value
    n = int(mission.starts.lat.shape[0])
    columns.update(
        start_lat=float(task.start.lat),
        start_lon=float(task.start.lon),
        goal_lat=float(task.goal_lat),
        goal_lon=float(task.goal_lon),
        n_agents=n,
        starts_lat=[float(x) for x in mission.starts.lat],
        starts_lon=[float(x) for x in mission.starts.lon],
        targets_lat=[float(x) for x in mission.targets_lat],
        targets_lon=[float(x) for x in mission.targets_lon],
        cost_km=float(flight.cost_km),
        summed_km=float(flight.summed_km),
        final_km=float(flight.final_km),
        closest_km=float(flight.closest_km),
        closest_move=int(flight.closest_move),
        farthest_km=float(flight.farthest_km),
        escape_final_km=float(flight.escape_final_km),
        arrived=bool(flight.complete),
        arrived_move=int(flight.complete_move),
        arrived_count=int(flight.arrived_count),
        visited_count=int(flight.visited_count),
        covered_fraction=float(flight.covered_fraction),
        seconds=seconds,
    )
    if keep_path:
        path = lambda a: [[round(float(x), 4) for x in one] for one in a]
        columns.update(path_lat=path(flight.visited.lat), path_lon=path(flight.visited.lon), path_altitude_km=path(flight.visited.altitude_km))
        if n == 1:  # one balloon: the flat list a single-balloon row has, so map_rows reads it
            columns.update({key: columns[key][0] for key in ("path_lat", "path_lon", "path_altitude_km")})
    columns.update(extra or {})
    return columns
