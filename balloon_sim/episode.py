"""Fly one controller in the true wind and return its trajectory and metrics"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from balloon_sim.belief import read
from balloon_sim.config import BalloonConfig
from balloon_sim.controllers import Controller
from balloon_sim.dynamics import State, arrival, clip, episode_cost, escape_cost, farthest, final_cost, step, summed_cost
from balloon_sim.wind import WindField


class Flight(NamedTuple):
    climbs_km: jax.Array  # (n_moves,) what the controller chose
    visited: State  # each field (n_moves,): where the balloon was after each move
    summed_km: jax.Array  # distance to the goal after each move, summed over every move
    final_km: jax.Array  # distance to the goal after the last move
    cost_km: jax.Array  # the objective: summed until the balloon arrives, nothing after (arrival_km = 0: summed_km)
    arrived: jax.Array  # whether it ever came within arrival_km
    arrived_move: jax.Array  # the move it first did, from 1; 0 if it never did
    farthest_km: jax.Array  # the farthest it ever got from its start
    escape_final_km: jax.Array  # how far from its start it finished


def fly(
    balloon: BalloonConfig,
    controller: Controller,
    truth: WindField,
    start: State,
    goal_lat: jax.Array,
    goal_lon: jax.Array,
    n_moves: int,
    key: jax.Array,
    arrival_km: float = 0.0,
    away: bool = False,
) -> Flight:
    """The simulator: each move, the balloon reads the true wind where it is, the controller picks a climb, the
    balloon moves in the true wind, and is held inside the record's latitudes.

    Every move is flown and recorded whatever happens, so a trajectory is always the same length; the arrival
    rule is applied to the cost afterwards rather than by stopping the scan, which keeps the shapes fixed for
    JAX. `summed_km` and `final_km` are kept as they were, so a run under the arrival rule can still be read
    against the sweeps that came before it.

    With `away` the objective is the other one (decision 45): get as far from the start as possible, and
    `cost_km` holds minus the summed distance from the start, so lower is still better. The goal-based
    columns are still filled in, since the goal is passed as the start there and they then say how far the
    balloon ranged rather than how near it came.
    """
    k_begin, k_moves = jax.random.split(key)

    def one_move(carry: tuple[State, Any], inputs: tuple[jax.Array, jax.Array]):
        state, memory = carry
        move, k = inputs
        climb, memory = controller.act(memory, k, state, move, read(truth, state))
        state = clip(truth, step(truth, state, climb, balloon.step_hours, balloon.substeps))
        return (state, memory), (climb, state)

    memory = controller.begin(k_begin, start)
    moves = (jnp.arange(n_moves), jax.random.split(k_moves, n_moves))  # each move gets its index and its own random key
    _, (climbs, visited) = jax.lax.scan(one_move, (start, memory), moves)
    ever, first = arrival(visited, goal_lat, goal_lon, arrival_km)
    objective = escape_cost(visited, start.lat, start.lon) if away else episode_cost(visited, goal_lat, goal_lon, arrival_km)
    return Flight(
        climbs,
        visited,
        summed_cost(visited, goal_lat, goal_lon),
        final_cost(visited, goal_lat, goal_lon),
        objective,
        ever,
        first,
        farthest(visited, start.lat, start.lon),
        final_cost(visited, start.lat, start.lon),
    )
