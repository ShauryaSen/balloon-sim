"""The three controllers: hindsight, fixed plan, MPC

A controller decides one climb per move, in two parts:
    begin(key, state)                       -> memory           once, before the first move
    act(memory, key, state, move, reading)  -> climb, memory    at every move
`reading` is the wind at the balloon's own point at that move. Memory is whatever the controller carries from
one move to the next: for the fixed plan the whole plan; for MPC (and hindsight, which is MPC on the truth) its
latest plan shifted by one move (the next solve starts from it) and every reading so far.
"""

import dataclasses
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from balloon_sim.belief import Belief, Readings, Truth
from balloon_sim.config import BalloonConfig, PlannerConfig
from balloon_sim.dynamics import State
from balloon_sim.planner import plan
from balloon_sim.wind import WindField


class Controller(NamedTuple):
    begin: Callable[[jax.Array, State], Any]
    act: Callable[[Any, jax.Array, State, jax.Array, jax.Array], tuple[jax.Array, Any]]


def _open_loop(
    balloon: BalloonConfig,
    planner: PlannerConfig,
    winds_for: Callable[[jax.Array], WindField],
    goal_lat: jax.Array,
    goal_lon: jax.Array,
    n_moves: int,
) -> Controller:
    """Plans all n_moves climbs once, on the winds winds_for gives, then flies them without looking"""
    whole_episode = dataclasses.replace(planner, horizon=n_moves)

    def begin(key: jax.Array, state: State) -> jax.Array:
        k_winds, k_plan = jax.random.split(key)
        return plan(balloon, whole_episode, k_plan, winds_for(k_winds), state, goal_lat, goal_lon, jnp.zeros(n_moves), jnp.array(n_moves))

    def act(memory: jax.Array, key: jax.Array, state: State, move: jax.Array, reading: jax.Array) -> tuple[jax.Array, jax.Array]:
        return memory[move], memory  # the planned climb for this move; the plan itself never changes

    return Controller(begin, act)


def fixed_plan(
    balloon: BalloonConfig,
    planner: PlannerConfig,
    belief: Belief,
    goal_lat: jax.Array,
    goal_lon: jax.Array,
    n_moves: int,
    n_winds: int,
) -> Controller:
    """Plans the whole episode once, on n_winds draws from the belief (its mean alone when n_winds is 0), before
    any reading is taken. The floor"""
    nothing_read = Readings.empty(1)
    return _open_loop(balloon, planner, lambda key: belief.winds(key, n_winds, nothing_read), goal_lat, goal_lon, n_moves)


class MPCMemory(NamedTuple):
    previous: jax.Array  # (horizon,) the last plan shifted by one move
    readings: Readings  # everything read so far


def mpc(
    balloon: BalloonConfig,
    planner: PlannerConfig,
    belief: Belief,
    goal_lat: jax.Array,
    goal_lon: jax.Array,
    n_winds: int,
    n_moves: int,
) -> Controller:
    """Every move: read the wind here and give it to the belief, draw n_winds fresh winds from the belief (its
    mean alone when n_winds is 0), plan the next planner.horizon moves from where the balloon really is (only
    the moves left in the episode count), fly the first climb, keep the rest of the plan as next move's
    starting point"""

    def begin(key: jax.Array, state: State) -> MPCMemory:
        return MPCMemory(jnp.zeros(planner.horizon), Readings.empty(n_moves))  # no plan and nothing read yet

    def act(memory: MPCMemory, key: jax.Array, state: State, move: jax.Array, reading: jax.Array) -> tuple[jax.Array, MPCMemory]:
        readings = memory.readings.add(state, reading)
        k_winds, k_plan = jax.random.split(key)
        climbs = plan(balloon, planner, k_plan, belief.winds(k_winds, n_winds, readings), state, goal_lat, goal_lon, memory.previous, n_moves - move)
        shifted = jnp.concatenate([climbs[1:], jnp.zeros(1)])  # the rest of this plan, then coast
        return climbs[0], MPCMemory(shifted, readings)

    return Controller(begin, act)


def hindsight(
    balloon: BalloonConfig, planner: PlannerConfig, truth: WindField, goal_lat: jax.Array, goal_lon: jax.Array, n_moves: int
) -> Controller:
    """Knows the true wind: replans every move on the truth over the whole episode. The ceiling.

    Replanning rather than one plan flown blind, because the dynamics are deterministic given the wind, so the
    two have the same optimum, and re-solving from the true position every move is the stronger search: one
    solve from a random pool left MPC on a perfect forecast beating it by a tenth. What is left is still only
    the best plan our optimiser found, which DESIGN.md names as hindsight's one approximation"""
    return mpc(balloon, dataclasses.replace(planner, horizon=n_moves), Truth(truth), goal_lat, goal_lon, 0, n_moves)
