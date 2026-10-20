"""The three controllers: hindsight, fixed plan, MPC

A controller decides one climb per move, in two parts:
    begin(key, state)              -> memory           once, before the first move
    act(memory, key, state, move)  -> climb, memory    at every move
Memory is whatever the controller carries from one move to the next: for the open-loop controllers the whole
plan, for MPC its latest plan shifted by one move (the next solve starts from it).
"""

import dataclasses
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from balloon_sim.belief import Belief
from balloon_sim.config import BalloonConfig, PlannerConfig
from balloon_sim.dynamics import State
from balloon_sim.planner import plan
from balloon_sim.wind import WindField


class Controller(NamedTuple):
    begin: Callable[[jax.Array, State], jax.Array]
    act: Callable[[jax.Array, jax.Array, State, jax.Array], tuple[jax.Array, jax.Array]]


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
        return plan(balloon, whole_episode, k_plan, winds_for(k_winds), state, goal_lat, goal_lon, jnp.zeros(n_moves))

    def act(memory: jax.Array, key: jax.Array, state: State, move: jax.Array) -> tuple[jax.Array, jax.Array]:
        return memory[move], memory  # the planned climb for this move; the plan itself never changes

    return Controller(begin, act)


def hindsight(
    balloon: BalloonConfig, planner: PlannerConfig, truth: WindField, goal_lat: jax.Array, goal_lon: jax.Array, n_moves: int
) -> Controller:
    """Knows the true wind: plans the whole episode once, on the truth. The ceiling"""
    truth_alone = truth._replace(uv=truth.uv[None])  # a batch of one wind, the shape the planner takes
    return _open_loop(balloon, planner, lambda key: truth_alone, goal_lat, goal_lon, n_moves)


def fixed_plan(
    balloon: BalloonConfig,
    planner: PlannerConfig,
    belief: Belief,
    goal_lat: jax.Array,
    goal_lon: jax.Array,
    n_moves: int,
    n_winds: int,
) -> Controller:
    """Plans the whole episode once, on n_winds draws from the belief (its mean alone when n_winds is 0). The floor"""
    return _open_loop(balloon, planner, lambda key: belief.winds(key, n_winds), goal_lat, goal_lon, n_moves)


def mpc(
    balloon: BalloonConfig, planner: PlannerConfig, belief: Belief, goal_lat: jax.Array, goal_lon: jax.Array, n_winds: int
) -> Controller:
    """Every move: draw n_winds winds from the belief (its mean alone when n_winds is 0), plan the next
    planner.horizon moves from where the balloon really is, fly the first climb, keep the rest for next time"""

    def begin(key: jax.Array, state: State) -> jax.Array:
        return jnp.zeros(planner.horizon)  # no previous plan yet

    def act(memory: jax.Array, key: jax.Array, state: State, move: jax.Array) -> tuple[jax.Array, jax.Array]:
        k_winds, k_plan = jax.random.split(key)
        climbs = plan(balloon, planner, k_plan, belief.winds(k_winds, n_winds), state, goal_lat, goal_lon, memory)
        shifted = jnp.concatenate([climbs[1:], jnp.zeros(1)])  # the rest of this plan, then coast
        return climbs[0], shifted

    return Controller(begin, act)
