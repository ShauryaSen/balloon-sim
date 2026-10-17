"""The three controllers: hindsight, fixed plan, MPC

A controller decides one climb per move, in two parts:
    begin(key, state)              -> memory           once, before the first move
    act(memory, key, state, move)  -> climb, memory    at every move
Memory is whatever the controller carries from one move to the next: for hindsight, the whole plan.
"""

import dataclasses
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from balloon_sim.config import BalloonConfig, PlannerConfig
from balloon_sim.dynamics import State
from balloon_sim.planner import plan
from balloon_sim.wind import WindField


class Controller(NamedTuple):
    begin: Callable[[jax.Array, State], jax.Array]
    act: Callable[[jax.Array, jax.Array, State, jax.Array], tuple[jax.Array, jax.Array]]


def hindsight(
    balloon: BalloonConfig, planner: PlannerConfig, truth: WindField, goal_lat: jax.Array, goal_lon: jax.Array, n_moves: int
) -> Controller:
    """Knows the true wind: plans all n_moves climbs once, on the truth, then flies them without looking"""
    whole_episode = dataclasses.replace(planner, horizon=n_moves)
    truth_alone = truth._replace(uv=truth.uv[None])  # a batch of one wind, the shape the planner takes

    def begin(key: jax.Array, state: State) -> jax.Array:
        return plan(balloon, whole_episode, key, truth_alone, state, goal_lat, goal_lon, jnp.zeros(n_moves))

    def act(memory: jax.Array, key: jax.Array, state: State, move: jax.Array) -> tuple[jax.Array, jax.Array]:
        return memory[move], memory  # the planned climb for this move; the plan itself never changes

    return Controller(begin, act)
