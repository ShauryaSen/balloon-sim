"""Myles's FOMPC planner: candidate plans, scoring, and refinement by a chosen optimiser

After Pasetsky, Lin, Guo and Dean (arXiv 2511.07761), section V.6 and algorithm 1:
    1. make a pool of plans: random ones, the previous plan, and coast
    2. score every plan once; keep the best few
    3. refine those by gradient descent
    4. return the refined plan with the lowest cost

Notation follows the bounds note: a = (a_1, ..., a_H) is the plan, w is one wind, J(a, w) is the cost of flying a in w.
A plan a is stored as H numbers in [-1, 1], each a fraction of the climb cap; the balloon climbs cap * a.
The optimisers work on z, any real numbers, with a = tanh(z / 2), so every z is a plan inside the cap.
"""

import functools

import jax
import jax.numpy as jnp
import optax

from balloon_sim.config import BalloonConfig, PlannerConfig
from balloon_sim.dynamics import State, counted, distance_km, rollout
from balloon_sim.wind import WindField

BATCHED = WindField(None, None, None, None, 0)
"""A batch of winds shares one grid, so only uv carries the extra leading axis (one entry per wind)"""


def expected_cost(
    balloon: BalloonConfig,
    planner: PlannerConfig,
    a: jax.Array,
    winds: WindField,
    state: State,
    goal_lat: jax.Array,
    goal_lon: jax.Array,
    remaining: jax.Array,
    arrival_km: float = 0.0,
) -> jax.Array:
    """J(a, w) for each wind w in the batch, averaged. Only moves the episode is still running for count: the
    rest of the horizon would be flown after the episode ends, so it must not steer the plan. Under the
    arrival rule (arrival_km > 0) the episode also ends on arrival, so the planner is minimising the same
    thing the simulator scores it on; with arrival_km = 0 this is the plain sum over the remaining moves.

    The mask depends on the plan, since where a plan arrives decides what it is charged for, so the objective
    is piecewise smooth rather than smooth. Inside a piece the gradient is the one L-BFGS wants, and the pool
    of 102 starting plans is what carries the search across the pieces.
    """
    last = jnp.clip(remaining - 1, 0, planner.horizon - 1)  # the move the episode ends on

    def in_one_wind(w: WindField) -> jax.Array:
        visited = rollout(w, state, balloon.climb_cap_km * a, balloon.step_hours, balloon.substeps)
        distance = distance_km(visited.lat, visited.lon, goal_lat, goal_lon)  # (H,) after each move
        keep = counted(distance, arrival_km, remaining)
        return jnp.sum(jnp.where(keep, distance, 0.0)) if planner.cost == "summed" else distance[last]

    return jnp.mean(jax.vmap(in_one_wind, in_axes=(BATCHED,))(winds))


def candidates(
    balloon: BalloonConfig, planner: PlannerConfig, key: jax.Array, state: State, previous_km: jax.Array, lowest_km: jax.Array, highest_km: jax.Array
) -> jax.Array:
    """The pool: M random plans, then the previous plan, then coast. Shape (M + 2, H), each row a plan a"""
    k_target, k_noise, k_coast = jax.random.split(key, 3)
    H, M, cap, p = planner.horizon, planner.candidates, balloon.climb_cap_km, planner.perturbation

    # a random plan picks a target altitude it could reach in H moves, inside the record's levels,
    # climbs toward it at nearly full rate until it would get there, then drifts with small random climbs
    low = jnp.maximum(state.altitude_km - cap * H, lowest_km)
    high = jnp.minimum(state.altitude_km + cap * H, highest_km)
    gap = jax.random.uniform(k_target, (M,), minval=low, maxval=high) - state.altitude_km
    moves_needed = jnp.ceil(jnp.abs(gap) / cap)  # (M,)
    heading = 0.95 * jnp.sign(gap)[:, None]  # (M, 1)
    drifting = jax.random.uniform(k_noise, (M, H), minval=-p, maxval=p)
    random = jnp.where(jnp.arange(H)[None, :] < moves_needed[:, None], heading, drifting)

    previous = jnp.clip(previous_km / cap, -1.0, 1.0)[None, :]  # last move's plan, already shifted by the caller
    coast = jax.random.uniform(k_coast, (1, H), minval=-p, maxval=p)  # small random climbs only
    return jnp.concatenate([random, previous, coast])


def _best_so_far(loss, z: jax.Array, improve, iterations: int, memory=None) -> jax.Array:
    """Runs `improve` for `iterations` steps from z and returns the best iterate seen, not the last: a line
    search or a fixed step can wander uphill, and a refinement must never hand back a worse plan than it was given"""

    def one(carry, _):
        z, memory, best_z, best_cost = carry
        z, memory = improve(z, memory)
        cost = loss(z)
        better = cost < best_cost
        return (z, memory, jnp.where(better, z, best_z), jnp.where(better, cost, best_cost)), None

    (_, _, best_z, _), _ = jax.lax.scan(one, (z, memory, z, loss(z)), None, length=iterations)
    return best_z


def _normalised_descent(planner: PlannerConfig, loss, z: jax.Array) -> jax.Array:
    """Myles's refinement: a step of fixed length against the gradient, z <- z - eta g / |g|, held once |g| is tiny"""

    def improve(z: jax.Array, memory) -> tuple[jax.Array, None]:
        g = jax.grad(loss)(z)
        norm = jnp.linalg.norm(g)
        moved = z - planner.step_size * g / (norm + 1e-12)
        return jnp.where(norm < planner.tolerance, z, moved), None

    return _best_so_far(loss, z, improve, planner.iterations)


def _lbfgs(planner: PlannerConfig, loss, z: jax.Array) -> jax.Array:
    """L-BFGS (optax's, with its line search), held once |g| is tiny"""
    solver = optax.lbfgs()
    value_and_grad = optax.value_and_grad_from_state(loss)

    def improve(z: jax.Array, memory):
        value, g = value_and_grad(z, state=memory)
        updates, next_memory = solver.update(g, memory, z, value=value, grad=g, value_fn=loss)
        done = jnp.linalg.norm(g) < planner.tolerance
        keep = lambda old, new: jnp.where(done, old, new)
        return jax.tree.map(keep, (z, memory), (optax.apply_updates(z, updates), next_memory))

    return _best_so_far(loss, z, improve, planner.iterations, solver.init(z))


@functools.partial(jax.jit, static_argnames=("balloon", "planner"))
def plan(
    balloon: BalloonConfig,
    planner: PlannerConfig,
    key: jax.Array,
    winds: WindField,
    state: State,
    goal_lat: jax.Array,
    goal_lon: jax.Array,
    previous_km: jax.Array,
    remaining: jax.Array,
    arrival_km: float = 0.0,
) -> jax.Array:
    """The H climbs (km) with the lowest cost averaged over the batch of winds, counting only the moves the
    episode is still running for: those left in it, and those before the plan arrives"""
    J = lambda a: expected_cost(balloon, planner, a, winds, state, goal_lat, goal_lon, remaining, arrival_km)

    # 1-2: the pool, every plan scored once, the best few kept
    pool = candidates(balloon, planner, key, state, previous_km, winds.altitude_km[0], winds.altitude_km[-1])
    best = pool[jnp.argsort(jax.vmap(J)(pool))[: planner.descend_top]]  # (k, H)

    # 3: refine in z, where a = tanh(z / 2), so any z gives a climb inside the cap and the descent needs no limits
    J_of_z = lambda z: J(jnp.tanh(z / 2.0))
    refine = _lbfgs if planner.optimizer == "lbfgs" else _normalised_descent
    start = 2.0 * jnp.arctanh(jnp.clip(best, -0.999, 0.999))  # back from a to z
    refined = jnp.tanh(jax.vmap(lambda z: refine(planner, J_of_z, z))(start) / 2.0)

    # 4: the refined plan with the lowest cost, as climbs in km
    return balloon.climb_cap_km * refined[jnp.argmin(jax.vmap(J)(refined))]
