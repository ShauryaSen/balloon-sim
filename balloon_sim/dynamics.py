"""Wind in one 3 h step: position and altitude update, a rollout, and the costs"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from balloon_sim.wind import WindField, wind_at

EARTH_RADIUS_M = 6_371_000.0


class State(NamedTuple):
    lat: jax.Array  # degrees
    lon: jax.Array  # degrees, 0 to 360
    altitude_km: jax.Array
    hours: jax.Array  # since the record starts


def step(field: WindField, state: State, climb_km: jax.Array, step_hours: float, substeps: int = 1) -> State:
    """One move: the balloon drifts with the wind for step_hours and climbs climb_km (negative sinks).

    Each substep reads the wind once at its start and moves in a straight line (Euler):
        lat += v dt / R
        lon += u dt / (R cos lat), wrapped round at 360
        altitude += climb_km / substeps, then held inside the record's levels
    Altitude is held here, so the planner and the simulator agree on it exactly; past the floor or ceiling a climb
    has no effect and no gradient. Position is not held here, so the planner's gradient never dies at the box's
    edge; the simulator holds it (see clip).
    """
    dt_s = step_hours * 3600.0 / substeps
    for _ in range(substeps):
        u, v = wind_at(field, state.lat, state.lon, state.altitude_km, state.hours)
        dlat = jnp.degrees(v * dt_s / EARTH_RADIUS_M)
        dlon = jnp.degrees(u * dt_s / (EARTH_RADIUS_M * jnp.cos(jnp.radians(state.lat))))
        state = State(
            state.lat + dlat,
            (state.lon + dlon) % 360.0,  # round the globe
            jnp.clip(state.altitude_km + climb_km / substeps, field.altitude_km[0], field.altitude_km[-1]),
            state.hours + step_hours / substeps,
        )
    return state


def rollout(field: WindField, state: State, climbs_km: jax.Array, step_hours: float, substeps: int = 1) -> State:
    """The states after each of the climbs in turn. Each field of the result has shape (n_steps,)"""

    def one_move(current: State, climb_km: jax.Array) -> tuple[State, State]:
        following = step(field, current, climb_km, step_hours, substeps)
        return following, following  # carry it forward, and record it

    _, visited = jax.lax.scan(one_move, state, climbs_km)
    return visited


def clip(field: WindField, state: State) -> State:
    """The simulator's limit on position: held inside the record's box (altitude is already held by step)"""
    return State(
        jnp.clip(state.lat, field.lat[0], field.lat[-1]),
        jnp.clip(state.lon, field.lon[0], field.lon[-1]),
        state.altitude_km,
        state.hours,
    )


def distance_km(lat1: jax.Array, lon1: jax.Array, lat2: jax.Array, lon2: jax.Array) -> jax.Array:
    """Great-circle distance (haversine).

    A tiny constant inside the square root keeps the gradient finite at zero distance; it adds about 13 m.
    """
    phi1, phi2 = jnp.radians(lat1), jnp.radians(lat2)
    dphi, dlam = phi2 - phi1, jnp.radians(lon2 - lon1)
    h = jnp.sin(dphi / 2) ** 2 + jnp.cos(phi1) * jnp.cos(phi2) * jnp.sin(dlam / 2) ** 2
    return 2.0 * EARTH_RADIUS_M / 1000.0 * jnp.arcsin(jnp.sqrt(h + 1e-12))


def counted(distance: jax.Array, arrival_km: float, remaining: jax.Array) -> jax.Array:
    """Which of a run of moves count toward the cost: a mask over the moves, shape (H,).

    A move counts if the episode is still running when it is flown. The episode ends the first time the
    balloon is within arrival_km of the goal, and the moves after that cost nothing, so arriving sooner is
    worth more than arriving at all. `remaining` cuts the moves a plan would make after the episode is over,
    which matters near the end, where a horizon reaches past the last move.

    The arrival move itself counts. With arrival_km = 0 nothing is ever within it, so every move up to
    `remaining` counts and the cost is the plain sum: the rule before this one, as a special case.
    """
    index = jnp.arange(distance.shape[0])
    arrived = distance < arrival_km
    first = jnp.where(jnp.any(arrived), jnp.argmax(arrived), distance.shape[0] - 1)
    return (index < remaining) & (index <= first)


def episode_cost(visited: State, goal_lat: jax.Array, goal_lon: jax.Array, arrival_km: float) -> jax.Array:
    """J under the arrival rule: distance to the goal after each move, summed until the balloon arrives"""
    distance = distance_km(visited.lat, visited.lon, goal_lat, goal_lon)
    return jnp.sum(jnp.where(counted(distance, arrival_km, distance.shape[0]), distance, 0.0))


def arrival(visited: State, goal_lat: jax.Array, goal_lon: jax.Array, arrival_km: float) -> tuple[jax.Array, jax.Array]:
    """Whether the balloon ever came within arrival_km, and the move it first did (from 1; 0 if it never did)"""
    distance = distance_km(visited.lat, visited.lon, goal_lat, goal_lon)
    arrived = distance < arrival_km
    ever = jnp.any(arrived)
    return ever, jnp.where(ever, jnp.argmax(arrived) + 1, 0)


def summed_cost(visited: State, goal_lat: jax.Array, goal_lon: jax.Array) -> jax.Array:
    """J = distance to the goal after each move, summed over the moves, in km"""
    return jnp.sum(distance_km(visited.lat, visited.lon, goal_lat, goal_lon))


def final_cost(visited: State, goal_lat: jax.Array, goal_lon: jax.Array) -> jax.Array:
    """Distance to the goal after the last move, in km"""
    return distance_km(visited.lat[-1], visited.lon[-1], goal_lat, goal_lon)
