"""The balloon: it drifts with the wind and can only choose its altitude.

Kinematic, like most balloon-control work: the balloon moves at the wind's speed (no inertia or lag), and climbs
or sinks at up to `climb_km_per_hour`, inside the field's altitude band. There is no power, battery or day-night
cycle; a balloon can always change altitude. The default climb, 1.1 km per 3 h, is the project's assumed limit
(unverified against a real balloon).

A command is a number in [-1, 1] per balloon: -1 sinks at the full rate, 0 holds, 1 climbs. The discrete
action space (down, stay, up) maps onto -1, 0, 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from balloongym.wind import WindField, wind_at

EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class BalloonConfig:
    dt_minutes: float = 30.0  # time between decisions
    substeps: int = 3  # Euler steps per decision
    climb_km_per_hour: float = 1.1 / 3.0


class Balloons(NamedTuple):
    """N balloons; every field has shape (N,)"""

    lat: jax.Array
    lon: jax.Array
    altitude_km: jax.Array
    hours: jax.Array


def _drift(field: WindField, lat, lon, altitude_km, hours, dt_s):
    u, v = wind_at(field, lat, lon, altitude_km, hours)
    lat_next = lat + jnp.degrees(v * dt_s / EARTH_RADIUS_M)
    lon_next = (lon + jnp.degrees(u * dt_s / (EARTH_RADIUS_M * jnp.cos(jnp.radians(lat))))) % 360.0
    return lat_next, lon_next


def step(field: WindField, config: BalloonConfig, balloons: Balloons, command: jax.Array) -> Balloons:
    """Every balloon one decision forward: climb by command * rate while drifting, held inside the field."""
    dt_h = config.dt_minutes / 60.0 / config.substeps
    climb = jnp.clip(command, -1.0, 1.0) * config.climb_km_per_hour * dt_h
    low, high = field.altitude_km[0], field.altitude_km[-1]

    def one(b: Balloons, _):
        lat, lon = jax.vmap(lambda *p: _drift(field, *p, dt_h * 3600.0))(b.lat, b.lon, b.altitude_km, b.hours)
        lat = jnp.clip(lat, field.lat[0], field.lat[-1])
        lon = jnp.where(field.is_global, lon, jnp.clip(lon, field.lon[0], field.lon[-1]))
        return Balloons(lat, lon, jnp.clip(b.altitude_km + climb, low, high), b.hours + dt_h), None

    balloons, _ = jax.lax.scan(one, balloons, None, length=config.substeps)
    return balloons


def distance_km(lat1, lon1, lat2, lon2) -> jax.Array:
    """Great-circle distance (haversine)"""
    p1, p2 = jnp.radians(lat1), jnp.radians(lat2)
    h = jnp.sin((p2 - p1) / 2) ** 2 + jnp.cos(p1) * jnp.cos(p2) * jnp.sin(jnp.radians(lon2 - lon1) / 2) ** 2
    return 2.0 * EARTH_RADIUS_M / 1000.0 * jnp.arcsin(jnp.sqrt(jnp.clip(h, 0.0, 1.0) + 1e-12))


def offset_km(lat0, lon0, lat1, lon1) -> jax.Array:
    """(east, north) km from point 0 to point 1, flat about point 0. Shape (..., 2)"""
    dlon = (lon1 - lon0 + 180.0) % 360.0 - 180.0
    r = EARTH_RADIUS_M / 1000.0
    return jnp.stack([r * jnp.radians(dlon) * jnp.cos(jnp.radians(lat0)), r * jnp.radians(lat1 - lat0)], -1)


def destination(lat, lon, bearing, distance) -> tuple[jax.Array, jax.Array]:
    """The point `distance` km from (lat, lon) along `bearing` (radians clockwise from north)"""
    d = distance / (EARTH_RADIUS_M / 1000.0)
    p1, l1 = jnp.radians(lat), jnp.radians(lon)
    p2 = jnp.arcsin(jnp.sin(p1) * jnp.cos(d) + jnp.cos(p1) * jnp.sin(d) * jnp.cos(bearing))
    l2 = l1 + jnp.arctan2(jnp.sin(bearing) * jnp.sin(d) * jnp.cos(p1), jnp.cos(d) - jnp.sin(p1) * jnp.sin(p2))
    return jnp.degrees(p2), jnp.degrees(l2) % 360.0
