"""A wind field on a (time, altitude, lat, lon) grid, loading one from a record file, and reading it at any point"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class WindField(NamedTuple):
    """Wind on a grid. Truth, forecast and every belief draw are this same type.

    Every coordinate axis is ascending. Longitude runs 0 to 360.
    """

    hours: jax.Array  # (n_t,) hours since the record starts
    altitude_km: jax.Array  # (n_alt,)
    lat: jax.Array  # (n_lat,) degrees
    lon: jax.Array  # (n_lon,) degrees
    uv: jax.Array  # (n_t, n_alt, n_lat, n_lon, 2) eastward and northward wind in m/s


def load_record(path: str) -> WindField:
    """The true wind in a record file (ERA5 or cBottle, same format), with every axis made ascending"""
    record = np.load(path)
    uv = np.stack([record["u"], record["v"]], axis=-1)
    axes = [record["hours"], record["altitude_km"], record["latitude"], record["longitude"]]
    for i, coordinate in enumerate(axes):
        if coordinate[0] > coordinate[-1]:  # ERA5 lists altitude top-down
            axes[i] = coordinate[::-1]
            uv = np.flip(uv, axis=i)
    hours, altitude_km, lat, lon = (jnp.asarray(a, dtype=jnp.float32) for a in axes)
    return WindField(hours, altitude_km, lat, lon, jnp.asarray(uv, dtype=jnp.float32))


def _bracket(coordinate: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The grid index just below x, and how far x sits toward the next one (0 to 1).

    Outside the grid, x is held at the nearest edge, so the wind there is the edge's wind.
    """
    i = jnp.clip(jnp.searchsorted(coordinate, x) - 1, 0, coordinate.shape[0] - 2)
    weight = (x - coordinate[i]) / (coordinate[i + 1] - coordinate[i])
    return i, jnp.clip(weight, 0.0, 1.0)


def wind_at(field: WindField, lat: jax.Array, lon: jax.Array, altitude_km: jax.Array, hours: jax.Array) -> jax.Array:
    """(u, v) in m/s at one point, blended linearly along all four axes from the 16 surrounding grid values.

    Differentiable in the point, which is what lets the planner take gradients through the wind.
    """
    t, wt = _bracket(field.hours, hours)
    a, wa = _bracket(field.altitude_km, altitude_km)
    y, wy = _bracket(field.lat, lat)
    x, wx = _bracket(field.lon, lon)
    corners = jax.lax.dynamic_slice(field.uv, (t, a, y, x, jnp.zeros_like(t)), (2, 2, 2, 2, 2))  # the 16 surrounding values
    pair = lambda w: jnp.stack([1.0 - w, w])  # weights of the lower and upper neighbour
    return jnp.einsum("i,j,k,l,ijklc->c", pair(wt), pair(wa), pair(wy), pair(wx), corners)
