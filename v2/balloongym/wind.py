"""Wind fields: a grid of (u, v) over time, altitude, latitude and longitude, read at any point.

The truth a balloon flies in is a `WindField`. It comes from an ERA5 record (`load_record`) or, so that
everything runs with no download, from `synthetic_wind`, a smooth random field with winds that change direction
with altitude.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class WindField(NamedTuple):
    """Wind on a grid. Every axis is ascending; a global field repeats its first longitude at +360."""

    hours: jax.Array  # (T,) hours since the field starts
    altitude_km: jax.Array  # (A,)
    lat: jax.Array  # (Y,) degrees north
    lon: jax.Array  # (X,) degrees east, 0 to 360
    uv: jax.Array  # (T, A, Y, X, 2) eastward and northward wind, m/s

    @property
    def is_global(self) -> jax.Array:
        """Whether longitude goes all the way round (a traced bool, so it works inside jit)"""
        return jnp.abs(self.lon[-1] - self.lon[0] - 360.0) < 1e-3


def load_record(path: str) -> WindField:
    """A field from an .npz with arrays u, v (T, A, Y, X), hours, altitude_km, latitude, longitude.

    This is the format of the ERA5 records the project builds. Axes are made ascending, and a global
    longitude axis gets its seam column.
    """
    record = np.load(path)
    uv = np.stack([record["u"], record["v"]], axis=-1)
    axes = [record["hours"], record["altitude_km"], record["latitude"], record["longitude"] % 360.0]
    for i, axis in enumerate(axes):
        if axis[0] > axis[-1]:
            axes[i] = axis[::-1]
            uv = np.flip(uv, axis=i)
    hours, altitude_km, lat, lon = axes
    step = float(lon[1] - lon[0])
    if abs(float(lon[-1] - lon[0]) + step - 360.0) < 1e-3:
        lon = np.append(lon, lon[0] + 360.0)
        uv = np.concatenate([uv, uv[:, :, :, :1]], axis=3)
    return WindField(*(jnp.asarray(a, dtype=jnp.float32) for a in (hours, altitude_km, lat, lon, uv)))


def _bracket(axis: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The grid index just below x and the fraction of the way to the next one. Held at the edges."""
    i = jnp.clip(jnp.searchsorted(axis, x) - 1, 0, axis.shape[0] - 2)
    weight = (x - axis[i]) / (axis[i + 1] - axis[i])
    return i, jnp.clip(weight, 0.0, 1.0)


def wind_at(field: WindField, lat, lon, altitude_km, hours) -> jax.Array:
    """(u, v) in m/s at one point, linear in all four axes. Differentiable in the point."""
    t, wt = _bracket(field.hours, hours)
    a, wa = _bracket(field.altitude_km, altitude_km)
    y, wy = _bracket(field.lat, lat)
    x, wx = _bracket(field.lon, lon % 360.0)
    corners = jax.lax.dynamic_slice(field.uv, (t, a, y, x, jnp.zeros_like(t)), (2, 2, 2, 2, 2))
    pair = lambda w: jnp.stack([1.0 - w, w])
    return jnp.einsum("i,j,k,l,ijklc->c", pair(wt), pair(wa), pair(wy), pair(wx), corners)


def synthetic_wind(
    seed: int = 0,
    hours: float = 240.0,
    step_hours: float = 3.0,
    levels: int = 8,
    altitude_km: tuple[float, float] = (15.0, 21.0),
    lat: tuple[float, float] = (-50.0, 50.0),
    resolution_deg: float = 2.5,
    variability_ms: float = 6.0,
) -> WindField:
    """A global, smooth, time-varying wind field for tests and quickstarts. Not a model of real weather.

    A mean zonal wind that turns from westerly at the bottom of the band to easterly at the top (so altitude
    choice matters, as in the real stratosphere), weakening toward the poles, plus a random field with about
    1,000 km, 1.5 km and 24 h correlation lengths. Deterministic in `seed`.
    """
    t = np.arange(0.0, hours + 1e-6, step_hours)
    alt = np.linspace(*altitude_km, levels)
    la = np.arange(lat[0], lat[1] + 1e-6, resolution_deg)
    lo = np.arange(0.0, 360.0, resolution_deg)

    s = (alt - alt[0]) / (alt[-1] - alt[0])  # 0 at the bottom, 1 at the top
    mean_u = (12.0 - 22.0 * s)[:, None] * np.cos(np.radians(la))[None, :]  # (A, Y)

    rng = np.random.default_rng(seed)
    n = 96
    radius = 6371.0
    phi, lam = np.radians(la)[:, None], np.radians(lo)[None, :]
    xyz = np.stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi) * np.ones_like(lam)], -1) * radius
    omega_xyz = rng.standard_normal((n, 3)) / 1000.0
    omega_alt = rng.standard_normal(n) / 1.5
    omega_t = rng.standard_normal(n) / 24.0
    phase = rng.uniform(0, 2 * np.pi, n)
    weights = rng.standard_normal((2, n)) * variability_ms * np.sqrt(2.0 / n)
    place = np.einsum("yxk,nk->yxn", xyz, omega_xyz)  # (Y, X, n)
    rest = t[:, None, None] * omega_t + alt[None, :, None] * omega_alt + phase  # (T, A, n)
    # cos(rest + place) = cos rest cos place - sin rest sin place, so the 5-D angle is never built
    noise = (np.einsum("tan,yxn,cn->tayxc", np.cos(rest), np.cos(place), weights)
             - np.einsum("tan,yxn,cn->tayxc", np.sin(rest), np.sin(place), weights))

    uv = noise
    uv[..., 0] += mean_u[None, :, :, None]
    lo = np.append(lo, 360.0)
    uv = np.concatenate([uv, uv[:, :, :, :1]], axis=3)
    return WindField(*(jnp.asarray(a, dtype=jnp.float32) for a in (t, alt, la, lo, uv)))
