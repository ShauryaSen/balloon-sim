"""Wind for the environments: an ERA5 record, or a synthetic field that needs no download.

A record is an .npz with u, v (time, altitude, lat, lon), hours, altitude_km, latitude, longitude, and
optionally forecast_u, forecast_v: the truth 24 h earlier, a persistence forecast (the project's ERA5 records
carry it). `Wind` holds the truth and, when there is one, that forecast.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from balloongym.sim.wind import WindField, load_record, wind_at

__all__ = ["Wind", "WindField", "load", "persistence", "synthetic", "wind_at"]


class Wind(NamedTuple):
    truth: WindField
    earlier: WindField  # the persistence forecast: the truth `lag` hours earlier at the same place


def persistence(truth: WindField, lag_hours: float = 24.0) -> WindField:
    """The truth lag_hours earlier, frame by frame. Frames before the record has lag_hours of history are held
    at the record's first frame."""
    step = float(truth.hours[1] - truth.hours[0])
    shift = int(round(lag_hours / step))
    uv = jnp.concatenate([jnp.repeat(truth.uv[:1], shift, axis=0), truth.uv[:-shift]]) if shift else truth.uv
    return truth._replace(uv=uv)


def load(path: str) -> Wind:
    """A record's truth, and its own persistence forecast if it carries one (else one built by `persistence`)"""
    truth = load_record(path)
    earlier = load_record(path, "forecast_") if "forecast_u" in np.load(path).files else persistence(truth)
    return Wind(truth, earlier)


def synthetic(
    seed: int = 0,
    hours: float = 240.0,
    step_hours: float = 3.0,
    levels: int = 10,
    altitude_km: tuple[float, float] = (14.3, 20.2),
    lat: tuple[float, float] = (-60.0, 60.0),
    resolution_deg: float = 2.5,
    variability_ms: float = 6.0,
) -> Wind:
    """A global, smooth, time-varying field for tests and getting started. Not a model of real weather.

    A mean zonal wind that turns from westerly at the bottom of the band to easterly at the top (so the choice
    of altitude matters, as in the real stratosphere), weaker toward the poles, plus a random field correlated
    over about 1,000 km, 1.5 km and 24 h. The band matches the project's ERA5 records; frames are 3-hourly.
    Deterministic in `seed`. Its forecast is persistence: the field 24 h earlier.
    """
    t = np.arange(0.0, hours + 1e-6, step_hours)
    alt = np.linspace(*altitude_km, levels)
    la = np.arange(lat[0], lat[1] + 1e-6, resolution_deg)
    lo = np.arange(0.0, 360.0, resolution_deg)

    s = (alt - alt[0]) / (alt[-1] - alt[0])
    mean_u = (12.0 - 22.0 * s)[:, None] * np.cos(np.radians(la))[None, :]  # (A, Y)

    rng = np.random.default_rng(seed)
    n = 96
    phi, lam = np.radians(la)[:, None], np.radians(lo)[None, :]
    xyz = 6371.0 * np.stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi) * np.ones_like(lam)], -1)
    omega_xyz = rng.standard_normal((n, 3)) / 1000.0
    omega_alt = rng.standard_normal(n) / 1.5
    omega_t = rng.standard_normal(n) / 24.0
    phase = rng.uniform(0, 2 * np.pi, n)
    weights = rng.standard_normal((2, n)) * variability_ms * np.sqrt(2.0 / n)
    place = np.einsum("yxk,nk->yxn", xyz, omega_xyz)  # (Y, X, n)
    rest = t[:, None, None] * omega_t + alt[None, :, None] * omega_alt + phase  # (T, A, n)
    # cos(rest + place) = cos rest cos place - sin rest sin place, so the 5-D angle array is never built
    uv = (np.einsum("tan,yxn,cn->tayxc", np.cos(rest), np.cos(place), weights)
          - np.einsum("tan,yxn,cn->tayxc", np.sin(rest), np.sin(place), weights))
    uv[..., 0] += mean_u[None, :, :, None]
    lo = np.append(lo, 360.0)  # the seam column, as load_record adds to a global record
    uv = np.concatenate([uv, uv[:, :, :, :1]], axis=3)
    truth = WindField(*(jnp.asarray(a, dtype=jnp.float32) for a in (t, alt, la, lo, uv)))
    return Wind(truth, persistence(truth))
