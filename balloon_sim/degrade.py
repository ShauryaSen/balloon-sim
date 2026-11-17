"""Copies of the true wind, each wrong in one way: the fidelity experiment (decisions 49 and 53)

A copy is what the controller plans on; the balloon still flies in the truth. Each copy changes one property
and leaves everything else as it was, so a drop in the score has one cause.

Every copy keeps the truth's grid, so the planner, the belief and every compiled function see the same shapes
whatever the copy is. Coarsening an axis keeps every k-th grid point along it and fills the points between by
a straight line from the kept ones. That is exactly what wind_at would read from a grid holding only the kept
points, since wind_at itself blends linearly, so "keep 1 point in k" is the same thing as "a grid k times
coarser".

Two kinds of copy:

    coarser     levels(n)        keep n altitude levels, both ends of the band and evenly spaced between
                frames(k)        keep every k-th frame in time; k = None freezes the first frame
                grid(k)          keep every k-th grid point in latitude and longitude
    wrong       offset(...)      one wind vector taken off everywhere, the same or ramping with altitude
                random(...)      a smooth random field taken off, the pilot's GP or the same at every altitude
                persistence      the truth as it was 24 h earlier, stored in the record itself

Every "wrong" copy is scaled to one size, written as a multiple of a day-old forecast's error: the RMS of the
persistence error in the launch band, the one real error among them.
"""

import dataclasses

import jax
import jax.numpy as jnp

from balloon_sim.belief import error_draws
from balloon_sim.config import BeliefConfig
from balloon_sim.wind import WindField

FLAT_KM = 1e6
"""The altitude lengthscale that makes a random field the same at every altitude: far longer than the band"""


# --- coarser ----------------------------------------------------------------------------------------------------


def _every(n: int, k: int) -> jax.Array:
    """Indices 0, k, 2k, ... of an axis of n points, with the last point always kept so nothing is extrapolated"""
    kept = list(range(0, n, k))
    if kept[-1] != n - 1:
        kept.append(n - 1)
    return jnp.array(kept)


def _resample(uv: jax.Array, axis: int, coordinate: jax.Array, keep: jax.Array) -> jax.Array:
    """uv with every point along `axis` replaced by the straight-line blend of the kept points either side"""
    kept = coordinate[keep]
    i = jnp.clip(jnp.searchsorted(kept, coordinate, side="right") - 1, 0, keep.shape[0] - 2)  # kept point at or below
    weight = (coordinate - kept[i]) / (kept[i + 1] - kept[i])  # 0 on a kept point, 1 on the next one
    shape = [1] * uv.ndim
    shape[axis] = -1
    weight = weight.reshape(shape)
    return jnp.take(uv, keep[i], axis=axis) * (1 - weight) + jnp.take(uv, keep[i + 1], axis=axis) * weight


def levels(truth: WindField, n: int) -> WindField:
    """The truth seen through n of its altitude levels.

    The kept levels always include the top and the bottom of the band and are spaced evenly between, because
    which levels are kept matters as much as how many: two levels at the ends of the band and two adjacent
    ones in the middle are both "two levels", and they lose very different amounts (docs/fidelity-design).
    One level is the wind averaged over the band, the same at every altitude: nothing to steer between.
    """
    n_alt = truth.altitude_km.shape[0]
    if n >= n_alt:
        return truth
    if n == 1:
        return truth._replace(uv=jnp.broadcast_to(jnp.mean(truth.uv, axis=1, keepdims=True), truth.uv.shape))
    keep = jnp.unique(jnp.round(jnp.linspace(0, n_alt - 1, n)).astype(int))
    return truth._replace(uv=_resample(truth.uv, 1, truth.altitude_km, keep))


def frames(truth: WindField, k: int | None) -> WindField:
    """The truth seen through every k-th frame, blended linearly in time between them; k = None is the first
    frame held for ever, a wind model that knows the weather at launch and nothing of how it changes"""
    if k is None:
        return truth._replace(uv=jnp.broadcast_to(truth.uv[:1], truth.uv.shape))
    if k <= 1:
        return truth
    return truth._replace(uv=_resample(truth.uv, 0, truth.hours, _every(truth.hours.shape[0], k)))


def grid(truth: WindField, k: int) -> WindField:
    """The truth seen through every k-th grid point in latitude and in longitude, blended linearly between.

    A global record carries its first longitude again at 360 (wind.load_record), and _every always keeps the
    last point, so the seam stays continuous. When k does not divide the number of longitudes, the last
    interval before the seam is shorter than the rest.
    """
    if k <= 1:
        return truth
    uv = _resample(truth.uv, 2, truth.lat, _every(truth.lat.shape[0], k))
    return truth._replace(uv=_resample(uv, 3, truth.lon, _every(truth.lon.shape[0], k)))


# --- wrong, at one size -----------------------------------------------------------------------------------------


def _ramp(n_alt: int, by_altitude: bool) -> jax.Array:
    """How much of the error each level gets: 1 everywhere, or a straight line from -1 at the bottom of the band
    to 1 at the top. Either way scaled so its mean square over the levels is 1, so the size is the same"""
    if not by_altitude:
        return jnp.ones(n_alt)
    r = jnp.linspace(-1.0, 1.0, n_alt)
    return r / jnp.sqrt(jnp.mean(r**2))


def offset(truth: WindField, key: jax.Array, size: float, by_altitude: bool) -> WindField:
    """The truth with one wind vector taken off, the same at every place and time: a mean-state bias.

    The vector points in a direction drawn from the key, and is sqrt(2) * size long, so its RMS per component
    is exactly `size`. With `by_altitude` it ramps from minus that vector at the bottom of the band to plus it
    at the top: the same size, but now it changes the differences between levels, which a uniform offset
    leaves untouched. That contrast is the sharpest test of the hypothesis that the balloon steers on the
    differences between levels.
    """
    angle = jax.random.uniform(key, maxval=2 * jnp.pi)
    vector = jnp.sqrt(2.0) * size * jnp.stack([jnp.cos(angle), jnp.sin(angle)])  # (2,)
    error = _ramp(truth.altitude_km.shape[0], by_altitude)[:, None] * vector  # (n_alt, 2)
    return truth._replace(uv=truth.uv - error[None, :, None, None, :])


def random(truth: WindField, key: jax.Array, kernel: BeliefConfig, size: float, by_altitude: bool) -> WindField:
    """The truth with one smooth random field taken off, of RMS `size` per component on average.

    With `by_altitude` it is the pilot's error: the GP with the fitted kernel, related across about 1.3 km in
    altitude. Without, the same field with an altitude lengthscale far longer than the band, so every level
    gets the same error at a given place and time and the differences between levels are left alone.
    """
    shaped = dataclasses.replace(kernel, sigma=size, altitude_km=kernel.altitude_km if by_altitude else FLAT_KM)
    return truth._replace(uv=truth.uv - error_draws(shaped, key, truth, 1)[0])


def persistence(truth: WindField, earlier: WindField) -> WindField:
    """The truth as it was 24 h earlier: a persistence forecast, the one error here that real weather made.

    `earlier` is the record's own forecast field (wind.load_record(path, "forecast_")), which the record
    builder fills from the frames 24 h before each of the truth's, so even the first day has one.
    """
    return truth._replace(uv=earlier.uv)


def error_size(truth: WindField, earlier: WindField, lat_band: tuple[float, float]) -> float:
    """The RMS per component of the persistence error over the launch band: 1x a day-old forecast's error.
    The seam column is left out, since it repeats the first"""
    rows = (truth.lat >= lat_band[0]) & (truth.lat <= lat_band[1])
    error = (truth.uv - earlier.uv)[:, :, rows, :-1]
    return float(jnp.sqrt(jnp.mean(error**2)))


# --- by name ----------------------------------------------------------------------------------------------------

COPIES = ("truth", "levels", "frames", "grid", "offset", "offset_altitude", "random", "random_altitude", "persistence")


def copy(name: str, truth: WindField, key: jax.Array, kernel: BeliefConfig, size: float, earlier: WindField | None = None) -> WindField:
    """A copy by its name, as a cell and a row write it: "truth", "levels:9", "frames:4", "frames:frozen",
    "grid:2", "offset", "offset_altitude", "random", "random_altitude" or "persistence". `size` is in m/s and
    only the wrong copies read it"""
    kind, _, setting = name.partition(":")
    if kind == "truth":
        return truth
    if kind == "levels":
        return levels(truth, int(setting))
    if kind == "frames":
        return frames(truth, None if setting == "frozen" else int(setting))
    if kind == "grid":
        return grid(truth, int(setting))
    if kind in ("offset", "offset_altitude"):
        return offset(truth, key, size, by_altitude=kind == "offset_altitude")
    if kind in ("random", "random_altitude"):
        return random(truth, key, kernel, size, by_altitude=kind == "random_altitude")
    if kind == "persistence":
        if earlier is None:
            raise ValueError("persistence needs the record's forecast field")
        return persistence(truth, earlier)
    raise ValueError(f"no such copy: {name}; the copies are {COPIES}")
