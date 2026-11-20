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

from balloongym.sim.belief import error_draws
from balloongym.sim.config import BeliefConfig
from balloongym.sim.wind import WindField

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


# --- forecast errors as people build them (2026-11-18) ---------------------------------------------------------
#
# Each is an error field with the shape a published simulator or our own gives it, scaled to a chosen size: the
# RMS per component over the launch band, as error_size measures persistence. The shape is the model's; the size
# is the experiment's, swept, so a loss can be put down to the shape at equal size.

KM_PER_DEG_LON, KM_PER_DEG_LAT = 111.32, 110.57
"""Flat km from degrees, taken at the equator: the launch band is within 10 degrees of it (cos 10 = 0.985);
BLE also measures its noise in flat km from the station (assumed adequate)"""

# BLE's simplex noise, balloon_learning_environment/env/simplex_wind_noise.py (google/balloon-learning-environment,
# read 2026-11-18): five harmonics per component, each (weight, km east, km north, Pa, hours) between grid points
BLE_SIMPLEX = {
    "u": ((0.1445, 702.269, 2116.987, 2587.802, 245.0), (0.2766, 1483.570, 752.124, 646.208, 16.39),
          (0.2627, 276.810, 147.040, 587.702, 3.836), (0.2137, 10214.525, 1512.216, 965.629, 41.780),
          (0.1025, 181.286, 420.942, 8500.0, 245.0)),
    "v": ((0.2716, 1974.228, 2028.814, 713.697, 26.435), (0.2684, 699.738, 541.845, 632.116, 9.530),
          (0.2348, 217.750, 196.522, 686.825, 3.546), (0.1186, 47.500, 43.048, 66.553, 8.424),
          (0.1066, 3663.291, 232.023, 7499.741, 225.0)),
}

# BLE's GP over the forecast's error, balloon_learning_environment/env/wind_gp.py (read 2026-11-18): Matern 1/2
# over the joint scaled distance of (km east, km north, Pa, hours)
BLE_GP_SCALES = (357.0, 357.0, 326.0, 34560.0 / 3600.0)


def pressure_pa(altitude_km: jax.Array) -> jax.Array:
    """The standard atmosphere's pressure in its isothermal layer, 11 to 20 km: 22,632 Pa at 11 km, scale height
    6.3416 km (ISO 2533). Our band is 14.3 to 20.2 km; the top 0.2 km is past the layer (assumed negligible)"""
    return 22632.0 * jnp.exp(-(jnp.asarray(altitude_km) - 11.0) / 6.3416)


def _to_size(truth: WindField, error: jax.Array, size: float, lat_band: tuple[float, float]) -> jax.Array:
    """error rescaled so its RMS per component over the launch band, seam left out, is `size`"""
    rows = (truth.lat >= lat_band[0]) & (truth.lat <= lat_band[1])
    return error * size / jnp.sqrt(jnp.mean(error[:, :, rows, :-1] ** 2))


def simplex_error(truth: WindField, key: jax.Array) -> jax.Array:
    """BLE's simplex noise on the record's grid, one independent field per component, in BLE's own units.
    Each harmonic is OpenSimplex 4-D noise on (km east, km north, Pa, hours) over its spacings, from a random
    seed and a random offset of the noise grid in [-1, 1), as BLE does; the harmonics are summed with BLE's
    weights and variance adjustment. The library's version differs from BLE's pin, which changes the noise's
    overall size but not its shape, and the size is set afterwards anyway"""
    import numpy as np
    import opensimplex

    x = np.asarray(truth.lon) * KM_PER_DEG_LON
    y = np.asarray(truth.lat) * KM_PER_DEG_LAT
    p = np.asarray(pressure_pa(truth.altitude_km))
    t = np.asarray(truth.hours)
    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
    fields = []
    for component in ("u", "v"):
        harmonics = BLE_SIMPLEX[component]
        total, weight, weight_sq = 0.0, sum(h[0] for h in harmonics), sum(h[0] ** 2 for h in harmonics)
        for w, xs, ys, ps, ts in harmonics:
            opensimplex.seed(int(rng.integers(0, 2**31 - 1)))
            dx, dy, dp, dt = rng.uniform(-1.0, 1.0, 4)
            total = total + w * opensimplex.noise4array(x / xs + dx, y / ys + dy, p / ps + dp, t / ts + dt)  # (t, alt, lat, lon)
        fields.append(total / weight * np.sqrt(weight / weight_sq))
    return jnp.asarray(np.stack(fields, axis=-1), dtype=truth.uv.dtype)


def ble_gp_error(truth: WindField, key: jax.Array, n_features: int = 512) -> jax.Array:
    """A draw of BLE's GP prior on the record's grid, by random Fourier features: Matern 1/2 is exp(-r) on the
    scaled distance r, whose spectrum is a multivariate t with one degree of freedom, so each frequency is a
    Gaussian over sqrt of a chi-squared(1) draw (Rahimi & Recht 2007). The feature sum is separable in the four
    axes, so the grid is a product of four small tables, contracted one axis at a time"""
    axes = (truth.hours, pressure_pa(truth.altitude_km), truth.lat * KM_PER_DEG_LAT, truth.lon * KM_PER_DEG_LON)
    scales = (BLE_GP_SCALES[3], BLE_GP_SCALES[2], BLE_GP_SCALES[1], BLE_GP_SCALES[0])
    fields = []
    for k in jax.random.split(key, 2):
        k_z, k_g, k_b, k_a = jax.random.split(k, 4)
        w = jax.random.normal(k_z, (n_features, 4)) / jnp.sqrt(jax.random.chisquare(k_g, 1.0, (n_features, 1)))
        b = jax.random.uniform(k_b, (n_features,), maxval=2 * jnp.pi)
        a = jax.random.normal(k_a, (n_features,))
        tables = [jnp.exp(1j * (jnp.asarray(axis)[:, None] / scale) * w[None, :, i]) for i, (axis, scale) in enumerate(zip(axes, scales))]
        field = jnp.einsum("tf,pf,yf,xf,f->tpyx", *tables, a * jnp.exp(1j * b), optimize=True).real
        fields.append(field * jnp.sqrt(2.0 / n_features))
    return jnp.stack(fields, axis=-1).astype(truth.uv.dtype)


def fitted_gp_error(truth: WindField, key: jax.Array, kernel: BeliefConfig) -> jax.Array:
    """A draw of our own GP, the kernel fitted to a day-old forecast's error on ERA5 (BeliefConfig): the
    forecast error every sweep before this one was built with"""
    return error_draws(kernel, key, truth, 1)[0]


ERROR_MODELS = ("simplex", "gp_ble", "gp_fitted", "persistence_scaled")


def error_model(name: str, truth: WindField, key: jax.Array, kernel: BeliefConfig, size: float, lat_band: tuple[float, float],
                earlier: WindField | None = None) -> WindField:
    """The truth with one error model's field taken off, at RMS `size` m/s per component over the launch band.
    "persistence_scaled" is the real error, truth minus the truth 24 h earlier, scaled up or down"""
    if name == "simplex":
        error = simplex_error(truth, key)
    elif name == "gp_ble":
        error = ble_gp_error(truth, key)
    elif name == "gp_fitted":
        error = fitted_gp_error(truth, key, kernel)
    elif name == "persistence_scaled":
        if earlier is None:
            raise ValueError("persistence_scaled needs the record's forecast field")
        error = truth.uv - earlier.uv
    else:
        raise ValueError(f"no such error model: {name}; they are {ERROR_MODELS}")
    return truth._replace(uv=truth.uv - _to_size(truth, error, size, lat_band))


def error_size(truth: WindField, earlier: WindField, lat_band: tuple[float, float]) -> float:
    """The RMS per component of the persistence error over the launch band: 1x a day-old forecast's error.
    The seam column is left out, since it repeats the first"""
    rows = (truth.lat >= lat_band[0]) & (truth.lat <= lat_band[1])
    error = (truth.uv - earlier.uv)[:, :, rows, :-1]
    return float(jnp.sqrt(jnp.mean(error**2)))


def errors(truth: WindField, copy: WindField, lat_band: tuple[float, float]) -> dict:
    """How wrong a copy is, in the launch band, measured two ways before anything is flown:

        wind_error_ms               RMS per component of the copy's wind minus the truth's
        level_difference_error_ms   the same for the differences between adjacent levels: what the balloon
                                    steers on, by the central hypothesis

    The primary test ranks the copies by each and asks which ranking control loss follows. The seam column is
    left out, since it repeats the first."""
    rows = (truth.lat >= lat_band[0]) & (truth.lat <= lat_band[1])
    wrong = (copy.uv - truth.uv)[:, :, rows, :-1]
    steer = wrong[:, 1:] - wrong[:, :-1]  # the change in each level-to-level difference
    return {"wind_error_ms": float(jnp.sqrt(jnp.mean(wrong**2))), "level_difference_error_ms": float(jnp.sqrt(jnp.mean(steer**2)))}


# --- by name ----------------------------------------------------------------------------------------------------

COPIES = ("truth", "levels", "frames", "grid", "offset", "offset_altitude", "random", "random_altitude", "persistence")


def copy(name: str, truth: WindField, key: jax.Array, kernel: BeliefConfig, size: float, earlier: WindField | None = None,
         lat_band: tuple[float, float] = (-10.0, 10.0)) -> WindField:
    """A copy by its name, as a cell and a row write it: "truth", "levels:9", "frames:4", "frames:frozen",
    "grid:2", "offset", "offset_altitude", "random", "random_altitude" or "persistence". `size` is in m/s and
    only the wrong copies read it"""
    kind, _, setting = name.partition(":")
    if kind in ERROR_MODELS:
        return error_model(kind, truth, key, kernel, size, lat_band, earlier)
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
