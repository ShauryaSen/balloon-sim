"""Fit the belief kernel's lengthscales and amplitude to a record's forecast error

    uv run scripts/fit_kernel.py records/cbottle_jul_s100.npz --stale-hours 24
    uv run scripts/fit_kernel.py records/era5_npac_wide.npz

The error fitted is that of a day-old forecast: the truth as it was `--stale-hours` earlier, which any wind
time series can supply (DESIGN.md, 26). Without the flag the record must carry its own forecast_u and
forecast_v, and the error is their difference from the truth. The belief's kernel is meant to describe that error, and its settings are zermelo's
hand-set numbers until this script replaces them with the record's own. It prints a BeliefConfig to paste.

    e = truth - forecast,   k(x, x') = sigma^2 * Matern52(place, altitude) * Matern32(time)

The kernel is a product over axes, so each axis is fitted on its own: hold three of the four coordinates
equal, measure how the correlation of the error falls off along the fourth, and pick the lengthscale whose
Matern curve matches. sigma is the error's own standard deviation, pooled over both components. Nothing here
is a maximum-likelihood fit of the full kernel; it is a moment fit per axis, which is what a product kernel
with one lengthscale per axis can absorb anyway.
"""

import argparse
import json
from pathlib import Path

import numpy as np

EARTH_RADIUS_KM = 6371.0
LAGS = 12  # lags per axis: beyond this the curve is noise, since few pairs are that far apart


def error(record, stale_hours: float | None) -> np.ndarray:
    """The error the forecast makes, as (n_t, n_alt, n_lat, n_lon, 2): against a forecast that is the truth
    stale_hours earlier, or the record's own forecast when stale_hours is None"""
    u, v = record["u"], record["v"]
    if stale_hours is not None:
        step = float(np.diff(record["hours"]).mean())
        frames = int(round(stale_hours / step))
        if frames < 1 or frames >= u.shape[0]:
            raise SystemExit(f"{stale_hours} h is {frames} frames of {step:.0f} h; the record has {u.shape[0]}")
        return np.stack([u[frames:] - u[:-frames], v[frames:] - v[:-frames]], axis=-1)  # truth now minus truth then
    missing = [k for k in ("forecast_u", "forecast_v") if k not in record]
    if missing:
        raise SystemExit(f"the record has no {', '.join(missing)}: pass --stale-hours to build a forecast from the truth")
    return np.stack([u - record["forecast_u"], v - record["forecast_v"]], axis=-1)


def matern(distance: np.ndarray, lengthscale: float, nu: float) -> np.ndarray:
    """Matern 5/2 or 3/2 correlation at a distance, as a fraction of the lengthscale"""
    r = np.abs(distance) / lengthscale
    if nu == 2.5:
        return (1 + np.sqrt(5) * r + 5 * r**2 / 3) * np.exp(-np.sqrt(5) * r)
    return (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)


def correlation_along(e: np.ndarray, axis: int) -> np.ndarray:
    """The error's correlation with itself at lag 0, 1, ... along one axis, the other three held equal.

    Every pair of points that far apart on that axis contributes, pooled over the whole grid and both wind
    components. The error is centred first, because a forecast with a steady bias would otherwise read as
    error that is correlated everywhere.
    """
    centred = e - e.mean()
    n = min(LAGS + 1, centred.shape[axis])
    variance = float((centred**2).mean())
    out = []
    for lag in range(n):
        a = np.take(centred, range(0, centred.shape[axis] - lag), axis=axis)
        b = np.take(centred, range(lag, centred.shape[axis]), axis=axis)
        out.append(float((a * b).mean()) / variance)
    return np.array(out)


def lengthscale(correlations: np.ndarray, spacing: float, nu: float) -> float:
    """The lengthscale whose Matern curve is closest to the measured correlations, by a scan then a refine.

    Only the lags where the correlation is still above 0.05 are fitted. Further out the measurement is mostly
    noise about zero, and every candidate lengthscale predicts roughly zero there, so those lags carry no
    information about which one is right but would still dominate a sum of squares.
    """
    keep = correlations > 0.05
    keep[0] = True
    distance = np.arange(correlations.size)[keep] * spacing
    measured = correlations[keep]
    if distance.size < 2:  # correlation gone by the first lag: the axis is as rough as the grid is coarse
        return float(spacing)
    scan = np.geomspace(spacing / 10, spacing * 10 * LAGS, 400)
    loss = [float(((matern(distance, candidate, nu) - measured) ** 2).sum()) for candidate in scan]
    return float(scan[int(np.argmin(loss))])


def fit(path: str, stale_hours: float | None = None, lat_band: tuple[float, float] | None = None) -> dict:
    record = np.load(path)
    e = error(record, stale_hours)
    hours, altitude_km = np.asarray(record["hours"], float), np.asarray(record["altitude_km"], float)
    lat, lon = np.asarray(record["latitude"], float), np.asarray(record["longitude"], float)
    if lat_band is not None:  # fit where the balloon flies, not the whole globe
        keep = (lat >= lat_band[0]) & (lat <= lat_band[1])
        lat, e = lat[keep], e[:, :, keep]

    # Horizontal spacing in km, measured at the middle of the record. A degree of longitude shrinks with
    # latitude while a degree of latitude does not, so the two axes are fitted separately and then pooled:
    # the kernel has one horizontal lengthscale, in km, for both.
    middle = np.radians(float(np.abs(lat).mean()))
    per_lat_km = EARTH_RADIUS_KM * np.radians(abs(float(np.diff(lat).mean())))
    per_lon_km = EARTH_RADIUS_KM * np.radians(abs(float(np.diff(lon).mean()))) * np.cos(middle)

    along_lat = lengthscale(correlation_along(e, 2), per_lat_km, nu=2.5)
    along_lon = lengthscale(correlation_along(e, 3), per_lon_km, nu=2.5)
    fitted = {
        "sigma": float(e.std()),
        "horizontal_km": float(np.sqrt(along_lat * along_lon)),  # geometric mean, so neither axis dominates
        "altitude_km": lengthscale(correlation_along(e, 1), abs(float(np.diff(altitude_km).mean())), nu=2.5),
        "hours": lengthscale(correlation_along(e, 0), abs(float(np.diff(hours).mean())), nu=1.5),
    }
    return {**fitted, "lat_km": along_lat, "lon_km": along_lon, "sigma_u": float(e[..., 0].std()), "sigma_v": float(e[..., 1].std())}


def main(argv: list[str] | None = None) -> None:
    ask = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ask.add_argument("record", help="the .npz to fit")
    ask.add_argument("--stale-hours", type=float, default=None, help="fit the error of the truth this many hours earlier")
    ask.add_argument("--lat", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"), help="fit within this latitude band only")
    ask.add_argument("--json", default=None, help="also write the fit to this file")
    asked = ask.parse_args(argv)

    fitted = fit(asked.record, asked.stale_hours, tuple(asked.lat) if asked.lat else None)
    print(f"{Path(asked.record).name}: error {fitted['sigma']:.3f} m/s "
          f"(u {fitted['sigma_u']:.3f}, v {fitted['sigma_v']:.3f})")
    print(f"  horizontal {fitted['horizontal_km']:.0f} km "
          f"(lat {fitted['lat_km']:.0f}, lon {fitted['lon_km']:.0f})")
    print(f"  altitude   {fitted['altitude_km']:.2f} km")
    print(f"  time       {fitted['hours']:.1f} h")
    print("\nBeliefConfig(")
    for field in ("sigma", "horizontal_km", "altitude_km", "hours"):
        print(f"    {field}={fitted[field]:.4g},")
    print(")")
    if asked.json:
        Path(asked.json).write_text(json.dumps(fitted, indent=2) + "\n")


if __name__ == "__main__":
    main()
