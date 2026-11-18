"""Measure how a day-old forecast's error is correlated across space, altitude and time, from the records
themselves, with no kernel shape assumed: K learned from past forecast errors (2026-11-18)

    uv run scripts/measure_error_covariance.py records/era5_tropics_2020-05-16_l18_1h.npz ... --out results/error-covariance.json

The error is the truth minus the record's forecast (the truth 24 h earlier), less its mean at each level. For each
displacement, the correlation is the mean product of the error at a point and at the displaced point, over every
point in the launch band, divided by the mean square error there:

    east, north      a whole number of grid cells (2 degrees, about 222 km), same level and hour
    along, across    any horizontal offset up to 8 cells, binned by distance and by its angle to the true wind
                     at the first point: along (within 30 degrees of the wind or against it) or across (within 30
                     degrees of square to it)
    up, down         a number of levels, at each level, so the reach above and below can differ
    time             a number of frames later, same place and level, reported in hours. Caution: the forecast
                     is the truth 24 h earlier, so the errors 24 h apart share one field with opposite signs
                     and anticorrelate by construction; a weather model's errors would not
    u with v         the east error at a point with the north error at the displaced point, same offsets as east

Correlations are vector ones, u and v pooled, except u with v. Each record gives one set; the output keeps every
record's set, so the spread across weeks is the variance (measured, not assumed).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from balloon_sim.wind import load_record

BAND = (-10.0, 10.0)
KM_PER_DEG = 111.2
MAX_CELLS = 8  # 16 degrees, about 1,780 km
MAX_HOURS = 36
DISTANCE_BINS_KM = np.arange(0.0, 1900.0, 150.0)


def error_field(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """(t, alt, lat, lon, 2) error with the seam column dropped, the truth, lat and altitudes"""
    truth, forecast = load_record(path), load_record(path, "forecast_")
    uv, fc = np.asarray(truth.uv)[..., :-1, :], np.asarray(forecast.uv)[..., :-1, :]
    error = uv - fc
    error = error - error.mean(axis=(0, 2, 3), keepdims=True)  # each level's mean taken off: K is about the spread
    step = float(np.diff(np.asarray(truth.hours))[0])
    return error, uv, np.asarray(truth.lat), np.asarray(truth.altitude_km), step


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))


def measure(path: str) -> dict:
    e, uv, lat, alt, step = error_field(path)
    rows = np.where((lat >= BAND[0]) & (lat <= BAND[1]))[0]
    out: dict = {"record": Path(path).name, "rms_ms": float(np.sqrt(np.mean(e[:, :, rows] ** 2)))}
    here = e[:, :, rows]

    out["east_km"] = [k * 2.0 * KM_PER_DEG for k in range(MAX_CELLS + 1)]
    out["east"] = [correlation(here, np.roll(e, -k, axis=3)[:, :, rows]) for k in range(MAX_CELLS + 1)]
    out["u_with_v"] = [correlation(here[..., 0], np.roll(e, -k, axis=3)[:, :, rows][..., 1]) for k in range(MAX_CELLS + 1)]
    out["north_km"] = [k * 2.0 * KM_PER_DEG for k in range(MAX_CELLS + 1)]
    out["north"] = [correlation(here, e[:, :, rows + k]) for k in range(MAX_CELLS + 1)]

    frames = range(int(MAX_HOURS / step) + 1)  # lags in frames, reported in hours
    out["hours"] = [k * step for k in frames]
    out["time"] = [correlation(here[: here.shape[0] - k], here[k:]) for k in frames]

    # altitude: record levels run top (index 0) to bottom; from each level, how far up and down it reaches
    order = np.argsort(alt)  # bottom to top
    e_up, alt_up = e[:, order][:, :, rows], alt[order]
    out["levels_km"] = alt_up.tolist()
    out["up"], out["down"] = [], []
    for i in range(len(alt_up)):
        out["up"].append([[float(alt_up[j] - alt_up[i]), correlation(e_up[:, i], e_up[:, j])] for j in range(i, len(alt_up))])
        out["down"].append([[float(alt_up[i] - alt_up[j]), correlation(e_up[:, i], e_up[:, j])] for j in range(i, -1, -1)])

    # along and across the true wind at the first point
    wind = uv[:, :, rows]
    wind_angle = np.arctan2(wind[..., 1], wind[..., 0])  # (t, alt, row, lon)
    sums = {k: np.zeros(len(DISTANCE_BINS_KM)) for k in ("along_num", "along_den", "across_num", "across_den")}
    norm = np.sum(here * here, axis=-1)
    for di in range(-MAX_CELLS, MAX_CELLS + 1):
        for dj in range(-MAX_CELLS, MAX_CELLS + 1):
            if di == 0 and dj == 0:
                continue
            dx = dj * 2.0 * KM_PER_DEG * np.cos(np.radians(lat[rows]))[None, None, :, None]
            dy = di * 2.0 * KM_PER_DEG
            distance = np.sqrt(dx**2 + dy**2) * np.ones_like(wind_angle)
            offset_angle = np.arctan2(dy, dx) * np.ones_like(wind_angle)
            to_wind = np.abs(np.cos(offset_angle - wind_angle))
            there = np.roll(e, -dj, axis=3)[:, :, rows + di]
            product = np.sum(here * there, axis=-1)
            bins = np.clip(np.digitize(distance, DISTANCE_BINS_KM) - 1, 0, len(DISTANCE_BINS_KM) - 1)
            for name, mask in (("along", to_wind > np.cos(np.radians(30))), ("across", to_wind < np.sin(np.radians(30)))):
                np.add.at(sums[f"{name}_num"], bins[mask], product[mask])
                np.add.at(sums[f"{name}_den"], bins[mask], norm[mask])
    out["direction_km"] = (DISTANCE_BINS_KM + 75.0).tolist()
    for name in ("along", "across"):
        with np.errstate(invalid="ignore", divide="ignore"):
            out[name] = (sums[f"{name}_num"] / sums[f"{name}_den"]).tolist()
    return out


def main() -> None:
    ask = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ask.add_argument("records", nargs="+")
    ask.add_argument("--out", default="results/error-covariance.json")
    a = ask.parse_args()
    results = []
    for path in a.records:
        results.append(measure(path))
        r = results[-1]
        print(f"{r['record']}: RMS {r['rms_ms']:.2f} m/s; east 222 km {r['east'][1]:.2f}, north {r['north'][1]:.2f}, "
              f"9 h {np.interp(9, r['hours'], r['time']):.2f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results))


if __name__ == "__main__":
    main()
