"""Turn the measured forecast-error correlations into a kernel a belief can condition with (2026-11-18)

    uv run scripts/fit_measured_kernel.py results/error-covariance-2026-11-18.json --out results/measured-kernel-2026-11-18.json

A belief needs K to be a valid covariance (positive definite), which raw measured curves need not be once they
are interpolated. So each part is put in a form that is valid by construction, as close to the measurements as
that allows:

    altitude     the measured level-by-level correlation matrix itself, mean over the weeks, with any negative
                 eigenvalues set to zero. Between levels it is read by straight-line weights, k(z, z') =
                 w(z)' M w(z'), which is valid for any valid M. Keeps the sign flip 2-3 km away.
    sideways     a Matern of smoothness nu with one length east and another north, fitted by least squares to
                 the measured east and north curves out to 1,500 km. nu is picked from 1/2, 3/2, 5/2.
    time         a Matern 3/2 fitted to the measured curve out to 12 h only: past that, the persistence forecast
                 makes errors anticorrelate by construction (docs/error-covariance-2026-11-18.md), which a
                 real forecast's errors would not.
    u and v      unrelated (measured +0.08)

The product of valid kernels is valid. sigma is not fitted here: the belief takes it from the record.
"""

import argparse
import json
from pathlib import Path

import numpy as np

FIT_KM = 1500.0
FIT_HOURS = 12.0


def matern(nu: float, r: np.ndarray) -> np.ndarray:
    r = np.abs(r)
    if nu == 0.5:
        return np.exp(-r)
    if nu == 1.5:
        return (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
    return (1 + np.sqrt(5) * r + 5 * r**2 / 3) * np.exp(-np.sqrt(5) * r)


def fit_length(nu: float, x: np.ndarray, y: np.ndarray, lengths: np.ndarray) -> tuple[float, float]:
    errors = [np.mean((matern(nu, x / length) - y) ** 2) for length in lengths]
    i = int(np.argmin(errors))
    return float(lengths[i]), float(errors[i])


def main() -> None:
    ask = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ask.add_argument("measured")
    ask.add_argument("--out", default="results/measured-kernel-2026-11-18.json")
    a = ask.parse_args()
    results = json.loads(Path(a.measured).read_text())

    east_x, north_x = np.array(results[0]["east_km"]), np.array(results[0]["north_km"])
    east, north = np.mean([r["east"] for r in results], 0), np.mean([r["north"] for r in results], 0)
    keep_e, keep_n = east_x <= FIT_KM, north_x <= FIT_KM
    lengths = np.arange(50.0, 5000.0, 5.0)
    best = None
    for nu in (0.5, 1.5, 2.5):
        le, ee = fit_length(nu, east_x[keep_e], east[keep_e], lengths)
        ln, en = fit_length(nu, north_x[keep_n], north[keep_n], lengths)
        if best is None or ee + en < best[3]:
            best = (nu, le, ln, ee + en)
    nu, east_km, north_km, error = best

    hours = np.array(results[0]["hours"])
    time = np.mean([r["time"] for r in results], 0)
    keep_t = hours <= FIT_HOURS
    hours_scale, time_error = fit_length(1.5, hours[keep_t], time[keep_t], np.arange(1.0, 60.0, 0.1))

    alt = np.array(results[0]["levels_km"])  # bottom to top
    n = len(alt)
    mats = []
    for r in results:
        m = np.eye(n)
        for i, row in enumerate(r["up"]):
            for k, (_, c) in enumerate(row):
                m[i, i + k] = m[i + k, i] = c
        mats.append(m)
    matrix = np.mean(mats, 0)
    values, vectors = np.linalg.eigh(matrix)
    clipped = int(np.sum(values < 0))
    matrix = (vectors * np.clip(values, 0, None)) @ vectors.T
    d = np.sqrt(np.diag(matrix))
    matrix = matrix / d[:, None] / d[None, :]  # back to unit diagonal

    out = {"nu": nu, "east_km": east_km, "north_km": north_km, "hours": hours_scale, "levels_km": alt.tolist(),
           "level_matrix": matrix.tolist(), "fit": {"sideways_mse": error, "time_mse": time_error,
           "negative_eigenvalues_clipped": clipped, "weeks": len(results)}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"sideways: Matern {nu}, east {east_km:.0f} km, north {north_km:.0f} km (mse {error:.4f})")
    print(f"time: Matern 1.5, {hours_scale:.1f} h (mse {time_error:.4f}, fitted to {FIT_HOURS:.0f} h)")
    print(f"altitude: measured {n}x{n} matrix, {clipped} negative eigenvalues clipped; "
          f"min off-diagonal {matrix[np.triu_indices(n, 1)].min():+.2f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
