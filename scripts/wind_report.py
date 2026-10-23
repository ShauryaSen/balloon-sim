"""Can a balloon hold a point in this record at all? A report on the wind itself, before any controller

    uv run scripts/wind_report.py records/era5_global_jul2021.npz --lat 20 50

A balloon only chooses altitude, so what it can do is bounded by the wind in its own column. Two numbers
decide it. The calmest level sets the drift it cannot avoid: if the slowest level at a point is 5 m/s, three
hours there carries it 54 km whatever it does. The spread of directions between levels sets whether it can
steer at all: levels that all blow the same way offer a choice of speed, not of heading, and a balloon that
can only pick its speed cannot hold station.

This answers a question the sweeps cannot, since a controller failing to hold a point does not say whether
holding was possible. It reads the record alone and runs in seconds.
"""

import argparse
from pathlib import Path

import numpy as np


def load(path: str) -> dict:
    """The record's wind with every axis ascending, as plain numpy"""
    record = np.load(path)
    u, v = np.asarray(record["u"], float), np.asarray(record["v"], float)
    axes = [np.asarray(record[name], float) for name in ("hours", "altitude_km", "latitude", "longitude")]
    for i, coordinate in enumerate(axes):
        if coordinate[0] > coordinate[-1]:
            axes[i] = coordinate[::-1]
            u, v = np.flip(u, axis=i), np.flip(v, axis=i)
    return {"u": u, "v": v, "hours": axes[0], "altitude_km": axes[1], "lat": axes[2], "lon": axes[3]}


def band(record: dict, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
    """The wind between two latitudes, as (u, v) each (times, levels, lats, lons)"""
    keep = (record["lat"] >= low) & (record["lat"] <= high)
    return record["u"][:, :, keep, :], record["v"][:, :, keep, :]


def report(path: str, low: float, high: float, step_hours: float) -> None:
    record = load(path)
    u, v = band(record, low, high)
    speed = np.hypot(u, v)  # (times, levels, lats, lons)

    print(f"{Path(path).name}: {speed.shape[0]} times, {speed.shape[1]} levels "
          f"{record['altitude_km'][0]:.1f} to {record['altitude_km'][-1]:.1f} km, latitudes {low:g} to {high:g}\n")

    print("per level, over the band:")
    for i, altitude in enumerate(record["altitude_km"]):
        at = speed[:, i]
        print(f"  {altitude:5.1f} km   mean {at.mean():5.1f}   median {np.median(at):5.1f}   "
              f"90th {np.percentile(at, 90):5.1f} m/s")

    # The column: at each point and time, the slowest level is the best the balloon can do for drift
    calmest = speed.min(axis=1)
    drift_km = calmest * step_hours * 3.6
    print(f"\ncalmest level in the column: median {np.median(calmest):.1f} m/s "
          f"(quartiles {np.percentile(calmest, 25):.1f} to {np.percentile(calmest, 75):.1f})")
    print(f"  under 5 m/s at {100 * (calmest < 5).mean():.0f}% of points, under 2 m/s at {100 * (calmest < 2).mean():.0f}%")
    print(f"  unavoidable drift in one {step_hours:g} h move: median {np.median(drift_km):.0f} km, "
          f"so about {np.median(drift_km) * 50:.0f} km over a 50-move episode")

    # Steering: the widest angle between any two levels at the same point and time
    angle = np.arctan2(u, v)  # (times, levels, lats, lons), 0 is north
    difference = np.abs(angle[:, :, None] - angle[:, None, :])  # (times, levels, levels, ...)
    difference = np.degrees(np.minimum(difference, 2 * np.pi - difference))
    widest = difference.reshape(difference.shape[0], -1, *difference.shape[3:]).max(axis=1)
    print(f"\nwidest angle between two levels in the column: median {np.median(widest):.0f} degrees")
    print(f"  levels oppose (over 90 degrees apart) at {100 * (widest > 90).mean():.0f}% of points")
    print(f"  near-opposed (over 150) at {100 * (widest > 150).mean():.0f}%")

    print("\nRead it as: the balloon cannot be slower than the calmest level, and cannot steer where no two")
    print("levels disagree. Holding a point needs both a low drift and a wide angle at the same place.")


def main(argv: list[str] | None = None) -> None:
    ask = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ask.add_argument("record")
    ask.add_argument("--lat", nargs=2, type=float, default=(20.0, 50.0), metavar=("LOW", "HIGH"))
    ask.add_argument("--step-hours", type=float, default=3.0)
    asked = ask.parse_args(argv)
    report(asked.record, asked.lat[0], asked.lat[1], asked.step_hours)


if __name__ == "__main__":
    main()
