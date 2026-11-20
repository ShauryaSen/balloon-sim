"""A picture of an episode: the paths on a map with the task's points, and altitude over time. Needs matplotlib."""

from __future__ import annotations

import numpy as np

from balloongym.tasks import Task

COLOURS = ["#2f6fde", "#e2852e", "#2a9d5c", "#c0392b", "#7d3cb5", "#8a6d3b"]
EARTH_RADIUS_KM = 6371.0


def _unwrap(lon: np.ndarray, centre: float) -> np.ndarray:
    return (lon - centre + 180.0) % 360.0 - 180.0 + centre


def _circle(lat: float, lon: float, radius_km: float) -> tuple[np.ndarray, np.ndarray]:
    bearing = np.linspace(0, 2 * np.pi, 90)
    d = radius_km / EARTH_RADIUS_KM
    p1, l1 = np.radians(lat), np.radians(lon)
    p2 = np.arcsin(np.sin(p1) * np.cos(d) + np.cos(p1) * np.sin(d) * np.cos(bearing))
    l2 = l1 + np.arctan2(np.sin(bearing) * np.sin(d) * np.cos(p1), np.cos(d) - np.sin(p1) * np.sin(p2))
    return np.degrees(p2), np.degrees(l2)


def draw(task: Task, truth, path: np.ndarray, points_lat: np.ndarray, points_lon: np.ndarray, score: float,
         size=(9.0, 4.0), dpi=80) -> np.ndarray:
    """path: (T, 3, N) of lat, lon, altitude. Returns an (H, W, 3) uint8 image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centre = float(path[0, 1, 0])
    fig, (ax, ax_alt) = plt.subplots(1, 2, figsize=size, dpi=dpi, gridspec_kw={"width_ratios": [1.6, 1]})
    for k, (la, lo) in enumerate(zip(points_lat, points_lon)):
        ring_lat, ring_lon = _circle(float(la), float(lo), task.radius_km)
        ax.plot(_unwrap(ring_lon, centre), ring_lat, color="#444", lw=1.2)
        if len(points_lat) > 1:
            ax.text(_unwrap(np.array(lo), centre), la, str(k + 1), ha="center", va="center", fontsize=9)
    for i in range(path.shape[2]):
        colour = COLOURS[i % len(COLOURS)]
        ax.plot(_unwrap(path[:, 1, i], centre), path[:, 0, i], color=colour, lw=1.4)
        ax.plot(_unwrap(path[-1:, 1, i], centre), path[-1:, 0, i], "o", color=colour, ms=5)
        ax_alt.plot(np.arange(path.shape[0]), path[:, 2, i], color=colour, lw=1.2)
    ax.plot(_unwrap(path[:1, 1, 0], centre), path[:1, 0, 0], "k^", ms=6)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_aspect(1.0 / max(np.cos(np.radians(path[0, 0, 0])), 0.2))
    ax.set_title(f"{task.name.replace('_', ' ')}   score {score:.2f}", fontsize=10)
    ax_alt.set_xlim(0, task.n_moves)
    ax_alt.set_ylim(float(truth.altitude_km[0]) - 0.2, float(truth.altitude_km[-1]) + 0.2)
    ax_alt.set_xlabel("step")
    ax_alt.set_ylabel("altitude (km)")
    fig.tight_layout()
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return image
