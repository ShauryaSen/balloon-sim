"""A picture of an episode: the paths on a map with the task's points, and altitude over time. Needs matplotlib."""

from __future__ import annotations

import numpy as np

from balloongym import tasks
from balloongym.env import BalloonEnv, EnvState
from balloongym.physics import destination
from balloongym.wind import WindField

COLOURS = ["#2f6fde", "#e2852e", "#2a9d5c", "#c0392b", "#7d3cb5", "#8a6d3b"]


def _unwrap(lon: np.ndarray, centre: float) -> np.ndarray:
    return (lon - centre + 180.0) % 360.0 - 180.0 + centre


def draw(env: BalloonEnv, field: WindField, state: EnvState, path: np.ndarray, size=(9.0, 4.0), dpi=80) -> np.ndarray:
    """path: (T, 3, N) of lat, lon, altitude. Returns an (H, W, 3) uint8 image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = env.config
    ts = state.task
    centre = float(path[0, 1, 0])
    fig, (ax, ax_alt) = plt.subplots(1, 2, figsize=size, dpi=dpi, gridspec_kw={"width_ratios": [1.6, 1]})

    points_lat, points_lon = np.asarray(ts.targets_lat), np.asarray(ts.targets_lon)
    if c.task.name != "max_distance":
        for k, (la, lo) in enumerate(zip(points_lat, points_lon)):
            ring_lat, ring_lon = destination(la, lo, np.linspace(0, 2 * np.pi, 90), c.task.radius_km)
            done = c.task.name == "waypoints" and bool(np.asarray(ts.visited)[k])
            ax.plot(_unwrap(np.asarray(ring_lon), centre), np.asarray(ring_lat), color="#2a9d5c" if done else "#444", lw=1.2)
            if c.task.name == "waypoints":
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
    ax.set_title(f"{c.task.name.replace('_', ' ')}   score {float(tasks.score(c.task, ts, state.balloons, max(int(state.t), 1))):.2f}", fontsize=10)
    ax_alt.set_xlim(0, c.n_steps)
    ax_alt.set_ylim(float(field.altitude_km[0]) - 0.2, float(field.altitude_km[-1]) + 0.2)
    ax_alt.set_xlabel("step")
    ax_alt.set_ylabel("altitude (km)")
    fig.tight_layout()
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return image
