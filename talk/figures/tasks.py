"""The two task pictures for the 'what we want balloons to do' slide. Schematic, not real flights.

  task-station.png     a balloon looping inside a ring around one point
  task-navigation.png  a path across the Pacific through two waypoints
  task-distance.png    a path that drifts away from its launch, and the straight line back to it

Run from the repo root:  uv run python talk/figures/tasks.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "talk" / "public" / "figures"
VIEW = (25.0, 190.0)  # the globe is seen from above this lat, lon
RED, INK = "#c8372d", "#1a1a1a"


def project(lat, lon):
    """Orthographic projection onto the unit disk"""
    p0, l0 = np.radians(VIEW)
    p, l = np.radians(lat), np.radians(lon)
    x = np.cos(p) * np.sin(l - l0)
    y = np.cos(p0) * np.sin(p) - np.sin(p0) * np.cos(p) * np.cos(l - l0)
    return x, y


def globe():
    figure, ax = plt.subplots(figsize=(4, 4))
    ax.add_patch(plt.Circle((0, 0), 1, color="#ececec", zorder=0))
    for lat in range(-60, 61, 30):
        lon = np.linspace(VIEW[1] - 90, VIEW[1] + 90, 181)
        ax.plot(*project(np.full_like(lon, lat), lon), color="#d4d4d4", lw=0.6, zorder=1)
    for lon in range(0, 360, 30):
        if abs(((lon - VIEW[1] + 180) % 360) - 180) < 90:
            lat = np.linspace(-90, 90, 181)
            ax.plot(*project(lat, np.full_like(lat, lon)), color="#d4d4d4", lw=0.6, zorder=1)
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="#999", lw=1, zorder=2))
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    ax.set_aspect("equal")
    ax.axis("off")
    return figure, ax


def save(figure, name):
    figure.savefig(OUT / name, dpi=220, bbox_inches="tight", transparent=True)
    plt.close(figure)
    print(f"wrote {OUT / name}")


# station keeping: a ring around one point, and a wandering loop that stays inside it (drawn far larger than real)
figure, ax = globe()
centre = (25.0, 190.0)
a = np.linspace(0, 2 * np.pi, 200)
ax.plot(*project(centre[0] + 15 * np.sin(a), centre[1] + 17 * np.cos(a)), color=INK, lw=1.4, ls=(0, (3, 2)), zorder=3)
t = np.linspace(0, 1, 400)
lat = centre[0] + 10 * np.sin(2 * np.pi * 1.3 * t) * np.cos(2 * np.pi * 0.5 * t)
lon = centre[1] + 12 * np.cos(2 * np.pi * 1.1 * t + 0.6) * np.sin(2 * np.pi * 0.6 * t + 1.0)
ax.plot(*project(lat, lon), color=RED, lw=2, zorder=4)
ax.plot(*project(*centre), marker="+", color=INK, ms=10, mew=2, zorder=5)
save(figure, "task-station.png")

# navigation: launch, two waypoints, a smooth path through them
figure, ax = globe()
stops = np.array([(35.0, 150.0), (18.0, 185.0), (38.0, 225.0)])  # launch, waypoint, goal
s = np.linspace(0, 1, len(stops))
u = np.linspace(0, 1, 300)
lat = np.interp(u, s, stops[:, 0]) + 5 * np.sin(np.pi * 2 * u) * np.sin(np.pi * u)
lon = np.interp(u, s, stops[:, 1])
ax.plot(*project(lat, lon), color=RED, lw=2, zorder=4)
ax.plot(*project(*stops[0]), marker="o", color=INK, ms=7, zorder=5)
for p in stops[1:]:
    ax.add_patch(plt.Circle(project(*p), 0.05, fill=False, color=INK, lw=1.6, zorder=5))
save(figure, "task-navigation.png")

# distance: launch, a path that drifts away, and the straight line from launch to where it ended up
figure, ax = globe()
launch, end = (30.0, 150.0), (22.0, 232.0)
u = np.linspace(0, 1, 300)
lat = np.interp(u, [0, 1], [launch[0], end[0]]) + 9 * np.sin(2 * np.pi * 1.25 * u) * np.sin(np.pi * u) ** 0.5
lon = np.interp(u, [0, 1], [launch[1], end[1]])
ax.plot(*project(lat, lon), color=RED, lw=2, zorder=4)
straight_lat, straight_lon = np.linspace(launch[0], lat[-1], 100), np.linspace(launch[1], lon[-1], 100)
ax.plot(*project(straight_lat, straight_lon), color=INK, lw=1.4, ls=(0, (3, 2)), zorder=3)
ax.plot(*project(*launch), marker="o", color=INK, ms=7, zorder=5)
ax.plot(*project(lat[-1], lon[-1]), marker="o", color=RED, ms=5, zorder=5)
save(figure, "task-distance.png")
