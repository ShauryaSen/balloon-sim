"""The globe figures for the motivation slide: one balloon drifting for ten days in ERA5, drawn over

  globe-ble.png  BLE's generator only has wind in a 1,000 x 1,000 km box around launch; the flight leaves it

Run from the repo root:  uv run python talk/figures/globe.py
"""

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from balloon_sim.wind import load_record, wind_at

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "talk" / "public" / "figures"
R = 6371.0

ALTITUDE_KM = 19.07  # the record's level nearest 19 km
START = (25.0, 200.0)  # lat, lon: 25N 160W, North Pacific
HOURS = 240  # ten days, the whole record
HALF_KM = 500.0  # BLE's generated field spans +-500 km around its target (latlng_displacement_km in generative/vae.py)
TILE = (START[0] - np.degrees(HALF_KM / R), START[0] + np.degrees(HALF_KM / R),
        START[1] - np.degrees(HALF_KM / (R * np.cos(np.radians(START[0])))), START[1] + np.degrees(HALF_KM / (R * np.cos(np.radians(START[0])))))  # lat min, lat max, lon min, lon max
VIEW = (20.0, 175.0)  # the globe is seen from above this lat, lon: the box near the centre, the end of the flight round the back


def drift(field) -> np.ndarray:
    """A balloon that never changes altitude, stepped hourly, as (lat, lon) points"""
    lat, lon = START
    points = [(lat, lon)]
    for hour in range(HOURS):
        u, v = np.asarray(wind_at(field, jnp.float32(lat), jnp.float32(lon), jnp.float32(ALTITUDE_KM), jnp.float32(hour)))
        lat += np.degrees(v * 3.6 / R)
        lon = (lon + np.degrees(u * 3.6 / (R * np.cos(np.radians(lat))))) % 360
        points.append((lat, lon))
    return np.array(points)


def distance_km(a, b) -> np.ndarray:
    p1, l1, p2, l2 = map(np.radians, (a[0], a[1], b[:, 0], b[:, 1]))
    return 2 * R * np.arcsin(np.sqrt(np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin((l2 - l1) / 2) ** 2))


def project(lat, lon):
    """Orthographic projection onto the unit disk; also whether the point faces the viewer"""
    p0, l0 = np.radians(VIEW)
    p, l = np.radians(lat), np.radians(lon)
    x = np.cos(p) * np.sin(l - l0)
    y = np.cos(p0) * np.sin(p) - np.sin(p0) * np.cos(p) * np.cos(l - l0)
    front = np.sin(p0) * np.sin(p) + np.cos(p0) * np.cos(p) * np.cos(l - l0) > 0
    return x, y, front


def unproject(x, y):
    """Each point of the disk back to (lat, lon)"""
    p0, l0 = np.radians(VIEW)
    rho = np.sqrt(x**2 + y**2)
    c = np.arcsin(np.clip(rho, 0, 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        lat = np.arcsin(np.cos(c) * np.sin(p0) + np.where(rho > 0, y * np.sin(c) * np.cos(p0) / rho, 0))
    lon = l0 + np.arctan2(x * np.sin(c), rho * np.cos(c) * np.cos(p0) - y * np.sin(c) * np.sin(p0))
    return np.degrees(lat), np.degrees(lon) % 360


def speed_image(field, n=900):
    """Wind speed at the start of the record, painted onto the disk"""
    xs = np.linspace(-1, 1, n)
    x, y = np.meshgrid(xs, xs[::-1])
    lat, lon = unproject(x, y)
    uv = np.asarray(field.uv[0, int(np.argmin(np.abs(np.asarray(field.altitude_km) - ALTITUDE_KM)))])
    lats, lons = np.asarray(field.lat), np.asarray(field.lon)
    fi = np.clip(np.interp(lat, lats, np.arange(len(lats))), 0, len(lats) - 1.001)  # fractional grid indices, for a smooth blend
    fj = np.clip(np.interp(lon, lons, np.arange(len(lons))), 0, len(lons) - 1.001)
    i, j = fi.astype(int), fj.astype(int)
    wi, wj = fi - i, fj - j
    grid = np.hypot(uv[..., 0], uv[..., 1])
    speed = (grid[i, j] * (1 - wi) * (1 - wj) + grid[i + 1, j] * wi * (1 - wj)
             + grid[i, j + 1] * (1 - wi) * wj + grid[i + 1, j + 1] * wi * wj)
    outside_disk = x**2 + y**2 > 1
    outside_record = np.abs(lat) > 80
    in_tile = (lat >= TILE[0]) & (lat <= TILE[1]) & (lon >= TILE[2]) & (lon <= TILE[3])
    return speed, outside_disk, outside_record, in_tile


def draw(name: str, path: np.ndarray, image, whole_globe: bool) -> None:
    speed, outside_disk, outside_record, in_tile = image
    shown = np.ma.masked_where(outside_disk | outside_record | (~in_tile if not whole_globe else False), speed)
    figure, ax = plt.subplots(figsize=(5, 5))
    ax.add_patch(plt.Circle((0, 0), 1, color="#ececec", zorder=0))
    ax.imshow(shown, extent=(-1, 1, -1, 1), cmap="viridis", vmin=0, vmax=40, zorder=1, interpolation="bilinear")
    for lat in range(-60, 61, 30):  # faint graticule
        lon = np.linspace(0, 360, 361)
        x, y, front = project(np.full_like(lon, lat), lon)
        ax.plot(np.where(front, x, np.nan), np.where(front, y, np.nan), color="white" if whole_globe else "#d4d4d4", lw=0.5, alpha=0.5 if whole_globe else 1, zorder=2)
    for lon in range(0, 360, 30):
        lat = np.linspace(-90, 90, 181)
        x, y, front = project(lat, np.full_like(lat, lon))
        ax.plot(np.where(front, x, np.nan), np.where(front, y, np.nan), color="white" if whole_globe else "#d4d4d4", lw=0.5, alpha=0.5 if whole_globe else 1, zorder=2)
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="#999", lw=1, zorder=3))

    if not whole_globe:  # the tile's edge
        lat0, lat1, lon0, lon1 = TILE
        edge_lat = np.r_[np.full(50, lat0), np.linspace(lat0, lat1, 50), np.full(50, lat1), np.linspace(lat1, lat0, 50)]
        edge_lon = np.r_[np.linspace(lon0, lon1, 50), np.full(50, lon1), np.linspace(lon1, lon0, 50), np.full(50, lon0)]
        x, y, _ = project(edge_lat, edge_lon)
        ax.plot(x, y, color="#1a1a1a", lw=1.2, zorder=4)

    x, y, front = project(path[:, 0], path[:, 1])
    x, y = np.where(front, x, np.nan), np.where(front, y, np.nan)
    inside = (path[:, 0] >= TILE[0]) & (path[:, 0] <= TILE[1]) & (path[:, 1] >= TILE[2]) & (path[:, 1] <= TILE[3])
    leaves = int(np.argmin(inside)) if not inside.all() else len(path)
    if whole_globe:
        ax.plot(x, y, color="#c8372d", lw=2.2, zorder=5)
    else:
        ax.plot(x[: leaves + 1], y[: leaves + 1], color="#c8372d", lw=2.2, zorder=5)
        ax.plot(x[leaves:], y[leaves:], color="#c8372d", lw=1.6, ls=(0, (2, 3)), zorder=5)
        ax.plot(x[leaves], y[leaves], marker="x", color="#1a1a1a", ms=9, mew=2, zorder=6)
    ax.plot(x[0], y[0], marker="o", color="#1a1a1a", ms=6, zorder=6)
    if front[-1]:
        ax.plot(x[-1], y[-1], marker="o", color="#c8372d", ms=7, zorder=6)

    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    ax.set_aspect("equal")
    ax.axis("off")
    figure.savefig(OUT / name, dpi=220, bbox_inches="tight", transparent=True)
    plt.close(figure)
    print(f"wrote {OUT / name}")


if __name__ == "__main__":
    field = load_record(str(REPO / "records" / "era5_global_jul2021.npz"))
    path = drift(field)
    far = distance_km(path[0], path)
    inside = (path[:, 0] >= TILE[0]) & (path[:, 0] <= TILE[1]) & (path[:, 1] >= TILE[2]) & (path[:, 1] <= TILE[3])
    print(f"farthest from launch: {far.max():,.0f} km; leaves the tile after {int(np.argmin(inside))} h")
    image = speed_image(field)
    draw("globe-ble.png", path, image, whole_globe=False)
