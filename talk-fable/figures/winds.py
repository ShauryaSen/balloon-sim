"""The wind pictures: truth and forecast, cBottle's levels against ERA5's, the route against a tile, and the
fidelity ladders. All drawn from the records the experiments flew on (records/*.npz).

    uv run python talk-fable/figures/winds.py        (from the repo root)
"""

import jax
import matplotlib.pyplot as plt
import numpy as np

from balloon_sim.belief import forecast
from balloon_sim.config import BeliefConfig
from balloon_sim.results import one_arm, read_rows
from balloon_sim.wind import WindField, load_record
from style import ACCENT, FAINT, FIXED, GOAL, INK, MPC, MUTED, REPO, WIND, save, save_layers

ERA5 = load_record(str(REPO / "records" / "era5_global_jul2021.npz"))
BOX = dict(lat=(14.0, 52.0), lon=(282.0, 348.0))  # the North Atlantic, where the example flight is
LEVEL = 3  # 17.9 km, mid-band
SPEED_MAX = 15.0


def crop(field: WindField, lat: tuple[float, float], lon: tuple[float, float]) -> WindField:
    y = (np.asarray(field.lat) >= lat[0]) & (np.asarray(field.lat) <= lat[1])
    x = (np.asarray(field.lon) >= lon[0]) & (np.asarray(field.lon) <= lon[1])
    return WindField(field.hours, field.altitude_km, field.lat[y], field.lon[x], field.uv[:, :, y][:, :, :, x])


def wind_map(ax, lat, lon, uv, arrows: int = 3, vmax: float = SPEED_MAX, arrow_scale: float = 260.0, interpolation: str = "bicubic",
             arrow_width: float = 0.0045):
    """Wind speed as colour with white arrows over it, the way the poster draws wind"""
    uv = np.asarray(uv)
    speed = np.hypot(uv[..., 0], uv[..., 1])
    extent = (float(lon[0]), float(lon[-1]), float(lat[0]), float(lat[-1]))
    ax.imshow(speed, origin="lower", extent=extent, cmap=WIND, vmin=0, vmax=vmax, interpolation=interpolation,
              aspect=1 / np.cos(np.radians(float(np.mean(lat)))))
    yy, xx = np.meshgrid(np.asarray(lat)[1::arrows], np.asarray(lon)[1::arrows], indexing="ij")
    ax.quiver(xx, yy, uv[1::arrows, 1::arrows, 0], uv[1::arrows, 1::arrows, 1], color="white", scale=arrow_scale, width=arrow_width,
              headwidth=3.6, headlength=4.2, alpha=0.9)
    ax.set_axis_off()


# --- truth, error, forecast ----------------------------------------------------------------------------------

def draw_forecast_split() -> None:
    truth = crop(ERA5, **BOX)
    truth = truth._replace(hours=truth.hours[:2], uv=truth.uv[:2])  # two frames are enough for a picture
    guess = forecast(BeliefConfig(sigma=3.2), jax.random.key(9), truth)
    pictures = {"truth": truth.uv[0, LEVEL], "forecast": guess.uv[0, LEVEL], "error": truth.uv[0, LEVEL] - guess.uv[0, LEVEL]}
    for name, uv in pictures.items():
        figure, ax = plt.subplots(figsize=(5.2, 3.6))
        wind_map(ax, truth.lat, truth.lon, uv)  # all three on the same colours and arrow lengths, so they can be compared
        figure.subplots_adjust(0, 0, 1, 1)
        save(figure, f"split-{name}.png")
        plt.close(figure)
    error = np.asarray(pictures["error"])
    print(f"  error field here: {np.sqrt((error ** 2).mean()):.1f} m/s rms per component")


# --- cBottle's levels against ERA5's --------------------------------------------------------------------------

def draw_levels() -> None:
    cbottle = np.load(REPO / "records" / "cbottle_jul_s100.npz")
    band = (np.asarray(ERA5.lat) >= 20) & (np.asarray(ERA5.lat) <= 50)
    era_speed = np.hypot(np.asarray(ERA5.uv[..., 0]), np.asarray(ERA5.uv[..., 1]))[:, :, band].mean(axis=(0, 2, 3))
    cb_band = (cbottle["latitude"] >= 20) & (cbottle["latitude"] <= 50)
    cb_speed = np.hypot(cbottle["u"], cbottle["v"])[:, :, cb_band].mean(axis=(0, 2, 3))
    print("  ERA5 mean speed by level:", dict(zip(np.round(np.asarray(ERA5.altitude_km), 1).tolist(), np.round(era_speed, 1).tolist())))
    for name, speed in (("ERA5", np.hypot(np.asarray(ERA5.uv[..., 0]), np.asarray(ERA5.uv[..., 1]))[:, :, band]),
                        ("cBottle", np.hypot(cbottle["u"], cbottle["v"])[:, :, cb_band])):
        calmest = speed.min(axis=1)  # the slowest level on offer at each place and time
        print(f"  {name}: calmest level is a median {np.median(calmest):.1f} m/s, under 5 m/s at {100 * (calmest < 5).mean():.0f}% of points")
    print("  cBottle mean speed by level:", dict(zip(cbottle["altitude_km"].tolist(), np.round(cb_speed, 1).tolist())))

    figure, axes = plt.subplots(1, 2, figsize=(6.4, 4.4), sharey=True, sharex=True)
    layers = []
    for ax, name, altitudes, speeds, colour in (
        (axes[0], "cBottle (released)", cbottle["altitude_km"], cb_speed, FIXED),
        (axes[1], "ERA5", np.asarray(ERA5.altitude_km), era_speed, ACCENT),
    ):
        parts = [ax.barh(altitudes, speeds, height=0.42, color=colour, zorder=3)]
        parts += [ax.text(s + 0.5, a, f"{s:.0f}", va="center", fontsize=11.5, color=INK) for a, s in zip(altitudes, speeds)]
        parts.append(ax.text(0.0, 1.03, f"{name}: {len(altitudes)} levels", transform=ax.transAxes, fontsize=13, fontweight=600, va="bottom"))
        ax.set_xlabel("average wind speed (m/s)")
        ax.set_xlim(0, 25)
        ax.set_xticks([0, 10, 20])
        layers.append([p for part in parts for p in (part.patches if hasattr(part, "patches") else [part])])
    axes[0].set_ylabel("altitude (km)")
    axes[0].set_ylim(10.5, 22)
    figure.tight_layout(w_pad=2.5, rect=(0, 0, 1, 0.95))
    save_layers(figure, "levels", layers)
    plt.close(figure)


# --- one flight against one generated tile --------------------------------------------------------------------

def draw_route() -> None:
    rows = one_arm(read_rows([REPO / "results" / "rows-2026-10-24.jsonl"]))
    flight = next(r for r in rows if r["controller"] == "fixed_plan" and r["seed"] == 9 and r["n_winds"] == 0 and r["belief.sigma"] == 3.2)
    lon = np.concatenate([[flight["start_lon"]], flight["path_lon"]])
    lat = np.concatenate([[flight["start_lat"]], flight["path_lat"]])
    field = crop(ERA5, **BOX)
    figure, ax = plt.subplots(figsize=(9.6, 5.3))
    wind_map(ax, field.lat, field.lon, field.uv[0, 4], arrows=4, vmax=26.0)
    # one tile of the summer model: 1,336 km by 1,023 km (the poster's zoom figure), drawn to scale at the launch
    height = 1023 / 111.2
    width = 1336 / (111.2 * np.cos(np.radians(lat[0])))
    tile = plt.Rectangle((lon[0] - 0.75 * width, lat[0] - 0.4 * height), width, height, fill=False, edgecolor="white", linewidth=2.6, linestyle=(0, (4, 2)))
    ax.add_patch(tile)
    label = ax.text(lon[0] - 0.75 * width + width / 2, lat[0] - 0.4 * height - 0.9, "one generated tile\n1,300 km across", color="white", fontsize=14,
                    fontweight=600, va="top", ha="center", linespacing=1.2)
    path = ax.plot(lon, lat, color="white", linewidth=3.4, solid_capstyle="round")[0]
    start = ax.plot(lon[0], lat[0], "o", color="white", markersize=11)[0]
    end = ax.text(lon[-1], lat[-1] + 1.6, "six days later, 4,300 km away", color="white", fontsize=14, fontweight=600, ha="left", va="bottom")
    figure.subplots_adjust(0, 0, 1, 1)
    save_layers(figure, "route", [[start, tile, label], [path, end]])
    plt.close(figure)
    km = 6371 * np.arccos(np.clip(np.sin(np.radians(lat[0])) * np.sin(np.radians(lat[-1])) + np.cos(np.radians(lat[0])) * np.cos(np.radians(lat[-1])) * np.cos(np.radians(lon[-1] - lon[0])), -1, 1))
    print(f"  the flight ends {km:,.0f} km from its launch")


# --- the fidelity ladders: the same wind, made coarser one property at a time ----------------------------------

def coarsen_grid(uv: np.ndarray, k: int) -> np.ndarray:
    """Keep every k-th grid point and fill in linearly between them"""
    if k == 1:
        return uv
    n_lat, n_lon = uv.shape[:2]
    kept_y, kept_x = np.arange(0, n_lat, k), np.arange(0, n_lon, k)
    rows = np.stack([[np.interp(np.arange(n_lon), kept_x, uv[y, kept_x, c]) for c in range(2)] for y in kept_y], axis=0)  # (kept_y, 2, n_lon)
    out = np.stack([[np.interp(np.arange(n_lat), kept_y, rows[:, c, x]) for x in range(n_lon)] for c in range(2)], axis=-1)  # (n_lon, n_lat, 2)
    return np.transpose(out, (1, 0, 2))


def draw_ladders() -> None:
    field = crop(ERA5, **BOX)
    lat, lon = np.asarray(field.lat), np.asarray(field.lon)
    uv = np.asarray(field.uv)

    # the grid: 2 (the record), 4, 8, 16 degrees
    figure, axes = plt.subplots(1, 4, figsize=(12.0, 2.15))
    for ax, k in zip(axes, (1, 2, 4, 8)):
        wind_map(ax, lat, lon, coarsen_grid(uv[0, LEVEL], k), arrows=4, interpolation="bilinear")
    figure.subplots_adjust(0, 0, 1, 1, wspace=0.04)
    save(figure, "ladder-grid.png")
    plt.close(figure)

    # the levels: the eastward wind up one column, from every level kept down to one
    y, x = int(np.argmin(abs(lat - 30))), int(np.argmin(abs(lon - 320)))
    altitude = np.asarray(field.altitude_km)
    column = uv[0, :, y, x, 0]
    fine = np.linspace(altitude[0], altitude[-1], 100)
    figure, axes = plt.subplots(1, 4, figsize=(12.0, 2.15), sharex=True, sharey=True)
    for ax, kept in zip(axes, ([0, 1, 2, 3, 4, 5], [0, 2, 5], [0, 5], [2])):
        ax.plot(np.interp(fine, altitude, column), fine, color=FAINT, linewidth=5, solid_capstyle="round")
        model = np.interp(fine, altitude[kept], column[kept]) if len(kept) > 1 else np.full_like(fine, column[kept[0]])
        ax.plot(model, fine, color=ACCENT, linewidth=2.6)
        ax.plot(column[kept], altitude[kept], "o", color=ACCENT, markersize=8)
        ax.axvline(0, color=MUTED, linewidth=0.8, zorder=0)
        ax.set_axis_off()
    figure.subplots_adjust(0.02, 0.06, 0.98, 0.94, wspace=0.12)
    save(figure, "ladder-levels.png")
    plt.close(figure)

    # the frames: the eastward wind at one place over ten days, from a frame every 3 hours to a frozen field
    hours = np.asarray(field.hours)
    series = uv[:, LEVEL, y, x, 0]
    figure, axes = plt.subplots(1, 4, figsize=(12.0, 2.15), sharex=True, sharey=True)
    for ax, k in zip(axes, (1, 4, 8, None)):
        ax.plot(hours, series, color=FAINT, linewidth=5, solid_capstyle="round")
        if k is None:
            ax.plot(hours, np.full_like(series, series[0]), color=ACCENT, linewidth=2.6)
            ax.plot(hours[0], series[0], "o", color=ACCENT, markersize=8)
        else:
            ax.plot(hours, np.interp(hours, hours[::k], series[::k]), color=ACCENT, linewidth=2.6)
            if k > 1:
                ax.plot(hours[::k], series[::k], "o", color=ACCENT, markersize=6)
        ax.set_axis_off()
    figure.subplots_adjust(0.02, 0.08, 0.98, 0.92, wspace=0.12)
    save(figure, "ladder-frames.png")
    plt.close(figure)


# --- whole-globe pictures: the cover, and cBottle's wind ------------------------------------------------------

def draw_globes() -> None:
    cbottle = np.load(REPO / "records" / "cbottle_jul_s100.npz")
    cb_uv = np.stack([cbottle["u"][0, -1], cbottle["v"][0, -1]], axis=-1)  # 20.6 km, the level inside the balloon's range
    if cbottle["latitude"][0] > cbottle["latitude"][-1]:
        cb_uv = cb_uv[::-1]
    for name, lat, lon, uv, vmax in (
        ("cover", np.asarray(ERA5.lat), np.asarray(ERA5.lon)[:-1], np.asarray(ERA5.uv[0, 4])[:, :-1], 30.0),
        ("cbottle-globe", np.sort(cbottle["latitude"]), cbottle["longitude"], cb_uv, 45.0),
    ):
        keep = (lat >= -62) & (lat <= 62)
        figure, ax = plt.subplots(figsize=(12.0, 4.1))
        wind_map(ax, lat[keep], lon, uv[keep], arrows=4, vmax=vmax, arrow_scale=1500.0, arrow_width=0.0013)
        ax.set_aspect("auto")
        figure.subplots_adjust(0, 0, 1, 1)
        figure.savefig(REPO / "talk-fable" / "public" / "fig" / f"{name}.png", transparent=False)
        print(f"wrote public/fig/{name}.png")
        plt.close(figure)


def draw_coarse() -> None:
    """The truth beside a copy on a 16 degree grid, for the slide that explains the method"""
    field = crop(ERA5, **BOX)
    figure, ax = plt.subplots(figsize=(5.2, 3.6))
    wind_map(ax, field.lat, field.lon, coarsen_grid(np.asarray(field.uv[0, LEVEL]), 8), interpolation="bilinear")
    figure.subplots_adjust(0, 0, 1, 1)
    save(figure, "split-coarse.png")
    plt.close(figure)


if __name__ == "__main__":
    draw_globes()
    draw_coarse()
    draw_forecast_split()
    draw_levels()
    draw_route()
    draw_ladders()
