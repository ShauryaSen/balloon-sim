"""One flight as a film: the same launch flown by hindsight, MPC and the fixed plan, over the wind it flew in.

The paths are the sweep's own saved trajectories (results/rows-2026-10-24.jsonl), not a re-flight, so the film
cannot disagree with the numbers. Writes public/fig/flight.mp4 and flight.png (its last frame, for the PDF).

    uv run python talk-fable/figures/flight.py [--seed 9] [--sigma 3.2]        (from the repo root)
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap

from balloon_sim.results import best_hindsight, one_arm, read_rows
from balloon_sim.wind import load_record
from style import CONTROLLER, FAINT, GOAL, INK, MUTED, NAME, OUT, REPO

ask = argparse.ArgumentParser()
ask.add_argument("--seed", type=int, default=9)
ask.add_argument("--sigma", type=float, default=3.2)
ask.add_argument("--horizon", type=int, default=20)
ask.add_argument("--per-move", type=int, default=5, help="film frames per three-hour move")
asked = ask.parse_args()

rows = one_arm(read_rows([REPO / "results" / "rows-2026-10-24.jsonl"]))


def pick(controller: str) -> dict:
    for r in rows:
        if r["controller"] != controller or r["seed"] != asked.seed or r["n_winds"] != 0:
            continue
        if controller == "hindsight" or (r["belief.sigma"] == asked.sigma and (controller == "fixed_plan" or r["planner.horizon"] == asked.horizon)):
            return r
    raise SystemExit(f"no row for {controller}")


ORDER = ["fixed_plan", "hindsight", "mpc"]  # drawn in this order, so MPC sits on top
flights = {c: pick(c) for c in ORDER}
any_row = flights["mpc"]
step = any_row["balloon.step_hours"]
radius = any_row["task.arrival_km"]
unwrap = lambda lon: np.degrees(np.unwrap(np.radians(np.asarray(lon, dtype=float))))

paths = {}
for c, r in flights.items():
    lon = unwrap(np.concatenate([[r["start_lon"]], r["path_lon"]]))
    lat = np.concatenate([[r["start_lat"]], r["path_lat"]])
    alt = np.concatenate([[r["path_altitude_km"][0]], r["path_altitude_km"]])
    last = int(r["arrived_move"]) if r["arrived"] else len(lon) - 1  # the episode ends on arrival
    paths[c] = (lon, lat, alt, last)

start_lon, start_lat = any_row["start_lon"], any_row["start_lat"]
goal_lat = any_row["goal_lat"]
goal_lon = start_lon + ((any_row["goal_lon"] - start_lon + 180) % 360 - 180)

every_lon = np.concatenate([p[0] for p in paths.values()] + [[goal_lon]])
every_lat = np.concatenate([p[1] for p in paths.values()] + [[goal_lat]])
west, east = every_lon.min() - 4, every_lon.max() + 4
middle = (every_lat.min() + every_lat.max()) / 2
aspect = 1 / np.cos(np.radians(middle))
half_height = (east - west) / aspect * (4.3 / 12.2) / 2  # fills the map panel exactly
south, north = middle - half_height, middle + half_height

wind = load_record(str(REPO / "records" / "era5_global_jul2021.npz"))
w_lat, w_lon = np.asarray(wind.lat), np.asarray(wind.lon)
ys = (w_lat >= south - 2) & (w_lat <= north + 2)
xs = (w_lon >= west - 2) & (w_lon <= east + 2)
level = 4  # 19 km, where these flights spend most of their time
uv = np.asarray(wind.uv[:, level][:, ys][:, :, xs])
hours = np.asarray(wind.hours)
pale = LinearSegmentedColormap.from_list("pale", ["#ffffff", "#dfe5ee"])

figure = plt.figure(figsize=(12.8, 7.2), facecolor="white")
where = figure.add_axes((0.03, 0.34, 0.94, 0.60))
when = figure.add_axes((0.085, 0.11, 0.80, 0.17))

extent = (w_lon[xs][0], w_lon[xs][-1], w_lat[ys][0], w_lat[ys][-1])
backdrop = where.imshow(np.hypot(uv[0, ..., 0], uv[0, ..., 1]), origin="lower", extent=extent, cmap=pale, vmin=0, vmax=18,
                        interpolation="bicubic", aspect=aspect)
gy, gx = np.meshgrid(w_lat[ys][::2], w_lon[xs][::2], indexing="ij")
arrows = where.quiver(gx, gy, uv[0, ::2, ::2, 0], uv[0, ::2, ::2, 1], color="#b3bccb", scale=330, width=0.0022, headwidth=3.8, headlength=4.4)
where.set_xlim(west, east)
where.set_ylim(south, north)
where.set_axis_off()

ring = np.linspace(0, 2 * np.pi, 160)
where.plot(goal_lon + np.degrees(radius / 6371) * np.sin(ring) / np.cos(np.radians(goal_lat)), goal_lat + np.degrees(radius / 6371) * np.cos(ring),
           color=GOAL, linewidth=1.6, linestyle=(0, (3, 2)))
where.plot(goal_lon, goal_lat, "*", color=GOAL, markersize=20)
where.text(goal_lon, goal_lat + np.degrees(radius / 6371) + 0.5, "goal", color=GOAL, fontsize=15, fontweight=600, ha="center", va="bottom")
where.plot(start_lon, start_lat, "o", color=INK, markersize=7)
where.text(start_lon + 0.6, start_lat - 0.5, "launch", color=INK, fontsize=15, fontweight=600, va="top")
km_per_deg = 111.2 * np.cos(np.radians(south + 1.2))
where.plot([west + 2, west + 2 + 1000 / km_per_deg], [south + 1.2] * 2, color=INK, linewidth=2)
where.text(west + 2 + 500 / km_per_deg, south + 1.6, "1,000 km", color=INK, fontsize=12, ha="center", va="bottom")

total = len(paths["mpc"][0]) - 1
when.set_xlim(0, total * step)
when.set_ylim(14.4, 20.6)
when.set_yticks([15, 20])
when.set_xticks(np.arange(0, total * step + 1, 24), [f"day {d}" for d in range(int(total * step // 24) + 1)])
when.set_ylabel("altitude (km)", fontsize=13)
when.tick_params(labelsize=12)

WIDTH = {"hindsight": 4.2, "mpc": 2.6, "fixed_plan": 2.6}
trail, dot, climb = {}, {}, {}
for c in ORDER:
    colour = CONTROLLER[c]
    trail[c] = where.plot([], [], color=colour, linewidth=WIDTH[c], solid_capstyle="round", alpha=0.95)[0]
    dot[c] = where.plot([], [], "o", color=colour, markersize=11, markeredgecolor="white", markeredgewidth=1.6, zorder=6)[0]
    climb[c] = when.plot([], [], color=colour, linewidth=WIDTH[c] - 0.4)[0]
legend_x = 0.905
for slot, c in enumerate(["hindsight", "mpc", "fixed_plan"]):
    figure.text(legend_x, 0.25 - 0.045 * slot, NAME[c], color=CONTROLLER[c], fontsize=15, fontweight=600, va="center")
clock = figure.text(0.03, 0.955, "", color=INK, fontsize=17, fontweight=600, va="center")
figure.text(0.97, 0.955, f"forecast error {asked.sigma:g} m/s, the size of a real day-old forecast's", color=MUTED, fontsize=13, va="center", ha="right")


def at(values: np.ndarray, move: float) -> float:
    low = int(np.floor(move))
    high = min(low + 1, len(values) - 1)
    return values[low] + (move - low) * (values[high] - values[low])


def draw(frame: int) -> None:
    move = min(frame / asked.per_move, total)
    hour = move * step
    t = float(np.interp(hour, hours, np.arange(len(hours))))
    low, high = int(np.floor(t)), min(int(np.floor(t)) + 1, len(hours) - 1)
    now = uv[low] + (t - low) * (uv[high] - uv[low])
    backdrop.set_data(np.hypot(now[..., 0], now[..., 1]))
    arrows.set_UVC(now[::2, ::2, 0], now[::2, ::2, 1])
    for c in ORDER:
        lon, lat, alt, last = paths[c]
        m = min(move, last)
        whole = int(np.floor(m))
        xs_ = np.append(lon[: whole + 1], at(lon, m))
        ys_ = np.append(lat[: whole + 1], at(lat, m))
        trail[c].set_data(xs_, ys_)
        dot[c].set_data([xs_[-1]], [ys_[-1]])
        climb[c].set_data(np.append(np.arange(whole + 1) * step, m * step), np.append(alt[: whole + 1], at(alt, m)))
    clock.set_text(f"day {int(hour // 24)}, hour {int(hour % 24):02d}")


writer = FFMpegWriter(fps=30, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])
frames = total * asked.per_move + 45  # hold the last frame a second and a half
with writer.saving(figure, str(OUT / "flight.mp4"), dpi=150):
    for frame in range(frames):
        draw(frame)
        writer.grab_frame()
draw(frames)
figure.savefig(OUT / "flight.png", dpi=150, transparent=False)
print("wrote public/fig/flight.mp4 and flight.png")
for c in ["hindsight", "mpc", "fixed_plan"]:
    r = flights[c]
    print(f"  {NAME[c]:12s} arrived: {('move %d (%d h)' % (r['arrived_move'], r['arrived_move'] * step)) if r['arrived'] else 'never'}, "
          f"average distance {r['summed_km'] / 50:,.0f} km")
