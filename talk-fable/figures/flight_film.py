"""One flight as a film, without the wind backdrop: the same launch flown by hindsight, MPC and the fixed plan.

Like flight.py, the paths are the sweep's own saved trajectories (results/rows-2026-10-24.jsonl), so the film
cannot disagree with the numbers. Unlike flight.py it needs no wind record, so it runs anywhere. The map is
framed on hindsight, MPC and the goal; a fixed plan that misses is allowed to run off the edge.
Writes public/fig/flight-s<seed>.mp4 and flight-s<seed>.png (its last frame).

    uv run python talk-fable/figures/flight_film.py [--seed 13] [--sigma 3.2]        (from the repo root)
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.path import Path

from balloon_sim.results import one_arm, read_rows
from style import CONTROLLER, GOAL, INK, NAME, OUT, REPO

ask = argparse.ArgumentParser()
ask.add_argument("--seed", type=int, default=13)
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
start_lon, start_lat = any_row["start_lon"], any_row["start_lat"]

paths = {}
for c, r in flights.items():
    lon = unwrap(np.concatenate([[start_lon], r["path_lon"]]))
    lat = np.concatenate([[start_lat], r["path_lat"]])
    alt = np.concatenate([[r["path_altitude_km"][0]], r["path_altitude_km"]])
    # arrived_move counts moves from 1, and lon[m] is where the balloon is after move m (lon[0] is the launch),
    # so the episode ends at lon[arrived_move]
    last = int(r["arrived_move"]) if r["arrived"] else len(lon) - 1
    paths[c] = (lon, lat, alt, last)

goal_lat = any_row["goal_lat"]
goal_lon = start_lon + ((any_row["goal_lon"] - start_lon + 180) % 360 - 180)

# frame the map on what reaches the goal, padded, and grown to the panel's shape
xs = np.concatenate([paths[c][0][: paths[c][3] + 1] for c in ("hindsight", "mpc")] + [[goal_lon]])
ys = np.concatenate([paths[c][1][: paths[c][3] + 1] for c in ("hindsight", "mpc")] + [[goal_lat]])
middle = (ys.min() + ys.max()) / 2
aspect = 1 / np.cos(np.radians(middle))
west, east, south, north = xs.min() - 2.5 * aspect, xs.max() + 2.5 * aspect, ys.min() - 2.5, ys.max() + 2.5
PANEL = (0.02, 0.33, 0.60, 0.62)
FIG = (12.8, 7.2)
panel_w, panel_h = PANEL[2] * FIG[0], PANEL[3] * FIG[1]
if (north - south) * aspect * panel_w / panel_h > east - west:
    half = (north - south) * aspect * panel_w / panel_h / 2
    west, east = (west + east) / 2 - half, (west + east) / 2 + half
else:
    half = (east - west) / aspect * panel_h / panel_w / 2
    south, north = (south + north) / 2 - half, (south + north) / 2 + half

figure = plt.figure(figsize=FIG, facecolor="white")
where = figure.add_axes(PANEL)
when = figure.add_axes((0.08, 0.09, 0.89, 0.20))
where.set_xlim(west, east)
where.set_ylim(south, north)
where.set_aspect(aspect, adjustable="box")
where.set_axis_off()

ring = np.linspace(0, 2 * np.pi, 160)
r_deg = np.degrees(radius / 6371)
where.plot(goal_lon + r_deg * np.sin(ring) * aspect, goal_lat + r_deg * np.cos(ring), color=GOAL, linewidth=1.8, linestyle=(0, (3, 2)))
where.plot(goal_lon, goal_lat, "*", color=GOAL, markersize=22)
where.text(goal_lon + r_deg * aspect + 0.4, goal_lat - 0.3, "goal", color=GOAL, fontsize=18, fontweight=600)
where.plot(start_lon, start_lat, "o", color=INK, markersize=8)
where.text(start_lon + 0.7, start_lat + 0.5, "launch", color=INK, fontsize=18, fontweight=600)
km_per_deg = 111.2 / aspect
where.plot([west + 1, west + 1 + 500 / km_per_deg], [south + 0.8] * 2, color=INK, linewidth=2.5)
where.text(west + 1 + 250 / km_per_deg, south + 1.2, "500 km", color=INK, fontsize=15, ha="center")

total = max(p[3] for p in paths.values())
when.set_xlim(0, total * step)
when.set_ylim(14.4, 20.6)
when.set_yticks([15, 20])
when.set_xticks(np.arange(0, total * step + 1, 24), [f"day {d}" for d in range(int(total * step // 24) + 1)])
when.set_ylabel("altitude (km)", fontsize=15)
when.tick_params(labelsize=14)

# a balloon: round envelope, a narrow neck down to a small basket, as one shape so it keeps together.
# Drawn as a marker so it keeps its size on screen whatever the map's scale.
def _balloon() -> Path:
    """Basket at the origin, so the balloon floats just above the point where it is"""
    theta = np.linspace(-0.5 * np.pi + 0.45, 1.5 * np.pi - 0.45, 48)  # the envelope, open at the bottom
    envelope = np.c_[np.cos(theta), np.sin(theta) + 2.2]
    envelope = np.vstack([envelope, [[-0.16, 1.05], [0.16, 1.05]], [envelope[0]]])
    rope = np.array([[-0.09, 1.1], [0.09, 1.1], [0.09, 0.42], [-0.09, 0.42], [-0.09, 1.1]])
    basket = np.array([[-0.34, 0.45], [0.34, 0.45], [0.28, 0.0], [-0.28, 0.0], [-0.34, 0.45]])
    parts = [envelope, rope, basket]
    vertices = np.vstack(parts)
    codes = np.concatenate([[Path.MOVETO] + [Path.LINETO] * (len(p) - 2) + [Path.CLOSEPOLY] for p in parts])
    return Path(vertices, codes)


BALLOON = _balloon()

WIDTH = {"hindsight": 7, "mpc": 3, "fixed_plan": 3}
trail, balloon, climb, dot = {}, {}, {}, {}
for c in ORDER:
    colour = CONTROLLER[c]
    trail[c] = where.plot([], [], color=colour, linewidth=WIDTH[c], solid_capstyle="round", solid_joinstyle="round")[0]
    balloon[c] = where.plot([], [], marker=BALLOON, markersize=80, color=colour, markeredgecolor="white", markeredgewidth=0.8, linestyle="none", zorder=7)[0]
    climb[c] = when.plot([], [], color=colour, linewidth=WIDTH[c] * 0.6, solid_joinstyle="round")[0]
    dot[c] = when.plot([], [], "o", color=colour, markersize=8, markeredgecolor="white", markeredgewidth=1.2, zorder=6)[0]

# the running cost: distance from the goal after each move, added up and divided by the 50 moves, so it ends
# at the flight's average distance. Moves after arrival cost nothing (the episode is over), so a bar stops
# growing when its balloon reaches the goal. Scaled to end exactly on the row's own cost_km.
moves = len(flights["mpc"]["path_lat"])
running = {}
for c, r in flights.items():
    lat, lon = np.radians(r["path_lat"]), np.radians(r["path_lon"])
    glat, glon = np.radians(goal_lat), np.radians(any_row["goal_lon"])
    d = 2 * 6371 * np.arcsin(np.sqrt(np.sin((glat - lat) / 2) ** 2 + np.cos(lat) * np.cos(glat) * np.sin((glon - lon) / 2) ** 2))
    if r["arrived"]:
        d[int(r["arrived_move"]):] = 0  # d[i] is after move i + 1, so this keeps moves 1..arrived_move
    cum = np.concatenate([[0.0], np.cumsum(d)])
    running[c] = cum * (r["cost_km"] / moves) / cum[-1]
final = {c: running[c][-1] for c in ORDER}

tally = figure.add_axes((0.66, 0.43, 0.31, 0.42))
tally.set_xlim(0, final["fixed_plan"] * 1.02)
tally.set_ylim(-0.6, 2.9)
tally.set_axis_off()
figure.text(0.66, 0.935, "Cost so far", color=INK, fontsize=19, fontweight=650, va="center")
figure.text(0.66, 0.895, "average distance from the goal", color=INK, fontsize=15, va="center")
ROW = {"hindsight": 2, "mpc": 1, "fixed_plan": 0}
bar, value = {}, {}
for c in ORDER:
    y = ROW[c]
    tally.text(0, y + 0.42, NAME[c], color=CONTROLLER[c], fontsize=18, fontweight=650, va="center")
    tally.barh(y, final["fixed_plan"] * 1.02, height=0.34, color="#f4f5f7")
    bar[c] = tally.barh(y, 0, height=0.34, color=CONTROLLER[c])[0]
    value[c] = tally.text(0, y, "", color=INK, fontsize=16, fontweight=600, va="center")
clock = figure.text(0.03, 0.965, "", color=INK, fontsize=19, fontweight=600, va="center")


def at(values: np.ndarray, move: float) -> float:
    low = int(np.floor(move))
    high = min(low + 1, len(values) - 1)
    return values[low] + (move - low) * (values[high] - values[low])


def draw(frame: int) -> None:
    move = min(frame / asked.per_move, total)
    clock.set_text(f"day {int(move * step // 24)}, hour {int(move * step % 24):02d}")
    for c in ORDER:
        lon, lat, alt, last = paths[c]
        m = min(move, last)
        whole = int(np.floor(m))
        x = np.append(lon[: whole + 1], at(lon, m))
        y = np.append(lat[: whole + 1], at(lat, m))
        trail[c].set_data(x, y)
        balloon[c].set_data([x[-1]], [y[-1]])
        climb[c].set_data(np.append(np.arange(whole + 1) * step, m * step), np.append(alt[: whole + 1], at(alt, m)))
        dot[c].set_data([m * step], [at(alt, m)])
        cost = at(running[c], min(move, moves))
        bar[c].set_width(cost)
        inside = cost > final["fixed_plan"] * 0.75  # a long bar carries its number inside, in white
        value[c].set_text(f"{cost:,.0f} km")
        value[c].set_position((cost - 40 if inside else cost + 40, ROW[c]))
        value[c].set_ha("right" if inside else "left")
        value[c].set_color("white" if inside else INK)


writer = FFMpegWriter(fps=30, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])
frames = total * asked.per_move + 45  # hold the last frame a second and a half
name = f"flight-s{asked.seed}"
with writer.saving(figure, str(OUT / f"{name}.mp4"), dpi=150):
    for frame in range(frames):
        draw(frame)
        writer.grab_frame()
draw(frames)
figure.savefig(OUT / f"{name}.png", dpi=150)
print(f"wrote public/fig/{name}.mp4 and {name}.png")
for c in ["hindsight", "mpc", "fixed_plan"]:
    r = flights[c]
    print(f"  {NAME[c]:12s} arrived: {('move %d (%d h)' % (r['arrived_move'], r['arrived_move'] * step)) if r['arrived'] else 'never'}")
