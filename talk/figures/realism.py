"""The realism results of Part 2: one headline number per kind of check, BLE's VAE against our summer model,
then the same with fitted simplex noise added.

  realism-1.png  BLE and ours
  realism-2.png  the same, plus simplex noise

Numbers are copied, not recomputed, from one source: the summer model at 4 years / 250k steps, the checkpoint the
poster declares.
  balloon-env-dev/docs/benchmark-reports/benchmark_v2_4yr_250000.md   (columns: self-split reference, simplex noise,
                                                                        ble_vae, idiff m2cond)
Change over time: balloon-env-dev/docs/benchmark-reports/temporal_4yr_250000.md (run 2026-11-12, the summer model
tiled in time by InfiniteDiffusion; BLE 6-hourly over 48 h). Simplex 0.53 is from the 250k report. Dispersion: log-MSD RMSE; simplex 0.70 from the 250k report.
Lower is better in every panel. The dotted line is the ERA5 reference: one half of held-out ERA5 scored against the other.

Run from the repo root:  uv run python talk/figures/realism.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2] / "talk" / "public" / "figures"
INK, MUTED = "#1a1a1a", "#666666"
NAMES = ["BLE's VAE", "Ours (summer)", "Simplex noise"]
COLOURS = ["#b7bcc6", "#4f7db3", "#a99bd1"]

#        title                   what is measured                                     floor  BLE    ours   simplex  label
CHECKS = [
    ("Wind speeds",          "distance between wind-speed\ndistributions (m/s)",       1.11,  4.56,  2.98,  1.69,  "{:.2f}"),
    ("Structure in space",   "finest scale it keeps (km)",                             56.12, 841.86, 336.74, 56.12, "{:.0f}"),
    ("Change with altitude", "error in shear between levels,\nmean of u and v ((m/s)/km)", 0.15, 2.115, 1.005, 0.64, "{:.2f}"),
    ("Change over time",     "error in how far drifting\nparticles spread (dispersion)",  0.18, 1.79, 0.81, 0.70, "{:.2f}"),
]


def draw(name: str, with_noise: bool) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(10.5, 3.6))
    for ax, (title, measured, floor, *scores, label) in zip(axes, CHECKS):
        shown = scores if with_noise else scores[:2]
        bars = ax.bar(range(len(shown)), shown, color=COLOURS[: len(shown)], width=0.7)
        for bar, score in zip(bars, shown):
            text = label.format(score)
            ax.text(bar.get_x() + bar.get_width() / 2, score, text, ha="center", va="bottom", fontsize=10.5, color=INK)
        ax.axhline(floor, color=INK, linewidth=1.2, linestyle=(0, (1.5, 2.5)))
        ax.text(2.6, floor, " ERA5", fontsize=10, color=INK, va="center", ha="left", clip_on=False)
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylim(0, max(scores + [floor]) * 1.15)
        ax.text(0.0, 1.32, title, transform=ax.transAxes, fontsize=13, fontweight=600, va="bottom")
        ax.text(0.0, 1.28, measured, transform=ax.transAxes, fontsize=10, color=MUTED, va="top", linespacing=1.2)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLOURS[: 3 if with_noise else 2]]
    figure.legend(handles, NAMES[: len(handles)], loc="lower center", ncol=3, frameon=False, fontsize=11)
    figure.subplots_adjust(left=0.01, right=0.96, bottom=0.14, top=0.68, wspace=0.45)
    figure.savefig(OUT / name, dpi=200, transparent=True)
    plt.close(figure)
    print(f"wrote {OUT / name}")


if __name__ == "__main__":
    draw("realism-1.png", with_noise=False)
    draw("realism-2.png", with_noise=True)
