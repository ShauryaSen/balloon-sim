"""The realism scorecard of Part 2: one number per kind of check, for each generator.

Numbers are copied from the generator repo's benchmark, not recomputed here:
  balloon-env-dev/docs/benchmark-reports/benchmark_v2_hpx.md   (every row but the two marked "poster")
  the summer poster's results table                             (change over time, for BLE-VAE and the summer model)
Lower is better in every panel. The dotted line is the ERA5 floor: the score one half of held-out ERA5 gets
against the other, so nothing can meaningfully beat it.

    uv run python talk-fable/figures/realism.py        (from the repo root)
"""

import matplotlib.pyplot as plt

from style import ACCENT, FAINT, INK, MUTED, save_layers

#            check title              what is measured                      floor  BLE-VAE  summer  global  fitted noise
CHECKS = [
    ("Wind speeds",          "distance between speed\ndistributions (m/s)",  1.11,  4.56,   2.89,   0.76,   1.69),
    ("Fine detail",          "smallest scale it\nresolves (km)",             56.0,  842.0,  374.0,  210.0,  56.0),
    ("Change with altitude", "error in the shear between\nlevels (m/s per km)", 0.20, 3.62,  1.03,   0.35,   0.31),
    ("Change over time",     "error in the\ntime spectrum",                  0.28,  4.19,   5.46,   2.60,   1.19),  # 4.19 and 5.46: poster
]
NAMES = ["BLE's VAE", "ours, summer", "ours, global", "fitted noise"]
COLOURS = ["#b7bcc6", "#8fd1c8", ACCENT, "#7a5cc2"]

figure, axes = plt.subplots(1, 4, figsize=(9.2, 3.7))
models, noise = [], []
for ax, (title, measured, floor, *scores) in zip(axes, CHECKS):
    bars = ax.bar(range(4), scores, color=COLOURS, width=0.72)
    models += list(bars.patches[:3])
    noise.append(bars.patches[3])
    for slot, (bar, score) in enumerate(zip(bars.patches, scores)):
        text = ax.text(bar.get_x() + bar.get_width() / 2, score, f"{score:g}", ha="center", va="bottom", fontsize=11, color=INK,
                       bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.0 if score > 1.6 * floor else 0.9))
        (noise if slot == 3 else models).append(text)
    ax.axhline(floor, color=INK, linewidth=1.2, linestyle=(0, (1.5, 2.5)))
    ax.text(3.62, floor, " ERA5", fontsize=10.5, color=INK, va="center", ha="left", clip_on=False)
    ax.text(0.0, 1.42, title, transform=ax.transAxes, fontsize=13.5, fontweight=600, va="bottom")
    ax.text(0.0, 1.38, measured, transform=ax.transAxes, fontsize=10.5, color=MUTED, va="top", linespacing=1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(0, max(scores) * 1.1)
# the colour key is in the slide itself (slides.md), so its last entry can arrive with the noise bars
figure.subplots_adjust(left=0.01, right=0.955, bottom=0.03, top=0.62, wspace=0.5)
save_layers(figure, "realism", [models, noise])
