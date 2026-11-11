"""One look for every figure in the talk: fonts, colours, and how a figure is saved.

Change a colour here and rerun `figures/build.sh`; the slides pick up the new files on their own.
The same colours are in `styles/index.css`, so the diagrams drawn in the slides match the plots.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).parent
REPO = HERE.parent.parent
OUT = HERE.parent / "public" / "fig"
OUT.mkdir(parents=True, exist_ok=True)

for ttf in sorted((HERE / "fonts").glob("Inter-*.ttf")):
    font_manager.fontManager.addfont(str(ttf))

INK = "#1c2433"  # text and axes
MUTED = "#8a93a3"  # secondary text, gridlines
FAINT = "#e6e8ec"
HINDSIGHT = "#1c2433"  # the best possible: drawn in ink
MPC = "#2f6fde"  # replans every move
MPC_LIGHT = "#9dbcf2"
FIXED = "#e2852e"  # commits at launch
GOAL = "#c8372d"
ACCENT = "#0f8a7a"  # the one extra colour, for "ours" and highlights
WIND = "viridis"  # every wind-speed map, as on the poster

CONTROLLER = {"hindsight": HINDSIGHT, "mpc": MPC, "mpc_blind": MPC_LIGHT, "fixed_plan": FIXED}
NAME = {"hindsight": "hindsight", "mpc": "MPC", "mpc_blind": "MPC, readings off", "fixed_plan": "fixed plan"}

plt.rcParams.update({
    "font.family": "Inter",
    "font.size": 13,
    "font.weight": 400,
    "axes.labelsize": 13,
    "axes.labelcolor": INK,
    "axes.labelweight": 500,
    "axes.edgecolor": INK,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 15,
    "axes.titleweight": 600,
    "axes.titlelocation": "left",
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelsize": 12.5,
    "ytick.labelsize": 12.5,
    "text.color": INK,
    "legend.frameon": False,
    "legend.fontsize": 12.5,
    "lines.linewidth": 2.6,
    "lines.solid_capstyle": "round",
    "figure.dpi": 100,
    "savefig.dpi": 280,
    "savefig.transparent": True,
})


def save(figure, name: str) -> None:
    figure.savefig(OUT / name)
    print(f"wrote public/fig/{name}")


def save_layers(figure, name: str, layers: list[list]) -> None:
    """Save one figure as a stack of transparent pictures the slide reveals one click at a time.

    `<name>-0.png` is the axes with none of the layers; `<name>-1.png` holds only the first layer's artists,
    and so on. Every file is the same size, so stacking them rebuilds the whole figure.
    """
    everything = [a for layer in layers for a in layer]
    for a in everything:
        a.set_visible(False)
    figure.savefig(OUT / f"{name}-0.png")
    axes_parts = []
    for ax in figure.axes:
        axes_parts += [ax.xaxis, ax.yaxis, ax.patch, *ax.spines.values(), ax.title, ax._left_title, ax._right_title]
        axes_parts += [c for c in ax.get_children() if c not in everything and c not in axes_parts and c is not ax.patch]
    for part in axes_parts:
        part.set_visible(False)
    hidden_texts = [t for t in figure.texts if t not in everything]
    for t in hidden_texts:
        t.set_visible(False)
    for i, layer in enumerate(layers, start=1):
        for a in layer:
            a.set_visible(True)
        figure.savefig(OUT / f"{name}-{i}.png")
        for a in layer:
            a.set_visible(False)
    print(f"wrote public/fig/{name}-0..{len(layers)}.png")
