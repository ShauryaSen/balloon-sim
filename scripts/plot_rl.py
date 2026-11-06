"""The learned policies' training curves: check these before reading anything into a policy's ranking

    uv run scripts/plot_rl.py policies --into figures/2026-11-07-rl

One panel per objective, a line per fleet size, solid learning from readings and dotted not. The curve is
the mean cost of the episodes each iteration sampled, in km as the sweep scores it, so it is noisy (32
episodes over every noise level at once) and higher than the evaluation, which acts on the policy's mean.
A curve still falling at its last iteration means the policy was stopped early and its rows understate it.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#12161b"
SHADES = {1: "#0072b2", 2: "#009e73", 4: "#d55e00", 8: "#cc79a7"}


def main(argv: list[str] | None = None) -> None:
    ask = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ask.add_argument("policies", nargs="?", default="policies")
    ask.add_argument("--into", default="figures/rl")
    asked = ask.parse_args(argv)
    runs = [json.loads(p.read_text()) for p in sorted(Path(asked.policies).glob("*.json"))]
    if not runs:
        sys.exit(f"no training curves under {asked.policies}")
    objectives = ["reach", "escape", "station", "targets"]
    figure, panels = plt.subplots(1, 4, figsize=(16, 3.8), dpi=200)
    for panel, objective in zip(panels, objectives):
        for run in runs:
            name = run["name"]
            if not name.startswith(objective + "-"):
                continue
            n = int(name.split("-n")[1].split("-")[0])
            learns = name.endswith("learns")
            x = [c["episodes"] for c in run["curve"]]
            y = np.array([c["cost_km"] for c in run["curve"]])
            panel.plot(x, y, color=SHADES.get(n, "#999999"), linestyle="-" if learns else ":", linewidth=1.4,
                       label=f"{n} balloon{'s' if n > 1 else ''}, {'learning' if learns else 'not learning'} ({run['seconds'] / 60:.0f} min)")
        panel.set_title(objective, fontsize=10, loc="left", color=INK)
        panel.set_xlabel("training episodes")
        panel.spines[["top", "right"]].set_visible(False)
        panel.legend(frameon=False, fontsize=6.5)
    panels[0].set_ylabel("mean cost of the sampled episodes (km)")
    figure.tight_layout()
    into = Path(asked.into)
    into.mkdir(parents=True, exist_ok=True)
    figure.savefig(into / "training_curves.png")
    print(f"wrote {into / 'training_curves.png'}")


if __name__ == "__main__":
    main()
