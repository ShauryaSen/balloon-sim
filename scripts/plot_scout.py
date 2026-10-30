"""Draw the scouting arm: which of the twenty candidate tasks had any room in them

    uv run scripts/plot_scout.py --into figures/

Two panels, both over goal distance with a line per direction.

    top     how often hindsight reaches the goal, against the 24-of-32 threshold the rule fixed.
            A task below the line cannot be flown by anything, because the best possible controller
            cannot fly it.
    bottom  how far above hindsight a blind fixed plan scores, on a typical seed, against the 25%
            the rule fixed. A task below that line is a free ride: the wind takes a balloon there
            whatever it does, so no controller can show anything.

A task has to clear both to be worth sweeping, and exactly one of the twenty does.

The numbers are read from the table in docs/scouting-rule.md, which is the committed record of the run
(Slurm arrays 577661 and 577662, 1,280 cells). The 1,280 rows themselves were never committed, so the
table is the source; parsing it rather than retyping it means the figure cannot drift from the document.
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#12161b"
COLOUR = {"downwind": "#0072b2", "crosswind": "#d55e00", "upwind": "#009e73", "random": "#cc79a7"}
REACH, MATERIAL = 24, 25  # of 32 seeds, and percent above hindsight: the thresholds docs/scouting-rule.md fixed

ROW = re.compile(
    r"^(?P<direction>\w+)\s+(?P<goal>\d+)\s+(?P<arrived>\d+)/32\s+(?P<move>\d+)\s+\d+/32\s+"
    r"(?P<hindsight>\d+)\s+(?P<gap>\d+)\s+(?P<percent>\d+)%\s+(?P<material>\d+)/32\s+\d+\s+(?P<verdict>\S+)"
)


def scouted(note: Path) -> list[dict]:
    """Every row of the result table in the scouting note"""
    out = []
    for line in note.read_text().splitlines():
        found = ROW.match(line.strip())
        if found:
            got = found.groupdict()
            out.append({k: (v if k in ("direction", "verdict") else int(v)) for k, v in got.items()})
    return out


def main(argv: list[str] | None = None) -> None:
    ask = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ask.add_argument("--note", default="docs/scouting-rule.md", help="the committed scouting note to read")
    ask.add_argument("--into", default="figures")
    asked = ask.parse_args(argv)

    rows = scouted(Path(asked.note))
    if len(rows) != 20:
        sys.exit(f"expected the note's twenty tasks, read {len(rows)}; has the table changed shape?")

    figure, (reach, room) = plt.subplots(2, 1, figsize=(7.0, 7.0), dpi=200, sharex=True)
    for direction in ("downwind", "crosswind", "upwind", "random"):
        mine = sorted((r for r in rows if r["direction"] == direction), key=lambda r: r["goal"])
        x = [r["goal"] for r in mine]
        reach.plot(x, [r["arrived"] for r in mine], marker="o", markersize=4.5, linewidth=1.6,
                   color=COLOUR[direction], label=direction)
        room.plot(x, [r["percent"] for r in mine], marker="o", markersize=4.5, linewidth=1.6,
                  color=COLOUR[direction], label=direction)

    chosen = next(r for r in rows if r["verdict"] == "QUALIFIES")
    for panel, value in ((reach, chosen["arrived"]), (room, chosen["percent"])):
        panel.plot(chosen["goal"], value, "o", markersize=13, markerfacecolor="none",
                   markeredgecolor="#d55e00", markeredgewidth=1.6, zorder=5)

    reach.axhspan(0, REACH, color="#999999", alpha=0.10, zorder=0)
    reach.axhline(REACH, color="#999999", linewidth=0.9, linestyle=":")
    reach.text(3000, REACH + 0.6, f"reachable: {REACH} of 32", fontsize=8, color="#666666", ha="right")
    reach.set_ylabel("days hindsight reaches the goal\n(of 32)")
    reach.set_ylim(0, 34)
    reach.set_title("Which tasks have any room in them: twenty candidates, one qualifier\n"
                    "shaded: ruled out. circled: the task the sweep flies",
                    fontsize=11, loc="left", color=INK)
    reach.legend(frameon=False, fontsize=8, ncol=4, loc="lower left")

    room.axhspan(0, MATERIAL, color="#999999", alpha=0.10, zorder=0)
    room.axhline(MATERIAL, color="#999999", linewidth=0.9, linestyle=":")
    room.text(3000, MATERIAL + 1.5, f"not free: {MATERIAL}% above hindsight", fontsize=8, color="#666666", ha="right")
    room.set_ylabel("a blind plan's cost above hindsight\non a typical day (%)")
    room.set_xlabel("how far away the goal is (km)")
    room.set_xticks([r["goal"] for r in rows if r["direction"] == "downwind"])

    for panel in (reach, room):
        panel.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    into = Path(asked.into)
    into.mkdir(parents=True, exist_ok=True)
    path = into / "scouting.png"
    figure.savefig(path)
    plt.close(figure)
    print(f"wrote {path}")
    print(f"  qualifies: {chosen['direction']} at {chosen['goal']:g} km, "
          f"reached on {chosen['arrived']}/32, blind plan {chosen['percent']}% above hindsight")


if __name__ == "__main__":
    main()
