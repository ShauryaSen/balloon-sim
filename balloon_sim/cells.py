"""Which flights a sweep is made of, and the keys one flight is drawn from

A cell names one flight. `cells` expands an ExperimentConfig into every flight the sweep needs, leaving out
the combinations that would repeat one: a controller is only given the axes it actually reads.

    hindsight    reads the truth, so sigma, n_winds and the horizon change nothing
    fixed_plan   plans the whole episode at once, so the MPC horizon changes nothing
    mpc          reads all of them

The keys matter as much as the cells. Everything an episode needs that is not the controller comes from the
seed alone: the task, the forecast's error and the flight's own randomness. So within a seed the start, the
goal and the true wind are the same for every controller and every sigma (decision 1), and the error field is
one pattern scaled by sigma rather than a fresh draw at each level. Two cells of the same seed are therefore
paired, and a difference between them is the controller or the noise, never the weather.
"""

from typing import NamedTuple

import jax

from balloon_sim.config import ExperimentConfig


class Cell(NamedTuple):
    controller: str
    seed: int
    sigma: float
    horizon: int  # for hindsight and the fixed plan, the whole episode: they do not use it
    n_winds: int


def cells(experiment: ExperimentConfig, n_moves: int) -> list[Cell]:
    """Every flight the sweep runs, with no two cells describing the same flight"""
    out: list[Cell] = []
    for seed in experiment.seeds:
        if "hindsight" in experiment.controllers:
            out.append(Cell("hindsight", seed, 0.0, n_moves, 0))  # the truth, whatever sigma says
        for sigma in experiment.sigmas:
            for n in experiment.n_winds:
                if "fixed_plan" in experiment.controllers:
                    out.append(Cell("fixed_plan", seed, sigma, n_moves, n))
                if "mpc" in experiment.controllers:
                    out.extend(Cell("mpc", seed, sigma, horizon, n) for horizon in experiment.horizons)
    return out


def keys(seed: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The task, the forecast error and the flight, each drawn from the seed and nothing else"""
    return tuple(jax.random.split(jax.random.key(seed), 3))  # type: ignore[return-value]
