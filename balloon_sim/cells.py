"""Which flights a sweep is made of, and the keys one flight is drawn from

A cell names one flight. `cells` expands an ExperimentConfig into every flight the sweep needs, leaving out
the combinations that would repeat one: a controller is only given the axes it actually reads.

    hindsight    reads the truth, so sigma, the draws and the horizon change nothing; the goal still does
    fixed_plan   plans the whole episode at once, so the MPC horizon changes nothing
    mpc          replans and learns from its own readings: reads every axis
    mpc_blind    replans on the forecast alone, never learning from its readings

The sweep is not a full grid. A full grid over noise, horizon, draws, goal distance and belief scale is tens
of thousands of flights, nearly all of them answering nothing: a cell that moves two knobs at once cannot say
which one did the work. So there is a main arm at one goal distance with the belief calibrated, and then one
small arm per extra question, each moving a single axis away from that base. Compare inside an arm.

The keys matter as much as the cells. Everything an episode needs that is not the controller comes from the
seed alone: the task, the forecast's error and the flight's own randomness. So within a seed the start, the
goal and the true wind are the same for every controller and every sigma (decision 1), and the error field is
one pattern scaled by sigma rather than a fresh draw at each level. Two cells of the same seed are therefore
paired, and a difference between them is the controller or the noise, never the weather.

One thing the goal distance does break: it changes where the goal is, so it changes the task. Cells at
different goal distances are not paired with each other, and each distance carries its own hindsight.
"""

from typing import NamedTuple

import jax

from balloon_sim.config import ExperimentConfig

BASE_HORIZON = 20
"""The horizon the side arms hold fixed: 60 h, the longest that was ever detectably better than the next one
down on the 1,000 km sweep"""

BASE_GOAL_KM = 2000.0
"""The goal distance the main arm and every side arm except the distance arm use. Chosen by the scouting
arm (docs/scouting-rule.md): the only task of twenty that is both reachable and not free"""

BASE_DIRECTION = "downwind"
"""Where the goal sits relative to the flow, for every arm but the scouting one. The scouting arm kept it:
crosswind, upwind and random were unreachable at every distance it tried, even 300 km"""


class Cell(NamedTuple):
    controller: str  # hindsight, fixed_plan, mpc or mpc_blind
    seed: int
    direction: str  # where the goal sits relative to the flow: downwind, crosswind, upwind or random
    sigma: float  # the error actually put into the forecast, in m/s
    horizon: int  # for hindsight and the fixed plan, the whole episode: they do not use it
    n_winds: int  # belief draws the plan is averaged over; 0 is the belief's mean
    goal_km: float  # how far downwind the goal is
    belief_scale: float  # what the belief assumes the error is, as a multiple of sigma; 1 is calibrated
    nested: bool  # whether the draws are a subset of one shared pool rather than drawn fresh


def _ladder(experiment: ExperimentConfig, fitted_sigma: float) -> list[float]:
    """The noise levels in m/s: multiples of the record's own day-old error"""
    return [round(multiple * fitted_sigma, 4) for multiple in experiment.sigma_multiples]


def main_arm(experiment: ExperimentConfig, n_moves: int, fitted_sigma: float) -> list[Cell]:
    """The sweep proper: every controller over the noise ladder, at the base goal with a calibrated belief.

    This is the 1,000 km sweep of 2026-10-22 again under the arrival rule, with mpc_blind added. Everything
    the headline claims are read off comes from here.
    """
    out: list[Cell] = []
    for seed in experiment.seeds:
        out.append(Cell("hindsight", seed, BASE_DIRECTION, 0.0, n_moves, 0, BASE_GOAL_KM, 1.0, False))
        for sigma in _ladder(experiment, fitted_sigma):
            out.extend(Cell("fixed_plan", seed, BASE_DIRECTION, sigma, n_moves, n, BASE_GOAL_KM, 1.0, False) for n in experiment.fixed_plan_n_winds)
            out.extend(
                Cell(which, seed, BASE_DIRECTION, sigma, horizon, n, BASE_GOAL_KM, 1.0, False)
                for which, draws in (("mpc", experiment.mpc_n_winds), ("mpc_blind", (0,)))
                for n in draws
                for horizon in experiment.horizons
            )
    return out


def distance_arm(experiment: ExperimentConfig, n_moves: int, fitted_sigma: float) -> list[Cell]:
    """Harder tasks: the same noise ladder at the other goal distances, one horizon and the mean alone.

    Each distance is its own task, so it carries its own hindsight. Only the base horizon and the belief's
    mean are run, since the question here is whether the ordering and the break survive a goal that is barely
    reachable, not how the horizon behaves at every distance.
    """
    out: list[Cell] = []
    for goal_km in experiment.goal_distances:
        if goal_km == BASE_GOAL_KM:
            continue  # the main arm already flies it
        for seed in experiment.seeds:
            out.append(Cell("hindsight", seed, BASE_DIRECTION, 0.0, n_moves, 0, goal_km, 1.0, False))
            for sigma in _ladder(experiment, fitted_sigma):
                out.append(Cell("fixed_plan", seed, BASE_DIRECTION, sigma, n_moves, 0, goal_km, 1.0, False))
                out.extend(Cell(which, seed, BASE_DIRECTION, sigma, BASE_HORIZON, 0, goal_km, 1.0, False) for which in ("mpc", "mpc_blind"))
    return out


def calibration_arm(experiment: ExperimentConfig, n_moves: int, fitted_sigma: float) -> list[Cell]:
    """A belief that is wrong about its own uncertainty, at the base goal and horizon.

    Both the mean and 16 draws are run, because the point of the arm is the two together: if hedging over
    draws only pays when the belief is miscalibrated, the gap between them should open here and nowhere else.
    Zero noise is skipped, since a belief scaled off nothing is still nothing.
    """
    out: list[Cell] = []
    for seed in experiment.seeds:
        for sigma in _ladder(experiment, fitted_sigma):
            if sigma == 0.0:
                continue
            for scale in experiment.belief_scales:
                out.extend(Cell("fixed_plan", seed, BASE_DIRECTION, sigma, n_moves, n, BASE_GOAL_KM, scale, False) for n in (0, 16))
                out.extend(Cell("mpc", seed, BASE_DIRECTION, sigma, BASE_HORIZON, n, BASE_GOAL_KM, scale, False) for n in (0, 16))
    return out


def nested_arm(experiment: ExperimentConfig, n_moves: int, fitted_sigma: float) -> list[Cell]:
    """Draw count with the draws held fixed, at the three noise levels where hedging could matter.

    Every cell here draws the full pool and uses the first n of it, so these arms cost what the largest one
    costs. That is why it is three noise levels and one horizon rather than the whole ladder.
    """
    out: list[Cell] = []
    loud = [sigma for sigma in _ladder(experiment, fitted_sigma) if sigma > 0][-3:]
    for seed in experiment.seeds:
        for sigma in loud:
            for n in (1, 4, experiment.nested_pool):
                out.append(Cell("fixed_plan", seed, BASE_DIRECTION, sigma, n_moves, n, BASE_GOAL_KM, 1.0, True))
                out.append(Cell("mpc", seed, BASE_DIRECTION, sigma, BASE_HORIZON, n, BASE_GOAL_KM, 1.0, True))
    return out


def scout_arm(experiment: ExperimentConfig, n_moves: int, fitted_sigma: float) -> list[Cell]:
    """Which tasks have any room in them: the ceiling and the floor alone, over every direction and distance.

    A task is worth sweeping only if a controller that knows the wind beats one that is flying blind. Where
    hindsight and the fixed plan score the same, the wind is doing all the work and no controller can show
    anything, however good it is; where hindsight cannot reach the goal either, the task is measuring the
    wind rather than the control. The gap between the two, and hindsight's own reach, are what this arm
    measures, at one noise level and one horizon. Two controllers over 4 directions and 5 distances is
    1,280 cells, against 7,904 for the sweep it chooses the setting for.

    docs/scouting-rule.md fixes how the answer is read, and was written before this ran.
    """
    out: list[Cell] = []
    sigma = round(fitted_sigma, 4)  # one level, the record's own error: the sweep varies it, this does not
    for direction in experiment.scout_directions:
        for goal_km in experiment.scout_distances:
            for seed in experiment.seeds:
                out.append(Cell("hindsight", seed, direction, 0.0, n_moves, 0, goal_km, 1.0, False))
                out.append(Cell("fixed_plan", seed, direction, sigma, n_moves, 0, goal_km, 1.0, False))
    return out


ARMS = {"scout": scout_arm, "main": main_arm, "distance": distance_arm, "calibration": calibration_arm, "nested": nested_arm}


SWEEP = ("main", "distance", "calibration", "nested")
"""The sweep proper: everything except the scouting arm, which is run first and on its own"""


def cells(experiment: ExperimentConfig, n_moves: int, fitted_sigma: float, arms: tuple[str, ...] = SWEEP) -> list[Cell]:
    """Every flight the sweep runs, with no two cells describing the same flight. fitted_sigma is the record's
    own day-old error (BeliefConfig.sigma); the sweep's levels are multiples of it"""
    seen: dict[Cell, None] = {}
    for name in arms:
        for cell in ARMS[name](experiment, n_moves, fitted_sigma):
            seen.setdefault(cell)
    return list(seen)


def keys(seed: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The task, the forecast error and the flight, each drawn from the seed and nothing else"""
    return tuple(jax.random.split(jax.random.key(seed), 3))  # type: ignore[return-value]
