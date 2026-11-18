"""The fidelity sweep: which flights it is made of, and the settings every one of them shares (decisions 49, 53-55)

Each flight is one balloon holding a station for 2 days in the true wind, steered by MPC planning on a copy
of that wind that is wrong in one way (balloon_sim.degrade), or by coast, which never changes altitude. The
control setup is Pasetsky et al.'s (decision 54); the score is how much of MPC's time within 50 km a copy
costs, against what coast keeps (decision 55, balloon_sim.fidelity.score).

Three arms, each moving one thing away from the base (MPC, 12 h horizon, base climb limit):

    base       every copy at its ladder's steps, and the five error shapes at 1x; the two offsets also at 2x
    climb      the truth and the levels ladder again at twice the climb limit: does a loss with fewer levels
               come from what the balloon knows, or from how far it can climb between them in one move?
    horizon    the truth and the frames ladder again at 6 h and 24 h horizons: coarse time should stop a long
               horizon from paying

Within a seed every cell flies the same start, station and truth, and a wrong copy draws the same error, so
any two cells of a seed are paired.
"""

from typing import NamedTuple

import numpy as np

from balloon_sim.config import BalloonConfig, PlannerConfig, TaskConfig

STEP_HOURS = 0.05
"""An action every 3 min (Pasetsky et al., Sec. VI-A)"""

CLIMB_KM_PER_3H = 1.1
"""Our climb limit, carried over from every sweep so far; unverified against a real balloon (decision 54)"""

BASE_HORIZON_HOURS = 12.0  # Pasetsky et al.'s best horizon (Fig. 3a)
REPLAN_STEPS = 24  # every 72 min (Fig. 3a)
EPISODE_STEPS = 960  # 2 days
STATION_KM = 50.0  # TWR50
LAUNCH_BAND = (-10.0, 10.0)  # within 10 degrees of the equator, as Pasetsky et al. and BLE

LEVELS = ("levels:10", "levels:9", "levels:6", "levels:3", "levels:2", "levels:1")
FRAMES = ("frames:3", "frames:6", "frames:12", "frames:24", "frames:frozen")  # on an hourly truth, k frames = k hours
GRID = ("grid:2", "grid:4", "grid:8")  # 4, 8 and 16 degrees on the 2 degree truth
SHAPES = ("offset", "offset_altitude", "random", "random_altitude", "persistence")
AT_TWICE = ("offset", "offset_altitude")
OTHER_HORIZONS = (6.0, 24.0)


class Cell(NamedTuple):
    controller: str  # "mpc" or "coast"
    seed: int
    copy: str  # what MPC plans on, a degrade.copy name; coast reads nothing
    size: float  # a wrong copy's error as a multiple of a day-old forecast's (degrade.error_size); 0 for the rest
    climb_scale: float  # the climb limit as a multiple of CLIMB_KM_PER_3H
    horizon_hours: float
    arm: str  # base, climb or horizon

    def args(self) -> str:
        return (f"--controller {self.controller} --seed {self.seed} --copy {self.copy} --size {self.size:g} "
                f"--climb-scale {self.climb_scale:g} --horizon-hours {self.horizon_hours:g} --arm {self.arm}")


def cells(seeds: tuple[int, ...] = tuple(range(32))) -> list[Cell]:
    """Every flight of the sweep, without repeats: one coast per seed, then each arm's MPC cells"""
    wrong = set(SHAPES) - {"persistence"}
    mpc = lambda seed, copy, arm, size=1.0, climb=1.0, horizon=BASE_HORIZON_HOURS: Cell(
        "mpc", seed, copy, size if copy in wrong else 0.0, climb, horizon, arm)
    out: list[Cell] = []
    for seed in seeds:
        out.append(Cell("coast", seed, "truth", 0.0, 1.0, BASE_HORIZON_HOURS, "base"))
        out += [mpc(seed, copy, "base") for copy in ("truth",) + LEVELS + FRAMES + GRID + SHAPES]
        out += [mpc(seed, copy, "base", size=2.0) for copy in AT_TWICE]
        out += [mpc(seed, copy, "climb", climb=2.0) for copy in ("truth",) + LEVELS]
        out += [mpc(seed, copy, "horizon", horizon=h) for h in OTHER_HORIZONS for copy in ("truth",) + FRAMES]
    return out


def configs(cell: Cell) -> tuple[BalloonConfig, PlannerConfig, TaskConfig]:
    """The balloon, planner and task a cell flies with"""
    balloon = BalloonConfig(step_hours=STEP_HOURS, substeps=1, climb_cap_km=CLIMB_KM_PER_3H * STEP_HOURS / 3.0 * cell.climb_scale)
    planner = PlannerConfig(horizon=int(round(cell.horizon_hours / STEP_HOURS)), replan_every=REPLAN_STEPS, candidates=100,
                            optimizer="normalised", iterations=100, step_size=1.0, tolerance=1e-7)
    task = TaskConfig(objective="station", n_agents=1, n_moves=EPISODE_STEPS, arrival_km=STATION_KM, lat_band=LAUNCH_BAND, distance_km=300.0)
    return balloon, planner, task


# --- the score --------------------------------------------------------------------------------------------------


def score(truth_twr: np.ndarray, copy_twr: np.ndarray, coast_twr: np.ndarray, draws: int = 10_000, seed: int = 0) -> dict:
    """The share of MPC's control a copy costs, in time within 50 km, with a paired bootstrap interval.

        score = (mean TWR on the truth − mean TWR on the copy) / (mean TWR on the truth − mean TWR of coast)

    0 is nothing lost, 1 is no better than never steering. A ratio of means over the seeds, not a mean of
    per-seed ratios, because a seed where MPC barely beats coast puts a near-zero denominator into a per-seed
    ratio (docs/dry-run-station-2026-11-17.md). The interval resamples the seeds, keeping each seed's three
    flights together.
    """
    truth_twr, copy_twr, coast_twr = (np.asarray(x, float) for x in (truth_twr, copy_twr, coast_twr))
    ratio = lambda t, c, k: (t.mean() - c.mean()) / (t.mean() - k.mean())
    picks = np.random.default_rng(seed).integers(0, len(truth_twr), (draws, len(truth_twr)))
    resampled = np.array([ratio(truth_twr[i], copy_twr[i], coast_twr[i]) for i in picks])
    low, high = np.percentile(resampled, [2.5, 97.5])
    difference = truth_twr - copy_twr
    return {"score": float(ratio(truth_twr, copy_twr, coast_twr)), "low": float(low), "high": float(high),
            "twr_drop": float(difference.mean()), "n": int(len(truth_twr))}
