"""One frozen dataclass of settings per piece"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class BalloonConfig:
    step_hours: float = 3.0  # one move: an ERA5 frame
    substeps: int = 1  # Euler pieces per move
    climb_cap_km: float = 1.1  # the most the balloon can climb or sink in one move


@dataclass(frozen=True)
class PlannerConfig:
    horizon: int = 20  # moves planned ahead
    cost: Literal["summed", "final"] = "summed"  # what the planner minimises
    candidates: int = 100  # random plans in the pool, beside the previous plan and coast
    descend_top: int = 1  # how many of the best candidates are refined
    perturbation: float = 0.2  # half-width of the small random climbs, as a fraction of the cap
    optimizer: Literal["lbfgs", "normalised"] = "lbfgs"  # how the best candidates are refined; "normalised" is Myles's
    iterations: int = 100  # refinement steps, at most
    step_size: float = 1.0  # normalised gradient descent's step
    tolerance: float = 1e-7  # stop refining once the gradient is this small


@dataclass(frozen=True)
class BeliefConfig:
    """The kernel shared by the forecast's error and the belief, so the belief is calibrated by construction.
    Fitted by scripts/fit_kernel.py to the error of a day-old forecast on records/era5_global_jul2021.npz
    between 20N and 50N, where the balloon flies (`--stale-hours 24 --lat 20 50`). The North Pacific box gave
    2.8 m/s, 544 km, 1.2 km, 9.4 h; cBottle gave 5.9, 578, 4.4, 9.9 (docs/cbottle-2026-10-21.md)"""

    sigma: float = 3.2  # m/s per wind component: the size of the forecast's error, the swept noise level
    horizontal_km: float = 544.0  # how far apart two places can be and still have related errors
    altitude_km: float = 1.3  # the same, between altitudes
    hours: float = 9.1  # the same, between times
    n_features: int = 256  # random Fourier features per draw: more is closer to the exact kernel, but slower
    reading_noise: float = 0.01  # m/s the belief assumes a reading is off by: zermelo's jitter, not a real error


@dataclass(frozen=True)
class TaskConfig:
    """Where an episode starts and where its goal is (decision 18)"""

    mode: Literal["random", "fixed"] = "random"
    n_moves: int = 50  # length of an episode: 50 three-hour moves is 6.25 days of the 10-day record
    arrival_km: float = 200.0
    """The episode ends the first time the balloon is this close, and the moves after it cost nothing, so the
    task is reach the goal and reach it soon rather than sit on it (decision 37). 200 km is the radius the
    goal-distance rule was fixed with: about one move of drift at a typical 10 m/s stratospheric wind, and
    about two cells of the 2 degree grid. Zero turns the rule off and gives the plain sum over every move,
    which is what the sweeps before 2026-10-23 scored"""

    start_hours: float = 0.0  # when in the record the episode starts
    # random mode: a start drawn inside the allowed area, the goal distance_km away in a random direction
    distance_km: float = 2000.0  # chosen by docs/scouting-rule.md; nearer, the wind carries the balloon there unaided
    direction: Literal["random", "downwind", "crosswind", "upwind"] = "downwind"
    """Where the goal is put relative to the flow at the start. downwind is carried there by the wind and so
    needs no steering; upwind needs a level that opposes the mean flow; crosswind a quarter turn off it,
    with the side drawn per try; random any bearing at all. Which of these the sweep should use is what
    docs/scouting-rule.md decides"""
    margin_deg: float = 3.0  # start and goal at least this far inside a regional record's edges
    lat_band: tuple[float, float] = (20.0, 50.0)  # launches and goals within these latitudes: summer mid-latitudes
    # fixed mode: these exact coordinates
    start_lat: float | None = None
    start_lon: float | None = None
    goal_lat: float | None = None
    goal_lon: float | None = None


@dataclass(frozen=True)
class WindConfig:
    """Which record an episode flies in"""

    path: str = "records/era5_global_jul2021.npz"  # a .npz holding u, v, hours, altitude_km, latitude, longitude


@dataclass(frozen=True)
class ExperimentConfig:
    """The grid of cells a sweep runs. One cell is one (controller, seed, sigma, horizon, n_winds) flight.

    Not every axis applies to every controller, so `cells` leaves out the combinations that would repeat a
    flight: hindsight reads the truth, so neither sigma nor n_winds nor the horizon changes what it does, and
    the fixed plan commits to the whole episode at once, so the MPC horizon does not reach it either.
    """

    seeds: tuple[int, ...] = tuple(range(32))

    scout_directions: tuple[str, ...] = ("downwind", "crosswind", "upwind", "random")
    scout_distances: tuple[float, ...] = (300.0, 600.0, 1000.0, 2000.0, 3000.0)
    """The scouting arm's grid. It flies only the ceiling and the floor, which is all that is needed to see
    whether a task has any room in it: where hindsight and a blind plan score the same, no controller can
    show anything, however good it is"""

    goal_distances: tuple[float, ...] = (1000.0, 2000.0, 3000.0)
    """How far downwind the goal is. 1,000 km is the rule-chosen distance hindsight reaches on 27 of 32 seeds;
    2,000 (21 of 32) and 3,000 (14 of 32) are there as a difficulty axis, to see whether the breaking point
    moves when the task stops being comfortably reachable"""

    belief_scales: tuple[float, ...] = (0.5, 2.0)
    """What the controller's belief assumes the error is, as a multiple of what it actually is. 1 is the
    calibrated belief every sweep so far has used, where noise and belief share a kernel by construction;
    these two arms are a belief that is wrong about its own uncertainty, under-confident at 2 and
    over-confident at 0.5. If averaging over draws ever pays, it should pay here"""

    nested_pool: int = 16
    """Draws in the shared pool of the nested-draw arm, where N = 1, 4 and 16 are subsets of one pool rather
    than fresh draws, so the arms differ only in how many winds the plan saw"""

    sigma_multiples: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
    """Forecast error as a multiple of BeliefConfig.sigma, the record's own day-old error, so the ladder is 0,
    a quarter, a half, one, two and four times a real 24 h forecast whatever record is flown. Denser below the
    real error than above it, because the zermelo sweep found performance flat up to about half of it and
    falling smoothly after"""

    horizons: tuple[int, ...] = (5, 10, 20, 50)
    """Moves the MPC plans ahead: a doubling ladder (1.25, 2.5 and 5 days) up to the whole 50-move episode,
    where MPC shares the bounds' horizon and differs from them only by its belief and its replanning"""

    mpc_n_winds: tuple[int, ...] = (0, 1, 4, 16)
    """Belief draws an MPC plan is averaged over: 0 is the belief's mean (the certainty-equivalent baseline
    of Myles and zermelo), 1 a single posterior sample, then two doublings to show saturation"""

    fixed_plan_n_winds: tuple[int, ...] = (0, 1, 4, 16, 64)
    """The same for the fixed plan, which solves once so a large N is nearly free; 64 is the best estimate of
    the proxy best-fixed-plan bound"""

    controllers: tuple[str, ...] = ("hindsight", "fixed_plan", "mpc")
