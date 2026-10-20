"""One frozen dataclass of settings per piece"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class BalloonConfig:
    step_hours: float = 3.0  # one move
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
    Lengthscales are zermelo's hand-set values until scripts/fit_kernel.py fits them to a record"""

    sigma: float = 2.8  # m/s per wind component: the size of the forecast's error, the swept noise level
    horizontal_km: float = 800.0  # how far apart two places can be and still have related errors
    altitude_km: float = 3.3  # the same, between altitudes (zermelo's 3 levels, about 1.1 km apart)
    hours: float = 24.0  # the same, between times
    n_features: int = 256  # random Fourier features per draw: more is closer to the exact kernel, but slower


@dataclass(frozen=True)
class TaskConfig:
    """Where an episode starts and where its goal is (decision 18)"""

    mode: Literal["random", "fixed"] = "random"
    n_moves: int = 80  # length of an episode
    start_hours: float = 0.0  # when in the record the episode starts
    # random mode: a start drawn inside the box, the goal distance_km away in a random direction
    distance_km: float = 1000.0
    margin_deg: float = 3.0  # start and goal at least this far inside the box's edges
    # fixed mode: these exact coordinates
    start_lat: float | None = None
    start_lon: float | None = None
    goal_lat: float | None = None
    goal_lon: float | None = None


@dataclass(frozen=True)
class WindConfig:
    """Which record an episode flies in"""

    path: str = "records/era5_npac_wide.npz"  # a .npz holding u, v, hours, altitude_km, latitude, longitude


@dataclass(frozen=True)
class ExperimentConfig:
    """The grid of cells a sweep runs. One cell is one (controller, seed, sigma, horizon, n_winds) flight.

    Not every axis applies to every controller, so `cells` leaves out the combinations that would repeat a
    flight: hindsight reads the truth, so neither sigma nor n_winds nor the horizon changes what it does, and
    the fixed plan commits to the whole episode at once, so the MPC horizon does not reach it either.
    """

    seeds: tuple[int, ...] = tuple(range(32))
    sigmas: tuple[float, ...] = (0.0, 0.7, 1.4, 2.8, 5.6, 11.2)
    """Forecast error in m/s per component. 2.8 is the record's own, so the ladder is 0, a quarter, a half,
    one, two and four times a real 24 h forecast. Denser below the real error than above it, because the
    zermelo sweep found performance flat up to about half of it and falling smoothly after"""

    horizons: tuple[int, ...] = (10, 20, 40)  # moves the MPC plans ahead
    n_winds: tuple[int, ...] = (0, 8)  # belief draws a plan is averaged over; 0 is the belief's mean alone
    controllers: tuple[str, ...] = ("hindsight", "fixed_plan", "mpc")
