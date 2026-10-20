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
