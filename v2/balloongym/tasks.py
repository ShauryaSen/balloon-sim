"""The benchmark's tasks, each a fixed set of the simulator's settings.

Two families, because the simulator has two control setups:

    station_keeping   one balloon holds a point for 2 days (balloongym.sim.station_env, decisions 54 and 57):
                      an action every 3 min (up, stay, down), BLE's observation and reward, launches within 10
                      degrees of the equator. Score: time within 50 km (TWR50). RL: QR-DQN (BLE's recipe).
                      Reference MPC: Pasetsky et al.'s setup, 12 h horizon, replan every 72 min, on the
                      forecast corrected by readings with the measured kernel.

    reach             get within 200 km of a goal 2,000 km downwind, and get there soon
    max_distance      get as far from launch as possible ("escape")
    waypoints         visit three points in a downwind chain, 2,000 km in all ("targets")
                      These three are the sweep's fleet objectives (balloongym.sim.fleet, decisions 45, 47, 48):
                      a climb every 3 h, 50 moves (6.25 days), launches 20N to 50N, one balloon or a fleet
                      sharing one belief. Cost is the objective's distance summed over the moves, in km (lower
                      is better); the reward is minus each move's cost per 1,000 km. RL: PPO (balloongym.sim.rl).
                      Reference MPC: the sweep's, 20-move horizon, on the GP belief around the forecast,
                      conditioned on every reading.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from balloongym.sim import fidelity
from balloongym.sim.config import BalloonConfig, BeliefConfig, PlannerConfig, TaskConfig

TASKS = ("station_keeping", "reach", "max_distance", "waypoints")
OBJECTIVE = {"station_keeping": "station", "reach": "reach", "max_distance": "escape", "waypoints": "targets"}
FLEET_HORIZON = 20  # ExperimentConfig.new_objective_horizon: the sweep's MPC horizon for the fleet objectives


@dataclass(frozen=True)
class Task:
    name: str
    objective: str  # the simulator's name for it
    task: TaskConfig
    balloon: BalloonConfig
    planner: PlannerConfig  # the reference MPC's
    belief: BeliefConfig  # the forecast error (fleet tasks); its sigma is the error's size

    @property
    def n_balloons(self) -> int:
        return self.task.n_agents

    @property
    def n_moves(self) -> int:
        return self.task.n_moves

    @property
    def radius_km(self) -> float:
        return self.task.arrival_km


def make_task(name: str, n_balloons: int = 1, sigma_ms: float | None = None, **task_overrides) -> Task:
    """A benchmark task. `sigma_ms` sets the fleet tasks' forecast error (default 3.2 m/s, a day-old forecast).
    Any simulator TaskConfig field can be overridden, e.g. distance_km=1000 or n_moves=80."""
    if name not in TASKS:
        raise ValueError(f"task {name!r} is not one of {TASKS}")
    belief = BeliefConfig() if sigma_ms is None else dataclasses.replace(BeliefConfig(), sigma=sigma_ms)
    if name == "station_keeping":
        if n_balloons != 1:
            raise ValueError("station keeping is one balloon (BLE's form); fleets are on the other tasks")
        balloon, planner, task = fidelity.configs(fidelity.Cell("mpc", 0, "truth", 0.0, 1.0, fidelity.BASE_HORIZON_HOURS, "base"))
        return Task(name, "station", dataclasses.replace(task, **task_overrides), balloon, planner, belief)
    task = TaskConfig(objective=OBJECTIVE[name], n_agents=n_balloons)
    planner = PlannerConfig(horizon=FLEET_HORIZON)
    return Task(name, OBJECTIVE[name], dataclasses.replace(task, **task_overrides), BalloonConfig(), planner, belief)
