"""balloongym: stratospheric balloon control with real wind and realistic forecasts.

Importing the package registers the Gymnasium environments:

    BalloonGym/StationKeeping-v0    stay within 50 km of the launch point for 2 days
    BalloonGym/Reach-v0             get within 100 km of a goal 1,500 km away in 6 days
    BalloonGym/MaxDistance-v0       get as far from launch as possible in 3 days
    BalloonGym/Waypoints-v0         visit 3 points in order, 1,800 km in all, in 6 days

and a fleet version of each with four balloons (StationKeepingFleet-v0 and so on).
"""

from gymnasium.envs.registration import register

from balloongym.env import BalloonEnv, EnvConfig, EnvState, rollout
from balloongym.forecast import ForecastConfig
from balloongym.physics import BalloonConfig
from balloongym.tasks import TASKS, TaskConfig, task_config
from balloongym.wind import WindField, load_record, synthetic_wind

__version__ = "0.1.0"

ENV_IDS = {
    "station_keeping": "StationKeeping",
    "reach": "Reach",
    "max_distance": "MaxDistance",
    "waypoints": "Waypoints",
}

for _task, _name in ENV_IDS.items():
    register(id=f"BalloonGym/{_name}-v0", entry_point="balloongym.gym_env:BalloonGymEnv", kwargs={"task": _task})
    register(id=f"BalloonGym/{_name}Fleet-v0", entry_point="balloongym.gym_env:BalloonGymEnv",
             kwargs={"task": _task, "n_balloons": 4})

__all__ = [
    "BalloonConfig", "BalloonEnv", "EnvConfig", "EnvState", "ForecastConfig", "TASKS", "TaskConfig", "WindField",
    "load_record", "rollout", "synthetic_wind", "task_config",
]
