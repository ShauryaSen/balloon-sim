"""balloongym: stratospheric balloon control with real wind and realistic forecasts, built on the project's
simulator (balloongym.sim).

Importing the package registers the Gymnasium environments:

    BalloonGym/StationKeeping-v0   hold within 50 km of a point for 2 days, an action every 3 min (BLE's form)
    BalloonGym/Reach-v0            get within 200 km of a goal 2,000 km downwind, soon; a climb every 3 h
    BalloonGym/MaxDistance-v0      get as far from launch as possible in 6.25 days
    BalloonGym/Waypoints-v0        visit three points in a 2,000 km downwind chain

and ReachFleet-v0, MaxDistanceFleet-v0 and WaypointsFleet-v0, with four balloons sharing one belief and reward.
"""

from gymnasium.envs.registration import register

from balloongym.env import FleetEnv, StationEnv, make_env, missions_for
from balloongym.tasks import TASKS, Task, make_task
from balloongym.wind import Wind, load, synthetic

__version__ = "0.2.0"

ENV_IDS = {"station_keeping": "StationKeeping", "reach": "Reach", "max_distance": "MaxDistance", "waypoints": "Waypoints"}

for _task, _name in ENV_IDS.items():
    register(id=f"BalloonGym/{_name}-v0", entry_point="balloongym.gym_env:BalloonGymEnv", kwargs={"task": _task})
    if _task != "station_keeping":
        register(id=f"BalloonGym/{_name}Fleet-v0", entry_point="balloongym.gym_env:BalloonGymEnv",
                 kwargs={"task": _task, "n_balloons": 4})

__all__ = ["FleetEnv", "StationEnv", "TASKS", "Task", "Wind", "load", "make_env", "make_task", "missions_for", "synthetic"]
