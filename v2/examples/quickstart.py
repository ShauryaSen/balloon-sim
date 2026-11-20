"""Fly one episode of each task with random actions, save a picture of each, and score coast and MPC on reach.

    python examples/quickstart.py
"""

import gymnasium as gym
import matplotlib.pyplot as plt

import balloongym
from balloongym import benchmark

for name in ["StationKeeping", "Reach", "MaxDistance", "Waypoints", "WaypointsFleet"]:
    env = gym.make(f"BalloonGym/{name}-v0", launches=16, render_mode="rgb_array")
    obs, info = env.reset(seed=0)
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())  # your policy(obs) here
        done = terminated or truncated
    print(f"{name:15s} random actions: {info}")
    plt.imsave(f"{name}.png", env.render())

# The reference controllers, on the first 4 test launches
task, wind = balloongym.make_task("reach"), balloongym.synthetic()
coast = benchmark.evaluate(task, wind, "coast", seeds=range(4))
mpc = benchmark.evaluate(task, wind, "mpc", seeds=range(4))
print("reach, summed distance to the goal (km): coast", coast["cost_km"].mean().round(), " MPC", mpc["cost_km"].mean().round())
