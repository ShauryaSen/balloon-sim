"""Fly one episode of each task with the MPC reference controller and save a picture of it.

    python examples/quickstart.py
"""

import gymnasium as gym
import matplotlib.pyplot as plt

import balloongym  # noqa: F401  (registers the environments)

for name in ["StationKeeping", "Reach", "MaxDistance", "Waypoints"]:
    env = gym.make(f"BalloonGym/{name}-v0", render_mode="rgb_array")
    obs, info = env.reset(seed=0)
    done = False
    while not done:
        action = env.unwrapped.reference_action("mpc")  # or env.action_space.sample(), or your policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    print(f"{name:15s} score {info['score']:.3f}")
    plt.imsave(f"{name}.png", env.render())
