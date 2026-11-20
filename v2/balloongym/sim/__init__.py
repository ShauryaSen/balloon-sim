"""Shaurya's simulator (balloon_sim), copied unchanged from ShauryaSen/balloon-sim at b50295b (2026-11-18).

Only the import path is changed (balloon_sim -> balloongym.sim). Everything balloongym runs is built on these
modules: the wind, the dynamics, the forecast and belief, the fleet tasks and the MPC planner, PPO (rl.py),
the BLE-form station-keeping environment and QR-DQN. DESIGN.md in the repo root documents the decisions.

To pick up a later change from balloon_sim, copy the module again and rerun the tests.
"""
