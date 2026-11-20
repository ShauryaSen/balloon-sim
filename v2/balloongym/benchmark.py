"""Scoring: every controller, learned or not, flown by the simulator's own fleet.fly on the same test launches.

    from balloongym import benchmark
    coast = benchmark.evaluate(task, wind, "coast")
    mpc = benchmark.evaluate(task, wind, "mpc")
    mine = benchmark.evaluate(task, wind, "ppo", params=params)
    benchmark.normalized(mine, coast, mpc)          # 0 = coast, 1 = MPC

Controllers (balloongym.sim.fleet unless noted):

    coast        never changes altitude: the floor
    fixed_plan   plans the whole flight once on the forecast, then flies it blind (fleet tasks)
    mpc          replans every move (station keeping: every 72 min) on the belief, the forecast corrected by the
                 readings: the GP belief on the fleet tasks, the measured kernel on station keeping
    mpc_blind    the same, readings set aside: plans on the forecast alone
    mpc_truth    plans on the true wind with the same horizon: what MPC would do with a perfect forecast
    hindsight    knows the true wind and plans the whole remaining flight every move: the ceiling (fleet tasks)
    ppo          a trained PPO policy (balloongym.sim.rl.controller), acting on its mean; pass params=
    qrdqn        a trained QR-DQN policy (station keeping, balloongym.sim.qrdqn.evaluate); pass params=

Each launch uses the sweeps' keys for its seed (balloongym.sim.cells.keys), so launch, forecast error and flight
are the same for every controller, and results compare with every sweep result on the same seeds.

`value`: what the benchmark ranks on, higher better: time within 50 km for station keeping, minus the summed
cost (km) for the fleet tasks. The normalized score is (agent - coast) / (reference - coast) on it, a ratio of
means over the launches (as balloongym.sim.fidelity.score).
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from balloongym.env import MEASURED_KERNEL, TEST_SEEDS, StationEnv
from balloongym.sim import fleet, qrdqn, rl
from balloongym.sim.belief import GPBelief, KernelBelief, MeasuredKernel, Truth, forecast
from balloongym.sim.cells import keys
from balloongym.sim.config import BeliefConfig
from balloongym.sim.qrdqn import missions as station_missions
from balloongym.tasks import Task
from balloongym.wind import Wind

CONTROLLERS = ("coast", "fixed_plan", "mpc", "mpc_blind", "mpc_truth", "hindsight", "ppo", "qrdqn")


def _controller(task: Task, wind: Wind, name: str, mission, k_noise, params=None, policy_spec=None):
    t = task
    truth = wind.truth
    if name == "coast":
        return fleet.coast(mission)
    if name == "mpc_truth":
        return fleet.mpc(t.balloon, t.planner, Truth(truth), t.objective, mission, 0, t.n_moves, t.radius_km, learns=False)
    if t.name == "station_keeping":
        sigma = StationEnv.build(t, wind).config.sigma
        believed = KernelBelief(wind.earlier, "measured", sigma, BeliefConfig(), MeasuredKernel.load(MEASURED_KERNEL))
        if name in ("mpc", "mpc_blind"):
            belief = believed if name == "mpc" else Truth(wind.earlier)
            return fleet.mpc(t.balloon, t.planner, belief, "station", mission, 0, t.n_moves, t.radius_km, learns=name == "mpc")
        raise ValueError(f"{name} is not run on station keeping")
    believed = GPBelief(t.belief, forecast(t.belief, k_noise, truth))
    if name in ("mpc", "mpc_blind"):
        return fleet.mpc(t.balloon, t.planner, believed, t.objective, mission, 0, t.n_moves, t.radius_km, learns=name == "mpc")
    if name == "fixed_plan":
        return fleet.fixed_plan(t.balloon, t.planner, believed, t.objective, mission, t.n_moves, 0, t.radius_km)
    if name == "hindsight":
        return fleet.hindsight(t.balloon, t.planner, truth, t.objective, mission, t.n_moves, t.radius_km)
    if name == "ppo":
        spec = policy_spec or rl.Spec(t.objective, t.n_balloons, True, t.n_moves, t.radius_km, t.balloon, t.belief)
        return rl.controller(spec, truth, params, mission, rl.noise_from_key(t.belief, k_noise), jnp.asarray(t.belief.sigma))
    raise ValueError(f"no controller {name!r}; one of {CONTROLLERS}")


def evaluate(task: Task, wind: Wind, controller: str, seeds=TEST_SEEDS, params=None, policy_spec=None, qr_config=None,
             log=None) -> dict[str, np.ndarray]:
    """Fly `controller` on each test launch. Returns per-launch arrays: value (higher better), cost_km, score
    (the task's own: time within the radius, balloons arrived, points visited, or final km from launch), seconds"""
    if controller == "qrdqn":
        return _evaluate_qrdqn(task, wind, params, seeds, qr_config or qrdqn.QRConfig())
    out = {"value": [], "cost_km": [], "score": [], "seconds": []}
    for seed in seeds:
        k_task, k_noise, k_fly = keys(int(seed))
        mission, _ = fleet.draw_mission(task.task, k_task, wind.truth)
        c = _controller(task, wind, controller, mission, k_noise, params, policy_spec)
        began = time.perf_counter()
        f = fleet.fly(task.balloon, c, wind.truth, mission, task.objective, task.n_moves, k_fly, task.radius_km)
        cost = float(f.cost_km)
        score = {"station": f.covered_fraction, "reach": f.arrived_count / task.n_balloons,
                 "targets": f.visited_count / task.task.n_targets, "escape": f.escape_final_km}[task.objective]
        out["cost_km"].append(cost)
        out["score"].append(float(score))
        out["value"].append(float(f.covered_fraction) if task.objective == "station" else -cost)
        out["seconds"].append(time.perf_counter() - began)
        if log:
            log(f"{task.name} {controller} seed {seed}: cost {cost:.0f} km, score {float(score):.3f}")
    return {k: np.array(v) for k, v in out.items()}


def _evaluate_qrdqn(task: Task, wind: Wind, params, seeds, c: qrdqn.QRConfig) -> dict[str, np.ndarray]:
    if task.name != "station_keeping":
        raise ValueError("QR-DQN is trained and scored on station keeping")
    env = StationEnv.build(task, wind)
    m = station_missions(task.task, wind.truth, seeds)
    twr = np.asarray(qrdqn.evaluate(jax.tree.map(jnp.asarray, params), c, env.config, wind.truth, wind.earlier, m))
    return {"value": twr, "score": twr, "cost_km": np.full(len(twr), np.nan), "seconds": np.zeros(len(twr))}


def normalized(agent: dict, floor: dict, reference: dict) -> float:
    """(agent - floor) / (reference - floor) on the mean value: 0 is the floor (coast), 1 the reference (MPC)"""
    a, f, r = (float(np.mean(x["value"])) for x in (agent, floor, reference))
    return (a - f) / (r - f)


def summary(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Mean, interquartile mean (Agarwal et al. 2021) and a 95% bootstrap interval of the mean"""
    x = np.sort(np.asarray(values, float))
    q = len(x) // 4
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, (n_boot, len(x))).mean(axis=1)
    return {"mean": float(x.mean()), "iqm": float(x[q: len(x) - q].mean()),
            "low": float(np.percentile(boots, 2.5)), "high": float(np.percentile(boots, 97.5)), "n": int(len(x))}
