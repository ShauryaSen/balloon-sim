"""Training the two RL baselines, each the simulator's own trainer:

    PPO      balloongym.sim.rl.train: one policy shared by every balloon (MAPPO with parameter sharing; plain PPO
             for one balloon), the fleet tasks. Settings: RLConfig (PPO's usual, untuned).
    QR-DQN   balloongym.sim.qrdqn: BLE's published recipe (decision 57), station keeping. Settings: QRConfig.

Training launches come from seeds TRAIN_SEED_OFFSET (100,000) up and never meet the test seeds.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from balloongym.env import TRAIN_SEED_OFFSET, StationEnv, draw_missions
from balloongym.sim import qrdqn, rl
from balloongym.sim.config import RLConfig
from balloongym.sim.qrdqn import QRConfig
from balloongym.sim.qrdqn import missions as station_missions
from balloongym.tasks import Task
from balloongym.wind import Wind

VALIDATION_SEED_OFFSET = 200_000  # QR-DQN picks its best weights on these, never on the test seeds


def train_ppo(task: Task, wind: Wind, config: RLConfig = RLConfig(), seed: int = 0, learns: bool = True,
              view: str = "column", sigmas: tuple[float, ...] | None = None, log=print) -> tuple[dict, rl.Spec, list[dict]]:
    """PPO on a fleet task. Each training episode draws a launch from config.train_seeds training seeds and a
    forecast error at one of `sigmas` (default: the task's own). Returns (params, spec, training curve)."""
    if task.name == "station_keeping":
        raise ValueError("station keeping trains QR-DQN (train_qrdqn); PPO is for the fleet tasks")
    spec = rl.Spec(task.objective, task.n_balloons, learns, task.n_moves, task.radius_km, task.balloon, task.belief, view)
    missions = draw_missions(task, wind.truth, range(TRAIN_SEED_OFFSET, TRAIN_SEED_OFFSET + config.train_seeds))
    params, curve = rl.train(spec, wind.truth, missions, sigmas or (task.belief.sigma,), config, jax.random.key(seed), log=log)
    return params, spec, curve


def train_qrdqn(task: Task, wind: Wind, config: QRConfig = QRConfig(), seed: int = 0, steps: int = 200_000, chunk: int = 500,
                eval_every: int = 5000, train_missions: int = 2048, log=print) -> tuple[list, list[dict]]:
    """QR-DQN on station keeping, as scripts/train_qrdqn.py runs it: `steps` vector steps of config.envs
    balloons, validated on 64 launches every `eval_every`. Returns the best-on-validation weights and the curve."""
    if task.name != "station_keeping":
        raise ValueError("QR-DQN is BLE's station-keeping agent")
    env = StationEnv.build(task, wind)
    train = station_missions(task.task, wind.truth, range(TRAIN_SEED_OFFSET, TRAIN_SEED_OFFSET + train_missions))
    valid = station_missions(task.task, wind.truth, range(VALIDATION_SEED_OFFSET, VALIDATION_SEED_OFFSET + 64))
    start, run = qrdqn.make(config, env.config, wind.truth, wind.earlier, train)
    key = jax.random.key(seed)
    carry = start(jax.random.fold_in(key, 0))
    curve, best, best_params, began = [], -1.0, None, time.time()
    for done in range(0, steps, chunk):
        carry, (reward, loss) = run(carry, jax.random.fold_in(key, done + 1), chunk)
        row = {"vector_steps": done + chunk, "transitions": (done + chunk) * config.envs, "reward": float(jnp.mean(reward)),
               "loss": float(loss[-1]), "minutes": (time.time() - began) / 60}
        if (done + chunk) % eval_every == 0 or done + chunk >= steps:
            v = float(jnp.mean(qrdqn.evaluate(carry[0].params, config, env.config, wind.truth, wind.earlier, valid)))
            row["validation_twr50"] = v
            if v > best:
                best, best_params = v, jax.tree.map(np.asarray, carry[0].params)
        curve.append(row)
        log(row)
    return best_params, curve
