"""Scoring a policy the same way every time.

    scores = evaluate(env, field, policy)                 # the task's score on each test launch
    ref = reference_scores(env, field)                    # coast and MPC on the same launches
    normalized_score(scores, ref)                         # 0 = coast, 1 = MPC

Test launches come from `TEST_SEEDS`; train on keys from `train_keys`, which never overlap them.
The normalized score is (agent - coast) / (MPC - coast), a ratio of means over the launches, so a policy that
beats MPC scores above 1. `summary` reports the mean, the interquartile mean (Agarwal et al. 2021) and a 95%
bootstrap interval.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from balloongym import controllers
from balloongym.env import BalloonEnv, Policy, rollout
from balloongym.wind import WindField

TEST_SEEDS = tuple(range(64))
TRAIN_SEED_OFFSET = 1_000_000


def test_keys(n: int = len(TEST_SEEDS)) -> jax.Array:
    return jnp.stack([jax.random.key(s) for s in TEST_SEEDS[:n]])


def train_keys(key: jax.Array, n: int) -> jax.Array:
    """n launch keys for training, drawn far from the test seeds"""
    seeds = TRAIN_SEED_OFFSET + jax.random.randint(key, (n,), 0, 2**30)
    return jax.vmap(jax.random.key)(seeds)


def evaluate(env: BalloonEnv, field: WindField, policy: Policy, n: int = len(TEST_SEEDS)) -> np.ndarray:
    """The task's score on each of the first n test launches"""
    run = jax.jit(jax.vmap(lambda k: rollout(env, field, k, policy).score))
    return np.asarray(run(test_keys(n)))


def reference_scores(env: BalloonEnv, field: WindField, n: int = len(TEST_SEEDS), **mpc_kwargs) -> dict[str, np.ndarray]:
    return {
        "coast": evaluate(env, field, controllers.coast(env), n),
        "mpc": evaluate(env, field, controllers.mpc(env, field, **mpc_kwargs), n),
    }


def normalized_score(scores: np.ndarray, reference: dict[str, np.ndarray]) -> float:
    floor, top = float(np.mean(reference["coast"])), float(np.mean(reference["mpc"]))
    return (float(np.mean(scores)) - floor) / (top - floor)


def iqm(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x))
    k = len(x) // 4
    return float(np.mean(x[k: len(x) - k]))


def summary(scores: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores)
    boots = [np.mean(rng.choice(scores, len(scores))) for _ in range(n_boot)]
    return {"mean": float(np.mean(scores)), "iqm": iqm(scores),
            "low": float(np.percentile(boots, 2.5)), "high": float(np.percentile(boots, 97.5))}
