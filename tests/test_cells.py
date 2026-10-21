import jax
import jax.numpy as jnp

from balloon_sim.cells import cells, keys
from balloon_sim.config import ExperimentConfig


def test_every_cell_is_a_distinct_flight_and_hindsight_runs_once_per_seed():
    experiment = ExperimentConfig(seeds=(0, 1), sigma_multiples=(0.0, 1.0), horizons=(10, 20), mpc_n_winds=(0, 8), fixed_plan_n_winds=(0, 8, 64))
    grid = cells(experiment, n_moves=50, fitted_sigma=5.9)
    assert len(grid) == len(set(grid))
    assert sum(c.controller == "hindsight" for c in grid) == 2  # one per seed, at sigma 0
    assert sum(c.controller == "fixed_plan" for c in grid) == 2 * 2 * 3  # seeds x sigmas x its n_winds
    assert sum(c.controller == "mpc" for c in grid) == 2 * 2 * 2 * 2  # seeds x sigmas x its n_winds x horizons
    assert {c.sigma for c in grid if c.controller != "hindsight"} == {0.0, 5.9}  # multiples of the fitted sigma


def test_a_seed_draws_the_same_task_noise_and_flight_keys_every_time():
    a, b = keys(3), keys(3)
    assert all(bool(jnp.array_equal(jax.random.key_data(x), jax.random.key_data(y))) for x, y in zip(a, b))
    assert not bool(jnp.array_equal(jax.random.key_data(keys(3)[0]), jax.random.key_data(keys(4)[0])))
