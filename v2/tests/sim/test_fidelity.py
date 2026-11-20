import jax
import numpy as np

from balloongym.sim import degrade, fidelity
from tests.sim.test_degrade import rough_field


def test_the_sweep_has_no_repeats_and_one_coast_per_seed():
    todo = fidelity.cells((0, 1))
    assert len(todo) == len(set(todo))
    assert sum(c.controller == "coast" for c in todo) == 2
    assert len(todo) == 2 * 42  # coast, 22 base, 7 climb, 12 horizon


def test_the_score_is_zero_with_nothing_lost_and_one_at_coast():
    truth, coast = np.array([0.6, 0.5, 0.7]), np.array([0.1, 0.2, 0.1])
    assert abs(fidelity.score(truth, truth, coast)["score"]) < 1e-9
    assert abs(fidelity.score(truth, coast, coast)["score"] - 1.0) < 1e-9
    half = fidelity.score(truth, (truth + coast) / 2, coast)
    assert abs(half["score"] - 0.5) < 1e-9 and half["low"] <= 0.5 <= half["high"]


def test_a_uniform_offset_has_wind_error_but_no_level_difference_error():
    field = rough_field()
    shifted = degrade.offset(field, jax.random.PRNGKey(0), 3.0, by_altitude=False)
    measured = degrade.errors(field, shifted, (-90.0, 90.0))
    assert abs(measured["wind_error_ms"] - 3.0) < 1e-3 and measured["level_difference_error_ms"] < 1e-4
