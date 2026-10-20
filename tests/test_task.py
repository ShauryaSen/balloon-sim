import jax
import jax.numpy as jnp
import pytest

from balloon_sim.config import TaskConfig
from balloon_sim.dynamics import distance_km
from balloon_sim.task import draw_task
from balloon_sim.wind import WindField

TRUTH = WindField(jnp.array([0.0, 240.0]), jnp.array([15.0, 20.0]), jnp.array([20.0, 50.0]), jnp.array([120.0, 240.0]), jnp.zeros((2, 2, 2, 2, 2)))


def test_random_goal_is_the_right_distance_away_and_inside_the_box():
    for seed in range(20):
        task = draw_task(TaskConfig(), jax.random.key(seed), TRUTH)
        assert jnp.allclose(distance_km(task.start.lat, task.start.lon, task.goal_lat, task.goal_lon), 1000.0, atol=1.0)
        for lat, lon in [(task.start.lat, task.start.lon), (task.goal_lat, task.goal_lon)]:
            assert 23.0 <= lat <= 47.0 and 123.0 <= lon <= 237.0  # inside the 3 degree margin
        assert task.start.altitude_km == 17.5 and task.start.hours == 0.0  # middle altitude, record start


def test_random_tasks_differ_by_seed():
    a, b = draw_task(TaskConfig(), jax.random.key(0), TRUTH), draw_task(TaskConfig(), jax.random.key(1), TRUTH)
    assert a.start.lat != b.start.lat


def test_fixed_mode_uses_the_given_coordinates():
    config = TaskConfig(mode="fixed", start_lat=35.0, start_lon=180.0, goal_lat=35.0, goal_lon=191.0)
    task = draw_task(config, jax.random.key(0), TRUTH)
    assert (float(task.start.lat), float(task.start.lon), float(task.goal_lat), float(task.goal_lon)) == (35.0, 180.0, 35.0, 191.0)


def test_fixed_mode_without_coordinates_is_an_error():
    with pytest.raises(ValueError):
        draw_task(TaskConfig(mode="fixed"), jax.random.key(0), TRUTH)


def test_a_goal_too_far_for_the_box_is_an_error():
    with pytest.raises(ValueError):
        draw_task(TaskConfig(distance_km=20_000.0), jax.random.key(0), TRUTH)
