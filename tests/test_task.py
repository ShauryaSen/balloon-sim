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
        task = draw_task(TaskConfig(direction="random", distance_km=1000.0), jax.random.key(seed), TRUTH)
        assert jnp.allclose(distance_km(task.start.lat, task.start.lon, task.goal_lat, task.goal_lon), 1000.0, atol=1.0)
        for lat, lon in [(task.start.lat, task.start.lon), (task.goal_lat, task.goal_lon)]:
            assert 23.0 <= lat <= 47.0 and 123.0 <= lon <= 237.0  # inside the 3 degree margin
        assert task.start.altitude_km == 17.5 and task.start.hours == 0.0  # middle altitude, record start


def test_random_tasks_differ_by_seed():
    a, b = draw_task(TaskConfig(direction="random"), jax.random.key(0), TRUTH), draw_task(TaskConfig(direction="random"), jax.random.key(1), TRUTH)
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
        draw_task(TaskConfig(direction="random", distance_km=20_000.0), jax.random.key(0), TRUTH)


def test_on_a_global_record_the_goal_may_cross_the_seam_but_stays_within_the_latitude_band():
    lon = jnp.arange(0.0, 360.0, 2.0)
    globe = WindField(jnp.array([0.0, 240.0]), jnp.array([11.8, 20.6]), jnp.arange(-80.0, 81.0, 2.0), lon, jnp.zeros((2, 2, 81, 180, 2)))
    seen_near_seam = False
    for seed in range(60):
        task = draw_task(TaskConfig(direction="random", distance_km=1000.0, lat_band=(-60.0, 60.0)), jax.random.key(seed), globe)
        assert -60.0 <= task.start.lat <= 60.0 and -60.0 <= task.goal_lat <= 60.0
        assert 0.0 <= task.goal_lon < 360.0
        assert jnp.allclose(distance_km(task.start.lat, task.start.lon, task.goal_lat, task.goal_lon), 1000.0, atol=1.0)
        seen_near_seam |= bool((task.start.lon > 350.0) | (task.start.lon < 10.0))
    assert seen_near_seam  # starts near the seam are allowed, nothing is reserved as a margin


def test_a_downwind_goal_lies_where_the_wind_blows():
    """Wind 10 m/s due east everywhere: the goal is 1,000 km east of the start, so same latitude, greater longitude"""
    uv = jnp.zeros((2, 2, 2, 2, 2)).at[..., 0].set(10.0)
    east = WindField(jnp.array([0.0, 240.0]), jnp.array([15.0, 20.0]), jnp.array([20.0, 50.0]), jnp.array([120.0, 240.0]), uv)
    for seed in range(5):
        task = draw_task(TaskConfig(direction="downwind"), jax.random.key(seed), east)
        assert jnp.allclose(task.goal_lat, task.start.lat, atol=1.0)  # due east: a great circle bends half a degree equatorward over 1,000 km
        assert task.goal_lon > task.start.lon


def test_launches_stay_in_the_latitude_band():
    for seed in range(20):
        task = draw_task(TaskConfig(), jax.random.key(seed), TRUTH)
        assert 20.0 <= task.start.lat <= 50.0 and 20.0 <= task.goal_lat <= 50.0


def test_the_goal_goes_where_the_direction_says():
    """Wind 10 m/s due east everywhere: downwind puts the goal east, upwind west, crosswind north or south"""
    uv = jnp.zeros((2, 2, 2, 2, 2)).at[..., 0].set(10.0)
    east = WindField(jnp.array([0.0, 240.0]), jnp.array([15.0, 20.0]), jnp.array([20.0, 50.0]), jnp.array([120.0, 240.0]), uv)
    key = jax.random.key(0)
    goals = {d: draw_task(TaskConfig(direction=d, distance_km=300.0), key, east) for d in ("downwind", "upwind", "crosswind")}
    along = lambda task: ((task.goal_lon - task.start.lon + 180.0) % 360.0) - 180.0  # degrees east of the start

    assert along(goals["downwind"]) > 0  # with the wind
    assert along(goals["upwind"]) < 0  # against it
    # a quarter turn off the flow is mostly north or south, so the goal barely moves along it
    assert abs(along(goals["crosswind"])) < abs(along(goals["downwind"])) / 2
    assert abs(goals["crosswind"].goal_lat - goals["crosswind"].start.lat) > 1.0  # but well away in latitude
