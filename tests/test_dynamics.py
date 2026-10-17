import jax
import jax.numpy as jnp
import numpy as np


from balloon_sim.dynamics import EARTH_RADIUS_M, State, clip, distance_km, final_cost, rollout, step, summed_cost
from balloon_sim.wind import WindField

STEP_HOURS = 3.0
DT_S = STEP_HOURS * 3600.0


def uniform_field(u: float, v: float) -> WindField:
    """The same wind everywhere, on a box from 0 to 70N and 100 to 200E"""
    hours, altitude, lat, lon = jnp.array([0.0, 100.0]), jnp.array([15.0, 20.0]), jnp.array([0.0, 70.0]), jnp.array([100.0, 200.0])
    uv = jnp.broadcast_to(jnp.array([u, v]), (2, 2, 2, 2, 2))
    return WindField(hours, altitude, lat, lon, uv)


def start(lat: float = 30.0) -> State:
    return State(jnp.array(lat), jnp.array(150.0), jnp.array(17.0), jnp.array(0.0))


def test_still_air_only_climbs():
    after = step(uniform_field(0.0, 0.0), start(), jnp.array(0.8), STEP_HOURS)
    assert jnp.allclose(jnp.array(after), jnp.array([30.0, 150.0, 17.8, 3.0]))


def test_northward_wind_moves_latitude_by_v_dt_over_r():
    after = step(uniform_field(0.0, 10.0), start(), jnp.array(0.0), STEP_HOURS)
    assert jnp.allclose(after.lat - 30.0, np.degrees(10.0 * DT_S / EARTH_RADIUS_M), atol=1e-5)
    assert jnp.allclose(after.lon, 150.0)


def test_eastward_step_in_degrees_doubles_at_60n():
    """A degree of longitude at 60N is half as long as at the equator"""
    field = uniform_field(10.0, 0.0)
    at_equator = step(field, start(0.0), jnp.array(0.0), STEP_HOURS).lon - 150.0
    at_60n = step(field, start(60.0), jnp.array(0.0), STEP_HOURS).lon - 150.0
    assert jnp.allclose(at_equator, np.degrees(10.0 * DT_S / EARTH_RADIUS_M), atol=1e-5)
    assert jnp.allclose(at_60n, 2.0 * at_equator, atol=1e-5)


def test_known_distances():
    one_degree_of_latitude = EARTH_RADIUS_M / 1000.0 * np.pi / 180.0  # 111.19 km
    assert jnp.allclose(distance_km(10.0, 150.0, 11.0, 150.0), one_degree_of_latitude, atol=0.05)
    assert jnp.allclose(distance_km(0.0, 100.0, 0.0, 190.0), EARTH_RADIUS_M / 1000.0 * np.pi / 2, atol=0.5)


def test_rollout_records_every_move_and_costs_use_it():
    climbs = jnp.array([0.5, 0.5, -1.0])
    visited = rollout(uniform_field(0.0, 0.0), start(), climbs, STEP_HOURS)
    assert jnp.allclose(visited.altitude_km, jnp.array([17.5, 18.0, 17.0]))
    assert jnp.allclose(visited.hours, jnp.array([3.0, 6.0, 9.0]))
    # in still air the balloon never moves, so each move is one degree of latitude from a goal 1 degree north
    assert jnp.allclose(summed_cost(visited, 31.0, 150.0), 3 * distance_km(30.0, 150.0, 31.0, 150.0), atol=0.05)
    assert jnp.allclose(final_cost(visited, 31.0, 150.0), distance_km(30.0, 150.0, 31.0, 150.0), atol=0.05)


def test_gradient_matches_finite_differences():
    """Wind that turns with altitude, so where the balloon climbs changes where it ends up"""
    with jax.enable_x64(True):
        hours, altitude = jnp.array([0.0, 100.0]), jnp.array([15.0, 20.0])
        lat, lon = jnp.array([0.0, 70.0]), jnp.array([100.0, 200.0])
        uv = jnp.zeros((2, 2, 2, 2, 2)).at[:, 0, ..., 0].set(15.0).at[:, 1, ..., 1].set(-8.0)  # east low, south high
        field = WindField(hours, altitude, lat, lon, uv)
        begin = State(jnp.array(30.0), jnp.array(150.0), jnp.array(17.0), jnp.array(0.0))

        def cost(climbs):
            return summed_cost(rollout(field, begin, climbs, STEP_HOURS), 28.0, 153.0)

        climbs = jnp.array([0.3, -0.4, 0.7, 0.1])
        analytic = jax.grad(cost)(climbs)
        eps = 1e-5
        numeric = jnp.array([(cost(climbs.at[i].add(eps)) - cost(climbs.at[i].add(-eps))) / (2 * eps) for i in range(4)])
        assert jnp.allclose(analytic, numeric, rtol=1e-5, atol=1e-6)


def test_clip_holds_the_balloon_inside_the_record():
    outside = State(jnp.array(80.0), jnp.array(90.0), jnp.array(25.0), jnp.array(6.0))
    held = clip(uniform_field(0.0, 0.0), outside)
    assert jnp.allclose(jnp.array(held), jnp.array([70.0, 100.0, 20.0, 6.0]))
