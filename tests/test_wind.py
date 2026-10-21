import jax
import jax.numpy as jnp
import numpy as np

from balloon_sim.wind import WindField, load_record, wind_at

HOURS = jnp.array([0.0, 3.0, 6.0])
ALTITUDE = jnp.array([15.0, 16.0, 17.5, 20.0])  # uneven spacing, as in ERA5
LAT = jnp.array([20.0, 22.0, 24.0])
LON = jnp.array([170.0, 172.0, 174.0, 176.0])
SLOPES = jnp.array([0.5, -1.0, 2.0, 0.25])  # per hour, km, degree of lat, degree of lon


def linear_field() -> WindField:
    """u rises linearly along every axis and v = -u, so linear interpolation should reproduce it exactly"""
    t, a, y, x = jnp.meshgrid(HOURS, ALTITUDE, LAT, LON, indexing="ij")
    u = SLOPES[0] * t + SLOPES[1] * a + SLOPES[2] * y + SLOPES[3] * x
    return WindField(HOURS, ALTITUDE, LAT, LON, jnp.stack([u, -u], axis=-1))


def expected_u(lat, lon, altitude, hours):
    return SLOPES[0] * hours + SLOPES[1] * altitude + SLOPES[2] * lat + SLOPES[3] * lon


def test_grid_point_returns_grid_value():
    field = linear_field()
    assert jnp.allclose(wind_at(field, LAT[1], LON[2], ALTITUDE[3], HOURS[1]), field.uv[1, 3, 1, 2], atol=1e-4)


def test_between_grid_points_is_exact_for_a_linear_field():
    point = (21.3, 173.1, 16.8, 4.4)  # lat, lon, altitude, hours
    u = expected_u(*point)
    assert jnp.allclose(wind_at(linear_field(), *point), jnp.array([u, -u]), atol=1e-3)


def test_outside_the_grid_holds_the_edge_value():
    field = linear_field()
    assert jnp.allclose(wind_at(field, 30.0, LON[0], ALTITUDE[0], HOURS[0]), wind_at(field, LAT[-1], LON[0], ALTITUDE[0], HOURS[0]))


def test_gradient_in_latitude_is_the_slope():
    du_dlat = jax.grad(lambda lat: wind_at(linear_field(), lat, 173.1, 16.8, 4.4)[0])(21.3)
    assert jnp.allclose(du_dlat, SLOPES[2], atol=1e-4)


def test_load_record_makes_altitude_ascending(tmp_path):
    shape = (len(HOURS), len(ALTITUDE), len(LAT), len(LON))
    u = np.random.default_rng(0).normal(size=shape)
    path = tmp_path / "record.npz"
    np.savez(path, hours=HOURS, altitude_km=ALTITUDE[::-1], latitude=LAT, longitude=LON, u=u, v=-u)  # top-down, as ERA5
    field = load_record(str(path))
    assert jnp.all(jnp.diff(field.altitude_km) > 0)
    assert jnp.allclose(field.uv[:, 0, :, :, 0], u[:, -1], atol=1e-6)  # the lowest level is the file's last


def test_a_global_record_blends_across_the_seam(tmp_path):
    """Longitude 0 to 358 by 2: reading at 359 blends the last column with the first, and 361 reads as 1"""
    lon = np.arange(0.0, 360.0, 2.0)
    u = np.zeros((2, 2, 2, len(lon))) + lon[None, None, None, :]  # u equals the longitude
    np.savez(tmp_path / "global.npz", hours=[0.0, 6.0], altitude_km=[15.0, 20.0], latitude=[0.0, 10.0], longitude=lon, u=u, v=0 * u)
    field = load_record(str(tmp_path / "global.npz"))
    assert field.lon.shape[0] == len(lon) + 1 and field.lon[-1] == 360.0
    assert jnp.allclose(wind_at(field, 5.0, 359.0, 15.0, 0.0)[0], (358.0 + 0.0) / 2, atol=1e-3)  # halfway round the seam
    assert jnp.allclose(wind_at(field, 5.0, 361.0, 15.0, 0.0)[0], wind_at(field, 5.0, 1.0, 15.0, 0.0)[0], atol=1e-3)
