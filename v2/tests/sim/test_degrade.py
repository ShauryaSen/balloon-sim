import jax
import jax.numpy as jnp
import numpy as np

from balloongym.sim import degrade
from balloongym.sim.config import BeliefConfig
from balloongym.sim.wind import WindField, load_record

HOURS = jnp.arange(0.0, 27.0, 3.0)  # 9 frames
ALTITUDE = jnp.linspace(14.0, 20.0, 7)
LAT = jnp.arange(20.0, 41.0, 2.0)  # 11 rows
LON = jnp.arange(170.0, 191.0, 2.0)  # 11 columns
SLOPES = jnp.array([0.5, -1.0, 2.0, 0.25])


def linear_field() -> WindField:
    """u rises linearly along every axis and v = -u: a straight-line blend between kept points reproduces it"""
    t, a, y, x = jnp.meshgrid(HOURS, ALTITUDE, LAT, LON, indexing="ij")
    u = SLOPES[0] * t + SLOPES[1] * a + SLOPES[2] * y + SLOPES[3] * x
    return WindField(HOURS, ALTITUDE, LAT, LON, jnp.stack([u, -u], axis=-1))


def rough_field() -> WindField:
    """A field with no structure at all, so a coarsened copy differs from it everywhere it can"""
    uv = jax.random.normal(jax.random.PRNGKey(0), (len(HOURS), len(ALTITUDE), len(LAT), len(LON), 2))
    return WindField(HOURS, ALTITUDE, LAT, LON, uv)


def level_differences(field: WindField) -> jax.Array:
    return field.uv[:, 1:] - field.uv[:, :-1]


def test_coarsening_a_linear_field_changes_nothing():
    field = linear_field()
    for copy in (degrade.levels(field, 3), degrade.frames(field, 2), degrade.frames(field, 4), degrade.grid(field, 2), degrade.grid(field, 4)):
        assert jnp.allclose(copy.uv, field.uv, atol=1e-3)


def test_coarsening_keeps_the_kept_points_and_both_ends():
    field = rough_field()
    two = degrade.levels(field, 2)
    assert jnp.allclose(two.uv[:, 0], field.uv[:, 0]) and jnp.allclose(two.uv[:, -1], field.uv[:, -1])  # both ends of the band
    assert not jnp.allclose(two.uv[:, 3], field.uv[:, 3])  # the middle is now a blend
    every_4 = degrade.frames(field, 4)
    assert all(jnp.allclose(every_4.uv[i], field.uv[i]) for i in (0, 4, 8))
    coarse = degrade.grid(field, 5)
    assert jnp.allclose(coarse.uv[:, :, ::5][:, :, :, ::5], field.uv[:, :, ::5][:, :, :, ::5])


def test_a_point_between_kept_ones_is_their_straight_line_blend():
    field = rough_field()
    copy = degrade.frames(field, 2)
    assert jnp.allclose(copy.uv[1], (field.uv[0] + field.uv[2]) / 2, atol=1e-6)


def test_one_level_is_the_band_average_everywhere_and_frozen_is_the_first_frame():
    field = rough_field()
    one = degrade.levels(field, 1)
    assert jnp.allclose(one.uv[:, 0], jnp.mean(field.uv, axis=1), atol=1e-6)
    assert jnp.allclose(level_differences(one), 0.0)
    frozen = degrade.frames(field, None)
    assert jnp.allclose(frozen.uv, field.uv[:1])


def test_a_uniform_offset_leaves_the_level_differences_alone_and_a_ramp_does_not():
    field, key = rough_field(), jax.random.PRNGKey(3)
    uniform = degrade.offset(field, key, 3.2, by_altitude=False)
    ramp = degrade.offset(field, key, 3.2, by_altitude=True)
    assert jnp.allclose(level_differences(uniform), level_differences(field), atol=1e-5)
    assert not jnp.allclose(level_differences(ramp), level_differences(field), atol=0.1)
    for copy in (uniform, ramp):  # both exactly the size asked for
        assert jnp.allclose(jnp.sqrt(jnp.mean((copy.uv - field.uv) ** 2)), 3.2, atol=1e-4)


def test_a_flat_random_error_is_the_same_at_every_level():
    field, key, kernel = rough_field(), jax.random.PRNGKey(4), BeliefConfig()
    flat = degrade.random(field, key, kernel, 3.2, by_altitude=False)
    error = field.uv - flat.uv
    assert jnp.allclose(error[:, 1:], error[:, :-1], atol=1e-3)
    shaped = degrade.random(field, key, kernel, 3.2, by_altitude=True)
    assert not jnp.allclose((field.uv - shaped.uv)[:, 1:], (field.uv - shaped.uv)[:, :-1], atol=0.1)


def test_copies_by_name_keep_the_grid():
    field, key, kernel = rough_field(), jax.random.PRNGKey(5), BeliefConfig()
    for name in ("truth", "levels:3", "frames:2", "frames:frozen", "grid:2", "offset", "offset_altitude", "random", "random_altitude"):
        copy = degrade.copy(name, field, key, kernel, 3.2)
        assert copy.uv.shape == field.uv.shape, name
    earlier = linear_field()
    assert jnp.allclose(degrade.copy("persistence", field, key, kernel, 3.2, earlier=earlier).uv, earlier.uv)


def test_persistence_reads_the_records_forecast_and_sizes_its_error(tmp_path):
    shape = (len(HOURS), len(ALTITUDE), len(LAT), len(LON))
    rng = np.random.default_rng(0)
    u, v = rng.normal(size=shape), rng.normal(size=shape)
    np.savez(tmp_path / "r.npz", hours=HOURS, altitude_km=ALTITUDE, latitude=LAT, longitude=LON,
             u=u, v=v, forecast_u=u + 2.0, forecast_v=v - 2.0)
    truth, earlier = load_record(str(tmp_path / "r.npz")), load_record(str(tmp_path / "r.npz"), "forecast_")
    assert jnp.allclose(degrade.persistence(truth, earlier).uv - truth.uv, jnp.array([2.0, -2.0]), atol=1e-5)
    assert abs(degrade.error_size(truth, earlier, (20.0, 50.0)) - 2.0) < 1e-4


def equator_field() -> WindField:
    """A zero wind on a grid across the equator, so a copy's wind is minus its error"""
    hours, lat, lon = jnp.arange(0.0, 25.0), jnp.arange(-10.0, 11.0, 2.0), jnp.arange(0.0, 41.0, 2.0)
    return WindField(hours, ALTITUDE, lat, lon, jnp.zeros((len(hours), len(ALTITUDE), len(lat), len(lon), 2)))


def test_every_error_model_comes_out_at_the_size_asked_and_keeps_the_grid():
    field = equator_field()
    earlier = field._replace(uv=jax.random.normal(jax.random.PRNGKey(1), field.uv.shape))
    for name in degrade.ERROR_MODELS:
        copy = degrade.copy(name, field, jax.random.PRNGKey(3), BeliefConfig(), 2.5, earlier, (-10.0, 10.0))
        assert copy.uv.shape == field.uv.shape
        assert np.isclose(float(jnp.sqrt(jnp.mean(copy.uv[:, :, :, :-1] ** 2))), 2.5, rtol=1e-4), name


def test_scaled_persistence_is_the_real_error_resized():
    field = equator_field()
    field = field._replace(uv=jax.random.normal(jax.random.PRNGKey(2), field.uv.shape))
    earlier = field._replace(uv=field.uv * 0.5)
    copy = degrade.error_model("persistence_scaled", field, jax.random.PRNGKey(0), BeliefConfig(), 1.0, (-10.0, 10.0), earlier)
    error = field.uv - copy.uv
    ratio = error / (field.uv - earlier.uv)
    assert jnp.allclose(ratio, ratio.reshape(-1)[0], atol=1e-4)  # one scale factor everywhere: the real error's shape, kept


def test_bles_gp_has_bles_correlations():
    """Matern 1/2 is exp(-r): two points 357 km apart east-west should correlate about exp(-1)"""
    field = equator_field()._replace(lon=jnp.arange(0.0, 361.0, 3.207), uv=jnp.zeros((25, 7, 11, 113, 2)))
    draws = jnp.stack([degrade.ble_gp_error(field, jax.random.PRNGKey(s)) for s in range(16)])
    near = float(jnp.mean(draws[..., :-1, :] * draws[..., 1:, :]) / jnp.mean(draws**2))  # one step is 357 km
    assert abs(near - np.exp(-1.0)) < 0.06
