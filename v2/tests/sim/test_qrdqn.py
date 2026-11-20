import jax
import jax.numpy as jnp
import numpy as np

from balloongym.sim import qrdqn

SMALL = qrdqn.QRConfig(layers=3, hidden=16, replay=40, envs=2, n_step=3, batch=8)


def test_n_step_returns_stop_at_an_episodes_end():
    r = qrdqn.replay_init(SMALL, 4)
    for t in range(6):
        end = jnp.array([t == 1, False])  # env 0's episode ends after its second transition
        r = qrdqn.replay_add(r, jnp.full((2, 4), float(t)), jnp.zeros(2, int), jnp.ones(2), jnp.full((2, 4), t + 0.5), end)
    r = r._replace(priority=jnp.zeros_like(r.priority).at[0, 0].set(1.0))  # sample only row 0 of env 0
    (obs, action, rewards, next_obs, discount, weights), (t, e) = qrdqn.replay_sample(jax.random.key(0), r, SMALL)
    assert jnp.all(t == 0) and jnp.all(e == 0)
    g = SMALL.gamma
    assert np.allclose(rewards, 1 + g)  # two rewards, then the end
    assert np.allclose(discount, g**2) and np.allclose(next_obs, 1.5)  # bootstrap from the episode's last observation


def test_the_newest_rows_are_not_sampled_before_their_n_steps_exist():
    r = qrdqn.replay_init(SMALL, 4)
    for t in range(5):
        r = qrdqn.replay_add(r, jnp.zeros((2, 4)), jnp.zeros(2, int), jnp.ones(2), jnp.zeros((2, 4)), jnp.zeros(2, bool))
    ok = qrdqn._sampleable(r, SMALL)
    assert list(np.asarray(ok[:6])) == [True, True, True, False, False, False]


def test_marco_polo_flies_the_policy_four_hours_then_walks_two():
    c = qrdqn.QRConfig()
    x = qrdqn.Explorer(jnp.array(True), jnp.array(False), jnp.array(0), jnp.array(0), jnp.array(8000.0))
    phases = []
    for i in range(130):
        x, a = qrdqn.explorer_step(jax.random.key(i), x, c, jnp.array(17.0), jnp.array(1))
        phases.append(bool(x.exploring))
    assert phases[:79] == [False] * 79 and phases[79:119] == [True] * 40 and not phases[119]


def test_the_quantile_loss_is_zero_only_when_the_quantiles_match():
    c = qrdqn.QRConfig()
    q = jnp.linspace(-1, 1, c.atoms)
    assert float(qrdqn.quantile_huber(q, q, c)) < float(qrdqn.quantile_huber(q, q + 0.5, c))


def test_training_runs_and_changes_the_weights():
    from tests.sim.test_station_env import config, fields
    from balloongym.sim.dynamics import State
    truth, forecast = fields()
    m = qrdqn.Missions(State(jnp.zeros(4), jnp.full(4, 100.0), jnp.full(4, 17.2), jnp.full(4, 2.0)), jnp.zeros(4), jnp.full(4, 100.2))
    c = qrdqn.QRConfig(layers=3, hidden=16, replay=400, envs=4, warmup=16, target_every=5)
    start, run = qrdqn.make(c, config(), truth, forecast, m)
    carry = start(jax.random.key(0))
    before = carry[0].params[0][0].copy()
    carry, (reward, loss) = run(carry, jax.random.key(1), 30)
    assert int(carry[0].updates) > 0 and not jnp.allclose(before, carry[0].params[0][0])
    assert jnp.all(jnp.isfinite(loss))
    scores = qrdqn.evaluate(carry[0].params, c, config(), truth, forecast, m)
    assert scores.shape == (4,) and jnp.all((scores >= 0) & (scores <= 1))
