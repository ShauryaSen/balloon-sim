"""QR-DQN for station keeping, BLE's recipe (decision 57), written in JAX so many balloons fly at once on a GPU

The agent (Dabney et al. 2018, as Dopamine's JaxQuantileAgent that BLE trains): a network gives, for each action,
51 quantiles of the return; Q is their mean; the action is the arg max (epsilon 0). It learns from a prioritized
replay of 5-step transitions, bootstrapping from a target network's quantiles at its own arg max, with the
quantile Huber loss (kappa 1). Settings are BLE's `agents/configs/quantile.gin` (QRConfig).

Exploration is BLE's MarcoPolo: 80% of episodes are exploratory, and in those the policy flies 4 h, then a random
walk in pressure flies 2 h, and so on. The walk keeps a target pressure that drifts by Gaussian steps whose size
grows with the time since the episode began (BLE's `random_walk_agent.py`, 0.1666 Pa per second elapsed), and
climbs or sinks toward it with 100 Pa of hysteresis. Pressure is the standard atmosphere's (degrade.pressure_pa).

Ours runs `envs` episodes side by side. Each vector step adds `envs` transitions, so it does envs / update_every
gradient updates to keep BLE's one update per 4 transitions. An episode is 960 steps and ends on time, not on a
terminal state, so the return bootstraps through the end (the observation has no clock). An n-step return stops
at an episode's end and bootstraps from that episode's last observation.

Missions are drawn as the sweep draws them (fleet.draw_mission), from seeds kept apart from the 32 test seeds:
training from 100,000, validation from 200,000.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from balloongym.sim import station_env as env
from balloongym.sim.degrade import pressure_pa
from balloongym.sim.dynamics import State, distance_km
from balloongym.sim.wind import WindField

N_ACTIONS = 3


class QRConfig(NamedTuple):
    layers: int = 8  # 7 hidden and the output
    hidden: int = 600
    atoms: int = 51
    gamma: float = 0.993
    n_step: int = 5
    learning_rate: float = 2e-6
    adam_eps: float = 2e-5
    batch: int = 32
    replay: int = 2_000_000
    update_every: int = 4
    target_every: int = 100  # gradient updates
    warmup: int = 500  # transitions before learning
    kappa: float = 1.0
    explore_episodes: float = 0.8
    policy_phase: int = 80  # steps: 4 h at 3 min
    explore_phase: int = 40  # 2 h
    walk_pa_per_s: float = 0.1666
    hysteresis_pa: float = 100.0
    envs: int = 128


# --- the network --------------------------------------------------------------------------------------------------


def init_params(key: jax.Array, n_in: int, c: QRConfig) -> list:
    """BLE's initializer: variance scaling 1/sqrt(3), fan in, uniform; zero biases"""
    sizes = [n_in] + [c.hidden] * (c.layers - 1) + [N_ACTIONS * c.atoms]
    params = []
    for k, (a, b) in zip(jax.random.split(key, len(sizes) - 1), zip(sizes[:-1], sizes[1:])):
        limit = jnp.sqrt(3.0 * (1.0 / jnp.sqrt(3.0)) / a)
        params.append((jax.random.uniform(k, (a, b), minval=-limit, maxval=limit), jnp.zeros(b)))
    return params


def quantiles(params: list, obs: jax.Array, c: QRConfig) -> jax.Array:
    """(..., actions, atoms)"""
    x = obs
    for w, b in params[:-1]:
        x = jax.nn.relu(x @ w + b)
    w, b = params[-1]
    return (x @ w + b).reshape(obs.shape[:-1] + (N_ACTIONS, c.atoms))


def act(params: list, obs: jax.Array, c: QRConfig) -> jax.Array:
    return jnp.argmax(jnp.mean(quantiles(params, obs, c), axis=-1), axis=-1)


def quantile_huber(theta: jax.Array, target: jax.Array, c: QRConfig) -> jax.Array:
    """theta (atoms,) predicted, target (atoms,): sum over predicted of mean over target"""
    tau = (jnp.arange(c.atoms) + 0.5) / c.atoms
    u = target[None, :] - theta[:, None]
    huber = jnp.where(jnp.abs(u) <= c.kappa, 0.5 * u**2, c.kappa * (jnp.abs(u) - 0.5 * c.kappa))
    return jnp.sum(jnp.mean(jnp.abs(tau[:, None] - (u < 0)) * huber / c.kappa, axis=1))


# --- missions -----------------------------------------------------------------------------------------------------


class Missions(NamedTuple):
    start: State  # each field (M,)
    station_lat: jax.Array
    station_lon: jax.Array


def missions(task_config, truth: WindField, seeds) -> Missions:
    from balloongym.sim.cells import keys
    from balloongym.sim.fleet import draw_mission

    drawn = [draw_mission(task_config, keys(int(s))[0], truth)[0] for s in seeds]
    return Missions(State(*(jnp.stack([getattr(m.starts, f)[0] for m in drawn]) for f in State._fields)),
                    jnp.stack([m.targets_lat[0] for m in drawn]), jnp.stack([m.targets_lon[0] for m in drawn]))


def _pick(m: Missions, i: jax.Array) -> tuple[State, jax.Array, jax.Array]:
    return State(*(f[i] for f in m.start)), m.station_lat[i], m.station_lon[i]


# --- exploration --------------------------------------------------------------------------------------------------


class Explorer(NamedTuple):
    exploratory: jax.Array  # this episode explores at all
    exploring: jax.Array  # in the walk's phase now
    phase_steps: jax.Array
    walk_steps: jax.Array  # steps the walk has taken this episode (its clock)
    target_pa: jax.Array


def explorer_reset(key: jax.Array, c: QRConfig, band_pa: tuple[float, float]) -> Explorer:
    k1, k2 = jax.random.split(key)
    return Explorer(jax.random.uniform(k1) < c.explore_episodes, jnp.array(False), jnp.array(0), jnp.array(0),
                    jax.random.uniform(k2, minval=band_pa[0], maxval=band_pa[1]))


def explorer_step(key: jax.Array, x: Explorer, c: QRConfig, altitude_km: jax.Array, policy_action: jax.Array) -> tuple[Explorer, jax.Array]:
    """MarcoPolo: advance the phase clock, switch phase when due, and act by the walk in its phase"""
    phase_steps = x.phase_steps + 1
    due = jnp.where(x.exploring, phase_steps >= c.explore_phase, phase_steps >= c.policy_phase)
    switch = x.exploratory & due
    exploring = jnp.where(switch, ~x.exploring, x.exploring)
    phase_steps = jnp.where(switch, 0, phase_steps)
    walk_steps = x.walk_steps + exploring
    target = x.target_pa + jnp.where(exploring, walk_steps * 180.0 * c.walk_pa_per_s * jax.random.normal(key), 0.0)
    p = pressure_pa(altitude_km)
    walk = jnp.where(p - c.hysteresis_pa > target, env.UP, jnp.where(p + c.hysteresis_pa < target, env.DOWN, env.STAY))
    return Explorer(x.exploratory, exploring, phase_steps, walk_steps, target), jnp.where(exploring, walk, policy_action)


# --- replay -------------------------------------------------------------------------------------------------------


class Replay(NamedTuple):
    obs: jax.Array  # (T, E, D)
    action: jax.Array  # (T, E)
    reward: jax.Array
    next_obs: jax.Array
    end: jax.Array  # the transition ends its episode
    priority: jax.Array  # (T, E)
    written: jax.Array  # rows written so far, all time
    max_priority: jax.Array


def replay_init(c: QRConfig, n_obs: int) -> Replay:
    rows = c.replay // c.envs
    z = lambda *shape: jnp.zeros((rows, c.envs) + shape)
    return Replay(z(n_obs), jnp.zeros((rows, c.envs), int), z(), z(n_obs), jnp.zeros((rows, c.envs), bool), z(), jnp.array(0), jnp.array(1.0))


def replay_add(r: Replay, obs, action, reward, next_obs, end) -> Replay:
    t = r.written % r.obs.shape[0]
    return Replay(r.obs.at[t].set(obs), r.action.at[t].set(action), r.reward.at[t].set(reward), r.next_obs.at[t].set(next_obs),
                  r.end.at[t].set(end), r.priority.at[t].set(r.max_priority), r.written + 1, r.max_priority)


def _sampleable(r: Replay, c: QRConfig) -> jax.Array:
    """(T,) rows whose next n - 1 rows are written and not yet overwritten"""
    rows = r.obs.shape[0]
    t = jnp.arange(rows)
    age = (r.written - 1 - t) % rows  # 0 for the newest row
    return (t < r.written) & (age >= c.n_step - 1)


def replay_sample(key: jax.Array, r: Replay, c: QRConfig):
    rows, envs = r.priority.shape
    p = r.priority * _sampleable(r, c)[:, None]
    p = p.reshape(-1) / jnp.sum(p)
    flat = jax.random.choice(key, rows * envs, (c.batch,), p=p)
    t, e = flat // envs, flat % envs
    ks = jnp.arange(c.n_step)
    tt = (t[:, None] + ks[None]) % rows  # (B, n)
    ends = r.end[tt, e[:, None]]
    included = jnp.concatenate([jnp.ones((c.batch, 1), bool), jnp.cumsum(ends[:, :-1], axis=1) == 0], axis=1)  # stop after an end
    n_used = jnp.sum(included, axis=1)
    rewards = jnp.sum(jnp.where(included, r.reward[tt, e[:, None]] * c.gamma ** ks[None], 0.0), axis=1)
    last = (t + n_used - 1) % rows
    weights = 1.0 / jnp.sqrt(p[flat] + 1e-10)
    return (r.obs[t, e], r.action[t, e], rewards, r.next_obs[last, e], c.gamma**n_used, weights / jnp.max(weights)), (t, e)


# --- training -----------------------------------------------------------------------------------------------------


class Learner(NamedTuple):
    params: list
    target: list
    opt: optax.OptState
    updates: jax.Array


def make(c: QRConfig, config: env.EnvConfig, truth: WindField, forecast: WindField, train: Missions):
    """The jitted pieces of training: start(key) and run(carry, key, steps)"""
    n_obs = env.obs_size(truth.altitude_km.shape[0])
    optimizer = optax.adam(c.learning_rate, eps=c.adam_eps)
    band_pa = (float(pressure_pa(truth.altitude_km[-1])), float(pressure_pa(truth.altitude_km[0])))
    n_missions = train.station_lat.shape[0]

    def new_episode(key):
        k1, k2 = jax.random.split(key)
        start, slat, slon = _pick(train, jax.random.randint(k1, (), 0, n_missions))
        s = env.reset(config, start, slat, slon)
        return s, env.observe(config, forecast, s), explorer_reset(k2, c, band_pa)

    def loss(params, target, batch):
        obs, action, rewards, next_obs, discount, weights = batch
        theta = jax.vmap(lambda q, a: q[a])(quantiles(params, obs, c), action)  # (B, atoms)
        next_q = quantiles(target, next_obs, c)  # (B, A, atoms)
        best = jnp.argmax(jnp.mean(next_q, -1), -1)
        bootstrap = jax.vmap(lambda q, a: q[a])(next_q, best)
        goal = jax.lax.stop_gradient(rewards[:, None] + discount[:, None] * bootstrap)
        each = jax.vmap(lambda th, g: quantile_huber(th, g, c))(theta, goal)
        return jnp.mean(weights * each), each

    def update(learner: Learner, replay: Replay, key):
        batch, (t, e) = replay_sample(key, replay, c)
        (value, each), grads = jax.value_and_grad(loss, has_aux=True)(learner.params, learner.target, batch)
        steps, opt = optimizer.update(grads, learner.opt, learner.params)
        params = optax.apply_updates(learner.params, steps)
        n = learner.updates + 1
        target = jax.tree.map(lambda a, b: jnp.where(n % c.target_every == 0, a, b), params, learner.target)
        new_priority = jnp.sqrt(each + 1e-10)
        replay = replay._replace(priority=replay.priority.at[t, e].set(new_priority), max_priority=jnp.maximum(replay.max_priority, jnp.max(new_priority)))
        return Learner(params, target, opt, n), replay, value

    def start(key):
        k1, k2 = jax.random.split(key)
        params = init_params(k1, n_obs, c)
        episodes = jax.vmap(new_episode)(jax.random.split(k2, c.envs))
        return (Learner(params, params, optimizer.init(params), jnp.array(0)), replay_init(c, n_obs), episodes, jnp.zeros(c.envs), jnp.array(0.0))

    def vector_step(carry, key):
        learner, replay, (s, obs, x), returns, last_loss = carry
        k_walk, k_reset, k_upd = jax.random.split(key, 3)
        policy = act(learner.params, obs, c)
        x, action = jax.vmap(lambda k, xi, a, al: explorer_step(k, xi, c, al, a))(jax.random.split(k_walk, c.envs), x, policy, s.balloon.altitude_km)
        s2, obs2, reward, done = jax.vmap(lambda si, ai: env.env_step(config, truth, forecast, si, ai))(s, action)
        replay = replay_add(replay, obs, action, reward, obs2, done)
        fresh = jax.vmap(new_episode)(jax.random.split(k_reset, c.envs))
        keep = lambda new, old: jax.tree.map(lambda n, o: jnp.where(done.reshape((-1,) + (1,) * (n.ndim - 1)), n, o), new, old)
        s, obs, x = keep(fresh, (s2, obs2, x))
        returns = returns + reward

        ready = (replay.written - (c.n_step - 1)) * c.envs >= jnp.maximum(c.warmup, 1)  # only rows whose n steps exist count

        def learn(args):
            learner, replay = args

            def one(carry, k):
                learner, replay, _ = carry
                learner, replay, value = update(learner, replay, k)
                return (learner, replay, value), None

            (learner, replay, value), _ = jax.lax.scan(one, (learner, replay, jnp.array(0.0)), jax.random.split(k_upd, c.envs // c.update_every))
            return learner, replay, value

        learner, replay, last_loss = jax.lax.cond(ready, learn, lambda a: (a[0], a[1], last_loss), (learner, replay))
        return (learner, replay, (s, obs, x), returns, last_loss), (jnp.mean(reward), last_loss)

    @functools.partial(jax.jit, static_argnums=2, donate_argnums=0)
    def run(carry, key, steps: int):
        return jax.lax.scan(vector_step, carry, jax.random.split(key, steps))

    return jax.jit(start), run


# --- evaluation ---------------------------------------------------------------------------------------------------


def evaluate(params: list, c: QRConfig, config: env.EnvConfig, truth: WindField, forecast: WindField, m: Missions) -> jax.Array:
    """The greedy policy flown from each mission for a whole episode: time within the radius, per mission"""

    def one(start, slat, slon):
        s = env.reset(config, start, slat, slon)

        def stepper(carry, _):
            s, obs = carry
            s, obs, _, _ = env.env_step(config, truth, forecast, s, act(params, obs, c))
            inside = distance_km(s.balloon.lat, s.balloon.lon, slat, slon) < config.radius_km
            return (s, obs), inside

        _, inside = jax.lax.scan(stepper, (s, env.observe(config, forecast, s)), None, length=config.n_moves)
        return jnp.mean(inside)

    return jax.jit(jax.vmap(one))(m.start, m.station_lat, m.station_lon)
