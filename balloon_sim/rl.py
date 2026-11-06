"""A learned controller for one balloon or a fleet: PPO with one policy shared by every balloon (decision 50)

What it is. Every balloon runs the same small network on its own observation and picks its own climb; the
network is trained by PPO (Schulman et al. 2017) on the fleet's one team reward. A shared policy with each
agent acting on its own view and a critic trained beside it is what Yu et al. (2022, "The surprising
effectiveness of PPO in cooperative multi-agent games") call MAPPO with parameter sharing; with N = 1 it is
plain PPO. It is the controller the sweep's other controllers are compared with, not a replacement for them.

What it sees, per balloon, all scaled to order one:
    its altitude, the fraction of the episode left, and the forecast error sigma the episode was drawn at
    the vector to the objective's point in local km (reach: the goal; station: the station; targets: the
        nearest stop not yet visited; escape: from its own start to where it is), its length, and a flag
        (reach: it has arrived; station: it is inside; targets: the fraction of stops left)
    the mean offset to the other balloons (zero alone)
    what its own anemometer reads minus what the forecast said there
    the belief's mean wind at its own position at every level of the record

The belief is the sweep's own: the GP around the forecast with the shared kernel, conditioned on every
reading the fleet has taken (decision 48). `learns = False` gives "rl_blind", which never uses a reading:
the belief is the forecast and the reading term is zero, as mpc_blind takes readings and sets them aside.
So "rl minus rl_blind" asks of the learned controller what "mpc minus mpc_blind" asks of MPC.

The forecast. forecast() builds a whole forecast grid, which for the global record is hundreds of MB, too
much to hold one per training episode. The error is a sum of random Fourier features, so it can be read at
any grid point without building the grid; forecast_at reads it at the 16 grid points round a position and
blends them exactly as wind_at blends the grid, so it is the same number forecast() then wind_at would give
(tests/test_rl.py checks it), at no memory. Training and evaluation both use it.

The reward is minus the objective's cost for that move, in thousands of km, so the return of an episode is
exactly minus fleet.fleet_cost of it (also checked). Training draws its tasks from seeds kept apart from the
32 the sweep is scored on (RLConfig.seed_offset), and the noise level of each episode from the sweep's
ladder, so one policy covers every sigma it is evaluated at and is told which one it is in.
"""

import dataclasses
import pickle
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from balloon_sim.belief import Features, Readings, _inputs, _kernel_readings, _matern32, _matern52, read
from balloon_sim.config import BalloonConfig, BeliefConfig, RLConfig, TaskConfig
from balloon_sim.controllers import Controller
from balloon_sim.dynamics import EARTH_RADIUS_M, State, clip, distance_km, step
from balloon_sim.fleet import Mission, draw_mission, reached
from balloon_sim.wind import WindField, _bracket

RADIUS_KM = EARTH_RADIUS_M / 1000.0


# --- the forecast, read at a point ------------------------------------------------------------------------------


class Noise(NamedTuple):
    """One forecast-error pattern, as forecast() draws it from its key; sigma scales it"""

    features: Features
    w: jax.Array  # (2, D)


def noise_from_key(config: BeliefConfig, key: jax.Array) -> Noise:
    """The pattern forecast(config, key, truth) subtracts, drawn the same way from the same key"""
    k_features, k_weights = jax.random.split(key)
    return Noise(Features.draw(config, k_features), jax.random.normal(k_weights, (1, 2, config.n_features))[0])


def _error_at_grid(config: BeliefConfig, noise: Noise, sigma: jax.Array, hours, altitude_km, lat, lon) -> jax.Array:
    """The error at grid points given by their coordinates (broadcast together). Shape (..., 2)"""
    a = _inputs(config, lat, lon, altitude_km) @ noise.features.omega_place.T  # (..., D)
    b = (hours / config.hours)[..., None] * noise.features.omega_time + noise.features.phase
    scale = sigma * jnp.sqrt(2.0 / config.n_features)
    return scale * jnp.einsum("cd,...d->...c", noise.w, jnp.cos(a + b))


def forecast_at(truth: WindField, config: BeliefConfig, noise: Noise, sigma: jax.Array, lat, lon, altitude_km, hours) -> jax.Array:
    """(u, v) the forecast gives at one point: the truth minus the error, both blended from the 16 grid points
    round it exactly as wind_at blends a grid"""
    t, wt = _bracket(truth.hours, hours)
    a, wa = _bracket(truth.altitude_km, altitude_km)
    y, wy = _bracket(truth.lat, lat)
    x, wx = _bracket(truth.lon, lon % 360.0)
    corners = jax.lax.dynamic_slice(truth.uv, (t, a, y, x, jnp.zeros_like(t)), (2, 2, 2, 2, 2))
    pick = lambda axis, i: jax.lax.dynamic_slice(axis, (i,), (2,))
    T, A, Y, X = jnp.meshgrid(pick(truth.hours, t), pick(truth.altitude_km, a), pick(truth.lat, y), pick(truth.lon, x), indexing="ij")
    corners = corners - _error_at_grid(config, noise, sigma, T, A, Y, X)
    pair = lambda w: jnp.stack([1.0 - w, w])
    return jnp.einsum("i,j,k,l,ijklc->c", pair(wt), pair(wa), pair(wy), pair(wx), corners)


def belief_mean_at(config: BeliefConfig, sigma: jax.Array, prior: jax.Array, lat, lon, altitude_km, hours, readings: Readings) -> jax.Array:
    """The GP's mean at points, given readings whose uv holds reading minus forecast:
    prior + k(x, X) K^-1 (y - F(X)). prior (P, 2) is the forecast there; the points are (P,) each"""
    config = dataclasses.replace(config, sigma=sigma)
    real = jnp.arange(readings.lat.shape[0]) < readings.count
    residual = jnp.where(real[:, None], readings.uv, 0.0)
    factor = jnp.linalg.cholesky(_kernel_readings(config, readings))
    alpha = jax.scipy.linalg.cho_solve((factor, True), residual)  # (R, 2)
    place = jnp.sqrt(jnp.sum((_inputs(config, lat, lon, altitude_km)[:, None] - _inputs(config, readings.lat, readings.lon, readings.altitude_km)[None]) ** 2, -1) + 1e-12)
    time = jnp.abs(hours[:, None] - readings.hours[None]) / config.hours
    k = sigma**2 * _matern52(place) * _matern32(time) * real[None]
    return prior + k @ alpha


# --- what a balloon sees ----------------------------------------------------------------------------------------


class Spec(NamedTuple):
    """Everything fixed about a policy's world: static, so it can be closed over"""

    objective: str
    n_agents: int
    learns: bool
    n_moves: int
    radius_km: float
    balloon: BalloonConfig
    belief: BeliefConfig


def obs_size(levels: int) -> int:
    return 11 + 2 * levels  # altitude, time left, sigma, vector (2), its length, flag, team (2), reading (2), belief (2 per level)


def _offset(lat0, lon0, lat1, lon1) -> jax.Array:
    """(east, north) km from point 0 to point 1, flat about point 0"""
    dlon = (lon1 - lon0 + 180.0) % 360.0 - 180.0
    return jnp.stack([RADIUS_KM * jnp.radians(dlon) * jnp.cos(jnp.radians(lat0)), RADIUS_KM * jnp.radians(lat1 - lat0)], -1)


class Flags(NamedTuple):
    arrived: jax.Array  # (N,) reach: this balloon has been inside the goal's circle
    visited: jax.Array  # (K,) targets: some balloon has been inside this stop's circle


def update_flags(spec: Spec, mission: Mission, state: State, flags: Flags) -> Flags:
    d = distance_km(state.lat, state.lon, mission.targets_lat[0], mission.targets_lon[0])
    return Flags(flags.arrived | (d < spec.radius_km), flags.visited | reached(state, mission, spec.radius_km))


def observe(spec: Spec, truth: WindField, noise: Noise, sigma: jax.Array, mission: Mission, state: State,
            readings: Readings, flags: Flags, residual: jax.Array, move: jax.Array) -> jax.Array:
    """(N, obs_size) one row per balloon. `residual` (N, 2) is reading minus forecast at each balloon"""
    n, levels = spec.n_agents, truth.altitude_km.shape[0]
    low, high = truth.altitude_km[0], truth.altitude_km[-1]
    altitude = (2.0 * state.altitude_km - low - high) / (high - low)
    left = jnp.full(n, 1.0 - move / spec.n_moves)
    noisy = jnp.full(n, sigma / 10.0)

    if spec.objective == "escape":
        vector = _offset(mission.starts.lat, mission.starts.lon, state.lat, state.lon)  # from the start, out
        flag = jnp.zeros(n)
    elif spec.objective == "targets":
        d = distance_km(state.lat[:, None], state.lon[:, None], mission.targets_lat[None], mission.targets_lon[None])  # (N, K)
        nearest = jnp.argmin(jnp.where(flags.visited[None], jnp.inf, d), axis=1)
        vector = _offset(state.lat, state.lon, mission.targets_lat[nearest], mission.targets_lon[nearest])
        vector = jnp.where(jnp.all(flags.visited), 0.0, vector)
        flag = jnp.full(n, 1.0 - jnp.mean(flags.visited))
    else:  # reach and station: the one point
        vector = _offset(state.lat, state.lon, mission.targets_lat[0], mission.targets_lon[0])
        inside = distance_km(state.lat, state.lon, mission.targets_lat[0], mission.targets_lon[0]) < spec.radius_km
        flag = (flags.arrived if spec.objective == "reach" else inside).astype(jnp.float32)
    length = jnp.linalg.norm(vector, axis=-1)

    if n > 1:
        pairwise = jax.vmap(lambda la, lo: _offset(la, lo, state.lat, state.lon))(state.lat, state.lon)  # (N, N, 2)
        team = jnp.sum(pairwise, axis=1) / (n - 1)
    else:
        team = jnp.zeros((1, 2))

    # the belief's mean at every level of the record at each balloon: N * levels points
    at = lambda f: jnp.repeat(f, levels)
    lat, lon, hours = at(state.lat), at(state.lon), at(state.hours)
    altitude_km = jnp.tile(truth.altitude_km, n)
    prior = jax.vmap(lambda la, lo, al, h: forecast_at(truth, spec.belief, noise, sigma, la, lo, al, h))(lat, lon, altitude_km, hours)
    mean = belief_mean_at(spec.belief, sigma, prior, lat, lon, altitude_km, hours, readings) if spec.learns else prior
    seen = residual if spec.learns else jnp.zeros_like(residual)

    return jnp.concatenate([
        altitude[:, None], left[:, None], noisy[:, None],
        vector / 1000.0, length[:, None] / 1000.0, flag[:, None], team / 1000.0,
        seen / 10.0, mean.reshape(n, 2 * levels) / 20.0,
    ], axis=1)


def move_cost(spec: Spec, mission: Mission, state: State, flags: Flags) -> jax.Array:
    """The objective's cost for one move, given where the fleet is after it and the flags from before it.
    Summed over an episode this is fleet.fleet_cost of the episode"""
    if spec.objective == "reach":
        d = distance_km(state.lat, state.lon, mission.targets_lat[0], mission.targets_lon[0])
        return jnp.mean(jnp.where(flags.arrived, 0.0, d))
    if spec.objective == "escape":
        return -jnp.mean(distance_km(state.lat, state.lon, mission.starts.lat, mission.starts.lon))
    if spec.objective == "station":
        d = distance_km(state.lat, state.lon, mission.targets_lat[0], mission.targets_lon[0])
        return jnp.min(jnp.maximum(d - spec.radius_km, 0.0))
    d = jnp.min(distance_km(state.lat[None], state.lon[None], mission.targets_lat[:, None], mission.targets_lon[:, None]), axis=1)
    return jnp.sum(jnp.where(flags.visited, 0.0, d))


# --- the network ------------------------------------------------------------------------------------------------


def _layer(key: jax.Array, fan_in: int, fan_out: int, scale: float) -> dict:
    return {"w": jax.random.normal(key, (fan_in, fan_out)) * scale / jnp.sqrt(fan_in), "b": jnp.zeros(fan_out)}


def init_params(key: jax.Array, size: int, config: RLConfig) -> dict:
    """Two networks of two hidden tanh layers: the policy's mean climb, and the value. One log std, shared"""
    k = jax.random.split(key, 6)
    h = config.hidden
    return {
        "pi": [_layer(k[0], size, h, 1.0), _layer(k[1], h, h, 1.0), _layer(k[2], h, 1, 0.01)],
        "v": [_layer(k[3], size, h, 1.0), _layer(k[4], h, h, 1.0), _layer(k[5], h, 1, 1.0)],
        "log_std": jnp.array(config.init_log_std),
    }


def _mlp(layers: list, x: jax.Array) -> jax.Array:
    for layer in layers[:-1]:
        x = jnp.tanh(x @ layer["w"] + layer["b"])
    return (x @ layers[-1]["w"] + layers[-1]["b"])[..., 0]


def policy(params: dict, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The mean of the pre-squash action and the value, one per row of obs. The climb is cap * tanh(action)"""
    return _mlp(params["pi"], obs), _mlp(params["v"], obs)


def _log_prob(u: jax.Array, mean: jax.Array, log_std: jax.Array) -> jax.Array:
    return -0.5 * ((u - mean) / jnp.exp(log_std)) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi)


# --- one episode, for training ----------------------------------------------------------------------------------


class Step(NamedTuple):
    obs: jax.Array  # (N, obs)
    u: jax.Array  # (N,) the pre-squash action taken
    log_prob: jax.Array  # (N,)
    value: jax.Array  # (N,)
    reward: jax.Array  # () the team's, in thousands of km


def episode(spec: Spec, truth: WindField, params: dict, mission: Mission, noise: Noise, sigma: jax.Array, key: jax.Array, scale_km: float) -> Step:
    """One episode with actions sampled from the policy. Each field of the result has a leading (T,) axis"""
    n, k = spec.n_agents, mission.targets_lat.shape[0]
    move_one = jax.vmap(lambda s, c: clip(truth, step(truth, s, c, spec.balloon.step_hours, spec.balloon.substeps)))

    def one_move(carry, inputs):
        state, readings, flags = carry
        move, key = inputs
        flags = update_flags(spec, mission, state, flags)
        reading = jax.vmap(lambda s: read(truth, s))(state)
        forecast = jax.vmap(lambda la, lo, al, h: forecast_at(truth, spec.belief, noise, sigma, la, lo, al, h))(state.lat, state.lon, state.altitude_km, state.hours)
        readings = _add_all(readings, state, reading - forecast) if spec.learns else readings
        obs = observe(spec, truth, noise, sigma, mission, state, readings, flags, reading - forecast, move)
        mean, value = policy(params, obs)
        u = mean + jnp.exp(params["log_std"]) * jax.random.normal(key, mean.shape)
        state = move_one(state, spec.balloon.climb_cap_km * jnp.tanh(u))
        reward = -move_cost(spec, mission, state, flags) / scale_km
        return (state, readings, flags), Step(obs, u, _log_prob(u, mean, params["log_std"]), value, reward)

    capacity = spec.n_moves * n if spec.learns else 1
    start = (mission.starts, Readings.empty(capacity), Flags(jnp.zeros(n, bool), jnp.zeros(k, bool)))
    _, steps = jax.lax.scan(one_move, start, (jnp.arange(spec.n_moves), jax.random.split(key, spec.n_moves)))
    return steps


def _add_all(readings: Readings, state: State, uv: jax.Array) -> Readings:
    for i in range(state.lat.shape[0]):
        readings = readings.add(State(state.lat[i], state.lon[i], state.altitude_km[i], state.hours[i]), uv[i])
    return readings


# --- training ---------------------------------------------------------------------------------------------------


def training_missions(task: TaskConfig, truth: WindField, config: RLConfig) -> Mission:
    """config.train_seeds tasks from seeds that the sweep never scores on, stacked on a leading axis.
    The task key is the one cells.keys gives a seed, so these are the sweep's kind of task, only other ones"""
    from balloon_sim.cells import keys

    drawn = [draw_mission(task, keys(config.seed_offset + i)[0], truth)[0] for i in range(config.train_seeds)]
    return jax.tree.map(lambda *xs: jnp.stack(xs), *drawn)


def _advantages(config: RLConfig, steps: Step, scale: jax.Array) -> tuple[jax.Array, jax.Array]:
    """GAE over each balloon's own run, every balloon given the team's reward divided by `scale`, the running
    spread of returns, so the value is learned in units of order one whatever the objective's size (the
    usual PPO reward normalisation). Shapes (T, B, N)"""
    reward = jnp.broadcast_to(steps.reward[..., None] / scale, steps.value.shape)

    def back(carry, inputs):
        gae, next_value = carry
        r, v = inputs
        delta = r + config.gamma * next_value - v
        gae = delta + config.gamma * config.gae_lambda * gae
        return (gae, v), gae

    zeros = jnp.zeros(steps.value.shape[1:])
    _, adv = jax.lax.scan(back, (zeros, zeros), (reward, steps.value), reverse=True)
    return adv, adv + steps.value


def train(spec: Spec, truth: WindField, missions: Mission, sigmas: tuple[float, ...], config: RLConfig, key: jax.Array,
          log=print, every: int = 10) -> tuple[dict, list[dict]]:
    """PPO. Returns the trained parameters and the training curve: the mean cost per episode (km, as the sweep
    scores it) of the sampled episodes, every `every` iterations"""
    k_init, key = jax.random.split(key)
    params = init_params(k_init, obs_size(truth.altitude_km.shape[0]), config)
    optimiser = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.learning_rate))
    opt_state = optimiser.init(params)
    ladder = jnp.array(sigmas)
    pool = missions.targets_lat.shape[0]

    @jax.jit
    def collect(params, key, truth, missions):  # the record is an argument, not a constant baked into the program
        k_pick, k_sigma, k_noise, k_act = jax.random.split(key, 4)
        chosen = jax.tree.map(lambda a: a[jax.random.randint(k_pick, (config.envs,), 0, pool)], missions)
        sigma = jax.random.choice(k_sigma, ladder, (config.envs,))
        noise = jax.vmap(lambda k: noise_from_key(spec.belief, k))(jax.random.split(k_noise, config.envs))
        run = lambda m, z, s, k: episode(spec, truth, params, m, z, s, k, config.reward_km)
        steps = jax.vmap(run)(chosen, noise, sigma, jax.random.split(k_act, config.envs))  # (B, T, ...)
        return jax.tree.map(lambda a: jnp.swapaxes(a, 0, 1), steps)  # (T, B, ...)

    def loss(params, batch):
        obs, u, old, adv, ret = batch
        mean, value = policy(params, obs)
        ratio = jnp.exp(_log_prob(u, mean, params["log_std"]) - old)
        clipped = jnp.clip(ratio, 1 - config.clip, 1 + config.clip)
        pg = -jnp.mean(jnp.minimum(ratio * adv, clipped * adv))
        entropy = params["log_std"] + 0.5 * jnp.log(2 * jnp.pi * jnp.e)
        return pg + config.value_coef * jnp.mean((value - ret) ** 2) - config.entropy_coef * entropy

    @jax.jit
    def update(params, opt_state, steps, key, scale):
        adv, ret = _advantages(config, steps, scale)
        flat = lambda a: a.reshape(-1, *a.shape[3:])
        data = (flat(steps.obs), flat(steps.u), flat(steps.log_prob), flat(adv), flat(ret))
        size = data[0].shape[0]
        per = size // config.minibatches

        def epoch(carry, k):
            params, opt_state = carry
            order = jax.random.permutation(k, size)

            def minibatch(carry, i):
                params, opt_state = carry
                idx = jax.lax.dynamic_slice(order, (i * per,), (per,))
                batch = tuple(d[idx] for d in data)
                a = batch[3]
                batch = batch[:3] + ((a - a.mean()) / (a.std() + 1e-8),) + batch[4:]
                grads = jax.grad(loss)(params, batch)
                updates, opt_state = optimiser.update(grads, opt_state, params)
                return (optax.apply_updates(params, updates), opt_state), None

            return jax.lax.scan(minibatch, (params, opt_state), jnp.arange(config.minibatches))[0], None

        (params, opt_state), _ = jax.lax.scan(epoch, (params, opt_state), jax.random.split(key, config.epochs))
        return params, opt_state

    curve = []
    spread = None  # running standard deviation of discounted returns, for the reward normalisation
    discount = config.gamma ** jnp.arange(spec.n_moves)
    for iteration in range(config.iterations):
        key, k_collect, k_update = jax.random.split(key, 3)
        steps = collect(params, k_collect, truth, missions)
        returns = jnp.cumsum((steps.reward * discount[:, None])[::-1], axis=0)[::-1] / discount[:, None]  # (T, B)
        now = float(jnp.std(returns)) + 1e-8
        spread = now if spread is None else 0.99 * spread + 0.01 * now
        params, opt_state = update(params, opt_state, steps, k_update, jnp.array(spread))
        if iteration % every == 0 or iteration == config.iterations - 1:
            cost = -float(jnp.mean(jnp.sum(steps.reward, axis=0))) * config.reward_km
            curve.append({"iteration": iteration, "episodes": (iteration + 1) * config.envs, "cost_km": cost,
                          "log_std": float(params["log_std"])})
            log(f"iteration {iteration:5d}  episodes {(iteration + 1) * config.envs:7d}  mean cost {cost:12.1f} km  "
                f"log std {float(params['log_std']):+.2f}")
    return params, curve


# --- the trained policy as a controller -------------------------------------------------------------------------


class RLMemory(NamedTuple):
    readings: Readings
    flags: Flags


def controller(spec: Spec, truth: WindField, params: dict, mission: Mission, noise: Noise, sigma: jax.Array) -> Controller:
    """The policy flying the sweep's episode through fleet.fly, acting on its mean (no sampling), so it is
    scored by the same simulator, on the same task and forecast, as every other controller of its seed"""
    n, k = spec.n_agents, mission.targets_lat.shape[0]

    def begin(key: jax.Array, state: State) -> RLMemory:
        return RLMemory(Readings.empty(spec.n_moves * n if spec.learns else 1), Flags(jnp.zeros(n, bool), jnp.zeros(k, bool)))

    def act(memory: RLMemory, key: jax.Array, state: State, move: jax.Array, reading: jax.Array):
        flags = update_flags(spec, mission, state, memory.flags)
        forecast = jax.vmap(lambda la, lo, al, h: forecast_at(truth, spec.belief, noise, sigma, la, lo, al, h))(state.lat, state.lon, state.altitude_km, state.hours)
        readings = _add_all(memory.readings, state, reading - forecast) if spec.learns else memory.readings
        mean, _ = policy(params, observe(spec, truth, noise, sigma, mission, state, readings, flags, reading - forecast, move))
        return spec.balloon.climb_cap_km * jnp.tanh(mean), RLMemory(readings, flags)

    return Controller(begin, act)


def save(path: str | Path, params: dict, spec: Spec, config: RLConfig, curve: list[dict], extra: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    blob = {"params": jax.tree.map(np.asarray, params), "spec": spec._replace(balloon=dataclasses.asdict(spec.balloon), belief=dataclasses.asdict(spec.belief))._asdict(),
            "rl": dataclasses.asdict(config), "curve": curve, **extra}
    with open(path, "wb") as file:
        pickle.dump(blob, file)


def load(path: str | Path) -> tuple[dict, Spec, dict]:
    with open(path, "rb") as file:
        blob = pickle.load(file)
    s = blob["spec"]
    spec = Spec(**{**s, "balloon": BalloonConfig(**s["balloon"]), "belief": BeliefConfig(**s["belief"])})
    return jax.tree.map(jnp.asarray, blob["params"]), spec, blob


def policy_name(objective: str, n_agents: int, learns: bool) -> str:
    return f"{objective}-n{n_agents}-{'learns' if learns else 'blind'}"


def policies(experiment) -> list[tuple[str, int, bool]]:
    """Every policy the sweep trains: each objective, alone and at each fleet size, learning and blind"""
    return [(objective, n, learns) for n in (1, *experiment.fleet_sizes) for objective in ("reach", "escape", "station", "targets") for learns in (True, False)]
