"""Reference controllers, to put a learned policy's score on a scale.

    coast   never changes altitude. The floor: what the wind does on its own.
    greedy  tries down, stay and up, each held for `lookahead` steps on the forecast, and takes the one that
            ends nearest the task's point.
    mpc     random-shooting model predictive control: plans `samples` altitude sequences `horizon` steps long
            on the forecast, flies the first step of the best, and replans every step.

Greedy and MPC plan on the forecast, never the truth, so they are as wrong as the forecast is. Each balloon of a
fleet is planned on its own (no coordination). With `forecast=ForecastConfig(kind="perfect")` MPC plans on the
truth, which makes it a hindsight-style ceiling.

Each returns a Policy: (obs, state, key) -> action, usable with env.rollout or the Gymnasium wrapper's
`reference_action`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from balloongym import tasks
from balloongym.env import BalloonEnv, EnvState, Policy
from balloongym.forecast import forecast_at
from balloongym.physics import EARTH_RADIUS_M
from balloongym.wind import WindField


def _to_action(env: BalloonEnv, command: jax.Array) -> jax.Array:
    if env.config.action == "discrete":
        return jnp.round(command).astype(jnp.int32) + 1
    return command


def coast(env: BalloonEnv) -> Policy:
    n = env.config.n_balloons
    return lambda obs, state, key: _to_action(env, jnp.zeros(n))


def _fly_forecast(env: BalloonEnv, field: WindField, state: EnvState, i: int | jax.Array, commands: jax.Array) -> jax.Array:
    """Balloon i flown on the forecast under a sequence of commands (H,); its cost-to-go after each step (H,)"""
    c = env.config
    b = state.balloons
    dt_h = c.balloon.dt_minutes / 60.0
    climb = c.balloon.climb_km_per_hour * dt_h
    low, high = field.altitude_km[0], field.altitude_km[-1]

    def one(p, command):
        lat, lon, alt, h = p
        u, v = forecast_at(field, c.forecast, state.error, lat, lon, alt, h)
        lat2 = jnp.clip(lat + jnp.degrees(v * dt_h * 3600.0 / EARTH_RADIUS_M), field.lat[0], field.lat[-1])
        lon2 = (lon + jnp.degrees(u * dt_h * 3600.0 / (EARTH_RADIUS_M * jnp.cos(jnp.radians(lat))))) % 360.0
        alt2 = jnp.clip(alt + command * climb, low, high)
        p = (lat2, lon2, alt2, h + dt_h)
        return p, p

    _, (lat, lon, _, _) = jax.lax.scan(one, (b.lat[i], b.lon[i], b.altitude_km[i], b.hours[i]), commands)
    task_state = state.task
    return jax.vmap(lambda la, lo: tasks.cost_to_go(c.task, task_state, la[None], lo[None])[0])(lat, lon)


def greedy(env: BalloonEnv, field: WindField, lookahead: int = 6) -> Policy:
    options = jnp.array([-1.0, 0.0, 1.0])

    def policy(obs, state, key):
        def per_balloon(i):
            final = jax.vmap(lambda c: _fly_forecast(env, field, state, i, jnp.full(lookahead, c))[-1])(options)
            return options[jnp.argmin(final)]

        return _to_action(env, jax.vmap(per_balloon)(jnp.arange(env.config.n_balloons)))

    return policy


def mpc(env: BalloonEnv, field: WindField, horizon: int = 24, samples: int = 64, hold: int = 4) -> Policy:
    """Random shooting over piecewise-constant plans: each plan picks down, stay or up every `hold` steps.
    The three constant plans are always among the candidates. Cost is the summed cost-to-go along the plan."""
    blocks = -(-horizon // hold)
    constant = jnp.repeat(jnp.array([-1.0, 0.0, 1.0])[:, None], blocks, axis=1)

    def policy(obs, state, key):
        def per_balloon(i, k):
            drawn = jax.random.randint(k, (samples - 3, blocks), -1, 2).astype(jnp.float32)
            plans = jnp.repeat(jnp.concatenate([constant, drawn]), hold, axis=1)[:, :horizon]
            cost = jax.vmap(lambda p: jnp.sum(_fly_forecast(env, field, state, i, p)))(plans)
            return plans[jnp.argmin(cost), 0]

        n = env.config.n_balloons
        return _to_action(env, jax.vmap(per_balloon)(jnp.arange(n), jax.random.split(key, n)))

    return policy
