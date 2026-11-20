"""The environments as pure JAX functions (reset and step), so episodes can be jitted, vmapped and scanned.

Both are thin shells round the simulator's own code, so a policy trained here is scored by the same simulator
as every controller in the sweeps.

FleetEnv (reach, max_distance, waypoints; one balloon or a fleet). One step is one 3 h move, exactly as
balloongym.sim.rl.episode flies it:

    the balloons move in the true wind by the climbs chosen
    reward = -(the objective's cost for that move) / 1000 km, so the return is -fleet_cost / 1000
    each balloon reads the true wind where it now is
    the belief is updated with those readings, and the next observation is built (rl.observe)

The observation (rl.observe), one row per balloon: altitude, time left, the error size, the vector to the
task's point and its length, a progress flag, the mean offset to the other balloons, the last reading minus
the forecast there, and the belief's mean wind at the balloon at every level.

What "the belief" means. The forecast is the truth minus a smooth random error (a Gaussian process with the
kernel fitted to a day-old forecast's error: 544 km, 1.3 km, 9.1 h). With `learns=True` the belief is that
forecast corrected by every reading the fleet has taken, the GP posterior mean

    belief(x) = forecast(x) + k(x, X) K^-1 (reading(X) - forecast(X))

so the wind near where a balloon has been is close to the truth and the correction fades over the kernel's
lengths. With `learns=False` the readings are set aside and the belief is the forecast.

StationEnv (station keeping, one balloon). One step is 3 min, exactly balloongym.sim.station_env.env_step:
up, stay or down; BLE's reward (1 inside 50 km, decaying outside); BLE's observation, whose wind features
come from the persistence forecast corrected by the latest 48 readings with the *measured* kernel
(decision 57), each level's posterior mean and standard deviation.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from balloongym.sim import rl, station_env
from balloongym.sim.belief import MeasuredKernel, Readings, read
from balloongym.sim.cells import keys
from balloongym.sim.degrade import error_size
from balloongym.sim.dynamics import State, clip, distance_km, step
from balloongym.sim.fidelity import LAUNCH_BAND
from balloongym.sim.fleet import Mission, draw_mission
from balloongym.sim.qrdqn import Missions as StationMissions
from balloongym.sim.qrdqn import missions as station_missions
from balloongym.sim.wind import WindField
from balloongym.tasks import Task
from balloongym.wind import Wind

MEASURED_KERNEL = str(__import__("pathlib").Path(__file__).parent / "sim" / "data" / "measured-kernel-2026-11-18.json")
TEST_SEEDS = tuple(range(32))  # the sweeps' seeds: every result so far is on these
TRAIN_SEED_OFFSET = 100_000  # RLConfig.seed_offset and QR-DQN's: training never sees a test launch
REWARD_KM = 1000.0


def draw_missions(task: Task, truth: WindField, seeds) -> Mission:
    """The launches (and goals) for these seeds, stacked on a leading axis, drawn as the sweeps draw them"""
    drawn = [draw_mission(task.task, keys(int(s))[0], truth)[0] for s in seeds]
    return jax.tree.map(lambda *xs: jnp.stack(xs), *drawn)


# --- reach, max distance, waypoints ------------------------------------------------------------------------------


class FleetState(NamedTuple):
    balloons: State  # (N,) each
    readings: Readings
    flags: rl.Flags
    mission: Mission
    noise: rl.Noise
    sigma: jax.Array
    residual: jax.Array  # (N, 2) the latest reading minus the forecast there
    move: jax.Array
    done: jax.Array
    cost_km: jax.Array  # the objective so far


@dataclass(frozen=True)
class FleetEnv:
    task: Task
    learns: bool = True  # the belief takes the readings
    view: str = "column"  # rl.VIEWS: what of the field the observation shows

    @property
    def spec(self) -> rl.Spec:
        t = self.task
        return rl.Spec(t.objective, t.n_balloons, self.learns, t.n_moves, t.radius_km, t.balloon, t.belief, self.view)

    def obs_size(self, n_levels: int) -> int:
        return rl.obs_size(n_levels, self.view)

    def _sense(self, truth: WindField, s: FleetState) -> tuple[jax.Array, FleetState]:
        """Flags, a reading at every balloon, the belief updated, and the observation: the start of a move"""
        spec, b = self.spec, s.balloons
        flags = rl.update_flags(spec, s.mission, b, s.flags)
        reading = jax.vmap(lambda one: read(truth, one))(b)
        forecast = jax.vmap(lambda la, lo, al, h: rl.forecast_at(truth, spec.belief, s.noise, s.sigma, la, lo, al, h))(b.lat, b.lon, b.altitude_km, b.hours)
        residual = reading - forecast
        readings = rl._add_all(s.readings, b, residual) if self.learns else s.readings
        s = s._replace(flags=flags, readings=readings, residual=residual)
        return rl.observe(spec, truth, s.noise, s.sigma, s.mission, b, readings, flags, residual, s.move), s

    @functools.partial(jax.jit, static_argnums=0)
    def reset(self, truth: WindField, missions: Mission, key: jax.Array) -> tuple[jax.Array, FleetState]:
        """A launch picked from `missions` (draw_missions) and a forecast error drawn from `key`"""
        k_pick, k_noise = jax.random.split(key)
        pool = missions.targets_lat.shape[0]
        mission = jax.tree.map(lambda a: a[jax.random.randint(k_pick, (), 0, pool)], missions)
        return self.reset_to(truth, mission, rl.noise_from_key(self.task.belief, k_noise))

    def reset_to(self, truth: WindField, mission: Mission, noise: rl.Noise) -> tuple[jax.Array, FleetState]:
        """A given launch and forecast error"""
        n, k = self.task.n_balloons, mission.targets_lat.shape[0]
        capacity = self.task.n_moves * n if self.learns else 1
        s = FleetState(mission.starts, Readings.empty(capacity), rl.Flags(jnp.zeros(n, bool), jnp.zeros(k, bool)), mission, noise,
                       jnp.asarray(self.task.belief.sigma, jnp.float32), jnp.zeros((n, 2)), jnp.array(0), jnp.array(False), jnp.array(0.0))
        return self._sense(truth, s)

    def climbs(self, action: jax.Array) -> jax.Array:
        """An action in [-1, 1] per balloon, as km climbed this move"""
        return self.task.balloon.climb_cap_km * jnp.clip(jnp.broadcast_to(action, (self.task.n_balloons,)), -1.0, 1.0)

    @functools.partial(jax.jit, static_argnums=0)
    def step(self, truth: WindField, s: FleetState, action: jax.Array):
        """(obs, state, reward, terminated, truncated, info). Steps after the end change nothing."""
        spec, balloon = self.spec, self.task.balloon
        move_one = jax.vmap(lambda one, c: clip(truth, step(truth, one, c, balloon.step_hours, balloon.substeps)))
        moved = move_one(s.balloons, self.climbs(action))
        cost = rl.move_cost(spec, s.mission, moved, s.flags)
        new = s._replace(balloons=moved, move=s.move + 1, cost_km=s.cost_km + cost)
        obs, new = self._sense(truth, new)
        if spec.objective == "reach":
            finished = jnp.all(new.flags.arrived)
        elif spec.objective == "targets":
            finished = jnp.all(new.flags.visited)
        else:
            finished = jnp.array(False)
        truncated = (new.move >= spec.n_moves) & ~finished
        new = new._replace(done=finished | truncated)
        new = jax.tree.map(lambda old, now: jnp.where(s.done, old, now), s, new)
        reward = jnp.where(s.done, 0.0, -cost / REWARD_KM)
        info = {"cost_km": new.cost_km, "score": self.score(new)}
        return obs, new, reward, finished & ~s.done, truncated & ~s.done, info

    def score(self, s: FleetState) -> jax.Array:
        """reach: balloons arrived (fraction); waypoints: points visited (fraction); max distance: mean km from
        launch now. The sweeps' headline is the cost, info["cost_km"]."""
        if self.task.objective == "reach":
            return jnp.mean(s.flags.arrived.astype(jnp.float32))
        if self.task.objective == "targets":
            return jnp.mean(s.flags.visited.astype(jnp.float32))
        return jnp.mean(distance_km(s.balloons.lat, s.balloons.lon, s.mission.starts.lat, s.mission.starts.lon))


# --- station keeping --------------------------------------------------------------------------------------------


class StationState(NamedTuple):
    env: station_env.EnvState
    inside: jax.Array  # steps ended within the radius
    done: jax.Array


@dataclass(frozen=True, eq=False)  # the kernel holds arrays: hashed by identity, so jit keys on the instance
class StationEnv:
    task: Task
    config: station_env.EnvConfig

    @staticmethod
    def build(task: Task, wind: Wind) -> "StationEnv":
        sigma = error_size(wind.truth, wind.earlier, LAUNCH_BAND)  # 1x: the forecast's own error in the launch band
        return StationEnv(task, station_env.EnvConfig(task.balloon, MeasuredKernel.load(MEASURED_KERNEL), sigma=sigma,
                                                      radius_km=task.radius_km, n_moves=task.n_moves))

    def obs_size(self, n_levels: int) -> int:
        return station_env.obs_size(n_levels)

    @functools.partial(jax.jit, static_argnums=0)
    def reset(self, wind: Wind, missions: StationMissions, key: jax.Array) -> tuple[jax.Array, StationState]:
        i = jax.random.randint(key, (), 0, missions.station_lat.shape[0])
        start = State(*(f[i] for f in missions.start))
        s = station_env.reset(self.config, start, missions.station_lat[i], missions.station_lon[i])
        return station_env.observe(self.config, wind.earlier, s), StationState(s, jnp.array(0), jnp.array(False))

    @functools.partial(jax.jit, static_argnums=0)
    def step(self, wind: Wind, s: StationState, action: jax.Array):
        """action: 0 up, 1 stay, 2 down (station_env.UP, STAY, DOWN)"""
        e, obs, reward, end = station_env.env_step(self.config, wind.truth, wind.earlier, s.env, jnp.asarray(action).reshape(()))
        inside = distance_km(e.balloon.lat, e.balloon.lon, e.station_lat, e.station_lon) <= self.config.radius_km
        new = StationState(e, s.inside + inside, end)
        new = jax.tree.map(lambda old, now: jnp.where(s.done, old, now), s, new)
        info = {"score": new.inside / jnp.maximum(new.env.move, 1)}
        return obs, new, jnp.where(s.done, 0.0, reward), jnp.array(False), end & ~s.done, info


def missions_for(task: Task, wind: Wind, seeds):
    """The launches for these seeds in the form the task's environment takes"""
    if task.name == "station_keeping":
        return station_missions(task.task, wind.truth, seeds)
    return draw_missions(task, wind.truth, seeds)


def make_env(task: Task, wind: Wind, learns: bool = True, view: str = "column"):
    return StationEnv.build(task, wind) if task.name == "station_keeping" else FleetEnv(task, learns, view)
