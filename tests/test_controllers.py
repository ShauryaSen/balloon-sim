import jax
import jax.numpy as jnp

from balloon_sim.belief import GPBelief, forecast
from balloon_sim.config import BalloonConfig, BeliefConfig, PlannerConfig
from balloon_sim.controllers import fixed_plan, hindsight, mpc
from balloon_sim.dynamics import State
from balloon_sim.episode import fly
from balloon_sim.wind import WindField

BALLOON = BalloonConfig()
MOVES = 8
SMALL = PlannerConfig(horizon=MOVES, candidates=30, iterations=50)
START = State(jnp.array(30.0), jnp.array(150.0), jnp.array(17.5), jnp.array(0.0))
GOAL = (jnp.array(30.0), jnp.array(156.0))  # about 580 km east


def east_low_west_high() -> WindField:
    """15 m/s east at 15 km, 15 m/s west at 20 km: to go east, the balloon has to sink"""
    hours, altitude, lat, lon = jnp.array([0.0, 100.0]), jnp.array([15.0, 20.0]), jnp.array([0.0, 70.0]), jnp.array([100.0, 200.0])
    uv = jnp.zeros((2, 2, 2, 2, 2)).at[:, 0, ..., 0].set(15.0).at[:, 1, ..., 0].set(-15.0)
    return WindField(hours, altitude, lat, lon, uv)


def belief_about(truth: WindField, sigma: float, key: jax.Array) -> GPBelief:
    config = BeliefConfig(sigma=sigma)
    return GPBelief(config, forecast(config, key, truth))


def test_with_a_perfect_forecast_the_fixed_plan_is_hindsight():
    """At sigma = 0 every belief draw is the truth, so the floor and the ceiling meet"""
    truth = east_low_west_high()
    key = jax.random.key(0)
    ceiling = fly(BALLOON, hindsight(BALLOON, SMALL, truth, *GOAL, MOVES), truth, START, *GOAL, MOVES, key)
    floor = fly(BALLOON, fixed_plan(BALLOON, SMALL, belief_about(truth, 0.0, key), *GOAL, MOVES, 2), truth, START, *GOAL, MOVES, key)
    # both reach the goal; the costs agree up to the optimiser's own slack (L-BFGS stops at a gradient threshold,
    # from a random pool, and its line search lands differently on different CPUs)
    assert floor.final_km < 1.0 and ceiling.final_km < 1.0
    assert jnp.allclose(floor.summed_km, ceiling.summed_km, rtol=0.10)


def test_mpc_with_a_perfect_forecast_gets_where_hindsight_does():
    truth = east_low_west_high()
    key = jax.random.key(0)
    ceiling = fly(BALLOON, hindsight(BALLOON, SMALL, truth, *GOAL, MOVES), truth, START, *GOAL, MOVES, key)
    replanned = fly(BALLOON, mpc(BALLOON, SMALL, belief_about(truth, 0.0, key), *GOAL, 0, MOVES), truth, START, *GOAL, MOVES, key)
    assert replanned.summed_km <= 1.05 * ceiling.summed_km


def test_mpc_flies_the_first_climb_and_keeps_the_rest():
    truth = east_low_west_high()
    controller = mpc(BALLOON, SMALL, belief_about(truth, 0.0, jax.random.key(0)), *GOAL, 0, MOVES)
    memory = controller.begin(jax.random.key(1), START)
    climb, kept = controller.act(memory, jax.random.key(2), START, jnp.array(0), jnp.array([0.0, 0.0]))
    assert kept.previous.shape == (MOVES,) and kept.previous[-1] == 0.0  # shifted by one, coast at the end
    assert kept.readings.count == 1  # and it remembered what it read
    assert climb < 0  # sinks first, toward the eastward wind


def test_on_average_hindsight_beats_mpc_beats_the_fixed_plan():
    """The bounds' order, V_hind <= V_policy <= V_plan, averaged over seeds with a noisy forecast"""
    truth = east_low_west_high()
    costs = {"hindsight": [], "mpc": [], "fixed_plan": []}
    for seed in range(6):
        key = jax.random.key(seed)
        belief = belief_about(truth, 6.0, key)
        controllers = {
            "hindsight": hindsight(BALLOON, SMALL, truth, *GOAL, MOVES),
            "mpc": mpc(BALLOON, SMALL, belief, *GOAL, 4, MOVES),
            "fixed_plan": fixed_plan(BALLOON, SMALL, belief, *GOAL, MOVES, 4),
        }
        for name, controller in controllers.items():
            costs[name].append(float(fly(BALLOON, controller, truth, START, *GOAL, MOVES, key).summed_km))
    mean = {name: sum(c) / len(c) for name, c in costs.items()}
    assert mean["hindsight"] <= mean["mpc"] <= mean["fixed_plan"], mean


def test_mpc_learns_the_wind_it_reads():
    """A forecast that is wrong everywhere by a constant: after readings at the balloon, the belief's mean there
    is the truth, not the forecast, so MPC plans on what it read. The start sits on a grid node, since the
    belief is conditioned on the grid and then blended, which is exact only at nodes"""
    from balloon_sim.belief import read
    from balloon_sim.wind import wind_at

    hours, altitude, lat, lon = jnp.array([0.0, 100.0]), jnp.array([15.0, 17.5, 20.0]), jnp.array([0.0, 30.0, 70.0]), jnp.array([100.0, 150.0, 200.0])
    uv = jnp.zeros((2, 3, 3, 3, 2)).at[:, 0, ..., 0].set(15.0).at[:, 2, ..., 0].set(-15.0)
    truth = WindField(hours, altitude, lat, lon, uv)
    wrong = GPBelief(BeliefConfig(sigma=6.0), truth._replace(uv=truth.uv + jnp.array([20.0, 0.0])))  # 20 m/s too eastward everywhere
    controller = mpc(BALLOON, SMALL, wrong, *GOAL, 0, MOVES)
    memory = controller.begin(jax.random.key(0), START)
    for move in range(2):
        _, memory = controller.act(memory, jax.random.key(move), START, jnp.array(move), read(truth, START))
    believed = wrong.winds(jax.random.key(9), 0, memory.readings)
    at_balloon = wind_at(believed._replace(uv=believed.uv[0]), START.lat, START.lon, START.altitude_km, START.hours)
    assert jnp.allclose(at_balloon, read(truth, START), atol=0.1)
