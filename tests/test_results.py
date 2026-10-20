import jax.numpy as jnp

from balloon_sim.config import BeliefConfig, PlannerConfig
from balloon_sim.dynamics import State
from balloon_sim.episode import Flight
from balloon_sim.results import read_rows, row, write_rows
from balloon_sim.task import Task

TASK = Task(State(jnp.array(30.0), jnp.array(150.0), jnp.array(17.5), jnp.array(0.0)), jnp.array(31.0), jnp.array(151.0))
FLIGHT = Flight(jnp.zeros(3), TASK.start, jnp.array(1234.5), jnp.array(67.8))


def test_a_row_flattens_every_setting_and_carries_the_metrics():
    r = row(FLIGHT, TASK, controller="mpc", n_winds=8, seed=3, seconds=1.5, settings={"planner": PlannerConfig(), "belief": BeliefConfig(sigma=4.8)})
    assert r["controller"] == "mpc" and r["n_winds"] == 8 and r["seed"] == 3
    assert r["planner.horizon"] == 20 and r["belief.sigma"] == 4.8
    assert r["summed_km"] == 1234.5 and abs(r["final_km"] - 67.8) < 1e-4
    assert (r["start_lat"], r["goal_lon"]) == (30.0, 151.0)


def test_rows_survive_writing_and_reading(tmp_path):
    rows = [row(FLIGHT, TASK, controller=c, n_winds=0, seed=0, seconds=0.1, settings={"planner": PlannerConfig()}) for c in ("hindsight", "mpc")]
    path = tmp_path / "cell.jsonl"
    write_rows(path, rows[:1])
    write_rows(path, rows[1:])  # appends
    assert read_rows([path]) == rows
