import json

import jax.numpy as jnp

from balloon_sim.config import BeliefConfig, PlannerConfig
from balloon_sim.dynamics import State
from balloon_sim.episode import Flight
from balloon_sim.results import best_hindsight, read_rows, row, write_rows
from balloon_sim.task import Task

TASK = Task(State(jnp.array(30.0), jnp.array(150.0), jnp.array(17.5), jnp.array(0.0)), jnp.array(31.0), jnp.array(151.0))
# a flight that never moves: three moves all at the start, so every move is equally close to the goal
FLIGHT = Flight(jnp.zeros(3), TASK.start, jnp.array(1234.5), jnp.array(67.8), jnp.array(1234.5), jnp.array(False), jnp.array(0))


def test_a_row_flattens_every_setting_and_carries_the_metrics():
    r = row(FLIGHT, TASK, controller="mpc", n_winds=8, seed=3, seconds=1.5, settings={"planner": PlannerConfig(), "belief": BeliefConfig(sigma=4.8)})
    assert r["controller"] == "mpc" and r["n_winds"] == 8 and r["seed"] == 3
    assert r["planner.horizon"] == 20 and r["belief.sigma"] == 4.8
    assert r["summed_km"] == 1234.5 and abs(r["final_km"] - 67.8) < 1e-4
    assert r["cost_km"] == 1234.5 and r["arrived"] is False and r["arrived_move"] == 0
    assert (r["start_lat"], r["goal_lon"]) == (30.0, 151.0)
    assert r["closest_move"] == 1 and r["closest_km"] > 0  # FLIGHT never moves, so every move is equally close


def test_rows_survive_writing_and_reading(tmp_path):
    rows = [row(FLIGHT, TASK, controller=c, n_winds=0, seed=0, seconds=0.1, settings={"planner": PlannerConfig()}) for c in ("hindsight", "mpc")]
    path = tmp_path / "cell.jsonl"
    write_rows(path, rows[:1])
    write_rows(path, rows[1:])  # appends
    assert read_rows([path]) == rows


def test_a_torn_line_is_skipped_and_a_directory_is_read(tmp_path):
    good = row(FLIGHT, TASK, controller="mpc", n_winds=0, seed=0, seconds=0.1, settings={"planner": PlannerConfig()})
    (tmp_path / "0.jsonl").write_text(json.dumps(good) + "\n4711}\n")
    (tmp_path / "1.jsonl").write_text(json.dumps(good) + "\n")
    assert read_rows([tmp_path]) == [good, good]


def test_hindsight_becomes_the_best_clairvoyant_flight_of_its_seed():
    base = {"seed": 0, "n_winds": 0, "planner.horizon": 50}
    rows = [
        {**base, "controller": "hindsight", "belief.sigma": 0.0, "summed_km": 500.0, "final_km": 90.0},
        {**base, "controller": "mpc", "belief.sigma": 0.0, "summed_km": 400.0, "final_km": 120.0},  # truth-informed
        {**base, "controller": "mpc", "belief.sigma": 3.2, "summed_km": 100.0, "final_km": 10.0},  # noisy: not a ceiling
    ]
    hind = [r for r in best_hindsight(rows) if r["controller"] == "hindsight"][0]
    assert (hind["summed_km"], hind["final_km"]) == (400.0, 90.0)  # each metric its own best, noisy flights ignored


def test_an_arm_is_chosen_before_anything_is_summarised():
    """The arms share every other setting, so rows from two of them must never land on one key"""
    from balloon_sim.results import one_arm

    rows = [
        {"controller": "mpc", "seed": 0, "goal_km": 2000.0, "belief_scale": 1.0, "nested": False},
        {"controller": "mpc", "seed": 0, "goal_km": 3000.0, "belief_scale": 1.0, "nested": False},
        {"controller": "mpc", "seed": 0, "goal_km": 2000.0, "belief_scale": 2.0, "nested": False},
        {"controller": "mpc", "seed": 0, "goal_km": 2000.0, "belief_scale": 1.0, "nested": True},
    ]
    assert one_arm(rows, 2000.0) == rows[:1]
    assert one_arm(rows, 3000.0) == rows[1:2]
    assert one_arm(rows, 2000.0, belief_scale=2.0) == rows[2:3]
    assert one_arm(rows, 2000.0, nested=True) == rows[3:4]


def test_hindsight_belongs_to_every_arm_at_its_goal():
    """It plans on the truth, so a miscalibrated belief and nested draws are settings it never reads: the
    side arms do not fly their own hindsight and take the one at their goal distance"""
    from balloon_sim.results import one_arm

    rows = [
        {"controller": "hindsight", "seed": 0, "goal_km": 2000.0, "belief_scale": 1.0, "nested": False},
        {"controller": "hindsight", "seed": 0, "goal_km": 3000.0, "belief_scale": 1.0, "nested": False},
    ]
    assert one_arm(rows, 2000.0, belief_scale=0.5) == rows[:1]
    assert one_arm(rows, 2000.0, nested=True) == rows[:1]
    assert one_arm(rows, 3000.0, belief_scale=2.0) == rows[1:]


def test_rows_from_before_the_arms_existed_are_kept():
    """A sweep with one arm is one arm; only a sweep with several has to choose"""
    from balloon_sim.results import one_arm

    old = [{"controller": "mpc", "seed": 0}, {"controller": "fixed_plan", "seed": 1}]
    assert one_arm(old) == old


def test_a_replanning_controller_is_keyed_by_its_horizon():
    """mpc_blind replans, so its horizons are four different controllers and must not share a key"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from summarise import keyed

    rows = [
        {"controller": "mpc_blind", "seed": 0, "n_winds": 0, "belief.sigma": 3.2, "planner.horizon": h}
        for h in (5, 10, 20, 50)
    ]
    assert len(keyed(rows)) == 4
    assert all(("mpc_blind", h, 0, 3.2) in keyed(rows) for h in (5, 10, 20, 50))
