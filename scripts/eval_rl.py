"""Fly a trained policy on the sweep's own seeds and noise levels, and write its rows (decision 50)

    uv run scripts/eval_rl.py policies/targets-n4-learns.pkl --out results/rows-2026-11-07/rl-targets-n4-learns.jsonl
    uv run scripts/eval_rl.py --index 5 --out-dir results/rows-2026-11-07     # the 6th of rl.policies()

Each (seed, sigma) is the sweep's cell for that seed: the same task key, the same forecast error, the same
flight key, flown by fleet.fly and written by fleet.fleet_row, with controller "rl" (learning from readings)
or "rl_blind". So its rows sit in the same arm as hindsight, the fixed plan and MPC of that seed and every
reader pairs them like any other controller. The policy acts on its mean, with no sampling.
"""

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from balloon_sim import fleet, rl
from balloon_sim.cells import keys
from balloon_sim.config import ExperimentConfig, PlannerConfig, RLConfig, TaskConfig, WindConfig
from balloon_sim.results import write_rows
from balloon_sim.wind import load_record


def main(argv: list[str] | None = None) -> None:
    ask = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ask.add_argument("policy", nargs="?", help="a .pkl written by train_rl.py")
    ask.add_argument("--index", type=int, help="the policy at this position of rl.policies(), from --policies")
    ask.add_argument("--policies", default="policies")
    ask.add_argument("--out", help="the .jsonl to add rows to")
    ask.add_argument("--out-dir", help="or a directory, where the rows go in rl-<policy>.jsonl")
    ask.add_argument("--record", help="the wind record, if not the one the policy was trained on")
    ask.add_argument("--seeds", type=int, nargs="*", help="only these seeds (default: the sweep's)")
    ask.add_argument("--sigmas", type=float, nargs="*", help="only these noise levels (default: the ladder trained on)")
    asked = ask.parse_args(argv)

    experiment = ExperimentConfig()
    path = asked.policy
    if asked.index is not None:
        path = Path(asked.policies) / f"{rl.policy_name(*rl.policies(experiment)[asked.index])}.pkl"
    if path is None:
        ask.error("give a policy, or --index")
    params, spec, blob = rl.load(path)
    out = Path(asked.out) if asked.out else Path(asked.out_dir) / f"rl-{blob['name']}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    record = asked.record or blob["record"]
    truth = load_record(record)
    task_config = TaskConfig(**{**blob["task"], "lat_band": tuple(blob["task"]["lat_band"])})
    controller_name = "rl" if spec.learns else "rl_blind"
    settings = {"wind": WindConfig(record), "balloon": spec.balloon, "planner": PlannerConfig(), "task": task_config,
                "rl": RLConfig(**{**blob["rl"]})}

    for seed in asked.seeds or experiment.seeds:
        k_task, k_noise, k_fly = keys(seed)
        mission, task = fleet.draw_mission(task_config, k_task, truth)
        for sigma in asked.sigmas or blob["sigmas"]:
            belief = dataclasses.replace(spec.belief, sigma=sigma)
            noise = rl.noise_from_key(belief, k_noise)  # the pattern run_cell's forecast() subtracts for this seed
            controller = rl.controller(spec, truth, params, mission, noise, jnp.array(sigma))
            began = time.perf_counter()
            flight = fleet.fly(spec.balloon, controller, truth, mission, spec.objective, spec.n_moves, k_fly, spec.radius_km)
            jax.block_until_ready(flight.cost_km)
            row = fleet.fleet_row(
                flight, mission, task, controller=controller_name, n_winds=0, seed=seed, seconds=time.perf_counter() - began,
                settings={**settings, "belief": belief},
                extra={"goal_km": task_config.distance_km, "direction": task_config.direction, "belief_scale": 1.0,
                       "nested": False, "objective": spec.objective, "policy": blob["name"],
                       "policy_episodes": blob["curve"][-1]["episodes"] if blob["curve"] else 0},
                keep_path=True,
            )
            write_rows(out, [row])
            print(f"{controller_name:8s} {blob['name']:22s} seed {seed:3d} sigma {sigma:5.2f}  cost {float(flight.cost_km):10.1f} km")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
