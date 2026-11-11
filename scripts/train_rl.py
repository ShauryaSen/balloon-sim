"""Train one learned policy (decision 50)

    uv run scripts/train_rl.py --objective targets --n-agents 4            # learns from the fleet's readings
    uv run scripts/train_rl.py --objective targets --n-agents 4 --blind    # never uses a reading
    uv run scripts/train_rl.py --index 5                                    # the 6th of rl.policies(), for a Slurm array

Writes policies/<objective>-n<N>-<learns|blind>.pkl (the parameters, the settings, the training curve) and
the curve again as .json beside it. Any RLConfig field can be overridden as --rl.<field>.
"""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import jax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from balloon_sim import rl
from balloon_sim.cells import BASE_DIRECTION, BASE_GOAL_KM
from balloon_sim.config import BalloonConfig, BeliefConfig, ExperimentConfig, RLConfig, TaskConfig, WindConfig
from balloon_sim.wind import load_record


def main(argv: list[str] | None = None) -> None:
    ask = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ask.add_argument("--objective", choices=("reach", "escape", "station", "targets"))
    ask.add_argument("--n-agents", type=int, default=1)
    ask.add_argument("--blind", action="store_true", help="never use a reading: the policy rl_blind")
    ask.add_argument("--view", choices=rl.VIEWS, default="column",
                     help="what of the field the policy sees: its own column, a patch round it, or where each level carries it")
    ask.add_argument("--index", type=int, help="train the policy at this position of rl.policies() instead")
    ask.add_argument("--record", default=WindConfig().path)
    ask.add_argument("--n-moves", type=int, default=TaskConfig().n_moves)
    ask.add_argument("--seed", type=int, default=0, help="the training run's own randomness")
    ask.add_argument("--out", default="policies")
    for field in dataclasses.fields(RLConfig):
        ask.add_argument(f"--rl.{field.name}", type=type(field.default), default=None, dest=f"rl.{field.name}")
    asked = vars(ask.parse_args(argv))

    experiment = ExperimentConfig()
    if asked["index"] is not None:
        objective, n_agents, learns = rl.policies(experiment)[asked["index"]]
    else:
        if asked["objective"] is None:
            ask.error("give --objective, or --index")
        objective, n_agents, learns = asked["objective"], asked["n_agents"], not asked["blind"]
    config = dataclasses.replace(RLConfig(), **{f.name: asked[f"rl.{f.name}"] for f in dataclasses.fields(RLConfig) if asked[f"rl.{f.name}"] is not None})
    task = TaskConfig(objective=objective, n_agents=n_agents, distance_km=BASE_GOAL_KM, direction=BASE_DIRECTION, n_moves=asked["n_moves"])
    spec = rl.Spec(objective, n_agents, learns, task.n_moves, task.arrival_km, BalloonConfig(), BeliefConfig(), asked["view"])
    sigmas = tuple(round(m * BeliefConfig().sigma, 4) for m in experiment.sigma_multiples)
    name = rl.policy_name(objective, n_agents, learns, asked["view"])
    print(f"training {name}: {config.iterations} iterations of {config.envs} episodes, {config.train_seeds} training tasks, sigma over {sigmas}")

    truth = load_record(asked["record"])
    began = time.perf_counter()
    missions = rl.training_missions(task, truth, config)
    print(f"  {config.train_seeds} training tasks drawn in {time.perf_counter() - began:.0f} s")
    began = time.perf_counter()
    params, curve = rl.train(spec, truth, missions, sigmas, config, jax.random.key(asked["seed"]))
    seconds = time.perf_counter() - began
    out = Path(asked["out"])
    rl.save(out / f"{name}.pkl", params, spec, config, curve,
            {"name": name, "record": asked["record"], "sigmas": sigmas, "seconds": seconds, "task": dataclasses.asdict(task)})
    (out / f"{name}.json").write_text(json.dumps({"name": name, "seconds": seconds, "curve": curve}, indent=1))
    print(f"wrote {out / name}.pkl after {seconds / 60:.1f} min")


if __name__ == "__main__":
    main()
