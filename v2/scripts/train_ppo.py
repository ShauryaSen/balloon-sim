"""Train PPO on a fleet task (reach, max_distance, waypoints) and save the policy.

    python scripts/train_ppo.py --task reach --out policies/reach.pkl
    python scripts/train_ppo.py --task waypoints --n-balloons 4 --wind records/era5_global_jul2021.npz --set iterations=500

Any RLConfig field can be set with --set field=value. Then score it with
    python scripts/evaluate.py --task reach --controller ppo --params policies/reach.pkl
"""

import argparse
import dataclasses
import json
import pickle
import sys
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from balloongym import wind as winds  # noqa: E402
from balloongym.sim.config import RLConfig  # noqa: E402
from balloongym.tasks import make_task  # noqa: E402
from balloongym.train import train_ppo  # noqa: E402


def main() -> None:
    ask = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ask.add_argument("--task", choices=("reach", "max_distance", "waypoints"), required=True)
    ask.add_argument("--wind", default="synthetic")
    ask.add_argument("--n-balloons", type=int, default=1)
    ask.add_argument("--sigma", type=float, default=None)
    ask.add_argument("--blind", action="store_true", help="the belief never takes a reading")
    ask.add_argument("--view", default="column", choices=("column", "patch", "ahead"))
    ask.add_argument("--seed", type=int, default=0)
    ask.add_argument("--set", action="append", default=[], help="an RLConfig field=value")
    ask.add_argument("--out", required=True)
    a = ask.parse_args()
    base = RLConfig()
    overrides = {k: type(getattr(base, k))(v) for k, v in (s.split("=") for s in a.set)}
    config = dataclasses.replace(base, **overrides)
    wind = winds.synthetic() if a.wind == "synthetic" else winds.load(a.wind)
    task = make_task(a.task, n_balloons=a.n_balloons, sigma_ms=a.sigma)
    params, spec, curve = train_ppo(task, wind, config, seed=a.seed, learns=not a.blind, view=a.view)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({"params": jax.tree.map(np.asarray, params), "spec": spec, "rl": dataclasses.asdict(config),
                     "task": a.task, "wind": a.wind, "curve": curve}, f)
    out.with_suffix(".json").write_text(json.dumps(curve, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
