"""Score a controller on a task's test launches and write one JSON line per launch.

    python scripts/evaluate.py --task reach --controller mpc
    python scripts/evaluate.py --task reach --controller ppo --params policies/reach.pkl
    python scripts/evaluate.py --task station_keeping --controller qrdqn --params runs/qrdqn/params_best.npz
    python scripts/evaluate.py --task reach --controller mpc --wind records/era5_global_jul2021.npz --seeds 32

Controllers: coast, fixed_plan, mpc, mpc_blind, mpc_truth, hindsight, ppo, qrdqn (balloongym.benchmark).
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from balloongym import benchmark  # noqa: E402
from balloongym import wind as winds
from balloongym.sim.qrdqn import QRConfig  # noqa: E402
from balloongym.tasks import TASKS, make_task  # noqa: E402


def load_params(path: str):
    if path.endswith(".pkl"):
        blob = pickle.load(open(path, "rb"))
        return blob["params"], blob.get("spec")
    flat = np.load(path)  # QR-DQN: layer i, part j saved as "i_j"
    layers = {}
    for name in flat.files:
        i, j = map(int, name.split("_"))
        layers.setdefault(i, {})[j] = jnp.asarray(flat[name])
    return [[layers[i][j] for j in sorted(layers[i])] for i in sorted(layers)], None


def main() -> None:
    ask = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ask.add_argument("--task", choices=TASKS, required=True)
    ask.add_argument("--controller", choices=benchmark.CONTROLLERS, required=True)
    ask.add_argument("--wind", default="synthetic", help="'synthetic' or a record .npz")
    ask.add_argument("--n-balloons", type=int, default=1)
    ask.add_argument("--sigma", type=float, default=None, help="forecast error, m/s (fleet tasks)")
    ask.add_argument("--seeds", type=int, default=32, help="test launches 0 .. n-1")
    ask.add_argument("--params", help="trained weights: a PPO .pkl (train_ppo.py) or a QR-DQN .npz (train_qrdqn.py)")
    ask.add_argument("--out", default=None, help="JSONL path (default: print)")
    a = ask.parse_args()

    wind = winds.synthetic() if a.wind == "synthetic" else winds.load(a.wind)
    task = make_task(a.task, n_balloons=a.n_balloons, sigma_ms=a.sigma)
    params, spec = load_params(a.params) if a.params else (None, None)
    qr_config = None
    if a.params and (Path(a.params).parent / "config.json").exists():
        qr_config = QRConfig(**json.loads((Path(a.params).parent / "config.json").read_text()))
    result = benchmark.evaluate(task, wind, a.controller, seeds=range(a.seeds), params=params, policy_spec=spec,
                                qr_config=qr_config, log=print)
    rows = [{"task": a.task, "controller": a.controller, "seed": s, **{k: float(v[i]) for k, v in result.items()}}
            for i, s in enumerate(range(a.seeds))]
    text = "\n".join(json.dumps(r) for r in rows) + "\n"
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
    print(json.dumps({"task": a.task, "controller": a.controller, **benchmark.summary(result["value"])}))


if __name__ == "__main__":
    main()
