"""Train QR-DQN (BLE's recipe) on station keeping and save the best-on-validation weights.

    python scripts/train_qrdqn.py --out runs/qrdqn/seed0
    python scripts/train_qrdqn.py --wind records/era5_tropics_jul2021_l18_1h.npz --steps 200000 --out runs/qrdqn/seed0

Any QRConfig field can be set with --set field=value (e.g. --set envs=256). Then score it with
    python scripts/evaluate.py --task station_keeping --controller qrdqn --params runs/qrdqn/seed0/params_best.npz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from balloongym import wind as winds  # noqa: E402
from balloongym.sim.qrdqn import QRConfig  # noqa: E402
from balloongym.tasks import make_task  # noqa: E402
from balloongym.train import train_qrdqn  # noqa: E402


def main() -> None:
    ask = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ask.add_argument("--wind", default="synthetic")
    ask.add_argument("--seed", type=int, default=0)
    ask.add_argument("--steps", type=int, default=200_000)
    ask.add_argument("--eval-every", type=int, default=5000)
    ask.add_argument("--set", action="append", default=[], help="a QRConfig field=value")
    ask.add_argument("--out", required=True)
    a = ask.parse_args()
    base = QRConfig()
    c = base._replace(**{k: type(getattr(base, k))(float(v) if type(getattr(base, k)) is float else int(float(v)))
                         for k, v in (s.split("=") for s in a.set)})
    wind = winds.synthetic() if a.wind == "synthetic" else winds.load(a.wind)
    params, curve = train_qrdqn(make_task("station_keeping"), wind, c, seed=a.seed, steps=a.steps, eval_every=a.eval_every)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "params_best.npz", **{f"{i}_{j}": x for i, layer in enumerate(params) for j, x in enumerate(layer)})
    (out / "config.json").write_text(json.dumps(c._asdict()))  # evaluate.py rebuilds the network from it
    (out / "curve.jsonl").write_text("\n".join(json.dumps(r) for r in curve) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
