"""Train BLE's QR-DQN on station keeping and score it (decision 57)

    uv run --extra gpu scripts/train_qrdqn.py --record records/era5_tropics_jul2021_l18_1h.npz --seed 0 --out results/qrdqn/seed0

Trains on missions from seeds 100,000 up, checks itself on 64 validation missions (seeds 200,000 up) every
--eval-every vector steps, and at the end flies the 32 test missions (seeds 0 to 31, the sweep's). Writes
curve.jsonl (mean reward, loss, validation time within 50 km, as training goes), params.npz (the last and the
best-on-validation weights) and test.json (time within 50 km per test seed, for the best-on-validation weights,
so the test seeds never choose anything).

Every QRConfig field can be overridden, e.g. --set learning_rate=1e-4 --set envs=256; each override is written
into curve.jsonl's first line, so a run says what it was.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from balloon_sim import qrdqn, station_env
from balloon_sim.belief import MeasuredKernel
from balloon_sim.degrade import error_size
from balloon_sim.fidelity import LAUNCH_BAND, MEASURED_KERNEL, Cell, configs
from balloon_sim.wind import load_record


def main() -> None:
    ask = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ask.add_argument("--record", required=True)
    ask.add_argument("--seed", type=int, default=0)
    ask.add_argument("--steps", type=int, default=200_000, help="vector steps, each one per env")
    ask.add_argument("--chunk", type=int, default=500)
    ask.add_argument("--eval-every", type=int, default=5000)
    ask.add_argument("--train-missions", type=int, default=2048)
    ask.add_argument("--set", action="append", default=[], help="a QRConfig field=value")
    ask.add_argument("--out", required=True)
    a = ask.parse_args()

    overrides = {}
    for item in a.set:
        k, v = item.split("=")
        overrides[k] = type(getattr(qrdqn.QRConfig(), k))(float(v) if "." in v or "e" in v else int(v))
    c = qrdqn.QRConfig()._replace(**overrides)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    truth, forecast = load_record(a.record), load_record(a.record, "forecast_")
    balloon, _, task = configs(Cell("mpc", 0, "truth", 0.0, 1.0, 12.0, "rl"))
    config = station_env.EnvConfig(balloon, MeasuredKernel.load(MEASURED_KERNEL), sigma=error_size(truth, forecast, LAUNCH_BAND))
    train = qrdqn.missions(task, truth, range(100_000, 100_000 + a.train_missions))
    valid = qrdqn.missions(task, truth, range(200_000, 200_064))
    test = qrdqn.missions(task, truth, range(32))

    start, run = qrdqn.make(c, config, truth, forecast, train)
    key = jax.random.key(a.seed)
    carry = start(jax.random.fold_in(key, 0))
    curve = out / "curve.jsonl"
    curve.write_text(json.dumps({"config": c._asdict(), "seed": a.seed, "steps": a.steps, "envs": c.envs, "device": str(jax.devices()[0])}) + "\n")
    best, best_params, began = -1.0, None, time.time()
    for done in range(0, a.steps, a.chunk):
        carry, (reward, loss) = run(carry, jax.random.fold_in(key, done + 1), a.chunk)
        row = {"vector_steps": done + a.chunk, "transitions": (done + a.chunk) * c.envs, "reward": float(jnp.mean(reward)),
               "loss": float(loss[-1]), "minutes": (time.time() - began) / 60}
        if (done + a.chunk) % a.eval_every == 0 or done + a.chunk >= a.steps:
            v = float(jnp.mean(qrdqn.evaluate(carry[0].params, c, config, truth, forecast, valid)))
            row["validation_twr50"] = v
            if v > best:
                best, best_params = v, jax.tree.map(np.asarray, carry[0].params)
        with curve.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    flat = lambda params: {f"{i}_{j}": x for i, layer in enumerate(params) for j, x in enumerate(layer)}
    np.savez(out / "params_last.npz", **flat(jax.tree.map(np.asarray, carry[0].params)))
    np.savez(out / "params_best.npz", **flat(best_params))
    scores = qrdqn.evaluate(jax.tree.map(jnp.asarray, best_params), c, config, truth, forecast, test)
    (out / "test.json").write_text(json.dumps({"seed": a.seed, "best_validation_twr50": best, "test_twr50": [float(x) for x in scores],
                                               "test_mean": float(jnp.mean(scores))}))
    print(f"test TWR50 mean {float(jnp.mean(scores)):.3f} (best validation {best:.3f})")


if __name__ == "__main__":
    main()
