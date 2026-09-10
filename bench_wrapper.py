"""What does the Python environment wrapper cost on top of raw MuJoCo?

The claim in the README -- that observation assembly, reward, push scheduling
and episode bookkeeping are free next to the physics -- is a ratio, so it needs
both halves measured the same way, back to back, in one process.

The bare loop below is deliberately not a fair "do nothing" baseline: it steps
the same model the same number of substeps and writes the same ctrl vector, so
the only difference between the two timings is the wrapper's own Python.

Single process, about 25 s, one core. Nothing here loads the machine.
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse           # noqa: E402
import json               # noqa: E402
import time               # noqa: E402

import mujoco             # noqa: E402
import numpy as np        # noqa: E402

import common             # noqa: E402
from env import MicroduckEnv   # noqa: E402


def bare(variant, seconds):
    """mj_step at the same substep count, with a ctrl write, and nothing else."""
    model, data = common.load(variant)
    common.reset_to(model, data, "STAND")
    ctrl = common.default_pose(model, "STAND").astype(np.float64)
    steps = 0
    t0 = time.perf_counter()
    while True:
        data.ctrl[:] = ctrl
        for _ in range(common.SUBSTEPS):
            mujoco.mj_step(model, data)
        steps += 1
        if (steps & 0xFF) == 0 and time.perf_counter() - t0 > seconds:
            break
    return steps / (time.perf_counter() - t0)


def wrapped(variant, seconds):
    """The real environment, zero action, resets included."""
    env = MicroduckEnv(variant=variant, seed=0)
    env.reset()
    action = np.zeros(common.N_ACT, dtype=np.float32)
    steps = 0
    t0 = time.perf_counter()
    while True:
        _, _, done, _ = env.step(action)
        if done:
            env.reset()
        steps += 1
        if (steps & 0xFF) == 0 and time.perf_counter() - t0 > seconds:
            break
    return steps / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--variants", nargs="+",
                    default=["groundcontact", "walk"])
    ap.add_argument("--out", default="runs/bench_wrapper.json")
    a = ap.parse_args()

    import bench
    for w in bench.box_is_busy():
        print(f"  WARNING  {w} -- numbers below are contention, not capability")

    print(f"\n  {'variant':<14} {'bare mj_step':>13} {'wrapped env':>13} "
          f"{'wrapper cost':>13}")
    rows = {}
    for v in a.variants:
        b = bare(v, a.seconds)
        w = wrapped(v, a.seconds)
        rows[v] = {"bare": b, "wrapped": w, "overhead": b / w - 1}
        print(f"  {v:<14} {b:13,.0f} {w:13,.0f} {100*(b/w-1):12.1f}%")

    print("\n  Both columns are env-steps/s, one env step = "
          f"{common.SUBSTEPS} physics steps at {common.CONTROL_HZ} Hz control.")
    print("  `groundcontact` enables ten ground-collidable geoms against "
          "`walk`'s two,\n  so it runs slower in absolute terms. It is the "
          "variant the task uses.")

    os.makedirs("runs", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"seconds": a.seconds, "rows": rows}, f, indent=2)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
