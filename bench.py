"""How fast can this laptop actually simulate the Microduck?

This is the feasibility gate for the whole project. Upstream trains on GPU with
4096 parallel environments; this box has no GPU and an 8-thread budget on a
company laptop. If the answer here is too slow, the project scope shrinks from
walking to standing, and that decision gets made on a measured number rather
than on a feeling.

MuJoCo's step for a 16-body model is single-threaded and small enough to stay
in cache, so parallelism has to come from processes, not threads. Hence
OMP_NUM_THREADS=1 below, set before mujoco loads -- otherwise every worker
would try to spawn its own thread pool and fight the others.

What is counted: one "env step" is one 50 Hz control decision, which is 10
physics steps. RL sample budgets are quoted in env steps, so that is the unit
the gate is written in.

Run:  python bench.py                 # 1, 2, 4, 8 processes
      python bench.py --seconds 30    # longer window
"""
import os

# Must precede the mujoco/numpy import. Parallelism here is by process.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse            # noqa: E402
import json                # noqa: E402
import multiprocessing as mp   # noqa: E402
import subprocess          # noqa: E402
import time                # noqa: E402

import mujoco              # noqa: E402
import numpy as np         # noqa: E402

import common              # noqa: E402

FALL_HEIGHT = 0.04         # m; trunk this low means it is down, reset


def sustained_worker(rank, window, nwindows, variant, barrier, q):
    m, d = common.load(variant)
    common.reset_to(m, d, "STAND")
    nominal = d.ctrl.copy()
    rng = np.random.default_rng(1234 + rank)
    for _ in range(200):
        d.ctrl[:] = nominal + rng.normal(0, 0.05, common.N_ACT)
        mujoco.mj_step(m, d)
    barrier.wait()
    for w in range(nwindows):
        steps = 0
        t0 = time.perf_counter()
        while True:
            d.ctrl[:] = nominal + rng.normal(0, 0.05, common.N_ACT)
            for _ in range(common.SUBSTEPS):
                mujoco.mj_step(m, d)
            steps += 1
            if common.trunk_height(m, d) < FALL_HEIGHT:
                common.reset_to(m, d, "STAND")
            if (steps & 0xFF) == 0 and time.perf_counter() - t0 > window:
                break
        q.put((w, steps / (time.perf_counter() - t0)))


def sustained(nproc, window, nwindows, variant):
    """Hold the full load without pause and watch the rate decay.

    The bracketed-reference sweep cannot see sustained-power limiting, because
    its single-process reference windows let the package cool between
    configurations. A real training run does not pause. This does what a
    training run does.
    """
    ctx = mp.get_context("fork")
    barrier, q = ctx.Barrier(nproc), ctx.Queue()
    ps = [ctx.Process(target=sustained_worker,
                      args=(r, window, nwindows, variant, barrier, q))
          for r in range(nproc)]
    for p in ps:
        p.start()
    agg = [[] for _ in range(nwindows)]
    for _ in range(nproc * nwindows):
        w, rate = q.get()
        agg[w].append(rate)
    for p in ps:
        p.join()
    return [float(sum(a)) for a in agg]


def worker(rank, seconds, variant, barrier, q, pin=None):
    if pin is not None:
        os.sched_setaffinity(0, {pin})
    m, d = common.load(variant)
    common.reset_to(m, d, "STAND")
    nominal = d.ctrl.copy()
    rng = np.random.default_rng(1234 + rank)

    # Warm up outside the timed window: first steps pay for lazy allocation
    # inside MjData and for the constraint solver finding its working set.
    for _ in range(200):
        d.ctrl[:] = nominal + rng.normal(0, 0.05, common.N_ACT)
        mujoco.mj_step(m, d)

    barrier.wait()
    env_steps = resets = 0
    t0 = time.perf_counter()
    while True:
        d.ctrl[:] = nominal + rng.normal(0, 0.05, common.N_ACT)
        for _ in range(common.SUBSTEPS):
            mujoco.mj_step(m, d)
        env_steps += 1
        if common.trunk_height(m, d) < FALL_HEIGHT:
            common.reset_to(m, d, "STAND")
            resets += 1
        if (env_steps & 0xFF) == 0 and time.perf_counter() - t0 > seconds:
            break
    el = time.perf_counter() - t0
    q.put((rank, env_steps, el, resets))


def run(nproc, seconds, variant, pins=None):
    ctx = mp.get_context("fork")
    barrier, q = ctx.Barrier(nproc), ctx.Queue()
    ps = [ctx.Process(target=worker,
                      args=(r, seconds, variant, barrier, q,
                            None if pins is None else pins[r]))
          for r in range(nproc)]
    for p in ps:
        p.start()
    res = [q.get() for _ in ps]
    for p in ps:
        p.join()
    rates = [s / e for _, s, e, _ in res]
    return {
        "procs": nproc,
        "env_steps_per_s": float(sum(rates)),
        "per_proc": float(np.mean(rates)),
        "per_proc_spread": float(max(rates) - min(rates)),
        "resets": sum(r[3] for r in res),
    }


def box_is_busy():
    """Contention and thermal throttling are indistinguishable from inside WSL,
    so at least rule out the one that is visible."""
    warn = []
    # Presence of another python process is not contention -- this box always
    # has two sleeping system daemons. Only CPU time actually being consumed
    # matters, so filter on %CPU instead of on the process list.
    try:
        out = subprocess.run(["ps", "-eo", "pid,pcpu,comm", "--no-headers"],
                             capture_output=True, text=True).stdout.splitlines()
        busy = []
        for line in out:
            pid, pcpu, comm = line.split(None, 2)
            if int(pid) != os.getpid() and float(pcpu) > 5.0:
                busy.append(f"{comm.strip()}({pcpu}%)")
        if busy:
            warn.append("CPU in use by " + ", ".join(busy[:4]))
    except (OSError, ValueError):
        pass
    la = os.getloadavg()[0]
    if la > 1.5:
        warn.append(f"1-min load average {la:.2f}")
    return warn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--ref-seconds", type=float, default=None,
                    help="length of the 1-process reference windows "
                         "(default: half of --seconds)")
    ap.add_argument("--variant", default="walk", choices=sorted(common.VARIANTS))
    ap.add_argument("--procs", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--pin", type=int, nargs="+", default=None,
                    help="pin worker i to CPU pin[i]; needs len(pin) >= max(procs)")
    ap.add_argument("--gate", type=float, default=5000.0,
                    help="env-steps/s below which walking leaves the scope")
    ap.add_argument("--allow-overcommit", action="store_true",
                    help="permit more than 8 processes (company-laptop cap)")
    ap.add_argument("--sustained", type=int, default=0, metavar="N",
                    help="instead of the sweep, hold N processes flat out and "
                         "report the rate per window (what a training run does)")
    ap.add_argument("--windows", type=int, default=12)
    ap.add_argument("--tag", default="", help="suffix for the output filename")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ref_seconds = a.ref_seconds if a.ref_seconds else a.seconds / 2
    out = a.out or f"runs/bench{('_' + a.tag) if a.tag else ''}.json"

    if max(a.procs) > 8 and not a.allow_overcommit:
        raise SystemExit("refusing >8 processes: this is a company laptop and "
                         "env.sh caps the budget at 8 of 14 cores. "
                         "Pass --allow-overcommit if that is really intended.")
    if a.pin and len(a.pin) < max(a.procs):
        raise SystemExit(f"--pin needs at least {max(a.procs)} cores")

    for w in box_is_busy():
        print(f"  WARNING  {w} -- numbers below are contention, not capability")

    if a.sustained:
        n, win = a.sustained, a.seconds
        print(f"\nsustained load: {n} processes, {a.windows} x {win:.0f} s "
              f"with no pause ({a.windows*win/60:.1f} min total)")
        rates = sustained(n, win, a.windows, a.variant)
        peak = max(rates)
        print(f"\n  {'window':>7} {'elapsed':>9} {'env-steps/s':>13} {'vs peak':>9}")
        for i, r in enumerate(rates):
            print(f"  {i:7d} {(i+1)*win:8.0f}s {r:13,.0f} {100*(r/peak-1):+8.1f}%")
        tail = float(np.mean(rates[-3:]))
        decay = tail / peak - 1
        print(f"\n  peak {peak:,.0f}   sustained (last 3 windows) {tail:,.0f}   "
              f"{100*decay:+.1f}%")
        if decay < -0.20:
            print(f"  The box does not hold its peak rate. Budget training on "
                  f"{tail:,.0f} env-steps/s,\n  not on {peak:,.0f}. WSL cannot "
                  f"read CPU temperature, so this decay curve is the\n  only "
                  f"evidence available that sustained power limiting is real here.")
        else:
            print(f"  The box holds its rate under continuous load.")
        os.makedirs("runs", exist_ok=True)
        with open(out, "w") as f:
            json.dump({"mode": "sustained", "procs": n, "window_s": win,
                       "rates": rates, "peak": peak, "sustained": tail,
                       "decay": decay}, f, indent=2)
        print(f"\n  wrote {out}")
        return

    m, _ = common.load(a.variant)
    print(f"\nvariant {a.variant}: nq {m.nq} nv {m.nv} nu {m.nu}, "
          f"{1/m.opt.timestep:.0f} Hz physics, {common.SUBSTEPS} substeps "
          f"per {common.CONTROL_HZ} Hz control step")
    print(f"{a.seconds:.0f} s per configuration, {ref_seconds:.0f} s reference "
          f"windows, {os.cpu_count()} cores present, 8 permitted"
          + (f", pinned to {a.pin}" if a.pin else ""))

    # A single-process reference is measured before the sweep and again after
    # every configuration. Inside WSL there is no CPU temperature and no
    # frequency readout, so a drifting reference is the only available evidence
    # that a row was measured on a different machine than the row above it.
    ref0 = run(1, ref_seconds, a.variant, a.pin)["env_steps_per_s"]
    print(f"\nreference (1 proc, before sweep): {ref0:,.0f} env-steps/s\n")

    print(f"  {'procs':>5} {'env-steps/s':>13} {'per proc':>10} {'speedup':>9} "
          f"{'eff':>6} {'resets':>7} {'ref after':>11} {'drift':>8}")
    # Speedup is measured against the 1-process row of this same sweep when
    # there is one, not against the reference window. The reference windows are
    # shorter and exist only to detect drift; using one as the baseline makes
    # the 1-process row report something other than exactly 1.00x, which reads
    # as a bug even though it is just the gap between two window lengths.
    rows = []
    baseline = None
    for n in a.procs:
        r = run(n, a.seconds, a.variant, a.pin)
        ref = run(1, ref_seconds, a.variant, a.pin)["env_steps_per_s"]
        if baseline is None:
            baseline = r["env_steps_per_s"] if n == 1 else ref0
        r["speedup"] = r["env_steps_per_s"] / baseline
        r["efficiency"] = r["speedup"] / n
        r["ref_after"] = ref
        r["ref_drift"] = ref / ref0 - 1
        rows.append(r)
        flag = "  <-- reference moved" if abs(r["ref_drift"]) > 0.10 else ""
        print(f"  {n:5d} {r['env_steps_per_s']:13,.0f} {r['per_proc']:10,.0f} "
              f"{r['speedup']:8.2f}x {100*r['efficiency']:5.0f}% {r['resets']:7d} "
              f"{ref:11,.0f} {100*r['ref_drift']:+7.1f}%{flag}")

    worst = max(abs(r["ref_drift"]) for r in rows)
    print()
    if worst > 0.10:
        print(f"  UNSTABLE: the 1-process reference moved by up to {100*worst:.0f}% "
              f"across the sweep.\n  Either the box is power-limiting or something "
              f"else used the CPU. Do not quote\n  these numbers; let it idle and "
              f"re-run. Both causes look identical from inside WSL.")
    else:
        print(f"  Stable: the 1-process reference held to within "
              f"{100*worst:.0f}% across the whole sweep.")

    best = max(r["env_steps_per_s"] for r in rows)
    print(f"\n  gate: {a.gate:,.0f} env-steps/s   measured: {best:,.0f}   "
          f"{'PASS' if best >= a.gate else 'FAIL'}")
    print(f"\n  wall-clock at {best:,.0f} env-steps/s:")
    for label, n in [("10M  (smoke / hyperparameter probe)", 10e6),
                     ("50M  (stand + push recovery, expected)", 50e6),
                     ("100M (stand, generous)", 100e6),
                     ("400M (walking gait, upstream-scale)", 400e6)]:
        h = n / best / 3600
        note = "" if h <= 2 else f"  -> {int(np.ceil(h/2))} chunks of <=2 h"
        print(f"    {label:<40} {h:6.2f} h{note}")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"variant": a.variant, "seconds": a.seconds,
                   "ref_seconds": ref_seconds, "cores": os.cpu_count(),
                   "pin": a.pin, "ref_before": ref0, "baseline": baseline,
                   "rows": rows,
                   "worst_ref_drift": worst, "stable": bool(worst <= 0.10),
                   "gate": a.gate, "pass": bool(best >= a.gate)}, f, indent=2)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
