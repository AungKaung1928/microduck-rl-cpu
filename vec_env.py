"""Run N Microducks in N processes, because threads do not help here.

Step 1 measured why: MuJoCo's step for this 16-body model is single-threaded
and fits in cache, so the only parallelism available is process-level, and it
tops out at 20,184 env-steps/s across 8 processes on this box. This is the
piece that turns that number into samples.

Two things this deliberately does not do:

  - It does not use gymnasium's SyncVectorEnv/AsyncVectorEnv. Those are fine,
    but they bring an API that has changed repeatedly and a step-return shape
    that changed under them, for a 150-line problem.
  - It does not send the full info dict every step. At 20k steps/s the pickle
    traffic starts to show up next to the physics. Four scalars go across by
    default; pass with_terms=True when logging reward composition.

OMP_NUM_THREADS is set below BEFORE mujoco or numpy load, exactly as bench.py
does it. Workers are forked, so they inherit an interpreter where that already
took effect. Import this module before `env` if you are importing both.
"""
import os

# Must precede the mujoco/numpy import, transitively via `env`.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import multiprocessing as mp   # noqa: E402

import numpy as np             # noqa: E402

import env as _env             # noqa: E402

MAX_WORKERS = 8   # the thread budget on this machine. bench.py enforces the same.


def _worker(conn, seed, kwargs, with_terms):
    e = _env.MicroduckEnv(seed=seed, **kwargs)
    obs = e.reset(seed=seed)
    conn.send(obs)
    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "step":
                obs, rew, done, info = e.step(payload)
                small = {"fallen": info["fallen"],
                         "upright_cos": info["upright_cos"],
                         "trunk_height": info["trunk_height"],
                         "pushed": info["pushed"]}
                if with_terms:
                    small["terms"] = info["terms"]
                if done:
                    # Autoreset. The observation the policy needs for its next
                    # decision is the new episode's, but the value function
                    # needs the one it just ended on, so both go across.
                    small["terminal_obs"] = obs
                    obs = e.reset()
                conn.send((obs, rew, done, small))
            elif cmd == "reset":
                conn.send(e.reset(seed=payload))
            elif cmd == "close":
                break
    finally:
        conn.close()


class VecEnv:
    """N independent envs, one per process. Env i is seeded `seed + i`.

    That seeding rule is not cosmetic -- it is what makes a run reproducible
    regardless of how many workers it was spread over, and test_env.py asserts
    that N workers produce exactly what N sequential envs produce.
    """

    def __init__(self, n=4, seed=0, with_terms=False, **env_kwargs):
        if n > MAX_WORKERS:
            raise ValueError(
                f"{n} workers requested; the budget on this machine is {MAX_WORKERS} "
                f"of 14 cores. Raise MAX_WORKERS only if you own the box."
            )
        ctx = mp.get_context("fork")
        self.n = n
        self.closed = False
        self._conns, self._procs = [], []
        for i in range(n):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, daemon=True,
                            args=(child, seed + i, env_kwargs, with_terms))
            p.start()
            child.close()
            self._conns.append(parent)
            self._procs.append(p)
        self._last_obs = np.stack([c.recv() for c in self._conns])

    def reset(self, seed=None):
        if seed is None:
            return self._last_obs.copy()
        for i, c in enumerate(self._conns):
            c.send(("reset", seed + i))
        self._last_obs = np.stack([c.recv() for c in self._conns])
        return self._last_obs.copy()

    def step(self, actions):
        actions = np.asarray(actions).reshape(self.n, _env.ACT_DIM)
        for c, a in zip(self._conns, actions):
            c.send(("step", a))
        out = [c.recv() for c in self._conns]
        obs = np.stack([o[0] for o in out])
        rew = np.array([o[1] for o in out], dtype=np.float64)
        done = np.array([o[2] for o in out], dtype=bool)
        self._last_obs = obs
        return obs.copy(), rew, done, [o[3] for o in out]

    def close(self):
        if self.closed:
            return
        self.closed = True
        for c in self._conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        for c in self._conns:
            c.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()
