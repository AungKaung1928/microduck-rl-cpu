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
import traceback              # noqa: E402

import numpy as np             # noqa: E402

import env as _env             # noqa: E402

MAX_WORKERS = 8   # the thread budget on this machine. bench.py enforces the same.


def _worker(conn, seed, kwargs, with_terms):
    """One env in one process.

    Every message to the parent is tagged ("ok", payload) or ("error", text).
    Without the tag a worker that dies leaves the parent blocked in recv(),
    which then raises a bare EOFError naming neither the worker nor the cause;
    the child's traceback does reach stderr, but in a redirected training log
    it lands nowhere near the failure. Construction is inside the guard too,
    because a bad env_kwargs fails there and produced the same bare EOFError
    out of VecEnv.__init__.
    """
    try:
        e = _env.MicroduckEnv(seed=seed, **kwargs)
        conn.send(("ok", e.reset(seed=seed)))
    except BaseException:
        try:
            conn.send(("error", traceback.format_exc()))
        except (BrokenPipeError, OSError):
            pass
        conn.close()
        return
    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "step":
                obs, rew, done, info = e.step(payload)
                small = {"fallen": info["fallen"],
                         "upright_cos": info["upright_cos"],
                         "trunk_height": info["trunk_height"],
                         "pushed": info["pushed"],
                         "t": info["t"],
                         # Carried across because a truncation is not a
                         # terminal state and the consumer cannot tell from
                         # `done` alone. See MicroduckEnv.step.
                         "truncated": info["truncated"],
                         "terminated": info["terminated"]}
                if with_terms:
                    small["terms"] = info["terms"]
                if done:
                    # Autoreset. The observation the policy needs for its next
                    # decision is the new episode's, but the value function
                    # needs the one it just ended on, so both go across.
                    small["terminal_obs"] = obs
                    obs = e.reset()
                conn.send(("ok", (obs, rew, done, small)))
            elif cmd == "reset":
                conn.send(("ok", e.reset(seed=payload)))
            elif cmd == "close":
                break
    except BaseException:
        try:
            conn.send(("error", traceback.format_exc()))
        except (BrokenPipeError, OSError):
            pass
    finally:
        conn.close()


class VecEnv:
    """N independent envs, one per process. Env i is seeded `seed + i`.

    That seeding rule is not cosmetic -- it is what makes a run reproducible
    regardless of how many workers it was spread over, and test_env.py asserts
    that N workers produce exactly what N sequential envs produce.
    """

    def __init__(self, n=4, seed=0, with_terms=False, **env_kwargs):
        # Before anything that can raise. __del__ calls close(), so a
        # constructor that fails validation would otherwise raise
        # AttributeError from inside __del__ and bury the real ValueError
        # under "Exception ignored in __del__".
        self.closed = False
        self._conns, self._procs = [], []
        if n > MAX_WORKERS:
            raise ValueError(
                f"{n} workers requested; the budget on this machine is {MAX_WORKERS} "
                f"of 14 cores. Raise MAX_WORKERS only if you own the box."
            )
        ctx = mp.get_context("fork")
        self.n = n
        for i in range(n):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, daemon=True,
                            args=(child, seed + i, env_kwargs, with_terms))
            p.start()
            child.close()
            self._conns.append(parent)
            self._procs.append(p)
        self._last_obs = np.stack([self._recv(i) for i in range(n)])

    def _recv(self, i):
        """Receive one tagged message, and name the worker when it goes wrong."""
        try:
            tag, payload = self._conns[i].recv()
        except EOFError:
            raise RuntimeError(
                f"worker {i} exited without replying. Its traceback went to "
                f"stderr -- in a redirected log, look above this line."
            ) from None
        if tag == "error":
            raise RuntimeError(f"worker {i} raised:\n{payload}")
        return payload

    def reset(self, seed=None):
        """Bare reset() returns the current observation; it does not restart.

        Passing `seed` restarts every worker on `seed + i` and REPLACES its
        RNG, so calling `reset(seed=cfg.seed)` once per training iteration
        replays one identical episode forever. Autoreset inside step() is the
        path a training loop should use.
        """
        if seed is None:
            return self._last_obs.copy()
        for i, c in enumerate(self._conns):
            c.send(("reset", seed + i))
        self._last_obs = np.stack([self._recv(i) for i in range(self.n)])
        return self._last_obs.copy()

    def step(self, actions):
        actions = np.asarray(actions).reshape(self.n, _env.ACT_DIM)
        for c, a in zip(self._conns, actions):
            c.send(("step", a))
        out = [self._recv(i) for i in range(self.n)]
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
