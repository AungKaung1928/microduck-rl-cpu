"""Contract tests for the environment. Every expected value is hand-derived.

These exist because the failure modes in an RL environment are quiet. A
mis-scaled observation, an action that saturates against a joint stop, a push
that never fires, an index that reads a passive joint on the evaluation model
-- none of them raise, and all of them produce training curves that look fine
and a policy that is measuring something other than what you think.

The most important check in this file is the third one. It rebuilds the
observation from the five sources the contract declares and asserts the
environment's output is exactly that, which is the only way to prove that no
sim-only quantity leaked in. The second most important is the last one, which
runs the observation builder on `walk_backlash` -- the held-out physics -- and
shows what the obvious qpos[7:21] shortcut would have read instead.

Run:  python test_env.py
"""
import numpy as np

import vec_env            # imported first: it sets OMP_NUM_THREADS before mujoco
import common
import env
import mujoco

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def rollout(e, n, actions):
    obs, rew = [], []
    for a in actions[:n]:
        o, r, d, i = e.step(a)
        obs.append(o); rew.append(r)
    return np.array(obs), np.array(rew)


rng = np.random.default_rng(0)
ACTIONS = rng.uniform(-1, 1, (300, env.ACT_DIM))

print("\n--- shape and layout ---")
e = env.MicroduckEnv(seed=0)
o = e.reset(seed=0)
check("observation is 48-dim float32 and finite",
      o.shape == (env.OBS_DIM,) and o.dtype == np.float32 and np.isfinite(o).all(),
      f"{o.shape} {o.dtype}")
check("action space is 14, matching the actuator count",
      env.ACT_DIM == common.N_ACT == e.model.nu)
covered = np.zeros(env.OBS_DIM, dtype=int)
for s in env.OBS_SLICES.values():
    covered[s] += 1
ndims = sum(len(range(*sl.indices(env.OBS_DIM))) for sl in env.OBS_SLICES.values())
check("the five observation slices tile [0,48) with no gap or overlap",
      bool((covered == 1).all()), f"{len(env.OBS_SLICES)} slices, {ndims} dims")
check("OBS_SCALE has one entry per observation dim", env.OBS_SCALE.shape == (env.OBS_DIM,))

print("\n--- the observation is EXACTLY the five declared sources, nothing else ---")
# Rebuilt here from scratch. If a sim-only quantity were smuggled in -- base
# linear velocity, world height, absolute yaw -- this comparison fails.
e = env.MicroduckEnv(seed=3)
e.reset(seed=3)
o, _, _, _ = e.step(ACTIONS[0])
qi, vi = common.actuated_qpos_index(e.model), common.actuated_qvel_index(e.model)
q = e.model.key_qpos[common.keyframe(e.model, "STAND")][qi]
quat = np.array(e.data.sensordata[common.sensor_slice(e.model, "orientation")], dtype=float)
mujoco.mju_normalize4(quat)
qinv = np.empty(4); mujoco.mju_negQuat(qinv, quat)
g = np.empty(3); mujoco.mju_rotVecQuat(g, np.array([0.0, 0.0, -1.0]), qinv)
expect = np.concatenate([
    e.data.qpos[qi] - q,
    e.data.qvel[vi] * 0.05,
    np.clip(ACTIONS[0], -1, 1),
    np.array(e.data.sensordata[common.sensor_slice(e.model, "angular-velocity")]) * 0.25,
    g,
]).astype(np.float32)
check("observation reproduces bit-for-bit from joint pos/vel, prev action, gyro, proj-gravity",
      np.allclose(o, expect, atol=1e-6), f"max |diff| {np.abs(o - expect).max():.2e}")

vel = np.array(e.data.sensordata[common.sensor_slice(e.model, "imu_lin_vel")])
check("the excluded velocimeter is live, so its absence is a real exclusion",
      float(np.abs(vel).max()) > 1e-4, f"|imu_lin_vel|max = {np.abs(vel).max():.4f} m/s")

print("\n--- projected gravity ---")
e = env.MicroduckEnv(seed=0, init_noise=0.0)
o = e.reset(seed=0)
pg = o[env.OBS_SLICES["proj_grav"]]
check("upright at STAND reads (0, 0, -1)",
      np.allclose(pg, [0, 0, -1], atol=1e-6), str(np.round(pg, 6)))
norms = []
for a in ACTIONS[:120]:
    o, *_ = e.step(a)
    norms.append(np.linalg.norm(o[env.OBS_SLICES["proj_grav"]]))
check("stays a unit vector through 120 steps of thrashing",
      max(abs(np.array(norms) - 1)) < 1e-5, f"max |norm-1| {max(abs(np.array(norms)-1)):.2e}")

print("\n--- actions ---")
e = env.MicroduckEnv(seed=0, init_noise=0.0)
e.reset(seed=0)
lo, hi = common.joint_limits(e.model)
e.step(np.zeros(env.ACT_DIM))
check("zero action commands the STAND pose exactly",
      np.allclose(e.data.ctrl, common.default_pose(e.model), atol=1e-12))
e.step(np.full(env.ACT_DIM, 5.0))
check("an out-of-range action is clipped to [-1,1] then to the joint limits",
      bool((e.data.ctrl >= lo - 1e-9).all() and (e.data.ctrl <= hi + 1e-9).all()),
      f"ctrlrange is [-10,10] rad; tightest limit {float((hi - lo).min()) / 2:+.3f}")
o, *_ = e.step(ACTIONS[5])
check("prev_action in the observation is the action just taken",
      np.allclose(o[env.OBS_SLICES["prev_action"]], ACTIONS[5], atol=1e-6))

print("\n--- episode ---")
e = env.MicroduckEnv(seed=0, episode_steps=50)
e.reset(seed=0)
dones = [e.step(a)[2] for a in ACTIONS[:50]]
check("done fires exactly once, on the last step of a fixed-length episode",
      sum(dones) == 1 and dones[-1] is True, f"{sum(dones)} done flags in 50 steps")
e = env.MicroduckEnv(seed=0)
e.reset(seed=0)
worst = 0.0
for a in ACTIONS[:60]:
    _, r, _, info = e.step(a)
    worst = max(worst, abs(r - sum(info["terms"].values())))
check("reward equals the sum of its logged terms, every step",
      worst < 1e-12 and set(info["terms"]) == set(env.REWARD_WEIGHTS),
      f"{len(info['terms'])} terms, worst residual {worst:.1e}: {', '.join(sorted(info['terms']))}")

print("\n--- determinism and seeding ---")
a_obs, a_rew = rollout(env.MicroduckEnv(seed=11), 200, ACTIONS)
e = env.MicroduckEnv(seed=11); e.reset(seed=11)
b_obs, b_rew = rollout(e, 200, ACTIONS)
e = env.MicroduckEnv(seed=11); e.reset(seed=11)
c_obs, c_rew = rollout(e, 200, ACTIONS)
check("same seed replays bit-for-bit",
      np.array_equal(b_obs, c_obs) and np.array_equal(b_rew, c_rew))
e1, e2 = env.MicroduckEnv(seed=1), env.MicroduckEnv(seed=2)
e1.reset(seed=1); e2.reset(seed=2)
check("different seeds draw different push schedules",
      not np.array_equal(e1._push_at, e2._push_at), f"{e1._push_at} vs {e2._push_at}")

print("\n--- pushes actually land, at the scheduled step ---")
ea = env.MicroduckEnv(seed=7, n_pushes=1, init_noise=0.0)
eb = env.MicroduckEnv(seed=7, n_pushes=1, init_noise=0.0)
ea.reset(seed=7); eb.reset(seed=7)
at = int(ea._push_at[0])
eb._push_vel[:] = 0.0            # same schedule, zero impulse
oa, _ = rollout(ea, at + 5, ACTIONS)
ob, _ = rollout(eb, at + 5, ACTIONS)
diff = np.abs(oa - ob).max(axis=1)
first = int(np.argmax(diff > 1e-6)) if (diff > 1e-6).any() else -1
check("trajectory is identical before the push and diverges on the push step",
      first == at, f"scheduled step {at}, first divergence {first}, "
                   f"|dv| {np.linalg.norm(ea._push_vel[0]):.3f} m/s")

print("\n--- vector env ---")
n = 3
with vec_env.VecEnv(n=n, seed=100) as v:
    v_obs, v_rew = [], []
    for a in ACTIONS[:120]:
        o, r, d, _ = v.step(np.repeat(a[None], n, axis=0))
        v_obs.append(o); v_rew.append(r)
v_obs, v_rew = np.array(v_obs), np.array(v_rew)
s_obs, s_rew = [], []
for i in range(n):
    e = env.MicroduckEnv(seed=100 + i); e.reset(seed=100 + i)
    oo, rr = rollout(e, 120, ACTIONS)
    s_obs.append(oo); s_rew.append(rr)
s_obs = np.stack(s_obs, axis=1); s_rew = np.stack(s_rew, axis=1)
check(f"{n} workers reproduce {n} sequential envs bit-for-bit",
      np.array_equal(v_obs, s_obs) and np.allclose(v_rew, s_rew, atol=0),
      "env i is seeded seed+i, so a run is reproducible at any worker count")
check("worker count is capped at the machine's 8-of-14 budget", vec_env.MAX_WORKERS == 8)

print("\n--- the held-out physics model, which is the point of all of this ---")
eb = env.MicroduckEnv(variant="walk_backlash", seed=0, init_noise=0.0)
ob = eb.reset(seed=0)
check("walk_backlash gives the same 48-dim observation and 14-dim action",
      ob.shape == (env.OBS_DIM,) and eb.model.nu == env.ACT_DIM,
      f"nq {eb.model.nq} vs {env.MicroduckEnv(seed=0).model.nq} on groundcontact")
eb.step(ACTIONS[0])
correct = eb.data.qpos[common.actuated_qpos_index(eb.model)]
naive = eb.data.qpos[7:21]
check("the qpos[7:21] shortcut would read the wrong 14 numbers here",
      not np.allclose(correct, naive),
      f"max |diff| {np.abs(correct - naive).max():.4f} rad across "
      f"{int((~np.isclose(correct, naive)).sum())} of 14")
check("and the actuator order is identical, so a policy transfers with no remapping",
      [mujoco.mj_id2name(eb.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(eb.model.nu)]
      == common.ACTUATOR_NAMES)

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: " + ", ".join(FAILED))
    raise SystemExit(1)
print("all environment contract checks passed")
