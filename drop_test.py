"""Does the shipped PD controller hold the robot up at all?

Two questions, both needed before writing a policy:

  --mode limp   Actuator gains zeroed, so the robot is a ragdoll. This checks
                that contact, mass and inertia are sane: it should fall, land,
                and come to rest rather than jitter, sink through the floor or
                explode. A model that cannot fall over correctly cannot be
                trained on.

  --mode hold   Actuators commanded to the STAND keyframe, which is where the
                policy will start every episode. If the shipped kp=0.55 holds
                that pose under gravity, the PD hold-pose controller is a
                legitimate baseline for the learned policy to beat. If it
                sags or topples, the baseline is "falls over in N ms" and the
                learning task is harder than balancing a stable pose.

Run:  python drop_test.py --mode hold
      python drop_test.py --mode limp --z0 0.25
"""
import argparse
import os

import mujoco
import numpy as np

import common

STILL_QVEL = 0.05      # rad/s and m/s; below this everywhere counts as at rest
STILL_FOR = 0.20       # s it must stay below before we call it settled


def make_limp(model):
    """Zero every actuator's force output.

    A MuJoCo `position` actuator produces gain*ctrl + bias1*qpos + bias2*qvel.
    Setting ctrl to zero does NOT disable it -- it commands the zero angle with
    full stiffness, which is a pose hold, not a ragdoll. All three coefficients
    have to go.
    """
    model.actuator_gainprm[:, 0] = 0.0
    model.actuator_biasprm[:, 1] = 0.0
    model.actuator_biasprm[:, 2] = 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="hold", choices=["hold", "limp"])
    ap.add_argument("--variant", default="groundcontact",
                    choices=sorted(common.VARIANTS))
    ap.add_argument("--keyframe", default="STAND")
    ap.add_argument("--z0", type=float, default=None,
                    help="override initial trunk height, m (default: keyframe)")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    out = a.out or f"out/drop_{a.variant}_{a.mode}.png"

    m, d = common.load(a.variant)
    touch = common.floor_contact_geoms(m)
    if len(touch) <= 2:
        print(f"  NOTE  on `{a.variant}` only {touch} can touch the floor. "
              f"A toppled robot\n        falls through it. That is correct for a "
              f"walk task that terminates on\n        a fall, and wrong for "
              f"anything that has to lie on the ground.")
    if a.mode == "limp":
        make_limp(m)
    common.reset_to(m, d, a.keyframe)
    hold_ctrl = d.ctrl.copy()
    if a.z0 is not None:
        d.qpos[2] = a.z0
        mujoco.mj_forward(m, d)

    n = int(a.seconds / m.opt.timestep)
    fell_at = None
    still_needed = int(STILL_FOR / m.opt.timestep)
    t_h = np.empty(n)
    t_up = np.empty(n)
    t_v = np.empty(n)
    settled_at = None
    still = 0
    frames_at = np.linspace(0, n - 1, 6).astype(int)
    frames = []

    trunk_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 0.55, 140.0, -15.0

    renderer = None
    if not a.no_render:
        try:
            renderer = mujoco.Renderer(m, 240, 320)
        except Exception as e:      # WSLg down, or no display
            print(f"  render unavailable ({type(e).__name__}: {e}); "
                  f"continuing headless")

    print(f"\n{a.variant} / {a.mode} / keyframe {a.keyframe}"
          + (f" / z0 {a.z0} m" if a.z0 is not None else ""))
    print(f"  start: trunk {common.trunk_height(m, d)*100:.1f} cm, "
          f"upright {common.upright_cos(m, d):.3f}")

    for i in range(n):
        d.ctrl[:] = hold_ctrl if a.mode == "hold" else 0.0
        mujoco.mj_step(m, d)
        t_h[i] = common.trunk_height(m, d)
        t_up[i] = common.upright_cos(m, d)
        t_v[i] = float(np.abs(d.qvel).max())
        still = still + 1 if t_v[i] < STILL_QVEL else 0
        if settled_at is None and still >= still_needed:
            settled_at = (i - still_needed) * m.opt.timestep
        if fell_at is None and t_up[i] < 0.7:
            fell_at = i * m.opt.timestep
        if renderer is not None and i in frames_at:
            # Track the trunk. The scene's free camera sits far back and the
            # robot is 25 cm tall, so an untracked shot is mostly floor.
            cam.lookat[:] = d.xpos[trunk_bid]
            renderer.update_scene(d, camera=cam)
            frames.append(renderer.render())

    if not np.all(np.isfinite(t_h)):
        raise SystemExit("  FAIL: simulation diverged (non-finite state)")

    fell = t_up[-1] < 0.7
    print(f"  end:   trunk {t_h[-1]*100:.1f} cm, upright {t_up[-1]:.3f}, "
          f"max|qvel| {t_v[-1]:.4f}")
    print(f"  min trunk height over the run: {t_h.min()*100:.1f} cm  "
          f"(floor is 0, so a negative value means it sank through)")
    print(f"  settled: " + (f"{settled_at:.2f} s" if settled_at is not None
                            else f"never within {a.seconds:.1f} s"))
    print(f"  contacts at rest: {int(np.count_nonzero(d.contact.dist < 0)) if d.ncon else 0}"
          f" of {d.ncon}, {len(touch)} geom(s) may touch the floor")
    print(f"  verdict: {'FELL OVER' if fell else 'still upright'}"
          f"  (upright cosine {t_up[-1]:.3f}, threshold 0.70)"
          + (f", crossed the 0.70 threshold at {fell_at:.2f} s" if fell_at else ""))

    if a.mode == "hold" and not fell:
        sag = (t_h[0] - t_h[-1]) * 1000
        print(f"  sag under gravity while holding STAND: {sag:+.1f} mm")

    if frames:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        strip = np.hstack(frames)
        try:
            import cv2
            cv2.imwrite(out, strip[:, :, ::-1])
        except ImportError:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.imsave(out, strip)
        print(f"  wrote {out}  (6 frames, t = 0 .. {a.seconds:.1f} s)")

    return 1 if not np.all(np.isfinite(t_h)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
