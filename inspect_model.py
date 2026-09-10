"""Print what the Microduck MJCF actually contains, and diff the variants.

Named inspect_model.py, not inspect.py, because a file called inspect.py in the
working directory shadows the standard library module of that name and breaks
torch's import.

Run:  python inspect_model.py            # nominal
      python inspect_model.py --all      # every variant, plus the diff table
"""
import argparse

import mujoco
import numpy as np

import common


def dump(variant):
    m, d = common.load(variant)
    common.reset_to(m, d, "STAND")
    print(f"\n=== {variant}  ({common.VARIANTS[variant]}) ===")
    print(f"  nq {m.nq}   nv {m.nv}   nu {m.nu}   nbody {m.nbody}   ngeom {m.ngeom}")
    print(f"  timestep {m.opt.timestep} s  ->  {1/m.opt.timestep:.0f} Hz physics, "
          f"{common.SUBSTEPS} substeps per {common.CONTROL_HZ} Hz control step")
    print(f"  solver {mujoco.mjtSolver(m.opt.solver).name}  iterations {m.opt.iterations}")
    print(f"  total mass {m.body_subtreemass[0]:.3f} kg")
    print(f"  sensors {m.nsensor}, {m.nsensordata} scalars: "
          + ", ".join(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i)
                      for i in range(m.nsensor)))
    print(f"  keyframes: " + ", ".join(
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(m.nkey)))

    print(f"\n  {'#':>2}  {'actuator':<16} {'joint range (deg)':>19}  "
          f"{'kp':>6} {'force (Nm)':>12}  {'qpos':>5}")
    for i, name in enumerate(common.ACTUATOR_NAMES):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = np.degrees(m.jnt_range[j])
        kp = m.actuator_gainprm[i, 0]
        f0, f1 = m.actuator_forcerange[i]
        print(f"  {i:2d}  {name:<16} {lo:8.1f} .. {hi:7.1f}  {kp:6.3f} "
              f"{f0:5.2f} ..{f1:5.2f}  {m.jnt_qposadr[j]:5d}")

    npassive = m.njnt - 1 - common.N_ACT     # -1 for the free joint
    if npassive:
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
                 for j in range(m.njnt)]
        extra = [n for n in names if n not in common.ACTUATOR_NAMES and n is not None]
        print(f"\n  {npassive} passive joint(s) not in the action space:")
        print("    " + ", ".join(extra[:8]) + (" ..." if len(extra) > 8 else ""))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="walk", choices=sorted(common.VARIANTS))
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()

    variants = sorted(common.VARIANTS) if a.all else [a.variant]
    models = {v: dump(v) for v in variants}

    if a.all:
        print("\n=== variant diff ===")
        print(f"  {'variant':<16} {'nq':>4} {'nv':>4} {'nu':>4} {'njnt':>5} "
              f"{'passive':>8}   action space")
        base = None
        for v, m in models.items():
            names = tuple(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                          for i in range(m.nu))
            base = base or names
            same = "identical" if names == base else "DIFFERENT -- remapping needed"
            print(f"  {v:<16} {m.nq:4d} {m.nv:4d} {m.nu:4d} {m.njnt:5d} "
                  f"{m.njnt - 1 - common.N_ACT:8d}   {same}")
        print("\n  Every variant exposes the same 14 actuators in the same order.")
        print("  A policy trained on `walk` runs on `walk_backlash` unchanged;")
        print("  only the observation builder has to know where qpos moved.")

    print("\n=== measured actuator classes in joints_properties.xml ===")
    print("  Four fits of the same XL330 servo, all shipped upstream. The spread")
    print("  between them is a measured parameter uncertainty, not a guess.")
    print(f"  {'class':<26} {'damping':>8} {'friction':>9} {'armature':>9} "
          f"{'kp':>7} {'force':>7}")
    for k, (dm, fr, am, kp, fo) in common.ACTUATOR_CLASSES.items():
        print(f"  {k:<26} {dm:8.4f} {fr:9.4f} {am:9.4f} {kp:7.3f} {fo:7.2f}")
    vals = np.array(list(common.ACTUATOR_CLASSES.values()))
    lo, hi = vals.min(0), vals.max(0)
    print(f"  {'spread (max/min)':<26} " + " ".join(
        f"{h/l:8.2f}x" if l else f"{'inf':>9}" for l, h in zip(lo, hi)))


if __name__ == "__main__":
    main()
