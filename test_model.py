"""Checks on the Microduck model and on this repo's assumptions about it.

Every expected value is derived from the MJCF or from arithmetic stated in the
check itself, not captured from a previous run. A test that records whatever
the code printed last time proves only that the code is deterministic.

The point of these is that all three of the failure modes below are silent:
an upstream actuator reordering permutes every action a policy emits, a
passive joint inserted into qpos shifts every joint angle a policy reads, and
a contact bitmask means the robot sinks through the floor with no error.

Run:  python test_model.py
"""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import common

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


print("\n--- action space is identical across variants ---")
# A policy is trained on one variant and evaluated on another. If the actuator
# list differs in content or order, action i means a different joint and the
# evaluation is meaningless while still producing plausible-looking numbers.
for v in sorted(common.VARIANTS):
    m, _ = common.load(v)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    check(f"{v}: 14 actuators, expected names in expected order",
          m.nu == common.N_ACT and names == common.ACTUATOR_NAMES)

print("\n--- degrees of freedom add up ---")
# One free joint (7 qpos, 6 qvel) plus one hinge per remaining joint. If this
# fails, some joint is not a hinge and the arithmetic elsewhere is wrong.
for v in sorted(common.VARIANTS):
    m, _ = common.load(v)
    hinges = m.njnt - 1
    check(f"{v}: nq == 7 + {hinges} hinges", m.nq == common.FREE_NQ + hinges,
          f"nq {m.nq}")
    check(f"{v}: nv == 6 + {hinges} hinges", m.nv == common.FREE_NV + hinges,
          f"nv {m.nv}")

print("\n--- passive joints shift qpos, so indices must be looked up ---")
m_w, _ = common.load("walk")
m_b, _ = common.load("walk_backlash")
iw = common.actuated_qpos_index(m_w)
ib = common.actuated_qpos_index(m_b)
check("walk: actuated qpos is the contiguous slice 7..20",
      np.array_equal(iw, np.arange(7, 21)), str(iw[:4]) + " ...")
check("backlash: actuated qpos is strided by 2, NOT 7..20",
      np.array_equal(ib, np.arange(7, 35, 2)) and not np.array_equal(ib, iw),
      str(ib[:4]) + " ...")
check("backlash inserts exactly 14 passive joints",
      m_b.njnt - m_w.njnt == 14, f"{m_w.njnt} -> {m_b.njnt}")

print("\n--- the STAND keyframe is self-consistent ---")
# These are position actuators: ctrl is a target angle. If the keyframe's ctrl
# did not equal its own qpos at those joints, resetting to STAND would command
# the robot away from the pose it was just placed in.
for v in ["walk", "groundcontact", "walk_backlash"]:
    m, d = common.load(v)
    common.reset_to(m, d, "STAND")
    q = d.qpos[common.actuated_qpos_index(m)]
    check(f"{v}: STAND ctrl == STAND qpos at the 14 actuated joints",
          np.allclose(d.ctrl, q, atol=1e-12),
          f"max diff {np.abs(d.ctrl - q).max():.2e}")

print("\n--- control rate ---")
m, _ = common.load("walk")
check("500 Hz physics divides into 50 Hz control with no remainder",
      abs(1.0 / (common.CONTROL_HZ * m.opt.timestep) - common.SUBSTEPS) < 1e-12,
      f"timestep {m.opt.timestep}, {common.SUBSTEPS} substeps")
check("PHYSICS_DT constant matches the MJCF", m.opt.timestep == common.PHYSICS_DT)

print("\n--- what can touch the floor ---")
m, _ = common.load("walk")
check("walk: exactly the two feet collide with the ground",
      sorted(common.floor_contact_geoms(m)) ==
      ["left_foot_collision", "right_foot_collision"])
m, _ = common.load("groundcontact")
check("groundcontact: more than the feet collide with the ground",
      len(common.floor_contact_geoms(m)) > 2,
      f"{len(common.floor_contact_geoms(m))} geoms")

# A toppled robot must come to rest on `groundcontact` and must NOT on `walk`.
# This is the concrete consequence of the bitmasks above.
for v, should_rest in [("groundcontact", True), ("walk", False)]:
    m, d = common.load(v)
    common.reset_to(m, d, "STAND")
    hold = d.ctrl.copy()
    for _ in range(2500):          # 5 s
        d.ctrl[:] = hold
        mujoco.mj_step(m, d)
    z = common.trunk_height(m, d)
    check(f"{v}: after toppling, trunk rests {'above' if should_rest else 'below'} the floor",
          (z > 0) == should_rest, f"trunk z {z*100:+.1f} cm")

print("\n--- limp mode really produces zero actuator force ---")
import drop_test
m, d = common.load("walk")
drop_test.make_limp(m)
common.reset_to(m, d, "STAND")
d.ctrl[:] = 5.0                    # a large command that must do nothing
mujoco.mj_step(m, d)
check("zeroed gain and bias give zero actuator force",
      np.abs(d.actuator_force).max() == 0.0,
      f"max |force| {np.abs(d.actuator_force).max():.2e}")

print("\n--- simulation is deterministic ---")
def roll(seed, n=500):
    m, d = common.load("walk")
    common.reset_to(m, d, "STAND")
    rng = np.random.default_rng(seed)
    nom = d.ctrl.copy()
    for _ in range(n):
        d.ctrl[:] = nom + rng.normal(0, 0.05, common.N_ACT)
        mujoco.mj_step(m, d)
    return d.qpos.copy()

check("same seed, bitwise-identical final state",
      np.array_equal(roll(7), roll(7)))
check("different seed, different final state", not np.array_equal(roll(7), roll(8)))

print("\n--- physical plausibility ---")
m, _ = common.load("walk")
check("total mass is 0.70-0.80 kg (spec says ~800 g)",
      0.70 <= m.body_subtreemass[0] <= 0.80, f"{m.body_subtreemass[0]:.3f} kg")
check("all 14 actuators share ctrlrange +-10",
      np.allclose(m.actuator_ctrlrange, np.tile([-10.0, 10.0], (14, 1))))
check("all 14 actuators share forcerange +-0.96 Nm",
      np.allclose(m.actuator_forcerange, np.tile([-0.96, 0.96], (14, 1))))

print("\n--- the hardcoded actuator table matches joints_properties.xml ---")
# common.ACTUATOR_CLASSES is transcribed by hand and used to set the
# domain-randomisation range later. If upstream refits a servo, this catches it.
root = ET.parse(f"{common.ASSETS}/joints_properties.xml").getroot()
for cls, (dm, fr, am, kp, fo) in common.ACTUATOR_CLASSES.items():
    node = root.find(f"./default[@class='{cls}']")
    if node is None:
        check(f"{cls} present in the MJCF", False)
        continue
    j, p = node.find("joint"), node.find("position")
    got = (float(j.get("damping")), float(j.get("frictionloss")),
           float(j.get("armature")), float(p.get("kp")),
           abs(float(p.get("forcerange").split()[1])))
    check(f"{cls} matches the MJCF", np.allclose(got, (dm, fr, am, kp, fo)),
          "" if np.allclose(got, (dm, fr, am, kp, fo)) else f"xml {got}")

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: " + ", ".join(FAILED))
    raise SystemExit(1)
print("all checks passed")
