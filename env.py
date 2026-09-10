"""The environment contract: what the policy sees, what it emits, when it ends.

Task: stand on `groundcontact`, and stay standing through pushes that arrive at
times the policy cannot predict. Fixed-length episodes, 50 Hz, 14 position
targets out, 48 numbers in.

WHY 48 AND NOT 61
-----------------
Step 1's README quoted a 61-dim observation. That was the wrong number, and the
reason it was wrong is the whole point of this file.

61 is everything the *simulator* can hand over about the robot's state. 48 is
everything the *robot* can measure about itself. The 13-dim difference is:

    imu_lin_vel     3   a velocimeter. The real duck has no state estimator,
                        so base linear velocity does not exist on hardware.
    trunk position  3   world-frame xyz. Same problem, plus there is no
                        external tracking system in the loop.
    orientation     4   the full quaternion. Yaw is not observable from a
                        gyro and an accelerometer alone -- no magnetometer is
                        declared in sensors.xml -- so absolute heading would
                        be a number the hardware can only integrate and drift.
    root_angmom     3   subtree angular momentum, a MuJoCo computation over
                        the whole body tree, not a sensor at all.

Every one of those is free in sim and impossible on the robot. A policy trained
on them learns to depend on them, and then has nothing to run on. Since the
entire claim this project can make is about transfer, the observation is
restricted to sensors the robot actually carries, up front, before any training
happens. This costs nothing now and cannot be retrofitted later.

What survives, and where it comes from on hardware:

    joint_pos    14   servo present-position, one per XL330
    joint_vel    14   servo present-velocity. Real and noisy; it is included
                      because the servos do report it, not because it is clean.
    prev_action  14   the policy's own last output. Free -- it is in RAM.
    gyro          3   IMU rate gyro, taken from the `angular-velocity` sensor
                      rather than its twin `imu_ang_vel`, because that is the
                      one upstream declared noise="0.005" on. Both read the
                      same site. MuJoCo only applies declared sensor noise when
                      mjENBL_SENSORNOISE is set, which it is not here -- so
                      step 4 turns the flag on and gets upstream's noise
                      magnitude instead of one invented for the occasion.
    proj_grav     3   world -Z expressed in the trunk frame: which way is down,
                      from the robot's point of view. This is the observable
                      part of attitude -- roll and pitch are gravity-referenced,
                      yaw is not -- and it is what every IMU with an onboard
                      complementary filter gives you. Derived here from the
                      `orientation` framequat so that the step-4 noise flag
                      perturbs it too.

WHAT ELSE IS DELIBERATE
-----------------------
Fixed-length episodes, no early termination on falling. The usual locomotion
setup ends the episode the moment the robot goes down, which is right when the
task is walking and wrong here: recovery is half the task, and terminating on a
fall makes falling unrecoverable by construction. It also keeps returns
comparable -- every episode is 250 steps, so the PD baseline that topples at
0.79 s has a well-defined return instead of a short episode that looks cheap.

Actions are residuals around the STAND pose, clipped to the joint limits. That
clip is not decoration: `ctrlrange` on all 14 actuators is [-10, 10] rad, while
the tightest joint range (hip roll) is +-0.384 rad. Handing a position actuator
a 10 rad target raises nothing -- it saturates against the joint stop and
spends the whole force range holding there.
"""
import mujoco
import numpy as np

import common

OBS_DIM = 48
ACT_DIM = common.N_ACT

# Layout is fixed and tested. Anything reading an observation slice imports
# these rather than writing the indices out again.
OBS_SLICES = {
    "joint_pos":   slice(0, 14),
    "joint_vel":   slice(14, 28),
    "prev_action": slice(28, 42),
    "gyro":        slice(42, 45),
    "proj_grav":   slice(45, 48),
}

# Per-group scaling, so no input dominates by unit choice alone. Joint residuals
# and projected gravity are already order 1. Joint velocity reaches ~20 rad/s in
# a fall and the gyro ~4 rad/s, so both are divided down to roughly unit range.
OBS_SCALE = np.concatenate([
    np.ones(14),          # joint_pos, rad, |.| <= ~1.5
    np.full(14, 0.05),    # joint_vel, rad/s
    np.ones(14),          # prev_action, already in [-1, 1]
    np.full(3, 0.25),     # gyro, rad/s
    np.ones(3),           # proj_grav, unit vector
]).astype(np.float32)

# A saturated action moves a joint 0.35 rad, about 22% of the median joint
# range, in one 20 ms decision. It is a hyperparameter, not a derived quantity;
# step 3 reports what happens at 0.2 and 0.5.
ACTION_SCALE = 0.35

CONTROL_DT = 1.0 / common.CONTROL_HZ
FALL_HEIGHT = 0.04     # m. Same threshold bench.py resets on.
STAND_HEIGHT = 0.12    # m. The STAND keyframe's trunk height, asserted at init.

# Per-step reward. Positive terms are bounded by 1 each, so a perfectly held
# stand scores 2.0 per step and 500 over an episode. The penalties are logged
# separately in info so step 3 can show which one is actually doing the work,
# rather than reporting a single scalar and calling it tuned.
REWARD_WEIGHTS = {
    "upright":     1.0,
    "height":      1.0,
    "posture":    -0.10,
    "action_rate": -0.05,
    "joint_vel":  -2.0e-4,
    "effort":     -0.02,
}


class MicroduckEnv:
    """One Microduck. Single process, no framework, no gym dependency.

    Deliberately not a gymnasium.Env. The only consumer is this repo's PPO, the
    surface is five methods, and a dependency whose API has changed three times
    is not worth taking on for a `.unwrapped` attribute.
    """

    def __init__(self, variant="groundcontact", seed=0, episode_steps=250,
                 action_scale=ACTION_SCALE, n_pushes=3,
                 push_speed=(0.15, 0.45), init_noise=0.02):
        self.variant = variant
        self.episode_steps = int(episode_steps)
        self.action_scale = float(action_scale)
        self.n_pushes = int(n_pushes)
        self.push_speed = tuple(push_speed)
        self.init_noise = float(init_noise)

        self.model, self.data = common.load(variant)

        # Index maps, resolved once. On walk_backlash these are strided, which
        # is the entire reason they exist -- see common.actuated_qpos_index.
        self._qpos_i = common.actuated_qpos_index(self.model)
        self._qvel_i = common.actuated_qvel_index(self.model)
        self._lo, self._hi = common.joint_limits(self.model)
        self._default = common.default_pose(self.model, "STAND")
        self._gyro = common.sensor_slice(self.model, "angular-velocity")
        self._quat = common.sensor_slice(self.model, "orientation")

        # The free joint must be joint 0 at qvel 0:6, or the push below lands
        # on some other body's velocity and nothing complains.
        if self.model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError("joint 0 is not the free joint; push code assumes it is")

        self.rng = np.random.default_rng(seed)
        self._prev_action = np.zeros(ACT_DIM)
        self._t = 0
        self._push_at = np.zeros(0, dtype=int)
        self._push_vel = np.zeros((0, 2))
        self._pushes_done = 0

        common.reset_to(self.model, self.data, "STAND")
        h = common.trunk_height(self.model, self.data)
        if abs(h - STAND_HEIGHT) > 1e-3:
            raise RuntimeError(f"STAND trunk height is {h:.4f}, expected {STAND_HEIGHT}")

    # -- observation ------------------------------------------------------

    def _proj_gravity(self):
        """World -Z in the trunk frame. (0, 0, -1) when upright."""
        q = np.array(self.data.sensordata[self._quat], dtype=float)
        mujoco.mju_normalize4(q)          # noise, when enabled, denormalises it
        qinv = np.empty(4)
        mujoco.mju_negQuat(qinv, q)       # conjugate: world -> site
        g = np.empty(3)
        mujoco.mju_rotVecQuat(g, np.array([0.0, 0.0, -1.0]), qinv)
        return g

    def observe(self):
        raw = np.concatenate([
            self.data.qpos[self._qpos_i] - self._default,
            self.data.qvel[self._qvel_i],
            self._prev_action,
            self.data.sensordata[self._gyro],
            self._proj_gravity(),
        ])
        return (raw * OBS_SCALE).astype(np.float32)

    # -- episode ----------------------------------------------------------

    def _schedule_pushes(self):
        """Draw this episode's kicks. Seeded, so an episode replays exactly.

        Kept away from the first and last half-second: a push at t=0 is a
        different initial condition rather than a disturbance, and one at the
        buzzer is never recovered from within the episode, so neither teaches
        anything about recovery.
        """
        margin = int(0.5 * common.CONTROL_HZ)
        lo, hi = margin, self.episode_steps - margin
        if self.n_pushes <= 0 or hi <= lo:
            self._push_at = np.zeros(0, dtype=int)
            self._push_vel = np.zeros((0, 2))
            return
        self._push_at = np.sort(self.rng.choice(np.arange(lo, hi),
                                                size=min(self.n_pushes, hi - lo),
                                                replace=False))
        speed = self.rng.uniform(*self.push_speed, size=self._push_at.size)
        theta = self.rng.uniform(0.0, 2 * np.pi, size=self._push_at.size)
        self._push_vel = np.stack([speed * np.cos(theta),
                                   speed * np.sin(theta)], axis=1)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        common.reset_to(self.model, self.data, "STAND")
        if self.init_noise > 0:
            n = self.init_noise
            self.data.qpos[self._qpos_i] += self.rng.uniform(-n, n, ACT_DIM)
            self.data.qvel[self._qvel_i] += self.rng.uniform(-5 * n, 5 * n, ACT_DIM)
            np.clip(self.data.qpos[self._qpos_i], self._lo, self._hi,
                    out=self.data.qpos[self._qpos_i])
            mujoco.mj_forward(self.model, self.data)
        self._prev_action = np.zeros(ACT_DIM)
        self._t = 0
        self._pushes_done = 0
        self._schedule_pushes()
        return self.observe()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float).reshape(ACT_DIM), -1.0, 1.0)
        target = np.clip(self._default + self.action_scale * action, self._lo, self._hi)
        self.data.ctrl[:] = target

        pushed = False
        if self._pushes_done < self._push_at.size and self._t == self._push_at[self._pushes_done]:
            self.data.qvel[0:2] += self._push_vel[self._pushes_done]
            self._pushes_done += 1
            pushed = True

        for _ in range(common.SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        terms = self._reward_terms(action)
        reward = float(sum(terms.values()))
        self._prev_action = action
        self._t += 1
        done = self._t >= self.episode_steps

        h = common.trunk_height(self.model, self.data)
        info = {"terms": terms, "trunk_height": h,
                "upright_cos": common.upright_cos(self.model, self.data),
                "fallen": h < FALL_HEIGHT, "pushed": pushed, "t": self._t}
        return self.observe(), reward, done, info

    def _reward_terms(self, action):
        w = REWARD_WEIGHTS
        q = self.data.qpos[self._qpos_i] - self._default
        dq = self.data.qvel[self._qvel_i]
        dh = common.trunk_height(self.model, self.data) - STAND_HEIGHT
        return {
            "upright":     w["upright"] * max(0.0, common.upright_cos(self.model, self.data)),
            "height":      w["height"] * float(np.exp(-(dh / 0.03) ** 2)),
            "posture":     w["posture"] * float(np.mean(q ** 2)),
            "action_rate": w["action_rate"] * float(np.mean((action - self._prev_action) ** 2)),
            "joint_vel":   w["joint_vel"] * float(np.mean(dq ** 2)),
            "effort":      w["effort"] * float(np.mean(self.data.actuator_force ** 2)),
        }

    # -- baselines --------------------------------------------------------

    def zero_action(self):
        """The PD hold-pose baseline: command the STAND pose and nothing else.

        This is the thing the learned policy has to beat, and step 1 already
        measured it -- it topples at 0.79 s.
        """
        return np.zeros(ACT_DIM)
