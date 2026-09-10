# microduck-rl-cpu — robot learning on a laptop with no GPU

Project 3 of a CPU-only ML track. The target is a balance-and-recover policy for
the [Microduck](https://github.com/pollen-robotics/microduck_rl) — a 25 cm,
737 g open-source biped with 14 position-controlled servos — trained in MuJoCo,
evaluated on physics it never trained on, and exported to ONNX.

The reason this project exists in this form: upstream trains the duck with
mjlab on MuJoCo Warp, which requires CUDA. There is no GPU on this machine and
no hardware to buy. So the whole thing runs the same MJCF in plain CPU MuJoCo,
parallel across processes, inside an 8-of-14-thread budget on a laptop that
has other work to do.

**Steps 1 and 2 are done.** Step 1 is the feasibility gate: is this machine
fast enough to train a policy at all. Step 2 is the environment contract — what
the policy sees, what it emits, when an episode ends — and the baseline it has
to beat. Steps 3-5 are not started.

Step 1's throughput numbers were re-measured on 2026-09-10 and came back 40%
higher across 8 processes. The original table had been taken on a machine in a
reduced-power state that is not visible from inside WSL. Both tables are kept
below, because the difference between them is the useful part.

---

## Step 1 — can this box simulate the duck fast enough?

One **env step** is one 50 Hz control decision, which is 10 physics steps at the
model's 500 Hz timestep. RL budgets are quoted in env steps, so the gate is
written in them. The gate was set before measuring: **below 5,000 env-steps/s at
8 processes, walking leaves the scope and the project becomes stand-only.**

MuJoCo's step for a 16-body model is single-threaded and small enough to stay in
cache, so parallelism has to come from processes. Every worker runs with
`OMP_NUM_THREADS=1`, set before MuJoCo loads, or the workers spawn thread pools
and fight each other.

### Scaling, `walk` variant, 20 s per configuration

| processes | env-steps/s | per process | speedup | efficiency |
|---|---|---|---|---|
| 1 | 6,521 | 6,521 | 1.00x | 100% |
| 2 | 12,555 | 6,278 | 1.93x | 96% |
| 4 | 21,424 | 5,356 | 3.29x | 82% |
| 8 | 28,749 | 3,594 | 4.41x | 55% |

**Gate: PASS, 5.7x over.** The single-process reference window between
configurations held to within 6% across the whole sweep, so the four rows were
measured on the same machine as each other. `runs/bench_main.json`.

The sweep was run twice, seven minutes apart, and the 8-process row came back
28,960 and 28,749 — 0.7% apart. `runs/bench_main2.json` is the first of the
two, kept because a single benchmark is an anecdote. Note that this is much
tighter than the ±10% run-to-run spread quoted further down, which was measured
on the machine in its degraded state; on a healthy box the sweep is repeatable
to about a percent.

### These are the second set of numbers, and the first set is the interesting one

Step 1 originally certified this:

| processes | env-steps/s | efficiency | vs the table above |
|---|---|---|---|
| 1 | 6,666 | 100% | +2% |
| 2 | 9,801 | 74% | −22% |
| 4 | 14,275 | 54% | −33% |
| 8 | 20,474 | 38% | −29% |

Same code, same variant, same 8-of-14 thread budget, same laptop on mains
power. The difference is a host state the guest cannot read.

Read the shape rather than the totals, because the shape is the whole
diagnosis: **single-process is flat within run-to-run spread, and every
multi-process row is roughly a third lower.** Thermal throttling decays with
time under load and would have moved the reference windows within the sweep.
Another process competing for CPU depresses every row, the single-process one
included. A sustained all-core power limit does exactly this and nothing else —
single-core turbo never depended on the all-core budget, every additional core
did.

The tell that should have caught it at the time is the 2-process row. **Two
processes on a fourteen-core machine returned 74% efficiency.** There is no
core-count explanation for that; two workers cannot contend for cores when
twelve are idle. I read 74% as a property of the chip and built a scaling story
on top of it, and it was a property of the machine's power state that
afternoon.

There was a deeper excursion in the same window. While step 2 was being built
the single-process rate fell to **3,069 env-steps/s**, 46% of certified, with
the box idle. `bench.py`'s reference bracket called that run stable, and it was
right to: the tool detects a reference that *moves during a sweep*, and this was
a machine that was uniformly slow for the whole sweep. Worth naming as a
limitation of the method — **a drift guard cannot catch a bias that is already
in place when the first window opens.** The only defence is an absolute
expectation, which is what the certified table now provides.

Both recovered after a check on the Windows side. **Which setting changed is
not recorded, and that is the finding rather than a hole in it:** a WSL2 guest
cannot read power mode, charger wattage, package power or core frequency, so
host state is unrecoverable after the fact and has to be written down at
measurement time or lost. The guest-visible substitute is the single-thread
versus all-core shape above.

**None of this is a reason to push the machine harder.** The thread budget
stays at 8 of 14.

### What a training run gets is not 28,749

Two separate reasons, and both cut the number.

**First, the sweep pauses and a training run does not.** In the reduced-power
state, holding 8 processes flat out for four minutes gave:

| elapsed | env-steps/s | vs first window |
|---|---|---|
| 20 s | 27,905 | — |
| 40 s | 20,076 | −28% |
| 60 s | 20,383 | −27% |
| 120 s | 20,451 | −27% |
| 180 s | 20,051 | −28% |
| 240 s | 20,132 | −28% |

Note what that first window is: **27,905, within 3% of today's 8-process sweep
row.** So a 20-second window sits inside a turbo budget the machine then spends.
That test has not been repeated since the recovery, so whether the decay is
still there is open.

**Second, `bench.py` does not measure the thing that gets trained.** Its worker
is a bare `mj_step` loop on `walk`. The task runs the Python environment
wrapper on `groundcontact`, and both of those cost:

| single process, 12 s windows | bare `mj_step` | through `MicroduckEnv` | wrapper cost |
|---|---|---|---|
| `walk` | 7,414 | 6,028 | 23% |
| `groundcontact` | 5,091 | 4,505 | **13%** |

`runs/bench_wrapper.json`, reproduced by `bench_wrapper.py`. `groundcontact`
enables ten ground-collidable geoms against `walk`'s two, so it is 31% slower
before any Python runs.

This corrects an earlier claim in this README that the wrapper cost **0%**,
measured at 2,771 against 2,776 env-steps/s. That measurement was taken during
the reduced-power window, where the physics was slow enough to hide a fixed
Python cost underneath it. On a healthy box the wrapper is visible. 13% is
still not where optimisation effort should go — the physics is 87% of the step
— but 0% was wrong, and it was wrong in the flattering direction.

So the rate a training budget should use is the measured single-process wrapped
rate on `groundcontact` times the measured 8-process speedup:
**4,505 × 4.41 = ~19,900 env-steps/s.** The assumption inside that product is
that a scaling factor measured on bare `walk` physics carries to wrapped
`groundcontact`; it is a projection, not a measurement, and it is flagged as
one every time it is used below.

| budget | at ~19,900 (projected) | if the −28% decay also returns (~14,300) |
|---|---|---|
| 10M (hyperparameter probe) | 8 min | 12 min |
| 50M (stand + push recovery, expected) | 42 min | 58 min |
| 100M (stand, generous) | 1.4 h | 1.9 h |
| 400M (walking gait, upstream-scale) | 5.6 h → 3 chunks | 7.8 h → 4 chunks |

**The scope conclusion is the same at either end of that range**, which is what
makes the range tolerable: the expected stand-and-recover run is a single
sitting, and walking is three or four chunked sessions rather than out of
reach. Two commands close the range, and neither has been run since the
recovery:

```bash
nice -n 10 python3 bench.py --sustained 8 --seconds 20 --windows 12 --tag sustained
nice -n 10 python3 bench.py --variant groundcontact --seconds 20 --ref-seconds 10 --tag gc
```

### Why efficiency still falls to 55% at 8 processes, and what I could not determine

96% at two processes and 82% at four are unremarkable. 55% at eight is not.
This chip is an Intel Core Ultra 5 225H: 14 cores, no hyperthreading, and
heterogeneous — a handful of performance cores alongside efficiency cores. The
obvious hypothesis is that workers 5-8 land on slower cores.

I tried to test it by pinning four workers to CPUs 0-3 and then to CPUs 10-13:

| pinned to | per process | 1-process reference |
|---|---|---|
| CPUs 0-3 | 3,883 | 6,129 |
| CPUs 10-13 | 3,379 | 5,769 |

A 15% difference under load, 6% single-threaded. Far too small for a
performance-core versus efficiency-core split, which would show roughly 2x.

The explanation is that **CPU affinity inside WSL2 does not pin to a physical
core.** `sched_setaffinity` binds the process to a virtual CPU, and the
hypervisor schedules virtual CPUs onto physical cores on its own. So this
experiment cannot answer the question, and neither can any other experiment run
from inside the guest. The remaining candidates — shared 18 MB L3, memory
bandwidth, hypervisor scheduling, sustained power — are not separable from here.

Those two rows predate the recovery described above and have not been repeated;
their single-process references, 6,129 and 5,769, sit between the degraded and
recovered rates, so the machine was somewhere in the middle when they were
taken. The conclusion they support is architectural rather than numeric, so it
survives, but the numbers themselves should not be quoted.

I am recording this as unresolved rather than picking the plausible-sounding
answer. What matters operationally is settled anyway: **8 processes is the
right choice**, because it delivers the most total throughput even at 55%
efficiency.

---

## Three things about the model that changed the plan

### 1. On the `walk` model, only the feet can touch the ground

MuJoCo lets two geoms collide only if one's `contype` shares a bit with the
other's `conaffinity`. In `robot_walk.xml` the floor is 1/1, the two foot geoms
are 1/1, and **every other body and limb geom is 2/2**. Two and one share no
bits. So the trunk, head and legs cannot touch the floor at all.

Hold the STAND pose and let it topple:

| variant | geoms that can touch the floor | trunk at rest |
|---|---|---|
| `walk` | 2 (the feet) | **−10.5 cm — through the floor** |
| `groundcontact` | 10 | +3.2 cm — resting on it |

This is deliberate upstream. A walking task terminates the episode the instant
the robot falls, so what happens afterwards never has to be physical, and
dropping the collision geometry makes the sim faster. It makes `walk` the wrong
model for anything that involves lying on the ground or getting back up.

The failure is silent. Nothing raises, no contact is reported, the robot just
sinks. A reward function reading trunk height would have been quietly rewarded
for falling through the world.

**Decision: the stand-and-recover task in step 2 uses `groundcontact`, not
`walk`.** Recovery requires the robot to actually rest on the floor.

### 2. The backlash variant renumbers qpos, and that is the variant policies get evaluated on

Four variants ship, all exposing the **same 14 actuators under the same names in
the same order**:

| variant | nq | nv | passive joints | action space |
|---|---|---|---|---|
| `walk` | 21 | 20 | 0 | identical |
| `groundcontact` | 21 | 20 | 0 | identical |
| `rollers` | 25 | 24 | 4 | identical |
| `walk_backlash` | 35 | 34 | 14 | identical |

That the action space is identical is what makes the sim-to-sim step possible: a
policy trained on one variant runs on another with no remapping. The held-out
physics is supplied by the robot's own authors rather than invented here, which
is a much stronger position than randomising parameters and testing on the same
randomisation.

But `walk_backlash` inserts a passive backlash joint in series with **every**
actuated joint, and those passive joints interleave. The actuated joint angles
sit at qpos indices 7, 9, 11 … 33, not the contiguous 7…20 they occupy on
`walk`. Any observation builder that hardcodes `qpos[7:21]` reads a mixture of
joint angles and backlash deflections on the evaluation model — plausible
numbers, wrong meaning, no error. Everything here goes through
`common.actuated_qpos_index()` instead, and a test asserts the stride.

### 3. Joint friction is the parameter the hardware people disagree about by 6.7x

`joints_properties.xml` ships four fitted actuator classes for the same XL330
servo — different people, different benches, all left in the file:

| class | damping | frictionloss | armature | kp | force limit |
|---|---|---|---|---|---|
| `chosen_actuator` | 0.053 | 0.0048 | 0.0018 | 0.550 | 0.96 |
| `chosen_actuator_old` | 0.048 | 0.0060 | 0.0020 | 0.520 | 0.91 |
| `chosen_actuator_new` | 0.041 | 0.0320 | 0.0020 | 0.386 | 0.67 |
| `chosen_actuator_antoine` | 0.044 | 0.0130 | 0.0017 | 0.430 | 0.75 |
| **spread (max/min)** | **1.29x** | **6.67x** | **1.18x** | **1.42x** | **1.43x** |

Damping, armature, gain and torque limit agree to within about 40%. Friction
loss disagrees by a factor of seven.

This is a measured parameter uncertainty, not a guess, and it is the honest
place to get a domain-randomisation range from in step 4 — rather than the usual
±20% around a nominal, chosen because it sounds reasonable. It also says where
to spend the randomisation budget: wide on friction, narrow on everything else.
A `test_model.py` check parses the XML and fails if upstream refits a servo.

---

## Do the physics behave?

Two drop tests, both on `groundcontact`, 5 s each.

**Limp** — actuator gains and biases zeroed so the robot is a ragdoll, dropped
from 25 cm. Note that commanding `ctrl = 0` does **not** do this: a MuJoCo
`position` actuator produces `gain·ctrl + bias₁·qpos + bias₂·qvel`, so zero ctrl
is a stiff hold at the zero pose. All three coefficients have to go.

It falls, lands, and comes to rest at trunk height 3.0 cm with 14 contacts and
max |qvel| 0.09 rad/s. No sinking, no jitter, no explosion.

![limp drop](out/drop_groundcontact_limp.png)

**Hold** — actuators commanded to the STAND keyframe, which is where every
training episode will begin.

It holds for **0.79 s**, then topples, and settles face-down at 4.30 s.

![hold drop](out/drop_groundcontact_hold.png)

That 0.79 s is the baseline number for step 3. The shipped PD controller at
kp = 0.55 does not hold the pose, so STAND is an unstable equilibrium and
standing is a real balancing problem rather than a pose hold. It also means the
PD baseline the learned policy has to beat is well defined and easy to state:
*falls over in 0.79 s.*

The same test on `walk` topples at the identical 0.79 s — the dynamics match
until the body reaches the floor — and then sinks to −10.5 cm, which is finding
1 again, visible as a number.

---

## Step 2 — the environment contract

### The observation is 48-dim. Step 1 said 61, and 61 was wrong

61 is everything the *simulator* knows about the robot. 48 is everything the
*robot* knows about itself. The difference is thirteen numbers that are free in
MuJoCo and do not exist on hardware:

| dropped | dims | why the real duck cannot produce it |
|---|---|---|
| `imu_lin_vel` | 3 | a velocimeter. There is no state estimator on the robot, so base linear velocity is not measurable |
| trunk position | 3 | world-frame xyz — same problem, and no external tracking rig in the loop |
| `orientation` (full quat) | 4 | yaw is not observable from a gyro and an accelerometer. `sensors.xml` declares no magnetometer, so heading can only be integrated, and drifts |
| `root_angmom` | 3 | subtree angular momentum: a MuJoCo computation over the body tree, not a sensor |

A policy that reads those learns to depend on them and then has nothing to run
on. Since the only claim this project can honestly make is about transfer, the
observation is cut down to sensors the robot carries *before* any training
happens. It costs nothing now and cannot be retrofitted later.

What is left, and where each part comes from on hardware:

| block | dims | source on the real robot |
|---|---|---|
| joint position | 14 | servo present-position |
| joint velocity | 14 | servo present-velocity — real, and noisy. Included because the servos report it, not because it is clean |
| previous action | 14 | the policy's own last output; it is in RAM |
| gyro | 3 | IMU rate gyro |
| projected gravity | 3 | world −Z in the trunk frame — the observable part of attitude, what an IMU with a complementary filter gives you |

The gyro is read from the `angular-velocity` sensor rather than its twin
`imu_ang_vel`. Both sit on the same site and read identically today, because
MuJoCo applies a sensor's declared `noise` only when `mjENBL_SENSORNOISE` is
set and it is not set here. But `angular-velocity` is the one upstream put
`noise="0.005"` on. Step 4 flips the flag and gets the noise magnitude the
robot's authors chose, instead of one invented to look reasonable.

`test_env.py` rebuilds the observation from those five sources and asserts the
environment's output matches bit-for-bit. That is the only way to prove nothing
sim-only leaked in — and it checks separately that the excluded velocimeter is
reading a live non-zero signal, so the exclusion is a real one rather than a
channel that happens to be zero.

### Actions, and a trap in the MJCF

14 outputs in [−1, 1], applied as residuals around the STAND pose:

```
target = STAND_pose + 0.35 * action        then clipped to the joint limits
```

A saturated action moves a joint 0.35 rad, about 22% of the median joint range,
in one 20 ms decision. That number is a hyperparameter, not a derived quantity;
step 3 reports what happens at 0.2 and 0.5.

The clip is not decoration. **`ctrlrange` on all 14 actuators is [−10, 10] rad,
while the tightest joint limit — hip roll — is ±0.384 rad.** Handing a position
actuator a 10 rad target raises nothing: it saturates against the joint stop and
spends the entire force range holding there. Nothing in the model prevents this
and nothing reports it.

### Episodes and pushes

250 steps at 50 Hz, five seconds, **fixed length with no early termination.**
The usual locomotion setup ends the episode the moment the robot falls. That is
right for walking and wrong here — recovery is half the task, and terminating
on a fall makes falling unrecoverable by construction. It also keeps returns
comparable: every episode is the same length, so a baseline that topples early
gets a low return rather than a short episode that looks cheap.

Three pushes per episode, drawn from the episode seed: magnitude uniform in
0.15–0.45 m/s applied to the trunk's linear velocity, direction uniform in
azimuth, timing uniform but held half a second clear of both ends. A push at
t=0 is an initial condition rather than a disturbance, and one at the buzzer is
never recovered from inside the episode.

### Reward

Six terms, all logged separately in `info` so step 3 can show which one is
doing the work rather than reporting one scalar and calling it tuned.

| term | weight | shape |
|---|---|---|
| upright | +1.0 | trunk z-axis vs world up, floored at 0 |
| height | +1.0 | Gaussian on trunk height about 0.12 m, σ = 3 cm |
| posture | −0.10 | mean squared joint deviation from STAND |
| action rate | −0.05 | mean squared change in action |
| joint velocity | −2e−4 | mean squared joint velocity |
| effort | −0.02 | mean squared actuator force |

A perfectly held stand scores 2.0 per step, so **500 is the episode ceiling.**

### The baseline the policy has to beat

The shipped PD controller commanded to hold STAND — which is what `zero_action`
emits — over 20 seeds of the full task, pushes and initial-state noise included:

| | value |
|---|---|
| return | **108.9 ± 2.8** of a 500 ceiling |
| fraction of the episode on the floor | **54%** |
| starts to tilt (upright cos < 0.9) | 0.64 s |
| trunk reaches the floor | 2.54 s |

Those last two reconcile step 1's drop test with this one: 0.79 s was the
toppling threshold, 2.54 s is ground contact. Same event, measured at two
points on the way down.

### The vector env

`vec_env.py` forks N workers over pipes, one env each, capped at the 8-of-14
budget. Env *i* is seeded `seed + i`, and a test asserts that **N workers
reproduce N sequential envs bit-for-bit** — so a run is reproducible at any
worker count, and a result cannot quietly depend on how it was parallelised.

Measured single-process on `groundcontact`, the environment wrapper —
observation assembly, reward, push scheduling, episode bookkeeping — costs
**13%** over a bare `mj_step` loop doing the same substeps: 4,505 against
5,091 env-steps/s. The physics is the other 87%, which is the expected answer
for a 16-body model and worth having as a number rather than an assumption.
`bench_wrapper.py` measures both halves back to back in one process.

An earlier version of this README reported that cost as 0%, from 2,771 against
2,776. That was measured while the machine was in the reduced-power state
described in step 1, where the physics was slow enough to hide a fixed Python
cost underneath it. Two lessons, both cheap: a ratio is not automatically safe
from a systematic slowdown, and a number with no run file behind it does not
get to survive a re-measurement it was never given.

### What the tests catch that would otherwise be silent

The last check runs the observation builder on `walk_backlash`, the held-out
model, and compares the correct strided index against the obvious shortcut:

| | result |
|---|---|
| `qpos[7:21]` on `walk_backlash` | wrong on **13 of 14** joints, by up to **0.938 rad** |
| actuator names and order | identical to `groundcontact` — a policy transfers with no remapping |

Reading a mixture of joint angles and backlash deflections produces entirely
plausible numbers and no error. That is exactly the failure that would make a
sim-to-sim transfer result meaningless while looking fine.

---

## Scope, now that the gate has been measured

- **In:** stand and recover from pushes, on `groundcontact`, ~50M env steps.
  42 minutes at the projected training rate, 58 minutes if the sustained decay
  returns. One sitting either way.
- **In:** domain randomisation over the four measured actuator classes,
  evaluated on `walk_backlash` as held-out physics.
- **In, was previously deferred:** a walking gait, 400M steps. 5.6 h at the
  projected rate, 7.8 h at the pessimistic one — three or four chunks of ≤2 h
  rather than the 36 h that had put it out of reach. It moved back into scope
  because the machine was measured properly, not because the plan changed.
- **Out:** anything requiring mjlab, MuJoCo Warp, or a GPU.

## Steps

1. **Feasibility gate — done.** Asset fetch, model inspection, CPU throughput,
   drop tests. This README.
2. **Environment contract — done.** 48-dim observation, 14-dim action, 50 Hz,
   fixed-length episodes with seeded pushes, hand-written multiprocess vector
   env, 20 contract tests.
3. PPO against a PD hold-pose baseline, reusing the implementation from
   [ppo-from-scratch](https://github.com/AungKaung1928/ppo-from-scratch).
   Metric: recovery rate under randomised pushes, n=100, seeds reported.
4. Domain randomisation, evaluated on `walk_backlash`. Report the gap.
5. ONNX export, single-thread latency, verified against PyTorch two ways.

## What steps 1 and 2 do not prove

- Nothing here trains anything. Throughput is measured with random actions
  around STAND. A real PPO loop adds policy forward passes, advantage
  computation and optimiser steps on top, so the measured rate is an upper
  bound on the rollout half only, not on training.
- **The training-rate figure is a projection, not a measurement.** ~19,900
  env-steps/s is a single-process wrapped rate on `groundcontact` multiplied by
  a scaling factor measured on bare `walk` physics. Two commands at the end of
  step 1 replace it with a measurement; neither has been run.
- **Sustained behaviour has not been re-measured since the machine recovered.**
  The four-minute test that produced the −28% decay was run in the
  reduced-power state. Every wall-clock number here is given as a range for
  that reason.
- The 0.79 s fall time is one deterministic rollout from one keyframe. It is a
  baseline to beat, not a distribution. The 108.9 ± 2.8 return is the
  distributional version of it, over 20 seeds, and that is the number step 3
  should be compared against.
- The reward weights are chosen, not tuned. Nothing has optimised against them
  yet, so they are a starting point and the first thing to suspect if step 3
  learns something strange.
- The environment is not validated by a policy learning in it. Twenty contract
  tests say it does what it claims; they cannot say the task is learnable. That
  is what step 3 is for.
- No observation noise, no domain randomisation, no actuator variation. The
  environment runs one nominal physics model. Step 4 adds the spread.
- Earlier revisions of this README described every rate as measured against a
  background process taking about 9% of one core. That figure came from
  `ps -o pcpu`, which reports CPU time averaged over a process's entire
  lifetime — a long-lived interactive process that was busy an hour ago and is
  idle now still reads several percent there. `bench.py` now samples
  `/proc/<pid>/stat` twice a fraction of a second apart and reports the rate
  *now*, in percent of one core, warning only above 20% of a core. The
  certified sweep raised no warning under the corrected check.
- Run-to-run spread on the certified sweep is about 1% at 8 processes, from
  two runs seven minutes apart. That is not enough samples to call it a
  distribution, and it says nothing about spread across days, where the host
  power state is the dominant term and has already moved these numbers by 40%.
  Two significant figures is still all any of this supports.

## Reproducing

From a fresh clone. The only hard dependencies are `mujoco` and `numpy`.

```bash
git clone https://github.com/AungKaung1928/microduck-rl-cpu.git
cd microduck-rl-cpu
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

./fetch_assets.sh        # pinned upstream commit, ~24 MB, gitignored
./verify.sh              # tiers 1-4 are cheap; tier 5 loads the box
```

The benchmark is the only part that needs the machine to itself:

```bash
nice -n 10 python3 bench.py --seconds 20 --ref-seconds 10 --tag main
nice -n 10 python3 bench.py --sustained 8 --seconds 20 --windows 12 --tag sustained
nice -n 10 python3 bench_wrapper.py --seconds 12        # 1 core, no load
```

Use the interpreter the virtual environment above provides. A bare `python` is
not a command on every system, and `nice` reports a missing interpreter as
`No such file or directory`, which reads like a missing script. `verify.sh`
prints the resolved interpreter for this reason.

`bench.py` brackets every configuration with a single-process reference window
and refuses to certify the table if that reference drifts more than 10%. It also
refuses to run more than 8 processes without an explicit flag: the thread budget
on this machine is 8 of 14 cores, and every number in this README was measured
inside it.

Rendering — only the drop-test filmstrips need it — goes through whatever
`MUJOCO_GL` names. If it fails, set it to `glfw`, `egl` or `osmesa`, whichever
your machine actually has.

### Measured environment

Intel Core Ultra 5 225H, 14 cores, no hyperthreading, 21 GB available to WSL2.
Windows 11 + WSL2 (kernel 6.18.33.2), Ubuntu 22.04, Python 3.10, MuJoCo 3.12,
`MUJOCO_GL=glfw` through WSLg (llvmpipe software rendering — `egl` and `osmesa`
both fail on this box). No CUDA anywhere.

## Licence

Upstream code is Apache-2.0; the 3D model files are **Creative Commons
BY-SA-NC**. `assets/` is therefore gitignored and reproduced by
`fetch_assets.sh` from pinned commit `53b8971`. Code in this repository is my
own.
