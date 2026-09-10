# microduck-rl-cpu — robot learning on a laptop with no GPU

Project 3 of a CPU-only ML track. The target is a balance-and-recover policy for
the [Microduck](https://github.com/pollen-robotics/microduck_rl) — a 25 cm,
737 g open-source biped with 14 position-controlled servos — trained in MuJoCo,
evaluated on physics it never trained on, and exported to ONNX.

The reason this project exists in this form: upstream trains the duck with
mjlab on MuJoCo Warp, which requires CUDA. There is no GPU on this machine and
no hardware to buy. So the whole thing runs the same MJCF in plain CPU MuJoCo,
parallel across processes, inside an 8-of-14-thread budget on a company laptop.

**Step 1 is done: the feasibility gate.** It answers one question — is this
machine fast enough to train a policy at all — and it turned up three things
about the model that changed the plan for step 2. Steps 2-5 are not started.

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
| 1 | 6,666 | 6,666 | 1.00x | 100% |
| 2 | 9,801 | 4,901 | 1.47x | 74% |
| 4 | 14,275 | 3,569 | 2.14x | 54% |
| 8 | 20,474 | 2,559 | 3.07x | 38% |

**Gate: PASS, 4.1x over.** Eight processes deliver about three times the
throughput of one, not eight. Scaling is poor and the reason is not visible from
inside WSL — see below.

### The number that actually matters is 28% lower than the peak

The sweep above pauses between configurations. A training run does not. Holding
8 processes flat out for four minutes:

| elapsed | env-steps/s | vs first window |
|---|---|---|
| 20 s | 27,905 | — |
| 40 s | 20,076 | −28% |
| 60 s | 20,383 | −27% |
| 120 s | 20,451 | −27% |
| 180 s | 20,051 | −28% |
| 240 s | 20,132 | −28% |

The box gives up 28% within the first 40 seconds and then holds **20,184
env-steps/s within ±3% for the remaining three and a half minutes**. That is a
short turbo budget followed by a flat sustained-power ceiling, and the flatness
is the useful part: the sustained rate is stable enough to plan a training run
against.

This resolves an ambiguity that has bitten this track before. WSL cannot read
CPU temperature — `/sys/class/thermal/` has no `thermal_zone*` and
`/proc/cpuinfo` reports a fixed base clock — so a throughput drop could be
thermal or it could be another process competing. Here they are separable by
shape: power limiting appears in the first 40 seconds and then holds flat,
while contention is erratic. Both were observed. During one sweep the system
file indexer woke up, the single-process reference moved 16%, and `bench.py`
refused to certify the numbers.

**Every training budget below uses 20,184 env-steps/s, not 27,905.**

| budget | wall clock | plan |
|---|---|---|
| 10M steps (hyperparameter probe) | 8 min | one sitting |
| 50M (stand + push recovery, expected) | 41 min | one sitting |
| 100M (stand, generous) | 1.4 h | one sitting |
| 400M (walking gait, upstream-scale) | 5.5 h | 3 chunks of ≤2 h |

So walking is not ruled out by compute. It is ruled out for now by the
no-overnight-runs rule, which makes it a three-session job rather than an
impossible one.

### Why the scaling is bad, and what I could not determine

Efficiency falls to 38% at 8 processes. This chip is an Intel Core Ultra 5 225H:
14 cores, no hyperthreading, and heterogeneous — a handful of performance cores
alongside efficiency cores. The obvious hypothesis is that workers 5-8 land on
slower cores.

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

I am recording this as unresolved rather than picking the plausible-sounding
answer. What matters operationally is settled anyway: **8 processes is the right
choice** because it delivers the most total throughput, even at 38% efficiency.

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

## Scope, now that the gate has been measured

- **In:** stand and recover from pushes, on `groundcontact`, ~50M env steps,
  about 41 minutes of training.
- **In:** domain randomisation over the four measured actuator classes,
  evaluated on `walk_backlash` as held-out physics.
- **Deferred, not blocked:** a walking gait. 400M steps is 5.5 h, which is three
  chunked sessions. Revisit if steps 2-4 land early.
- **Out:** anything requiring mjlab, MuJoCo Warp, or a GPU.

## Steps

1. **Feasibility gate — done.** Asset fetch, model inspection, CPU throughput,
   drop tests. This README.
2. Environment contract: 61-dim observation, 14-dim action, 50 Hz, hand-written
   multiprocess vector env, tests.
3. PPO against a PD hold-pose baseline, reusing the implementation from
   [ppo-from-scratch](https://github.com/AungKaung1928/ppo-from-scratch).
   Metric: recovery rate under randomised pushes, n=100, seeds reported.
4. Domain randomisation, evaluated on `walk_backlash`. Report the gap.
5. ONNX export, single-thread latency, verified against PyTorch two ways.

## What step 1 does not prove

- Nothing here trains anything. Throughput is measured with random actions
  around STAND. A real PPO loop adds policy forward passes, advantage
  computation and optimiser steps on top, so 20,184 env-steps/s is an upper
  bound on the rollout half only, not on training.
- The 0.79 s fall time is one deterministic rollout from one keyframe. It is a
  baseline to beat, not a distribution.
- Every rate here was measured with the Claude Code process consuming about 9%
  of one core. That is a systematic offset present in all rows equally, so the
  comparisons hold, but the absolute numbers are a few percent pessimistic.
- Run-to-run spread on the sweep is roughly ±10%. Two significant figures is all
  these numbers support.

## Reproducing

```bash
cd ~/personal/ml/microduck-rl-cpu
./fetch_assets.sh        # pinned upstream commit, ~24 MB, gitignored
./verify.sh              # tiers 1-3 are cheap; tier 4 loads the box
```

The benchmark is the only part that needs the machine to itself:

```bash
cd ~/personal/ml/microduck-rl-cpu
source ~/personal/ml/env.sh
nice -n 10 python bench.py --seconds 20 --ref-seconds 10 --tag main
nice -n 10 python bench.py --sustained 8 --seconds 20 --windows 12 --tag sustained
```

`bench.py` brackets every configuration with a single-process reference window
and refuses to certify the table if that reference drifts more than 10%. It also
refuses to run more than 8 processes without an explicit flag, because this is a
company laptop and the budget is 8 of 14 cores.

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
