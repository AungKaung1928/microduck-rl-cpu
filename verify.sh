#!/usr/bin/env bash
# Reproduce the README's claims from a clean checkout.
#
# Tiered. Tiers 1-4 are cheap, so a reader can check every structural claim --
# action space, dof arithmetic, contact masks, the observation contract,
# determinism -- in a couple of minutes without loading the box. Tier 2 briefly
# forks three worker processes; that is the heaviest of them. Tier 5 is the CPU
# benchmark and is the only part that needs the machine to itself.
set -u
cd "$(dirname "$0")"

# Python comes from whatever environment is active. A venv is expected but not
# required; the only hard dependencies are mujoco and numpy.
PY="${PYTHON:-python3}"
if ! "$PY" -c 'import mujoco, numpy' 2>/dev/null; then
  echo "mujoco/numpy are not importable with '$PY'. From the repo root:" >&2
  echo "    python3 -m venv .venv && . .venv/bin/activate" >&2
  echo "    pip install -r requirements.txt" >&2
  exit 1
fi

hr() { printf '\n=== %s ===\n' "$1"; }

if [ ! -f assets/scene_walk.xml ]; then
  cat <<'MSG'
assets/ is missing. It is gitignored on purpose: the meshes are CC BY-SA-NC.

    ./fetch_assets.sh

pulls them from a pinned upstream commit (~24 MB, one shallow clone).
MSG
  exit 1
fi

hr "1/5  model and repo assumptions -- hand-derived checks"
# Three silent failure modes live here: a reordered action space, a passive
# joint shifting qpos, and a contact bitmask that drops the robot through the
# floor. None of them raise.
"$PY" test_model.py || exit 1

hr "2/5  environment contract -- 48-dim observation, actions, pushes, vector env"
# The quiet failures here are a leaked sim-only observation, an action that
# saturates against a joint stop, and an index that reads a passive backlash
# joint on the evaluation model. None of them raise. Forks 3 workers briefly.
"$PY" test_env.py || exit 1

hr "3/5  what the MJCF contains"
"$PY" inspect_model.py --all || exit 1

hr "4/5  drop tests -- does the physics behave, does the shipped PD hold"
"$PY" drop_test.py --mode limp --z0 0.25 --seconds 5 || exit 1
"$PY" drop_test.py --mode hold --seconds 5 || exit 1
"$PY" drop_test.py --mode hold --variant walk --seconds 5 --no-render || exit 1

hr "5/5  CPU throughput -- the feasibility gate"
cat <<'MSG'
This one loads the machine: up to 8 processes for a few minutes. Close other
work first. WSL cannot read CPU temperature, so bench.py brackets every
configuration with a single-process reference and refuses to call the numbers
usable if that reference drifts more than 10%.

Short version (about 90 s):
    nice -n 10 python bench.py --seconds 12 --ref-seconds 6 --tag quick

The README's table (about 4 min, box otherwise idle):
    nice -n 10 python bench.py --seconds 20 --ref-seconds 10 --tag main
MSG
if [ -f runs/bench_main.json ]; then
  "$PY" - <<'PY'
import json
b = json.load(open("runs/bench_main.json"))
print(f"\nlast recorded run: gate {'PASS' if b['pass'] else 'FAIL'}, "
      f"stable {b['stable']}, worst reference drift {100*b['worst_ref_drift']:+.1f}%")
for r in b["rows"]:
    print(f"  {r['procs']} proc  {r['env_steps_per_s']:>10,.0f} env-steps/s  "
          f"{100*r['efficiency']:3.0f}% efficient")
PY
fi
