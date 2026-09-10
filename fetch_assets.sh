#!/usr/bin/env bash
# Fetch the Microduck MJCF and meshes from the upstream repo into assets/.
#
# Nothing under assets/ is committed here. Two reasons:
#   - The 3D model files (86 STL, 23 MB) are Creative Commons BY-SA-NC.
#     This repo is a public portfolio; redistributing NC-licensed geometry
#     from it is not something to be casual about.
#   - Vendoring 23 MB of someone else's binary meshes into a learning repo
#     hides where the robot came from. A pinned fetch script does not.
#
# The commit is pinned so the physics this repo measures cannot drift under it.
set -euo pipefail
cd "$(dirname "$0")"

UPSTREAM="https://github.com/pollen-robotics/microduck_rl.git"
COMMIT="53b8971b61baf5b7f3c16d135dd7cac37623de4b"   # 2026-09-09
SUBDIR="src/mjlab_microduck/robot/microduck"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "cloning $UPSTREAM @ ${COMMIT:0:8}"
git clone --quiet --filter=blob:none --no-checkout "$UPSTREAM" "$TMP/src"
git -C "$TMP/src" -c advice.detachedHead=false checkout --quiet "$COMMIT"

rm -rf assets
mkdir -p assets
cp -r "$TMP/src/$SUBDIR/." assets/
cp "$TMP/src/LICENSE" assets/LICENSE.upstream
rm -rf assets/__pycache__

N_XML=$(find assets -maxdepth 1 -name '*.xml' | wc -l)
N_MESH=$(find assets/assets -name '*.stl' | wc -l)
echo "assets/: $N_XML xml, $N_MESH stl, $(du -sh assets | cut -f1)"

# Load-test the three scenes this project actually uses. A copied tree that
# does not compile is worse than no tree, and it fails silently until training.
source ~/personal/ml/env.sh
python - <<'PY'
import mujoco, sys
for s in ("scene_walk.xml", "scene_walk_backlash.xml", "scene_rollers.xml"):
    m = mujoco.MjModel.from_xml_path(f"assets/{s}")
    print(f"  ok  {s:26s} nq={m.nq:3d} nv={m.nv:3d} nu={m.nu:3d}")
PY
echo "done"
