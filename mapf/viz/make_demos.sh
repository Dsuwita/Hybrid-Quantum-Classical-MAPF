#!/usr/bin/env bash
# Regenerate the MAPF demo plans and GIFs shown in the README.
#
# Requires: the project built (build/solve_mapf) and Python with
# matplotlib + pillow available as `python3` (used for rendering only).
#
# Run from the repository root:
#   ./mapf/viz/make_demos.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SOLVE=build/solve_mapf
if [[ ! -x "$SOLVE" ]]; then
  echo "build/solve_mapf not found; run: cmake -S . -B build && cmake --build build" >&2
  exit 1
fi

echo "Solving crossing demo..."
"$SOLVE" mapf/viz/demo_crossing.map mapf/viz/demo_crossing.scen 6 \
  --sweeps 4000 --replicas 12 --out mapf/viz/demo_crossing.plan

echo "Solving rooms demo..."
"$SOLVE" mapf/viz/demo_rooms.map mapf/viz/demo_rooms.scen 5 \
  --sweeps 4000 --replicas 12 --out mapf/viz/demo_rooms.plan

echo "Solving lifelong warehouse demo (real-time replanning)..."
build/demo_rolling data/maps/warehouse-34-22.map 18 --steps 130 --window 10 \
  --execute 3 --deadline 25 --seed 3 --out mapf/viz/demo_lifelong.plan

echo "Rendering GIFs (needs matplotlib + pillow)..."
python3 mapf/viz/render_plan.py mapf/viz/demo_crossing.plan --out mapf/viz/demo_crossing.gif --fps 3
python3 mapf/viz/render_plan.py mapf/viz/demo_rooms.plan --out mapf/viz/demo_rooms.gif --fps 3
python3 mapf/viz/render_plan.py mapf/viz/demo_lifelong.plan \
  --map data/maps/warehouse-34-22.map --out mapf/viz/demo_lifelong.gif --fps 6

echo "Done: mapf/viz/demo_crossing.gif, mapf/viz/demo_rooms.gif, mapf/viz/demo_lifelong.gif"
echo "Browse them in the GUI with: python3 mapf/viz/serve.py"
