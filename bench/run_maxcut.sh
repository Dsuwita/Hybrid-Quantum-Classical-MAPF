#!/usr/bin/env bash
# Reproduce the Max-Cut quality numbers on Gset instances.
#
# Downloads the instances if needed, then solves each with a fixed budget
# and seed and prints cut value and percent of the best-known value. The
# best-known values are the standard published bests for these instances
# (see the table in the README for the source).
#
# Usage (from the repo root, after building):
#   cmake -S . -B build && cmake --build build
#   ./bench/run_maxcut.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SOLVE=build/solve_maxcut
[[ -x "$SOLVE" ]] || { echo "build build/solve_maxcut first" >&2; exit 1; }
[[ -s data/gset/G1 ]] || ./data/download_gset.sh

# instance best-known
INSTANCES=(
  "G1 11624"
  "G22 13359"
  "G39 2408"
  "G55 10294"
)

REPLICAS=16
SWEEPS=20000
SEED=1

printf "%-6s %8s %10s %10s %9s %10s\n" instance nodes cut best percent wall_ms
for row in "${INSTANCES[@]}"; do
  set -- $row
  g=$1; best=$2
  out=$("$SOLVE" "data/gset/$g" --best "$best" --replicas $REPLICAS --sweeps $SWEEPS --seed $SEED)
  nodes=$(echo "$out" | awk '/^vertices/{print $2}')
  cut=$(echo "$out"   | awk '/^cut/{print $2}')
  pct=$(echo "$out"   | awk -F'[()%]' '/^best-known/{print $2}')
  wall=$(echo "$out"  | awk '/^wall_ms/{print $2}')
  printf "%-6s %8s %10s %10s %8s%% %10s\n" "$g" "$nodes" "$cut" "$best" "$pct" "$wall"
done

echo
echo "Budget: $REPLICAS replicas x $SWEEPS sweeps, seed $SEED, geometric T0=3 alpha=0.9995."
