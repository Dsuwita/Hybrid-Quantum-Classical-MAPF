#!/usr/bin/env python3
"""Compare this project's annealer against dwave-neal on Gset Max-Cut.

D-Wave's `neal` (SimulatedAnnealingSampler, now shipped in dwave-samplers)
is the reference open-source simulated annealer for Ising/QUBO problems.
This script runs both solvers on the same Gset instances with a comparable
budget and reports cut quality (percent of best-known) and wall time, side
by side. It reports whichever way the numbers come out.

Budget matching: our ParallelAnnealer runs `replicas` independent restarts
of `sweeps` sweeps each and keeps the best; neal is asked for the same
`num_reads` (= replicas) and `num_sweeps` (= sweeps), so both do the same
count of single-spin-flip sweeps over the same graph.

Setup:
    pip install dwave-neal           # or dwave-samplers
    ./data/download_gset.sh          # fetch G1, G22, G39, G55
    cmake --build build              # so compare_maxcut exists
    python3 bench/neal_compare.py    # prints a table, writes bench/neal.md

Only the standard library plus dwave-neal is used here; the C++ solver has
no dependencies at all.
"""

import json
import pathlib
import subprocess
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
GSET = ROOT / "data" / "gset"
COMPARE = ROOT / "build" / "compare_maxcut"

# Published best-known cut values (Benlic & Hao 2013; also the values used
# elsewhere in this repo).
BEST = {"G1": 11624, "G22": 13359, "G39": 2408, "G55": 10294}

SWEEPS = 2000
REPLICAS = 16


def read_gset(path):
    """Return (n, edges) where edges is a list of (u, v, w), 0-indexed."""
    with open(path) as f:
        n, m = (int(x) for x in f.readline().split())
        edges = []
        for line in f:
            if not line.strip():
                continue
            u, v, w = (int(x) for x in line.split())
            edges.append((u - 1, v - 1, w))
    return n, edges


def run_ours(name):
    """Our annealer via compare_maxcut; returns (cut, wall_ms)."""
    out = subprocess.run(
        [str(COMPARE), str(GSET / name), "--best", str(BEST[name]),
         "--sweeps", str(SWEEPS), "--replicas", str(REPLICAS), "--restarts", "1"],
        capture_output=True, text=True, timeout=600).stdout
    data = json.loads(out.strip().splitlines()[-1])
    return data["annealer"]["cut"], data["annealer"]["wall_ms"]


def run_neal(n, edges):
    """dwave-neal on the same Ising model; returns (cut, wall_ms).

    Max-Cut -> Ising: J_uv = w for each edge, h = 0. The cut of a sample is
    (total_edge_weight - ising_energy) / 2, so minimizing energy maximizes
    the cut (same mapping the C++ library uses).
    """
    import neal
    h = {i: 0.0 for i in range(n)}
    J = {(u, v): float(w) for (u, v, w) in edges}
    total_w = sum(w for (_, _, w) in edges)

    sampler = neal.SimulatedAnnealingSampler()
    t0 = time.perf_counter()
    sampleset = sampler.sample_ising(h, J, num_reads=REPLICAS, num_sweeps=SWEEPS)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    best_energy = sampleset.first.energy
    cut = (total_w - best_energy) / 2.0
    return int(round(cut)), wall_ms


def main():
    if not COMPARE.exists():
        raise SystemExit("build/compare_maxcut not found; run cmake --build build")
    instances = [n for n in ("G1", "G22", "G39", "G55") if (GSET / n).is_file()]
    if not instances:
        raise SystemExit("no Gset instances in data/gset; run ./data/download_gset.sh")

    rows = []
    header = ("instance", "n", "best", "ours cut", "ours %", "ours ms",
              "neal cut", "neal %", "neal ms")
    print(f"budget: {REPLICAS} restarts x {SWEEPS} sweeps\n")
    print("  ".join(f"{h:>9}" for h in header))
    for name in instances:
        n, edges = read_gset(GSET / name)
        best = BEST[name]
        oc, oms = run_ours(name)
        nc, nms = run_neal(n, edges)
        op, npc = 100 * oc / best, 100 * nc / best
        rows.append((name, n, best, oc, op, oms, nc, npc, nms))
        print(f"{name:>9}  {n:>9}  {best:>9}  {oc:>9}  {op:>8.2f}  {oms:>8.1f}  "
              f"{nc:>9}  {npc:>8.2f}  {nms:>8.1f}")

    # Write a markdown table for the README.
    lines = [
        "# dwave-neal comparison (Milestone 6)",
        "",
        f"Both solvers run {REPLICAS} restarts of {SWEEPS} single-spin-flip sweeps on the",
        "same Gset instance. `neal` is D-Wave's SimulatedAnnealingSampler",
        "(`pip install dwave-neal`). Cut percentages are of the published best-known",
        "value. Reproduce with `python3 bench/neal_compare.py`.",
        "",
        "| instance | n | best-known | ours cut | ours % | ours ms | neal cut | neal % | neal ms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for (name, n, best, oc, op, oms, nc, npc, nms) in rows:
        lines.append(f"| {name} | {n} | {best} | {oc} | {op:.2f}% | {oms:.1f} | "
                     f"{nc} | {npc:.2f}% | {nms:.1f} |")
    (HERE / "neal.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {HERE / 'neal.md'}")


if __name__ == "__main__":
    main()
