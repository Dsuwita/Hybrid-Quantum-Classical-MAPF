#!/usr/bin/env python3
"""Plot the Milestone 4 thread-scaling curve.

Usage:
    ./bench_scaling > bench/scaling.csv
    python3 bench/plot_scaling.py          # writes bench/scaling.png

Requires matplotlib (used for plotting only; the library itself has no
dependencies).
"""

import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent


def main() -> None:
    threads, wall_ms = [], []
    with open(HERE / "scaling.csv", newline="") as f:
        for row in csv.DictReader(f):
            threads.append(int(row["threads"]))
            wall_ms.append(float(row["wall_ms"]))

    speedup = [wall_ms[0] / ms for ms in wall_ms]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(threads, threads, "--", color="gray", label="ideal (linear)")
    ax.plot(threads, speedup, "o-", color="tab:blue", label="measured")
    ax.set_xlabel("threads")
    ax.set_ylabel("speedup vs 1 thread")
    ax.set_title("Parallel restarts: 24 replicas, n=2000 3-regular, 10000 sweeps")
    ax.set_xticks(threads)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = HERE / "scaling.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
