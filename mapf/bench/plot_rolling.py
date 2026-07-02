#!/usr/bin/env python3
"""Plot rolling-horizon replan cycle time and throughput vs agents.

Usage:
    ./bench_rolling > mapf/bench/rolling.csv
    python3 mapf/bench/plot_rolling.py     # writes rolling.png
"""

import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent


def main() -> None:
    agents, cycle_ms, thru = [], [], []
    with open(HERE / "rolling.csv", newline="") as f:
        for row in csv.DictReader(f):
            agents.append(int(row["agents"]))
            cycle_ms.append(float(row["avg_cycle_ms"]))
            thru.append(float(row["throughput_goals_per_step"]))

    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
    ax1.plot(agents, cycle_ms, "o-", color="tab:red", label="cycle time")
    ax1.set_xlabel("number of agents")
    ax1.set_ylabel("avg replan cycle time (ms)", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(agents, thru, "s--", color="tab:blue", label="throughput")
    ax2.set_ylabel("throughput (goals / step)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    ax1.set_title("Rolling-horizon: cycle time and throughput vs agents\n"
                  "(20x20, 4 moving obstacles, lifelong)")
    fig.tight_layout()
    out = HERE / "rolling.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
