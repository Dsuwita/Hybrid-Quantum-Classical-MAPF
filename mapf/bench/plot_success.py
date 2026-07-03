#!/usr/bin/env python3
"""Plot MAPF solver success rate vs number of agents, per map.

Usage:
    ./bench_mapf > mapf/bench/results.csv
    python3 mapf/bench/plot_success.py     # writes success_rate.png
"""

import csv
import pathlib
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent


def main() -> None:
    series = defaultdict(lambda: ([], []))  # map -> (agents, success_rate)
    with open(HERE / "results.csv", newline="") as f:
        for row in csv.DictReader(f):
            name = pathlib.Path(row["map"]).stem
            series[name][0].append(int(row["agents"]))
            series[name][1].append(float(row["success_rate"]))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, (agents, rate) in sorted(series.items()):
        ax.plot(agents, rate, "o-", label=name)
    ax.set_xlabel("number of agents (k)")
    ax.set_ylabel("success rate (%)")
    ax.set_title("Hybrid anneal MAPF: success vs congestion (16x16 maps)")
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = HERE / "success_rate.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
