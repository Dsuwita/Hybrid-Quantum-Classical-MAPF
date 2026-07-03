#!/usr/bin/env python3
"""Plot the Milestone 6 quality-vs-sweeps curve.

Usage:
    ./build/bench_quality --out bench/quality.csv
    python3 bench/plot_quality.py          # writes bench/quality.png

Shows best energy reached versus sweep budget on a fixed instance: the mean
over 10 seeds as a line, with the seed-to-seed min/max drawn as a shaded
band. Lower energy is better. Requires matplotlib (plotting only; the
library itself has no dependencies).
"""

import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent


def main() -> None:
    sweeps, mean, lo, hi = [], [], [], []
    with open(HERE / "quality.csv", newline="") as f:
        for row in csv.DictReader(f):
            sweeps.append(int(row["sweeps"]))
            mean.append(float(row["mean_energy"]))
            lo.append(float(row["min_energy"]))
            hi.append(float(row["max_energy"]))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.fill_between(sweeps, lo, hi, color="tab:blue", alpha=0.2, label="min/max over seeds")
    ax.plot(sweeps, mean, "o-", color="tab:blue", label="mean of 10 seeds")
    ax.set_xscale("log")
    ax.set_xlabel("sweeps (log scale)")
    ax.set_ylabel("best energy (lower is better)")
    ax.set_title("Quality vs sweeps: random 3-regular, n=800, single-run anneal")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    out = HERE / "quality.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
