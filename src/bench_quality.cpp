// bench_quality.cpp
//
// Milestone 6 quality-vs-effort benchmark. On one fixed instance it runs a
// single simulated-annealing trajectory at each sweep budget in
// {10, 100, 1000, 10000, 100000}, repeats over 10 seeds, and reports the
// best energy reached: mean and the min/max band across seeds. Plotting
// bench/plot_quality.py turns the CSV into the diminishing-returns curve.
//
// Each (budget, seed) is one independent single-replica anneal (no
// parallel best-of), so the curve shows what one run of the solver buys per
// decade of sweeps, not what throwing more restarts at it buys. Lower
// energy is better; for the Max-Cut mapping used here it corresponds to a
// larger cut.
//
// The instance defaults to a seeded random 3-regular graph (n = 800, the
// size of Gset G1) so the benchmark is self-contained and reproducible with
// no downloads. Pass a Gset file to run on a published instance instead.
//
// Usage:
//   bench_quality [gset-file] [--seeds N] [--out FILE]
// Output CSV columns: sweeps,mean_energy,min_energy,max_energy,mean_cut,seeds

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "anneal/fast_annealer.hpp"
#include "anneal/problems/maxcut.hpp"
#include "anneal/rng.hpp"
#include "anneal/schedule.hpp"

using namespace anneal;

namespace {

long argl(int argc, char** argv, const char* f, long d) {
    for (int i = 1; i + 1 < argc; ++i)
        if (!std::strcmp(argv[i], f)) return std::atol(argv[i + 1]);
    return d;
}
const char* args(int argc, char** argv, const char* f, const char* d) {
    for (int i = 1; i + 1 < argc; ++i)
        if (!std::strcmp(argv[i], f)) return argv[i + 1];
    return d;
}

}  // namespace

int main(int argc, char** argv) {
    // Load the instance: a Gset file if given as a bare first argument,
    // otherwise a seeded random 3-regular graph the size of G1.
    Graph g;
    std::string instance_name;
    if (argc > 1 && argv[1][0] != '-') {
        std::ifstream in(argv[1]);
        if (!in) {
            std::fprintf(stderr, "cannot open %s\n", argv[1]);
            return 1;
        }
        g = parse_gset(in);
        instance_name = argv[1];
    } else {
        g = random_d_regular(800, 3, 12345);
        instance_name = "random-3-regular-n800-seed12345";
    }
    const BQM bqm = maxcut_to_bqm(g);

    const std::size_t seeds = static_cast<std::size_t>(argl(argc, argv, "--seeds", 10));
    const char* out_path = args(argc, argv, "--out", nullptr);
    const std::vector<std::size_t> budgets = {10, 100, 1000, 10000, 100000};

    // Same schedule shape as the Max-Cut runner, just re-derived per budget
    // so the geometric decay spans the whole run at every budget.
    const double t0 = 3.0;

    std::vector<std::string> rows;
    rows.push_back("sweeps,mean_energy,min_energy,max_energy,mean_cut,seeds");
    std::fprintf(stderr, "instance: %s (n=%zu, m=%zu)\n", instance_name.c_str(), g.n,
                 g.edges.size());
    std::fprintf(stderr, "%8s  %12s  %12s  %12s  %10s\n", "sweeps", "mean_E", "min_E", "max_E",
                 "mean_cut");

    for (std::size_t budget : budgets) {
        // alpha chosen so the schedule cools from t0 to ~0.05 over the run.
        const double alpha = std::pow(0.05 / t0, 1.0 / static_cast<double>(std::max<std::size_t>(1, budget)));
        std::vector<double> energies;
        double cut_sum = 0.0;
        for (std::size_t s = 0; s < seeds; ++s) {
            GeometricSchedule schedule(t0, alpha);
            FastAnnealer<GeometricSchedule, Xoshiro256pp, 4> annealer(bqm, schedule, budget,
                                                                      1000 + s);
            SolveResult r = annealer.solve();
            energies.push_back(r.best_energy);
            cut_sum += cut_value(g, r.best_state);
        }
        const double mean = std::accumulate(energies.begin(), energies.end(), 0.0) / seeds;
        const double lo = *std::min_element(energies.begin(), energies.end());
        const double hi = *std::max_element(energies.begin(), energies.end());
        const double mean_cut = cut_sum / seeds;
        char line[256];
        std::snprintf(line, sizeof line, "%zu,%.2f,%.2f,%.2f,%.2f,%zu", budget, mean, lo, hi,
                      mean_cut, seeds);
        rows.push_back(line);
        std::fprintf(stderr, "%8zu  %12.1f  %12.1f  %12.1f  %10.1f\n", budget, mean, lo, hi,
                     mean_cut);
    }

    if (out_path) {
        std::ofstream out(out_path);
        for (const auto& r : rows) out << r << "\n";
        std::fprintf(stderr, "wrote %s\n", out_path);
    } else {
        for (const auto& r : rows) std::printf("%s\n", r.c_str());
    }
    return 0;
}
