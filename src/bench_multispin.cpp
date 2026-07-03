// bench_multispin.cpp
//
// Milestone 7(b) throughput: multi-spin coding versus the scalar fast
// annealer, in flips per nanosecond (the units of the Isakov paper). The
// scalar annealer does one replica per pass; the multi-spin annealer does 64
// at once with bitwise updates, so its flip count is 64x the passes. Same
// instance and sweep budget for both. Reported alongside the single-thread
// numbers in bench/opt_log.md.
//
// Instance: random 3-regular graph, n=2000, +/-1 couplings, h=0 (the
// multi-spin technique's target case), matching the opt_log throughput setup.
//
// Usage: bench_multispin [--n N] [--sweeps S] [--reps R]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "anneal/bqm.hpp"
#include "anneal/fast_annealer.hpp"
#include "anneal/multispin.hpp"
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
double now_ns() {
    return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now().time_since_epoch())
                                   .count());
}
}  // namespace

int main(int argc, char** argv) {
    const std::size_t n = static_cast<std::size_t>(argl(argc, argv, "--n", 2000));
    const std::size_t sweeps = static_cast<std::size_t>(argl(argc, argv, "--sweeps", 3000));
    const int reps = static_cast<int>(argl(argc, argv, "--reps", 3));

    // Random 3-regular graph with +/-1 couplings.
    Graph g = random_d_regular(n, 3, 12345);
    BQM bqm(n, Vartype::Spin);
    {
        std::mt19937_64 rng(999);
        for (const auto& e : g.edges) bqm.add_interaction(e.u, e.v, (rng() & 1u) ? 1.0 : -1.0);
    }

    const double t0 = 3.0, alpha = 0.999;

    // Scalar fast annealer (level 4, xoshiro): best of `reps` timed runs.
    double scalar_best_ns = 1e300;
    for (int r = 0; r < reps; ++r) {
        GeometricSchedule sched(t0, alpha);
        FastAnnealer<GeometricSchedule, Xoshiro256pp, 4> ann(bqm, sched, sweeps, 100 + r);
        const double a = now_ns();
        volatile double sink = ann.solve().best_energy;
        (void)sink;
        scalar_best_ns = std::min(scalar_best_ns, now_ns() - a);
    }
    const double scalar_flips = static_cast<double>(n) * sweeps;
    const double scalar_fpns = scalar_flips / scalar_best_ns;

    // Multi-spin annealer (64 replicas/word). Disable interior sampling so
    // the timed loop is pure updates.
    double ms_best_ns = 1e300;
    for (int r = 0; r < reps; ++r) {
        GeometricSchedule sched(t0, alpha);
        MultiSpinAnnealer<GeometricSchedule> ann(bqm, sched, sweeps, 100 + r, sweeps + 1);
        const double a = now_ns();
        volatile double sink = ann.solve().best_energy;
        (void)sink;
        ms_best_ns = std::min(ms_best_ns, now_ns() - a);
    }
    const double ms_flips = 64.0 * static_cast<double>(n) * sweeps;
    const double ms_fpns = ms_flips / ms_best_ns;

    std::printf("instance: random 3-regular, n=%zu, +/-1 couplings, %zu sweeps\n", n, sweeps);
    std::printf("%-28s %12s %12s\n", "method", "flips/ns", "speedup");
    std::printf("%-28s %12.3f %12s\n", "scalar FastAnnealer (L4)", scalar_fpns, "1.0x");
    std::printf("%-28s %12.3f %11.1fx\n", "multi-spin (64/word)", ms_fpns, ms_fpns / scalar_fpns);
    return 0;
}
