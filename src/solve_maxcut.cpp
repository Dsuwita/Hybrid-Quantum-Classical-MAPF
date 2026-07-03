// solve_maxcut.cpp
//
// Command-line Max-Cut solver over a Gset-format instance. Loads the
// graph, maps it to an Ising BQM, runs the parallel annealer, and reports
// the cut value (recounted independently from the edge list, never trusted
// from the energy), the percent of a supplied best-known value, and wall
// time.
//
// Usage:
//   solve_maxcut <file> [options]
// Options:
//   --threads N    worker threads (default: hardware)
//   --replicas N   parallel restarts (default 16)
//   --sweeps N     sweeps per replica (default 20000)
//   --t0 F         schedule start temperature (default 3.0)
//   --alpha F      geometric cooling factor (default 0.9995)
//   --seed N       base RNG seed (default 1)
//   --best V       best-known cut, to print percent achieved

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#include "anneal/parallel.hpp"
#include "anneal/problems/maxcut.hpp"
#include "anneal/schedule.hpp"

using namespace anneal;

namespace {
double argf(int argc, char** argv, const char* flag, double fallback) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return std::atof(argv[i + 1]);
    return fallback;
}
long argl(int argc, char** argv, const char* flag, long fallback) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return std::atol(argv[i + 1]);
    return fallback;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <gset-file> [options]\n", argv[0]);
        return 2;
    }
    std::ifstream f(argv[1]);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }
    Graph g = parse_gset(f);

    const std::size_t threads = static_cast<std::size_t>(argl(argc, argv, "--threads", 0));
    const std::size_t replicas = static_cast<std::size_t>(argl(argc, argv, "--replicas", 16));
    const std::size_t sweeps = static_cast<std::size_t>(argl(argc, argv, "--sweeps", 20000));
    const double t0 = argf(argc, argv, "--t0", 3.0);
    const double alpha = argf(argc, argv, "--alpha", 0.9995);
    const std::uint64_t seed = static_cast<std::uint64_t>(argl(argc, argv, "--seed", 1));
    const double best = argf(argc, argv, "--best", 0.0);

    BQM bqm = maxcut_to_bqm(g);
    GeometricSchedule schedule(t0, alpha);

    auto start = std::chrono::steady_clock::now();
    ParallelAnnealer<GeometricSchedule> annealer(bqm, schedule, sweeps, seed, replicas, threads);
    ParallelResult r = annealer.solve();
    double wall_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

    double cut = cut_value(g, r.best.best_state);

    std::printf("file        %s\n", argv[1]);
    std::printf("vertices    %zu\n", g.n);
    std::printf("edges       %zu\n", g.edges.size());
    std::printf("replicas    %zu   sweeps %zu   T0 %.3g   alpha %.5g\n", replicas, sweeps, t0,
                alpha);
    std::printf("cut         %.0f\n", cut);
    if (best > 0.0) std::printf("best-known  %.0f   (%.2f%%)\n", best, 100.0 * cut / best);
    std::printf("wall_ms     %.1f\n", wall_ms);
    return 0;
}
