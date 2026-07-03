// compare_maxcut.cpp
//
// Run the annealer and the classical local-search solver on the same
// Max-Cut instance and report both, side by side. Emits JSON by default
// (consumed by the GUI); pass --human for a table.
//
// Usage:
//   compare_maxcut <gset-file> [--best V] [--sweeps N] [--replicas N]
//                  [--restarts N] [--threads N] [--seed N] [--human]

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#include "anneal/parallel.hpp"
#include "anneal/problems/classical_maxcut.hpp"
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
bool has(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return true;
    return false;
}
double pct(double cut, double best) { return best > 0.0 ? 100.0 * cut / best : 0.0; }
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

    const double best = argf(argc, argv, "--best", 0.0);
    const std::size_t sweeps = static_cast<std::size_t>(argl(argc, argv, "--sweeps", 20000));
    const std::size_t replicas = static_cast<std::size_t>(argl(argc, argv, "--replicas", 16));
    const std::size_t restarts = static_cast<std::size_t>(argl(argc, argv, "--restarts", 200));
    const std::size_t threads = static_cast<std::size_t>(argl(argc, argv, "--threads", 0));
    const std::uint64_t seed = static_cast<std::uint64_t>(argl(argc, argv, "--seed", 1));

    // Annealer.
    BQM bqm = maxcut_to_bqm(g);
    GeometricSchedule schedule(3.0, 0.9995);
    auto a0 = std::chrono::steady_clock::now();
    ParallelAnnealer<GeometricSchedule> annealer(bqm, schedule, sweeps, seed, replicas, threads);
    ParallelResult ar = annealer.solve();
    double a_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a0).count();
    double a_cut = cut_value(g, ar.best.best_state);

    // Classical local search.
    auto c0 = std::chrono::steady_clock::now();
    ClassicalResult cr = multistart_local_search_maxcut(g, restarts, seed);
    double c_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();
    double c_cut = cr.cut;

    if (has(argc, argv, "--human")) {
        std::printf("instance %s  (%zu nodes, %zu edges)  best-known %.0f\n", argv[1], g.n,
                    g.edges.size(), best);
        std::printf("  %-10s cut %8.0f  %6.2f%%  %8.1f ms\n", "annealer", a_cut, pct(a_cut, best),
                    a_ms);
        std::printf("  %-10s cut %8.0f  %6.2f%%  %8.1f ms\n", "classical", c_cut, pct(c_cut, best),
                    c_ms);
    } else {
        std::printf(
            "{\"nodes\":%zu,\"edges\":%zu,\"best\":%.0f,"
            "\"annealer\":{\"cut\":%.0f,\"percent\":%.2f,\"wall_ms\":%.1f,\"replicas\":%zu,"
            "\"sweeps\":%zu},"
            "\"classical\":{\"cut\":%.0f,\"percent\":%.2f,\"wall_ms\":%.1f,\"restarts\":%zu}}\n",
            g.n, g.edges.size(), best, a_cut, pct(a_cut, best), a_ms, replicas, sweeps, c_cut,
            pct(c_cut, best), c_ms, restarts);
    }
    return 0;
}
