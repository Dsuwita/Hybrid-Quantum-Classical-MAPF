// bench_scaling.cpp
//
// Milestone 4 scaling benchmark: fixed problem, fixed per-replica sweep
// budget, fixed replica count; only the number of worker threads varies.
// Perfect scaling would halve wall time with every doubling of threads
// (until physical cores run out). Emits CSV to stdout:
//   threads,wall_ms,best_energy
// plus a human-readable speedup table on stderr, so
//   ./bench_scaling > bench/scaling.csv
// captures clean data while still showing progress in the terminal.
//
// Best energy must be IDENTICAL in every row: thread count only changes
// who runs which replica, never the replicas' results (that is the
// permutation-stability guarantee tested in tests/test_parallel.cpp).

#include "anneal/parallel.hpp"
#include "anneal/rng.hpp"
#include "anneal/schedule.hpp"

#include "bench_common.hpp"

#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

using namespace anneal;

int main() {
    const std::size_t n = 2000;
    BQM bqm = anneal_bench::random_3_regular_pm1(n, 12345);

    const double t0 = 3.0, alpha = 0.999;
    const std::size_t sweeps = 10000;   // per replica
    const std::size_t replicas = 24;    // divisible by 1..4, 6, 8, 12
    const std::uint64_t seed_base = 42;
    GeometricSchedule sched(t0, alpha);

    // Thread counts: 1, 2, 4, ... up to hardware_concurrency, plus
    // hw/2 (the physical core count on SMT machines, where the scaling
    // curve is expected to bend) and hardware_concurrency itself.
    std::vector<std::size_t> thread_counts;
    const std::size_t hw = std::thread::hardware_concurrency();
    for (std::size_t t = 1; t < hw; t *= 2) thread_counts.push_back(t);
    if (hw >= 2) thread_counts.push_back(hw / 2);
    thread_counts.push_back(hw);
    std::sort(thread_counts.begin(), thread_counts.end());
    thread_counts.erase(std::unique(thread_counts.begin(), thread_counts.end()),
                        thread_counts.end());

    std::fprintf(stderr,
                 "Scaling benchmark: n=%zu 3-regular +/-1, %zu replicas x %zu sweeps, "
                 "xoshiro256++, hardware_concurrency=%zu\n\n",
                 n, replicas, sweeps, hw);

    // Warmup (untimed): boost clocks before the single-thread reference.
    {
        ParallelAnnealer<GeometricSchedule, Xoshiro256pp> w(bqm, sched, sweeps, seed_base,
                                                            replicas, hw);
        volatile double sink = w.solve().best.best_energy;
        (void)sink;
    }

    std::printf("threads,wall_ms,best_energy\n");
    double t1_ms = 0.0;
    for (std::size_t threads : thread_counts) {
        // Best of 3 runs per thread count, same noise rationale as
        // bench_throughput.cpp: OS interference only ever adds time.
        double wall_ms = 1e18;
        double best_energy = 0.0;
        for (int rep = 0; rep < 3; ++rep) {
            ParallelAnnealer<GeometricSchedule, Xoshiro256pp> pa(bqm, sched, sweeps, seed_base,
                                                                 replicas, threads);
            auto start = std::chrono::steady_clock::now();
            ParallelResult result = pa.solve();
            auto stop = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(stop - start).count();
            if (ms < wall_ms) wall_ms = ms;
            best_energy = result.best.best_energy;  // identical across reps and thread counts
        }
        if (threads == 1) t1_ms = wall_ms;

        std::printf("%zu,%.1f,%.1f\n", threads, wall_ms, best_energy);
        std::fflush(stdout);
        std::fprintf(stderr, "%2zu threads: %8.1f ms  speedup %5.2fx  efficiency %5.1f%%  E=%.0f\n",
                     threads, wall_ms, t1_ms / wall_ms, 100.0 * t1_ms / wall_ms / threads,
                     best_energy);
    }
    return 0;
}
