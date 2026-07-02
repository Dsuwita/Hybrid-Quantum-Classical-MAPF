// bench_throughput.cpp
//
// Milestone 3 throughput benchmark: flips attempted per second on a
// G-set-sized problem, measured for the naive annealer and for each
// optimization level of FastAnnealer in turn. Results are recorded in
// bench/opt_log.md.
//
// Problem: random 3-regular graph, n = 2000, couplings ±1, h = 0 (a
// standard spin-glass benchmark shape, matching Isakov et al.'s setup).
// The metric is proposals per wall-clock second: n * num_sweeps / time.
// Every configuration runs the SAME schedule and sweep budget: the mix
// of accepted vs rejected and uphill vs downhill proposals depends on
// temperature, so unequal budgets would give each config a different
// work profile and make the throughput ratios meaningless.

#include "anneal/annealer.hpp"
#include "anneal/bqm.hpp"
#include "anneal/fast_annealer.hpp"
#include "anneal/rng.hpp"
#include "anneal/schedule.hpp"

#include "bench_common.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <random>
#include <set>
#include <utility>
#include <vector>

using namespace anneal;

namespace {

// Frozen copy of the Milestone 2 Annealer exactly as it shipped (commit
// 799aef): std::map adjacency walk per proposal AND the original
// std::uniform_real_distribution draw. The in-tree Annealer has since
// adopted the shared uniform01() helper (part of optimization 5, needed
// so the differential test can compare decision streams), which already
// speeds it up; keeping this verbatim copy preserves the true Milestone 2
// baseline that the >= 10x acceptance criterion is measured against.
class M2BaselineAnnealer {
public:
    M2BaselineAnnealer(const BQM& bqm, GeometricSchedule schedule, std::size_t num_sweeps,
                       std::uint64_t seed)
        : bqm_(bqm), schedule_(schedule), num_sweeps_(num_sweeps), rng_(seed) {}

    SolveResult solve() {
        std::size_t n = bqm_.num_variables();
        std::vector<std::int8_t> state(n);
        std::uniform_int_distribution<int> spin_dist(0, 1);
        for (std::size_t i = 0; i < n; ++i) {
            state[i] = spin_dist(rng_) ? std::int8_t(1) : std::int8_t(-1);
        }
        double current_energy = bqm_.energy(state);
        std::vector<std::int8_t> best_state = state;
        double best_energy = current_energy;
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        for (std::size_t sweep = 0; sweep < num_sweeps_; ++sweep) {
            double t = schedule_.temperature(sweep);
            for (std::size_t i = 0; i < n; ++i) {
                double field = bqm_.linear(i);
                for (const auto& [j, coupling] : bqm_.neighbors(i)) {
                    field += coupling * static_cast<double>(state[j]);
                }
                double d_energy = -2.0 * static_cast<double>(state[i]) * field;
                bool accept = (d_energy <= 0.0) || (unif(rng_) < std::exp(-d_energy / t));
                if (accept) {
                    state[i] = static_cast<std::int8_t>(-state[i]);
                    current_energy += d_energy;
                    if (current_energy < best_energy) {
                        best_energy = current_energy;
                        best_state = state;
                    }
                }
            }
        }
        return SolveResult{best_state, best_energy, state, current_energy, num_sweeps_};
    }

private:
    BQM bqm_;
    GeometricSchedule schedule_;
    std::size_t num_sweeps_;
    std::mt19937_64 rng_;
};

using anneal_bench::random_3_regular_pm1;

// Each config runs kRepetitions times and reports the fastest run: for a
// fixed workload the minimum wall time is the measurement least polluted
// by OS scheduling noise (which only ever adds time, never removes it).
// Repetitions are interleaved round-robin across configs (rep 1 of every
// config, then rep 2 of every config, ...) rather than run back to back:
// throttling and background load come in multi-second waves, and
// interleaving keeps one wave from eating all of a single config's reps.
constexpr int kRepetitions = 5;

struct BenchRow {
    const char* name;
    std::size_t sweeps;
    double seconds = 1e18;
    double best_energy = 0.0;
};

template <typename SolverFactory>
void run_once(BenchRow& row, const BQM& bqm, SolverFactory make) {
    auto solver = make(bqm, row.sweeps);
    auto start = std::chrono::steady_clock::now();
    SolveResult result = solver.solve();
    auto stop = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double>(stop - start).count();
    if (seconds < row.seconds) {
        row.seconds = seconds;
        row.best_energy = result.best_energy;
    }
}

void print_row(const BenchRow& row, std::size_t n, double baseline_flips_per_sec) {
    double flips = static_cast<double>(n) * static_cast<double>(row.sweeps);
    double fps = flips / row.seconds;
    std::printf("%-28s %8zu sweeps  %8.3f s  %12.3e flips/s  %7.4f flips/ns  %6.1fx  E=%.0f\n",
                row.name, row.sweeps, row.seconds, fps, fps / 1e9,
                fps / baseline_flips_per_sec, row.best_energy);
}

}  // namespace

int main() {
    const std::size_t n = 2000;
    BQM bqm = random_3_regular_pm1(n, 12345);

    // T decays from 3.0 to ~1e-4 over the run: the budget covers the full
    // hot-to-frozen range every real anneal goes through.
    const double t0 = 3.0, alpha = 0.999;
    const std::size_t sweeps = 10000;
    const std::uint64_t seed = 42;

    // Warmup: let the CPU reach its boost clock before anything is timed,
    // so the first measured config isn't penalized for starting cold.
    {
        FastAnnealer<GeometricSchedule, Xoshiro256pp, 4> w(bqm, GeometricSchedule(t0, alpha),
                                                           sweeps, seed);
        volatile double sink = w.solve().best_energy;
        (void)sink;
    }

    std::vector<BenchRow> rows(7);
    rows[0] = {"M2 baseline (verbatim)", sweeps};
    rows[1] = {"naive + fast uniform01", sweeps};
    rows[2] = {"L1 flat CSR storage", sweeps};
    rows[3] = {"L2 + local field cache", sweeps};
    rows[4] = {"L3 + downhill early-exit", sweeps};
    rows[5] = {"L4 + acceptance table", sweeps};
    rows[6] = {"L4 + xoshiro256++", sweeps};

    GeometricSchedule sched(t0, alpha);
    for (int rep = 0; rep < kRepetitions; ++rep) {
        run_once(rows[0], bqm, [&](const BQM& b, std::size_t s) {
            return M2BaselineAnnealer(b, sched, s, seed);
        });
        run_once(rows[1], bqm, [&](const BQM& b, std::size_t s) {
            return Annealer<GeometricSchedule>(b, sched, s, seed);
        });
        run_once(rows[2], bqm, [&](const BQM& b, std::size_t s) {
            return FastAnnealer<GeometricSchedule, std::mt19937_64, 1>(b, sched, s, seed);
        });
        run_once(rows[3], bqm, [&](const BQM& b, std::size_t s) {
            return FastAnnealer<GeometricSchedule, std::mt19937_64, 2>(b, sched, s, seed);
        });
        run_once(rows[4], bqm, [&](const BQM& b, std::size_t s) {
            return FastAnnealer<GeometricSchedule, std::mt19937_64, 3>(b, sched, s, seed);
        });
        run_once(rows[5], bqm, [&](const BQM& b, std::size_t s) {
            return FastAnnealer<GeometricSchedule, std::mt19937_64, 4>(b, sched, s, seed);
        });
        run_once(rows[6], bqm, [&](const BQM& b, std::size_t s) {
            return FastAnnealer<GeometricSchedule, Xoshiro256pp, 4>(b, sched, s, seed);
        });
    }

    double baseline_fps = static_cast<double>(n) * static_cast<double>(rows[0].sweeps) / rows[0].seconds;

    std::printf("Throughput benchmark: random 3-regular graph, n=%zu, couplings +/-1\n", n);
    std::printf("Schedule: geometric T0=%.1f alpha=%.4f, seed=%llu\n\n", t0, alpha,
                static_cast<unsigned long long>(seed));
    for (const auto& row : rows) print_row(row, n, baseline_fps);

    return 0;
}
