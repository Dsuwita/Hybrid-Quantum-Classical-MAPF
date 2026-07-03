// parallel.hpp
//
// Parallel restarts: the simplest useful way to parallelize simulated
// annealing. Anneals are embarrassingly parallel — k independent runs
// from different seeds explore different basins, and the final answer is
// the best result among them. There is no communication between runs, so
// the parallel version must produce EXACTLY the same best-of answer as
// running the same seeds one after another (the permutation-stability
// test relies on this).
//
// Threading model:
//  - The CompactBQM view is built once and shared read-only by every
//    worker; nothing in a solve writes to it, so no synchronization is
//    needed to read it.
//  - Each replica r gets its own FastAnnealer with seed_base + r and its
//    own heap-allocated working state (spins, field cache), so worker
//    threads never write to memory another worker reads (no false
//    sharing of hot data, no locks anywhere in the hot loop).
//  - num_replicas may exceed num_threads: replicas form a work queue and
//    workers pull the next index with a single atomic counter. That
//    atomic is touched once per replica, not per sweep, so contention is
//    irrelevant.
//  - Workers write their result only into their own replica's slot of a
//    preallocated results vector; the best-of reduction happens on the
//    calling thread after all workers have joined.
#pragma once

#include "anneal/bqm.hpp"
#include "anneal/fast_annealer.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <random>
#include <thread>
#include <vector>

namespace anneal {

// Result of a parallel solve: the winning replica plus every replica's
// full result. Keeping all of them lets callers (and tests) audit each
// replica's energy independently instead of trusting the reduction.
struct ParallelResult {
    SolveResult best;
    std::size_t best_replica = 0;  // index (== seed offset) of the winner
    std::vector<SolveResult> replicas;
};

template <typename Schedule, typename Rng = std::mt19937_64>
class ParallelAnnealer {
public:
    // num_threads = 0 means "use hardware_concurrency". max_wall_ms > 0
    // makes each replica anytime: it stops at that per-replica deadline and
    // contributes its best-so-far. Used by the rolling-horizon MAPF driver.
    ParallelAnnealer(const BQM& bqm, Schedule schedule, std::size_t num_sweeps,
                     std::uint64_t seed_base, std::size_t num_replicas,
                     std::size_t num_threads = 0, double max_wall_ms = 0.0)
        : view_(bqm),
          schedule_(schedule),
          num_sweeps_(num_sweeps),
          seed_base_(seed_base),
          num_replicas_(num_replicas),
          num_threads_(num_threads == 0 ? std::thread::hardware_concurrency() : num_threads),
          max_wall_ms_(max_wall_ms) {}

    ParallelResult solve() {
        ParallelResult out;
        out.replicas.resize(num_replicas_);

        // Replica seeding is by REPLICA index, not worker index: replica
        // r always anneals with seed_base + r no matter which thread
        // happens to pick it up. This is what makes results independent
        // of thread count and scheduling order.
        std::atomic<std::size_t> next_replica{0};
        auto worker = [&]() {
            while (true) {
                const std::size_t r = next_replica.fetch_add(1, std::memory_order_relaxed);
                if (r >= num_replicas_) return;
                FastAnnealer<Schedule, Rng> annealer(view_, schedule_, num_sweeps_,
                                                     seed_base_ + r, max_wall_ms_);
                out.replicas[r] = annealer.solve();
            }
        };

        const std::size_t k = std::min(num_threads_, num_replicas_);
        std::vector<std::thread> threads;
        threads.reserve(k);
        for (std::size_t i = 0; i < k; ++i) threads.emplace_back(worker);
        for (auto& t : threads) t.join();

        // Best-of reduction, ties broken by lowest replica index so the
        // answer is deterministic regardless of completion order.
        out.best_replica = 0;
        for (std::size_t r = 1; r < num_replicas_; ++r) {
            if (out.replicas[r].best_energy < out.replicas[out.best_replica].best_energy) {
                out.best_replica = r;
            }
        }
        out.best = out.replicas[out.best_replica];
        return out;
    }

    const CompactBQM& view() const { return view_; }

private:
    CompactBQM view_;
    Schedule schedule_;
    std::size_t num_sweeps_;
    std::uint64_t seed_base_;
    std::size_t num_replicas_;
    std::size_t num_threads_;
    double max_wall_ms_ = 0.0;
};

}  // namespace anneal
