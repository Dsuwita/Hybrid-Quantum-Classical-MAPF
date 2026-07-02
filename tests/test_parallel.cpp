// test_parallel.cpp
//
// Milestone 4 tests. Parallel restarts must be a pure best-of over
// independent runs: same seeds -> same answer, no matter how many
// threads or in what order replicas execute. Also built to run cleanly
// under -fsanitize=thread (see the milestone notes for the command).

#include "anneal/bqm.hpp"
#include "anneal/fast_annealer.hpp"
#include "anneal/parallel.hpp"
#include "anneal/schedule.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace anneal;

namespace {

bool close(double a, double b, double eps = 1e-9) {
    return std::fabs(a - b) < eps;
}

BQM random_ising(std::size_t n, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> bias(-2.0, 2.0);
    BQM bqm(n, Vartype::Spin);
    for (std::size_t i = 0; i < n; ++i) bqm.add_linear(i, bias(rng));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            if (std::uniform_real_distribution<double>(0, 1)(rng) < 0.5) {
                bqm.add_interaction(i, j, bias(rng));
            }
        }
    }
    return bqm;
}

// The core guarantee: k threads over R replicas produce exactly the same
// per-replica results (not just the same best) as running the same R
// seeds sequentially on one thread.
void test_permutation_stable_best_of() {
    std::mt19937_64 rng(4242);
    BQM bqm = random_ising(40, rng);
    GeometricSchedule schedule(10.0, 0.99);
    const std::size_t sweeps = 500, replicas = 8;
    const std::uint64_t seed_base = 900;

    // Sequential reference: same seeds, one at a time, shared view.
    CompactBQM view(bqm);
    std::vector<SolveResult> reference;
    for (std::size_t r = 0; r < replicas; ++r) {
        FastAnnealer<GeometricSchedule> a(view, schedule, sweeps, seed_base + r);
        reference.push_back(a.solve());
    }

    // Parallel runs with several thread counts, including more threads
    // than replicas and fewer threads than replicas (work queue).
    for (std::size_t threads : {1u, 3u, 4u, 12u}) {
        ParallelAnnealer<GeometricSchedule> pa(bqm, schedule, sweeps, seed_base, replicas,
                                               threads);
        ParallelResult result = pa.solve();
        assert(result.replicas.size() == replicas);
        for (std::size_t r = 0; r < replicas; ++r) {
            assert(result.replicas[r].best_energy == reference[r].best_energy);
            assert(result.replicas[r].best_state == reference[r].best_state);
            assert(result.replicas[r].final_state == reference[r].final_state);
        }
        // The reduction picks the true minimum, deterministically.
        for (std::size_t r = 0; r < replicas; ++r) {
            assert(result.best.best_energy <= result.replicas[r].best_energy);
        }
        assert(result.best.best_energy == result.replicas[result.best_replica].best_energy);
    }

    std::printf("test_permutation_stable_best_of passed (threads in {1,3,4,12})\n");
}

// The Milestone 2 energy-audit tripwire, applied to EVERY replica of a
// parallel run: each reported best_energy must equal BQM::energy of the
// reported best_state.
void test_energy_audit_every_replica() {
    std::mt19937_64 rng(777);
    for (int t = 0; t < 5; ++t) {
        BQM bqm = random_ising(30, rng);
        GeometricSchedule schedule(8.0, 0.98);
        ParallelAnnealer<GeometricSchedule> pa(bqm, schedule, 300,
                                               static_cast<std::uint64_t>(7000 + t), 6, 3);
        ParallelResult result = pa.solve();
        for (const SolveResult& r : result.replicas) {
            assert(close(bqm.energy(r.best_state), r.best_energy, 1e-9));
            assert(close(bqm.energy(r.final_state), r.final_energy, 1e-9));
        }
    }
    std::printf("test_energy_audit_every_replica passed\n");
}

void test_determinism() {
    std::mt19937_64 rng(31);
    BQM bqm = random_ising(25, rng);
    GeometricSchedule schedule(10.0, 0.99);

    ParallelAnnealer<GeometricSchedule> a(bqm, schedule, 400, 5, 10, 4);
    ParallelAnnealer<GeometricSchedule> b(bqm, schedule, 400, 5, 10, 4);
    ParallelResult ra = a.solve();
    ParallelResult rb = b.solve();
    assert(ra.best.best_energy == rb.best.best_energy);
    assert(ra.best_replica == rb.best_replica);
    assert(ra.best.best_state == rb.best.best_state);

    std::printf("test_determinism passed\n");
}

}  // namespace

int main() {
    test_permutation_stable_best_of();
    test_energy_audit_every_replica();
    test_determinism();
    std::printf("All parallel annealer tests passed.\n");
    return 0;
}
