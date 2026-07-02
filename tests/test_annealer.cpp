// test_annealer.cpp
//
// Assert-based tests for the naive sequential Annealer (Milestone 2).
// Exit code 0 means pass.

#include "anneal/annealer.hpp"
#include "anneal/bqm.hpp"
#include "anneal/bruteforce.hpp"
#include "anneal/schedule.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
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

// Best-of-k-seeds solve: run the annealer with several seeds and keep the
// lowest-energy result. This is the standard way to trade wall time for
// solution quality with a stochastic local-search solver.
SolveResult best_of_seeds(const BQM& bqm, int num_seeds, std::uint64_t seed_base) {
    SolveResult best;
    best.best_energy = std::numeric_limits<double>::infinity();
    for (int s = 0; s < num_seeds; ++s) {
        GeometricSchedule schedule(10.0, 0.99);
        Annealer<GeometricSchedule> annealer(bqm, schedule, 2000, seed_base + static_cast<std::uint64_t>(s));
        SolveResult r = annealer.solve();
        if (r.best_energy < best.best_energy) best = r;
    }
    return best;
}

void test_finds_ground_state_most_of_the_time() {
    std::mt19937_64 rng(123);
    int hits = 0;
    const int trials = 30;

    for (int t = 0; t < trials; ++t) {
        std::uniform_int_distribution<int> n_dist(8, 16);
        std::size_t n = static_cast<std::size_t>(n_dist(rng));
        BQM bqm = random_ising(n, rng);

        BruteForceResult exact = brute_force_ground_state(bqm);
        SolveResult result = best_of_seeds(bqm, 5, static_cast<std::uint64_t>(1000 + t));

        // Never below the true ground energy: that would mean an energy
        // accounting bug, not just a search that didn't converge.
        assert(result.best_energy >= exact.ground_energy - 1e-9);

        if (close(result.best_energy, exact.ground_energy, 1e-6)) {
            ++hits;
        }
    }

    std::printf("test_finds_ground_state_most_of_the_time: %d/%d exact hits\n", hits, trials);
    assert(hits >= 28);
    std::printf("test_finds_ground_state_most_of_the_time passed\n");
}

void test_determinism() {
    std::mt19937_64 rng(7);
    BQM bqm = random_ising(12, rng);

    GeometricSchedule schedule(10.0, 0.99);
    Annealer<GeometricSchedule> a1(bqm, schedule, 500, 42);
    Annealer<GeometricSchedule> a2(bqm, schedule, 500, 42);

    SolveResult r1 = a1.solve();
    SolveResult r2 = a2.solve();

    assert(r1.best_energy == r2.best_energy);
    assert(r1.best_state == r2.best_state);
    assert(r1.final_state == r2.final_state);

    std::printf("test_determinism passed\n");
}

void test_number_partition_smoke() {
    // 14 numbers with a known perfect split (sum is even and achievable).
    std::vector<double> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15};
    // total = 106, need a subset summing to 53.
    // Build minimize (sum a_i s_i)^2 as an Ising model:
    // J_ij = 2 a_i a_j for all i<j, h_i = 0, offset = sum a_i^2.
    std::size_t n = numbers.size();
    BQM bqm(n, Vartype::Spin);
    double offset = 0.0;
    for (std::size_t i = 0; i < n; ++i) offset += numbers[i] * numbers[i];
    bqm.add_offset(offset);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            bqm.add_interaction(i, j, 2.0 * numbers[i] * numbers[j]);
        }
    }

    SolveResult result = best_of_seeds(bqm, 5, 999);

    // Recover the actual signed sum from the best state to report the
    // partition difference (should be 0 for a perfect split).
    double signed_sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) signed_sum += numbers[i] * result.best_state[i];

    std::printf("test_number_partition_smoke: |signed_sum| = %.6f\n", std::fabs(signed_sum));
    assert(close(signed_sum, 0.0, 1e-6));

    std::printf("test_number_partition_smoke passed\n");
}

void test_energy_audit() {
    std::mt19937_64 rng(55);
    for (int t = 0; t < 10; ++t) {
        BQM bqm = random_ising(10, rng);
        GeometricSchedule schedule(8.0, 0.98);
        Annealer<GeometricSchedule> annealer(bqm, schedule, 300, static_cast<std::uint64_t>(2000 + t));
        SolveResult result = annealer.solve();

        double recomputed = bqm.energy(result.best_state);
        assert(close(recomputed, result.best_energy, 1e-9));
    }
    std::printf("test_energy_audit passed\n");
}

}  // namespace

int main() {
    test_finds_ground_state_most_of_the_time();
    test_determinism();
    test_number_partition_smoke();
    test_energy_audit();
    std::printf("All annealer tests passed.\n");
    return 0;
}
