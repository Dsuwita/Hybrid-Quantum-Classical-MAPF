// test_fast_annealer.cpp
//
// Milestone 3 tests. The naive Annealer's suite (test_annealer.cpp) still
// runs unchanged; this file checks that the optimized FastAnnealer is
// (a) just as correct against the brute-force oracle, and (b) makes
// bit-identical accept/reject decisions to the naive implementation on
// the same seed — the differential test that pins the optimizations to
// the slow-but-obvious baseline.

#include "anneal/annealer.hpp"
#include "anneal/bqm.hpp"
#include "anneal/bruteforce.hpp"
#include "anneal/fast_annealer.hpp"
#include "anneal/rng.hpp"
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

// Integer-valued instance: exercises the level-4 acceptance lookup table,
// and makes the naive-vs-fast comparison exact (integer arithmetic in
// doubles has no rounding, so cached local fields match recomputed ones
// bit for bit).
BQM random_integer_ising(std::size_t n, std::mt19937_64& rng) {
    std::uniform_int_distribution<int> bias(-5, 5);
    BQM bqm(n, Vartype::Spin);
    for (std::size_t i = 0; i < n; ++i) bqm.add_linear(i, bias(rng));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            if (std::uniform_real_distribution<double>(0, 1)(rng) < 0.5) {
                int c = bias(rng);
                if (c != 0) bqm.add_interaction(i, j, c);
            }
        }
    }
    return bqm;
}

void test_finds_ground_state() {
    std::mt19937_64 rng(321);
    int hits = 0;
    const int trials = 30;

    for (int t = 0; t < trials; ++t) {
        std::uniform_int_distribution<int> n_dist(8, 16);
        std::size_t n = static_cast<std::size_t>(n_dist(rng));
        BQM bqm = random_ising(n, rng);

        BruteForceResult exact = brute_force_ground_state(bqm);

        double best = std::numeric_limits<double>::infinity();
        for (int s = 0; s < 5; ++s) {
            GeometricSchedule schedule(10.0, 0.99);
            FastAnnealer<GeometricSchedule> annealer(bqm, schedule, 2000,
                                                     static_cast<std::uint64_t>(4000 + t * 10 + s));
            best = std::min(best, annealer.solve().best_energy);
        }

        assert(best >= exact.ground_energy - 1e-9);
        if (close(best, exact.ground_energy, 1e-6)) ++hits;
    }

    std::printf("test_finds_ground_state: %d/%d exact hits\n", hits, trials);
    assert(hits >= 28);
    std::printf("test_finds_ground_state passed\n");
}

void test_determinism() {
    std::mt19937_64 rng(9);
    BQM bqm = random_ising(12, rng);

    GeometricSchedule schedule(10.0, 0.99);
    FastAnnealer<GeometricSchedule> a1(bqm, schedule, 500, 77);
    FastAnnealer<GeometricSchedule> a2(bqm, schedule, 500, 77);

    SolveResult r1 = a1.solve();
    SolveResult r2 = a2.solve();
    assert(r1.best_energy == r2.best_energy);
    assert(r1.best_state == r2.best_state);
    assert(r1.final_state == r2.final_state);

    std::printf("test_determinism passed\n");
}

void test_energy_audit() {
    std::mt19937_64 rng(66);
    for (int t = 0; t < 10; ++t) {
        BQM bqm = random_ising(10, rng);
        GeometricSchedule schedule(8.0, 0.98);
        FastAnnealer<GeometricSchedule> annealer(bqm, schedule, 300,
                                                 static_cast<std::uint64_t>(3000 + t));
        SolveResult result = annealer.solve();
        double recomputed = bqm.energy(result.best_state);
        assert(close(recomputed, result.best_energy, 1e-9));
    }
    std::printf("test_energy_audit passed\n");
}

// The differential test: same seed, same RNG type -> the fully optimized
// path (level 4) must make the exact same accept/reject decision on every
// proposal as the naive Annealer. Any divergence means an optimization
// changed the algorithm, not just its speed.
void test_differential_naive_vs_fast() {
    std::mt19937_64 rng(1234);

    // 5 integer instances (lookup-table path, exact arithmetic).
    for (int t = 0; t < 5; ++t) {
        BQM bqm = random_integer_ising(12, rng);
        GeometricSchedule schedule(10.0, 0.99);
        std::uint64_t seed = static_cast<std::uint64_t>(5000 + t);

        std::vector<std::uint8_t> naive_log, fast_log;
        Annealer<GeometricSchedule> naive(bqm, schedule, 500, seed);
        FastAnnealer<GeometricSchedule> fast(bqm, schedule, 500, seed);

        SolveResult rn = naive.solve(&naive_log);
        SolveResult rf = fast.solve(&fast_log);

        assert(naive_log == fast_log);
        assert(rn.final_state == rf.final_state);
        assert(rn.best_state == rf.best_state);
        assert(rn.best_energy == rf.best_energy);  // exact: integer arithmetic
    }

    // 2 float instances (exp fallback path). Decisions still match because
    // both paths draw the uniform only on uphill proposals; energies match
    // to rounding (the field cache accumulates in a different order).
    for (int t = 0; t < 2; ++t) {
        BQM bqm = random_ising(12, rng);
        GeometricSchedule schedule(10.0, 0.99);
        std::uint64_t seed = static_cast<std::uint64_t>(6000 + t);

        std::vector<std::uint8_t> naive_log, fast_log;
        Annealer<GeometricSchedule> naive(bqm, schedule, 500, seed);
        FastAnnealer<GeometricSchedule> fast(bqm, schedule, 500, seed);

        SolveResult rn = naive.solve(&naive_log);
        SolveResult rf = fast.solve(&fast_log);

        assert(naive_log == fast_log);
        assert(rn.final_state == rf.final_state);
        assert(rn.best_state == rf.best_state);
        assert(close(rn.best_energy, rf.best_energy, 1e-9));
    }

    std::printf("test_differential_naive_vs_fast passed (5 integer + 2 float instances)\n");
}

// All optimization levels are the same Metropolis algorithm; levels 1-2
// consume the RNG differently (they draw on every proposal) but must
// still pass the brute-force check and the energy audit.
void test_all_levels_correct() {
    std::mt19937_64 rng(2718);
    BQM bqm = random_integer_ising(12, rng);
    BruteForceResult exact = brute_force_ground_state(bqm);

    auto check = [&](double best_energy, const std::vector<std::int8_t>& best_state) {
        assert(best_energy >= exact.ground_energy - 1e-9);
        assert(close(bqm.energy(best_state), best_energy, 1e-9));
    };

    GeometricSchedule schedule(10.0, 0.99);
    {
        FastAnnealer<GeometricSchedule, std::mt19937_64, 1> a(bqm, schedule, 1000, 1);
        SolveResult r = a.solve();
        check(r.best_energy, r.best_state);
    }
    {
        FastAnnealer<GeometricSchedule, std::mt19937_64, 2> a(bqm, schedule, 1000, 1);
        SolveResult r = a.solve();
        check(r.best_energy, r.best_state);
    }
    {
        FastAnnealer<GeometricSchedule, std::mt19937_64, 3> a(bqm, schedule, 1000, 1);
        SolveResult r = a.solve();
        check(r.best_energy, r.best_state);
    }
    {
        FastAnnealer<GeometricSchedule, std::mt19937_64, 4> a(bqm, schedule, 1000, 1);
        SolveResult r = a.solve();
        check(r.best_energy, r.best_state);
    }
    std::printf("test_all_levels_correct passed\n");
}

// The xoshiro256++ RNG behind the same interface: deterministic, audited.
void test_xoshiro_variant() {
    std::mt19937_64 rng(31337);
    BQM bqm = random_integer_ising(12, rng);

    GeometricSchedule schedule(10.0, 0.99);
    FastAnnealer<GeometricSchedule, Xoshiro256pp> a1(bqm, schedule, 1000, 55);
    FastAnnealer<GeometricSchedule, Xoshiro256pp> a2(bqm, schedule, 1000, 55);

    SolveResult r1 = a1.solve();
    SolveResult r2 = a2.solve();
    assert(r1.best_energy == r2.best_energy);
    assert(r1.best_state == r2.best_state);
    assert(close(bqm.energy(r1.best_state), r1.best_energy, 1e-9));

    std::printf("test_xoshiro_variant passed\n");
}

}  // namespace

int main() {
    test_finds_ground_state();
    test_determinism();
    test_energy_audit();
    test_differential_naive_vs_fast();
    test_all_levels_correct();
    test_xoshiro_variant();
    std::printf("All fast annealer tests passed.\n");
    return 0;
}
