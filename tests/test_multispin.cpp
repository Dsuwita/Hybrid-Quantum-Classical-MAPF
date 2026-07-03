// test_multispin.cpp
//
// Milestone 7(b) tests for the multi-spin-coded annealer. The critical test
// is that the bit-sliced per-lane flip arithmetic matches a plain scalar
// recomputation for all 64 replicas on random states (this is the part most
// likely to have a bug). Then the usual project discipline: it reaches the
// brute-force ground state on small instances, and the reported best energy
// is re-audited from the returned state.

#include "anneal/bqm.hpp"
#include "anneal/multispin.hpp"
#include "anneal/problems/maxcut.hpp"
#include "anneal/schedule.hpp"

#include <cassert>
#include <cstdio>
#include <random>
#include <vector>

using namespace anneal;

namespace {

// Build a +/-1 coupling, zero-field Ising BQM from a random graph, flipping
// a fraction of edge signs to -1 so both coupling signs are exercised.
BQM pm1_bqm(std::size_t n, double p, std::uint64_t seed) {
    Graph g = erdos_renyi(n, p, seed);
    std::mt19937_64 rng(seed ^ 0x5555);
    BQM bqm(n, Vartype::Spin);
    for (auto& e : g.edges) {
        double w = (rng() & 1u) ? 1.0 : -1.0;
        bqm.add_interaction(e.u, e.v, w);
    }
    return bqm;
}

void test_bitsliced_k_matches_scalar() {
    // For random packed states, the bit-sliced k_r (unsatisfied-edge count)
    // must equal a direct scalar count for every one of the 64 lanes and
    // every site.
    BQM bqm = pm1_bqm(30, 0.3, 4242);
    GeometricSchedule sched(3.0, 0.99);
    MultiSpinAnnealer<GeometricSchedule> ann(bqm, sched, 1, 1);
    const CompactBQM& v = ann.view();
    const std::size_t n = v.num_variables;

    std::mt19937_64 rng(9);
    std::vector<std::uint64_t> word(n);
    for (int trial = 0; trial < 20; ++trial) {
        for (auto& w : word) w = rng();
        for (std::size_t i = 0; i < n; ++i) {
            std::vector<std::size_t> sliced = ann.lane_k(i, word);
            for (std::size_t r = 0; r < 64; ++r) {
                // scalar k_r: number of neighbours with J s_i s_j = +1.
                const double si = ((word[i] >> r) & 1u) ? 1.0 : -1.0;
                std::size_t k = 0;
                for (std::size_t e = v.row_start[i]; e < v.row_start[i + 1]; ++e) {
                    const double sj = ((word[v.nbr_index[e]] >> r) & 1u) ? 1.0 : -1.0;
                    if (v.nbr_coupling[e] * si * sj > 0.0) ++k;
                }
                assert(sliced[r] == k);
            }
        }
    }
    std::printf("test_bitsliced_k_matches_scalar passed (20 states x 30 sites x 64 lanes)\n");
}

void test_finds_ground_state() {
    // Unweighted Max-Cut (all +1 couplings): 64 lanes should collectively
    // reach the exact brute-force ground energy on small instances.
    int hits = 0;
    const int trials = 15;
    for (int t = 0; t < trials; ++t) {
        Graph g = erdos_renyi(16, 0.35, 1000 + t);
        BQM bqm = maxcut_to_bqm(g);
        const double ground = g.total_weight() - 2.0 * brute_force_max_cut(g);
        GeometricSchedule sched(4.0, 0.995);
        MultiSpinAnnealer<GeometricSchedule> ann(bqm, sched, 400, 7 + t);
        MultiSpinResult r = ann.solve();
        assert(r.best_energy >= ground - 1e-9);  // never below true ground
        if (std::fabs(r.best_energy - ground) < 1e-9) ++hits;
    }
    assert(hits >= trials - 1);
    std::printf("test_finds_ground_state passed (%d/%d exact)\n", hits, trials);
}

void test_energy_audit() {
    Graph g = erdos_renyi(18, 0.4, 55);
    BQM bqm = maxcut_to_bqm(g);
    GeometricSchedule sched(4.0, 0.99);
    MultiSpinAnnealer<GeometricSchedule> ann(bqm, sched, 500, 3);
    MultiSpinResult r = ann.solve();
    assert(bqm.energy(r.best_state) == r.best_energy);
    std::printf("test_energy_audit passed (E=%.1f)\n", r.best_energy);
}

void test_determinism() {
    BQM bqm = pm1_bqm(24, 0.35, 8);
    GeometricSchedule sched(3.0, 0.99);
    MultiSpinAnnealer<GeometricSchedule> a(bqm, sched, 300, 100);
    MultiSpinAnnealer<GeometricSchedule> b(bqm, sched, 300, 100);
    MultiSpinResult ra = a.solve(), rb = b.solve();
    assert(ra.best_energy == rb.best_energy);
    assert(ra.best_state == rb.best_state);
    std::printf("test_determinism passed\n");
}

void test_rejects_bad_problem() {
    // Non +/-1 coupling must be rejected (the arithmetic assumes unit edges).
    BQM bqm(3, Vartype::Spin);
    bqm.add_interaction(0, 1, 2.0);
    bool threw = false;
    try {
        GeometricSchedule sched(3.0, 0.99);
        MultiSpinAnnealer<GeometricSchedule> ann(bqm, sched, 10, 1);
    } catch (const std::exception&) {
        threw = true;
    }
    assert(threw);
    std::printf("test_rejects_bad_problem passed\n");
}

}  // namespace

int main() {
    test_bitsliced_k_matches_scalar();
    test_finds_ground_state();
    test_energy_audit();
    test_determinism();
    test_rejects_bad_problem();
    std::printf("All multi-spin tests passed.\n");
    return 0;
}
