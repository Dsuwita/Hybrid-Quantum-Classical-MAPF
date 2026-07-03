// test_tempering.cpp
//
// Milestone 7(a) tests for parallel tempering. Same discipline as the rest
// of the project: correctness is judged by exhaustive brute force on small
// instances, and the reported best energy is re-audited by recomputing it
// from the state with BQM::energy (the tripwire for incremental-update bugs
// in the field cache and the swap bookkeeping).

#include "anneal/bqm.hpp"
#include "anneal/parallel_tempering.hpp"
#include "anneal/problems/maxcut.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace anneal;

namespace {

void test_finds_ground_state() {
    // On small random Max-Cut instances, parallel tempering should reach the
    // exact brute-force ground energy essentially every time.
    int hits = 0;
    const int trials = 20;
    for (int t = 0; t < trials; ++t) {
        Graph g = erdos_renyi(14, 0.35, 500 + t);
        BQM bqm = maxcut_to_bqm(g);
        double ground = -brute_force_max_cut(g) * 2.0 + g.total_weight() * 0.0;
        // Ising ground energy: cut = (total_w - E)/2 => E = total_w - 2*cut.
        double ground_energy = g.total_weight() - 2.0 * brute_force_max_cut(g);

        auto ladder = geometric_ladder(0.3, 8.0, 8);
        ParallelTempering<> pt(bqm, ladder, 500, 1234 + t);
        TemperingResult r = pt.solve();

        // Never below the true ground (that would be an energy-accounting bug).
        assert(r.best.best_energy >= ground_energy - 1e-9);
        if (std::fabs(r.best.best_energy - ground_energy) < 1e-9) ++hits;
        (void)ground;
    }
    assert(hits >= trials - 1);
    std::printf("test_finds_ground_state passed (%d/%d exact)\n", hits, trials);
}

void test_energy_audit() {
    // The reported best_energy must equal an independent recount of the
    // returned state's energy, exactly.
    Graph g = erdos_renyi(16, 0.4, 99);
    BQM bqm = maxcut_to_bqm(g);
    auto ladder = geometric_ladder(0.2, 6.0, 10);
    ParallelTempering<> pt(bqm, ladder, 800, 7);
    TemperingResult r = pt.solve();
    double recomputed = bqm.energy(r.best.best_state);
    assert(recomputed == r.best.best_energy);
    std::printf("test_energy_audit passed (E=%.1f)\n", r.best.best_energy);
}

void test_determinism() {
    Graph g = erdos_renyi(20, 0.3, 3);
    BQM bqm = maxcut_to_bqm(g);
    auto ladder = geometric_ladder(0.3, 5.0, 8);
    ParallelTempering<> a(bqm, ladder, 400, 42);
    ParallelTempering<> b(bqm, ladder, 400, 42);
    auto ra = a.solve(), rb = b.solve();
    assert(ra.best.best_energy == rb.best.best_energy);
    assert(ra.best.best_state == rb.best.best_state);
    std::printf("test_determinism passed\n");
}

void test_swaps_happen() {
    // With a sensible ladder, adjacent swaps should be accepted a healthy
    // fraction of the time (otherwise the ladder is mis-spaced and tempering
    // reduces to independent fixed-T chains).
    Graph g = erdos_renyi(18, 0.4, 11);
    BQM bqm = maxcut_to_bqm(g);
    auto ladder = geometric_ladder(0.3, 6.0, 8);
    ParallelTempering<> pt(bqm, ladder, 600, 5);
    TemperingResult r = pt.solve();
    double avg = 0.0;
    for (double s : r.swap_rate) avg += s;
    avg /= r.swap_rate.size();
    assert(avg > 0.05);  // some exchange is happening
    std::printf("test_swaps_happen passed (mean swap rate %.2f)\n", avg);
}

}  // namespace

int main() {
    test_finds_ground_state();
    test_energy_audit();
    test_determinism();
    test_swaps_happen();
    std::printf("All parallel-tempering tests passed.\n");
    return 0;
}
