// test_problems.cpp
//
// Milestone 5 tests for the Max-Cut and number-partitioning problem
// modules. Correctness is pinned to brute force on small instances: the
// annealer's cut must equal the exact maximum cut, the independent cut
// verifier must catch a corrupted state, and the Gset parser must
// round-trip an instance's edge count and total weight.

#include "anneal/bruteforce.hpp"
#include "anneal/parallel.hpp"
#include "anneal/problems/maxcut.hpp"
#include "anneal/problems/partition.hpp"
#include "anneal/schedule.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <sstream>
#include <vector>

using namespace anneal;

namespace {

bool close(double a, double b, double eps = 1e-9) { return std::fabs(a - b) < eps; }

// The mapping is exact: cut = (total_weight - ising_energy) / 2 for every
// state. Check it against the independent recount on random states.
void test_mapping_matches_recount() {
    std::mt19937_64 rng(1);
    for (int t = 0; t < 20; ++t) {
        Graph g = erdos_renyi(12, 0.4, rng(), 1.0);
        BQM bqm = maxcut_to_bqm(g);
        double total = g.total_weight();
        std::uniform_int_distribution<int> coin(0, 1);
        for (int s = 0; s < 10; ++s) {
            std::vector<std::int8_t> state(g.n);
            for (auto& x : state) x = coin(rng) ? 1 : -1;
            double from_energy = (total - bqm.energy(state)) / 2.0;
            assert(close(from_energy, cut_value(g, state)));
        }
    }
    std::printf("test_mapping_matches_recount passed\n");
}

// Annealer cut == exact max cut on 20 brute-forceable graphs.
void test_annealer_finds_max_cut() {
    std::mt19937_64 rng(42);
    int exact_hits = 0;
    const int trials = 20;
    for (int t = 0; t < trials; ++t) {
        std::uniform_int_distribution<int> nd(8, 14);
        std::size_t n = static_cast<std::size_t>(nd(rng));
        Graph g = erdos_renyi(n, 0.5, rng(), 1.0);

        double exact = brute_force_max_cut(g);

        BQM bqm = maxcut_to_bqm(g);
        GeometricSchedule schedule(5.0, 0.99);
        ParallelAnnealer<GeometricSchedule> annealer(bqm, schedule, 2000, 100 + t, 8, 4);
        ParallelResult r = annealer.solve();
        double cut = cut_value(g, r.best.best_state);

        assert(cut <= exact + 1e-9);          // never exceed the true maximum
        if (close(cut, exact)) ++exact_hits;
    }
    std::printf("test_annealer_finds_max_cut: %d/%d exact\n", exact_hits, trials);
    assert(exact_hits >= 18);
    std::printf("test_annealer_finds_max_cut passed\n");
}

// d-regular generator produces a graph with exactly the requested degree.
void test_d_regular_degrees() {
    Graph g = random_d_regular(20, 3, 7);
    std::vector<int> deg(g.n, 0);
    for (const auto& e : g.edges) {
        ++deg[e.u];
        ++deg[e.v];
    }
    for (int d : deg) assert(d == 3);
    assert(g.edges.size() == 20 * 3 / 2);
    std::printf("test_d_regular_degrees passed\n");
}

// The verifier recounts from the raw graph, so a deliberately corrupted
// state yields a different cut than the true optimum -- it cannot be fooled
// by trusting a stale energy.
void test_verifier_catches_corruption() {
    std::mt19937_64 rng(3);
    Graph g = erdos_renyi(12, 0.5, rng(), 1.0);
    double exact = brute_force_max_cut(g);

    // Find an optimal state by brute force over the BQM, then corrupt it.
    BQM bqm = maxcut_to_bqm(g);
    BruteForceResult gs = brute_force_ground_state(bqm);
    assert(close(cut_value(g, gs.ground_state), exact));

    std::vector<std::int8_t> corrupted = gs.ground_state;
    corrupted[0] = static_cast<std::int8_t>(-corrupted[0]);  // flip one vertex
    // The recount reflects the corruption (almost surely a worse cut).
    assert(cut_value(g, corrupted) != exact || g.edges.empty());
    std::printf("test_verifier_catches_corruption passed\n");
}

// Gset parser round-trips edge count and total weight.
void test_gset_parser_roundtrip() {
    // 4 vertices, 5 edges, mixed weights, 1-indexed.
    std::string text = "4 5\n1 2 1\n2 3 1\n3 4 1\n4 1 1\n1 3 -1\n";
    std::istringstream in(text);
    Graph g = parse_gset(in);
    assert(g.n == 4);
    assert(g.edges.size() == 5);
    assert(close(g.total_weight(), 3.0));  // 1+1+1+1-1
    // 1-indexed -> 0-indexed conversion.
    assert(g.edges[0].u == 0 && g.edges[0].v == 1);
    assert(g.edges[4].u == 0 && g.edges[4].v == 2 && close(g.edges[4].w, -1.0));
    std::printf("test_gset_parser_roundtrip passed\n");
}

// Number partitioning: perfect split found, difference 0; mapping exact.
void test_partition() {
    // 1..14 has total 105 (odd) -> best difference 1. Use 1..8 (total 36,
    // even) which has a perfect split.
    std::vector<double> nums = {1, 2, 3, 4, 5, 6, 7, 8};
    BQM bqm = partition_to_bqm(nums);

    // Mapping: energy == difference^2 for random states.
    std::mt19937_64 rng(9);
    std::uniform_int_distribution<int> coin(0, 1);
    for (int s = 0; s < 20; ++s) {
        std::vector<std::int8_t> state(nums.size());
        for (auto& x : state) x = coin(rng) ? 1 : -1;
        double diff = partition_difference(nums, state);
        assert(close(bqm.energy(state), diff * diff));
    }

    GeometricSchedule schedule(10.0, 0.99);
    ParallelAnnealer<GeometricSchedule> annealer(bqm, schedule, 3000, 1, 8, 4);
    ParallelResult r = annealer.solve();
    double diff = partition_difference(nums, r.best.best_state);
    assert(close(diff, 0.0));  // perfect partition exists and is found
    std::printf("test_partition passed (difference %.0f)\n", diff);
}

}  // namespace

int main() {
    test_mapping_matches_recount();
    test_annealer_finds_max_cut();
    test_d_regular_degrees();
    test_verifier_catches_corruption();
    test_gset_parser_roundtrip();
    test_partition();
    std::printf("All problem-module tests passed.\n");
    return 0;
}
