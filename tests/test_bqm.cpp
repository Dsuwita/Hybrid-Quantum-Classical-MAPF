// test_bqm.cpp
//
// Assert-based tests for the BQM container. Exit code 0 means pass.
//
// The gold standard for correctness here is exhaustive enumeration: for
// n <= 8 we can just try all 2^n states and check that vartype conversion
// preserves the energy of every single one, not just a few spot checks.

#include "anneal/bqm.hpp"

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

void test_hand_computed_energy() {
    // Two spins with h_0=1, h_1=-1, J_01=2, offset=0.5
    // state (+1,+1): 0.5 + 1*1 + (-1*1) + 2*1*1 = 0.5 + 1 - 1 + 2 = 2.5
    BQM bqm(2, Vartype::Spin);
    bqm.add_linear(0, 1.0);
    bqm.add_linear(1, -1.0);
    bqm.add_interaction(0, 1, 2.0);
    bqm.add_offset(0.5);

    std::vector<std::int8_t> s{+1, +1};
    assert(close(bqm.energy(s), 2.5));

    s = {+1, -1};
    // 0.5 + 1*1 + (-1*-1) + 2*1*-1 = 0.5 + 1 + 1 - 2 = 0.5
    assert(close(bqm.energy(s), 0.5));

    std::printf("test_hand_computed_energy passed\n");
}

void test_duplicate_edge_accumulation() {
    BQM bqm(2, Vartype::Spin);
    bqm.add_interaction(0, 1, 1.0);
    bqm.add_interaction(0, 1, 2.0);
    bqm.add_interaction(1, 0, 0.5);  // order shouldn't matter
    // total coupling should be 3.5
    std::vector<std::int8_t> s{+1, +1};
    assert(close(bqm.energy(s), 3.5));

    std::printf("test_duplicate_edge_accumulation passed\n");
}

void test_triangle_maxcut_ground_state() {
    // Max-Cut on a triangle: best possible cut is 2 of 3 edges.
    // Ising energy = -(cut edges) + (uncut edges) for unit weights,
    // so ground energy should be -1 (2 cut, 1 uncut: -2+1=-1).
    BQM bqm(3, Vartype::Spin);
    bqm.add_interaction(0, 1, 1.0);
    bqm.add_interaction(1, 2, 1.0);
    bqm.add_interaction(0, 2, 1.0);

    double best = 1e18;
    for (int mask = 0; mask < 8; ++mask) {
        std::vector<std::int8_t> s(3);
        for (int i = 0; i < 3; ++i) s[i] = (mask & (1 << i)) ? +1 : -1;
        best = std::min(best, bqm.energy(s));
    }
    assert(close(best, -1.0));

    std::printf("test_triangle_maxcut_ground_state passed\n");
}

// Enumerate all 2^n states for a BQM of the given vartype and return the
// energies indexed by bitmask (bit i = 1 means spin +1 / bit value 1).
std::vector<double> enumerate_energies(const BQM& bqm, Vartype vartype) {
    std::size_t n = bqm.num_variables();
    std::vector<double> energies(std::size_t(1) << n);
    for (std::size_t mask = 0; mask < (std::size_t(1) << n); ++mask) {
        std::vector<std::int8_t> s(n);
        for (std::size_t i = 0; i < n; ++i) {
            bool bit = (mask >> i) & 1u;
            if (vartype == Vartype::Spin) {
                s[i] = bit ? +1 : -1;
            } else {
                s[i] = bit ? 1 : 0;
            }
        }
        energies[mask] = bqm.energy(s);
    }
    return energies;
}

void test_exhaustive_conversion_preserves_energy() {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> bias_dist(-3.0, 3.0);

    for (int trial = 0; trial < 25; ++trial) {
        std::uniform_int_distribution<int> n_dist(2, 8);
        std::size_t n = static_cast<std::size_t>(n_dist(rng));

        BQM bqm(n, Vartype::Spin);
        for (std::size_t i = 0; i < n; ++i) {
            bqm.add_linear(i, bias_dist(rng));
        }
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                // Not every pair gets an edge; sparsify a bit.
                if (std::uniform_real_distribution<double>(0, 1)(rng) < 0.6) {
                    bqm.add_interaction(i, j, bias_dist(rng));
                }
            }
        }
        bqm.add_offset(bias_dist(rng));

        // Every state's energy under x=(s+1)/2 must match before/after
        // conversion to Binary, since mask bit meaning changes (spin +1
        // corresponds to bit 1, same mask), so we compare mask-by-mask.
        std::vector<double> spin_energies = enumerate_energies(bqm, Vartype::Spin);

        BQM as_binary = bqm;
        as_binary.change_vartype(Vartype::Binary);
        std::vector<double> binary_energies = enumerate_energies(as_binary, Vartype::Binary);

        for (std::size_t mask = 0; mask < spin_energies.size(); ++mask) {
            assert(close(spin_energies[mask], binary_energies[mask], 1e-9));
        }

        // Round trip back to Spin should reproduce the original energies.
        BQM round_trip = as_binary;
        round_trip.change_vartype(Vartype::Spin);
        std::vector<double> round_trip_energies = enumerate_energies(round_trip, Vartype::Spin);
        for (std::size_t mask = 0; mask < spin_energies.size(); ++mask) {
            assert(close(spin_energies[mask], round_trip_energies[mask], 1e-9));
        }
    }

    std::printf("test_exhaustive_conversion_preserves_energy passed (25 random instances)\n");
}

}  // namespace

int main() {
    test_hand_computed_energy();
    test_duplicate_edge_accumulation();
    test_triangle_maxcut_ground_state();
    test_exhaustive_conversion_preserves_energy();
    std::printf("All BQM tests passed.\n");
    return 0;
}
