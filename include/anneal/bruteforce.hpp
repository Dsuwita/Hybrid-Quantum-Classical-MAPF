// bruteforce.hpp
//
// Exact ground-state finder by exhaustive enumeration of all 2^n states.
// This is the "gold standard" correctness oracle used throughout the
// project: anything an approximate solver claims can be checked against
// this for small n. Only usable up to n ~ 24 (2^24 = ~16M states); beyond
// that it becomes too slow to be a practical test oracle.
#pragma once

#include "anneal/bqm.hpp"

#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>

namespace anneal {

struct BruteForceResult {
    double ground_energy;
    std::vector<std::int8_t> ground_state;
};

// Enumerates every assignment of the BQM's variables and returns the one
// with minimum energy (ties broken by first occurrence in mask order).
inline BruteForceResult brute_force_ground_state(const BQM& bqm) {
    std::size_t n = bqm.num_variables();
    assert(n <= 24 && "brute force enumeration only practical up to n=24");

    std::int8_t low = (bqm.vartype() == Vartype::Spin) ? -1 : 0;
    std::int8_t high = 1;

    BruteForceResult best;
    best.ground_energy = std::numeric_limits<double>::infinity();
    best.ground_state.resize(n);

    std::size_t total = std::size_t(1) << n;
    std::vector<std::int8_t> state(n);
    for (std::size_t mask = 0; mask < total; ++mask) {
        for (std::size_t i = 0; i < n; ++i) {
            bool bit = (mask >> i) & 1u;
            state[i] = bit ? high : low;
        }
        double e = bqm.energy(state);
        if (e < best.ground_energy) {
            best.ground_energy = e;
            best.ground_state = state;
        }
    }
    return best;
}

}  // namespace anneal
