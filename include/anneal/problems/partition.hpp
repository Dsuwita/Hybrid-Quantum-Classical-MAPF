// partition.hpp
//
// Number partitioning as an Ising problem. Given numbers a_0..a_{n-1},
// split them into two groups so the two group sums are as close as
// possible. Assigning spin s_i = +/-1 to put a_i in one group or the
// other, the signed sum sum_i a_i s_i is the difference between the two
// group totals, so we minimize its square:
//
//     E(s) = (sum_i a_i s_i)^2
//          = sum_i a_i^2 + 2 sum_{i<j} a_i a_j s_i s_j.
//
// That is an Ising model with J_ij = 2 a_i a_j and a constant offset
// sum_i a_i^2 (no linear fields). Its ground energy is the squared
// difference of the best split; energy 0 means a perfect partition
// exists and was found.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "anneal/bqm.hpp"

namespace anneal {

// Build a Spin-vartype BQM whose ground state is a best two-way partition
// of `numbers`.
inline BQM partition_to_bqm(const std::vector<double>& numbers) {
    const std::size_t n = numbers.size();
    BQM bqm(n, Vartype::Spin);
    double offset = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        offset += numbers[i] * numbers[i];
        for (std::size_t j = i + 1; j < n; ++j) {
            bqm.add_interaction(i, j, 2.0 * numbers[i] * numbers[j]);
        }
    }
    bqm.add_offset(offset);
    return bqm;
}

// Independent objective recount: the absolute difference between the two
// group sums for a given spin assignment. Equals sqrt(energy) of the BQM
// above, but recomputed straight from the numbers so a mapping bug cannot
// hide.
inline double partition_difference(const std::vector<double>& numbers,
                                   const std::vector<std::int8_t>& state) {
    double signed_sum = 0.0;
    for (std::size_t i = 0; i < numbers.size(); ++i) {
        signed_sum += numbers[i] * static_cast<double>(state[i]);
    }
    return std::fabs(signed_sum);
}

}  // namespace anneal
