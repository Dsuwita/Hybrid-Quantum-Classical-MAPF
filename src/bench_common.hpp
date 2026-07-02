// bench_common.hpp
//
// Shared helpers for the benchmark executables (not part of the library).
#pragma once

#include "anneal/bqm.hpp"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <set>
#include <utility>
#include <vector>

namespace anneal_bench {

// Random 3-regular graph via three random perfect matchings on the vertex
// set: shuffle the vertices, pair them up, repeat 3 times. A matching
// can't create self-loops; if a round duplicates an existing edge, retry
// that round. Simple and good enough for a benchmark instance.
inline anneal::BQM random_3_regular_pm1(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::set<std::pair<std::size_t, std::size_t>> edges;
    std::vector<std::size_t> perm(n);
    std::iota(perm.begin(), perm.end(), std::size_t{0});

    for (int round = 0; round < 3; ++round) {
        for (int attempt = 0; attempt < 1000; ++attempt) {
            std::shuffle(perm.begin(), perm.end(), rng);
            std::set<std::pair<std::size_t, std::size_t>> round_edges;
            bool ok = true;
            for (std::size_t k = 0; k + 1 < n; k += 2) {
                auto e = std::minmax(perm[k], perm[k + 1]);
                std::pair<std::size_t, std::size_t> edge{e.first, e.second};
                if (edges.count(edge)) {
                    ok = false;
                    break;
                }
                round_edges.insert(edge);
            }
            if (ok) {
                edges.insert(round_edges.begin(), round_edges.end());
                break;
            }
        }
    }

    anneal::BQM bqm(n, anneal::Vartype::Spin);
    std::uniform_int_distribution<int> sign(0, 1);
    for (const auto& [u, v] : edges) {
        bqm.add_interaction(u, v, sign(rng) ? 1.0 : -1.0);
    }
    return bqm;
}

}  // namespace anneal_bench
