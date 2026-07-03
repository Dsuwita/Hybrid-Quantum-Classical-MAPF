// classical_maxcut.hpp
//
// A purely classical Max-Cut heuristic, to sit next to the annealer as a
// baseline. This is multi-start 1-opt local search (a standard, strong
// classical method for Max-Cut): from a random assignment, repeatedly move
// any vertex to the other side while that increases the cut, until no such
// move exists (a local optimum); do this from many random starts and keep
// the best.
//
// The move gains are cached and updated incrementally, the same trick the
// annealer uses for local fields: the gain of flipping vertex i is
//     gain_i = sum_{j~i} w_ij * (s_i == s_j ? +1 : -1)
// (positive means flipping i moves same-side edges to cross, raising the
// cut). Flipping i negates gain_i and shifts each neighbor's gain by
// -2 w_ij * (s_j == s_i_old ? +1 : -1), so a full local-search step costs
// only the degree of the flipped vertex.
//
// This shares nothing with the annealer's acceptance logic on purpose: it
// is greedy (never accepts a worse move), so comparing the two shows what
// simulated annealing's uphill moves buy over plain hill-climbing.
#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "anneal/problems/maxcut.hpp"

namespace anneal {

struct ClassicalResult {
    std::vector<std::int8_t> state;
    double cut = 0.0;
    std::size_t restarts = 0;
};

// Multi-start 1-opt local search. `restarts` random starts; each is hill-
// climbed to a local optimum. Deterministic given `seed`.
inline ClassicalResult multistart_local_search_maxcut(const Graph& g, std::size_t restarts,
                                                      std::uint64_t seed) {
    const std::size_t n = g.n;

    // CSR adjacency (both directions) for fast neighbor scans.
    std::vector<std::size_t> deg(n, 0);
    for (const auto& e : g.edges) {
        ++deg[e.u];
        ++deg[e.v];
    }
    std::vector<std::size_t> row(n + 1, 0);
    for (std::size_t i = 0; i < n; ++i) row[i + 1] = row[i] + deg[i];
    std::vector<std::size_t> nbr(row[n]);
    std::vector<double> wgt(row[n]);
    std::vector<std::size_t> cursor(row.begin(), row.end() - 1);
    for (const auto& e : g.edges) {
        nbr[cursor[e.u]] = e.v;
        wgt[cursor[e.u]] = e.w;
        ++cursor[e.u];
        nbr[cursor[e.v]] = e.u;
        wgt[cursor[e.v]] = e.w;
        ++cursor[e.v];
    }

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> coin(0, 1);

    ClassicalResult best;
    best.cut = -1.0;
    best.restarts = restarts;

    std::vector<std::int8_t> s(n);
    std::vector<double> gain(n);

    for (std::size_t r = 0; r < restarts; ++r) {
        for (std::size_t i = 0; i < n; ++i) s[i] = coin(rng) ? std::int8_t(1) : std::int8_t(-1);

        // Initial gains and cut.
        double cut = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double gi = 0.0;
            for (std::size_t e = row[i]; e < row[i + 1]; ++e)
                gi += wgt[e] * (s[i] == s[nbr[e]] ? 1.0 : -1.0);
            gain[i] = gi;
        }
        for (const auto& e : g.edges)
            if (s[e.u] != s[e.v]) cut += e.w;

        // Hill-climb: keep sweeping and flipping improving vertices until a
        // full sweep flips nothing (local optimum).
        bool improved = true;
        while (improved) {
            improved = false;
            for (std::size_t i = 0; i < n; ++i) {
                if (gain[i] > 1e-12) {
                    const std::int8_t si_old = s[i];
                    cut += gain[i];
                    for (std::size_t e = row[i]; e < row[i + 1]; ++e) {
                        gain[nbr[e]] -= 2.0 * wgt[e] * (s[nbr[e]] == si_old ? 1.0 : -1.0);
                    }
                    s[i] = static_cast<std::int8_t>(-si_old);
                    gain[i] = -gain[i];
                    improved = true;
                }
            }
        }

        if (cut > best.cut) {
            best.cut = cut;
            best.state = s;
        }
    }
    return best;
}

}  // namespace anneal
