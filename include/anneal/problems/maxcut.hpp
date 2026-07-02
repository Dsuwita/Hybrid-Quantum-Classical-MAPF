// maxcut.hpp
//
// Max-Cut as an Ising problem, plus the graph plumbing around it. Max-Cut
// asks for a partition of a weighted graph's vertices into two sets that
// maximizes the total weight of edges crossing between the sets. It is the
// canonical benchmark for Ising solvers (and the native problem of D-Wave
// hardware), so it is how we validate the annealer against published
// best-known values.
//
// Mapping (section 3 of the project spec). For each edge (u, v, w) add an
// Ising interaction J_uv = +w with zero linear fields. Then the Ising
// energy is
//     E(s) = sum_edges w * s_u * s_v
// and since s_u s_v is +1 for an uncut edge and -1 for a cut edge,
//     E = W_total - 2 * W_cut   =>   W_cut = (W_total - E) / 2.
// Minimizing energy maximizes the cut. The annealer only ever sees the
// Ising energy; the cut value is recovered by an INDEPENDENT recount over
// the raw edge list (cut_value below), never trusted from the energy, so a
// mapping bug cannot masquerade as a good cut.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "anneal/bqm.hpp"

namespace anneal {

struct Edge {
    std::size_t u;
    std::size_t v;
    double w;
};

struct Graph {
    std::size_t n = 0;
    std::vector<Edge> edges;

    double total_weight() const {
        double t = 0.0;
        for (const auto& e : edges) t += e.w;
        return t;
    }
};

// Build a Spin-vartype BQM whose ground state is a maximum cut of `g`.
inline BQM maxcut_to_bqm(const Graph& g) {
    BQM bqm(g.n, Vartype::Spin);
    for (const auto& e : g.edges) bqm.add_interaction(e.u, e.v, e.w);
    return bqm;
}

// Independent cut verifier: recount the weight of edges whose endpoints
// land in different partitions, straight from the edge list. `state` is a
// spin assignment (+1/-1); the sign splits the two sides.
inline double cut_value(const Graph& g, const std::vector<std::int8_t>& state) {
    double cut = 0.0;
    for (const auto& e : g.edges) {
        if (state[e.u] != state[e.v]) cut += e.w;
    }
    return cut;
}

// Exact maximum cut by brute force (n <= ~24), for tests. Enumerates every
// bipartition and recounts each; the gold-standard oracle.
inline double brute_force_max_cut(const Graph& g) {
    double best = 0.0;
    std::vector<std::int8_t> state(g.n);
    const std::size_t total = std::size_t(1) << g.n;
    for (std::size_t mask = 0; mask < total; ++mask) {
        for (std::size_t i = 0; i < g.n; ++i) state[i] = (mask >> i) & 1u ? 1 : -1;
        best = std::max(best, cut_value(g, state));
    }
    return best;
}

// --- generators ----------------------------------------------------------

// Erdos-Renyi G(n, p): each of the n(n-1)/2 possible edges is present
// independently with probability p, with the given weight (default 1).
inline Graph erdos_renyi(std::size_t n, double p, std::uint64_t seed, double weight = 1.0) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    Graph g;
    g.n = n;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j)
            if (u(rng) < p) g.edges.push_back({i, j, weight});
    return g;
}

// Random d-regular graph via d random perfect matchings (retrying a round
// that would repeat an edge). n*d must be even; d < n.
inline Graph random_d_regular(std::size_t n, std::size_t d, std::uint64_t seed,
                              double weight = 1.0) {
    if ((n * d) % 2 != 0 || d >= n) throw std::runtime_error("random_d_regular: bad n,d");
    std::mt19937_64 rng(seed);
    std::set<std::pair<std::size_t, std::size_t>> edges;
    std::vector<std::size_t> perm(n);
    std::iota(perm.begin(), perm.end(), std::size_t{0});

    for (std::size_t round = 0; round < d; ++round) {
        for (int attempt = 0; attempt < 1000; ++attempt) {
            std::shuffle(perm.begin(), perm.end(), rng);
            std::set<std::pair<std::size_t, std::size_t>> round_edges;
            bool ok = true;
            for (std::size_t k = 0; k + 1 < n; k += 2) {
                auto mm = std::minmax(perm[k], perm[k + 1]);
                std::pair<std::size_t, std::size_t> e{mm.first, mm.second};
                if (edges.count(e)) {
                    ok = false;
                    break;
                }
                round_edges.insert(e);
            }
            if (ok) {
                edges.insert(round_edges.begin(), round_edges.end());
                break;
            }
        }
    }
    Graph g;
    g.n = n;
    for (const auto& [u, v] : edges) g.edges.push_back({u, v, weight});
    return g;
}

// --- Gset benchmark format ------------------------------------------------

// Parse the Gset format: first line "n m", then m lines "u v w" with u, v
// 1-indexed. Returns a 0-indexed Graph.
inline Graph parse_gset(std::istream& in) {
    Graph g;
    std::size_t m = 0;
    if (!(in >> g.n >> m)) throw std::runtime_error("parse_gset: bad header");
    g.edges.reserve(m);
    for (std::size_t k = 0; k < m; ++k) {
        std::size_t u = 0, v = 0;
        double w = 0.0;
        if (!(in >> u >> v >> w)) throw std::runtime_error("parse_gset: bad edge line");
        if (u < 1 || v < 1 || u > g.n || v > g.n)
            throw std::runtime_error("parse_gset: vertex index out of range");
        g.edges.push_back({u - 1, v - 1, w});
    }
    return g;
}

}  // namespace anneal
