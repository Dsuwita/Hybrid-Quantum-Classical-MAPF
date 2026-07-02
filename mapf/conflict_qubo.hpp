// conflict_qubo.hpp
//
// The hybrid step: classical A* proposed a menu of candidate paths per
// agent (path_gen.hpp); here we turn "pick one path per agent, avoiding
// collisions, minimizing travel" into a QUBO for the anneal library to
// solve. This is the interface between project 2 (MAPF) and project 1
// (the annealer).
//
// Variables. One binary variable x_p per candidate path p. x_p = 1 means
// "agent a walks path p". A valid selection sets exactly one variable to
// 1 per agent.
//
// Objective (minimize). Three parts:
//
//   1. Travel cost.   sum_p cost_p * x_p
//      Cheaper paths lower the energy, so among conflict-free selections
//      the annealer prefers the shortest total travel.
//
//   2. Conflict penalty.   P * x_p * x_q  for every pair (p, q) of paths
//      from DIFFERENT agents that collide when replayed in lockstep.
//      Selecting both raises the energy by P, so collisions are pushed
//      out of the ground state.
//
//   3. One-hot per agent.   P1 * (sum_{p in a} x_p - 1)^2  for each agent
//      a. This is minimized (at 0) exactly when the agent picks one path.
//      Expanding, with x^2 = x for binary variables:
//        (S - 1)^2 = -sum_p x_p + 2 sum_{p<q} x_p x_q + 1
//      so each variable in the agent gets linear bias -P1, each within-
//      agent pair gets coupling +2 P1, and a constant +P1 goes to the
//      offset. Different agents' one-hot terms never share a variable.
//
// Penalty weights. The constraints must outweigh any travel-cost saving a
// violation could buy, or the "cheapest" state would cheat by dropping an
// agent or ignoring a collision. Choosing P = P1 = (sum over agents of
// that agent's most expensive candidate) + 1 guarantees this: it exceeds
// the largest possible total travel cost, so no cost saving can ever pay
// for a single constraint violation, and the ground state is a valid,
// conflict-free, minimum-cost selection whenever one exists. That is the
// safe DEFAULT; callers may pass smaller weights.
//
// Dynamic-range tradeoff. Bigger penalties make the formulation "more
// correct" but harder to anneal: the energy landscape develops tall
// walls between feasible states, and the acceptance-probability dynamic
// range widens, so the annealer needs lower temperatures / more sweeps to
// cross them. src/mapf_penalty_experiment.cpp measures this; results are
// in mapf/bench/penalty_experiment.md. The takeaway: use the smallest
// penalty that still makes feasible selections win, not the largest you
// can write down.
#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "anneal/bqm.hpp"
#include "mapf/grid.hpp"
#include "mapf/path_gen.hpp"

namespace mapf {

// Do two candidate paths (from different agents) collide? Replay them in
// lockstep over their common horizon, parking each on its final cell once
// its path ends (position_at semantics from plan.hpp). Vertex conflict:
// same cell at the same time. Swap conflict: the two exchange cells across
// one step. Same logic as the verifier's pairwise loop, specialized to a
// single pair of raw paths.
inline bool paths_conflict(const std::vector<Cell>& a, const std::vector<Cell>& b) {
    if (a.empty() || b.empty()) return false;
    auto at = [](const std::vector<Cell>& p, std::size_t t) {
        return t < p.size() ? p[t] : p.back();
    };
    const std::size_t horizon = std::max(a.size(), b.size()) - 1;
    for (std::size_t t = 0; t <= horizon; ++t) {
        Cell pa = at(a, t), pb = at(b, t);
        if (pa == pb) return true;  // vertex conflict
        if (t > 0) {
            Cell pa_prev = at(a, t - 1), pb_prev = at(b, t - 1);
            if (pa == pb_prev && pb == pa_prev) return true;  // swap conflict
        }
    }
    return false;
}

// A built QUBO plus the bookkeeping needed to read a solution back out.
struct SelectionQubo {
    anneal::BQM bqm{0, anneal::Vartype::Binary};

    // var_index[a][i] is the global variable index of agent a's candidate
    // i; var_owner[v] = {agent, candidate} is the inverse.
    std::vector<std::vector<std::size_t>> var_index;
    std::vector<std::pair<std::size_t, std::size_t>> var_owner;

    double penalty_conflict = 0.0;
    double penalty_onehot = 0.0;
    std::size_t num_conflict_edges = 0;
};

// The safe default penalty (see the header comment): larger than any
// achievable total travel cost.
inline double default_penalty(const std::vector<std::vector<Candidate>>& pools) {
    double sum_of_max = 0.0;
    for (const auto& agent_pool : pools) {
        double agent_max = 0.0;
        for (const auto& c : agent_pool) agent_max = std::max(agent_max, c.cost);
        sum_of_max += agent_max;
    }
    return sum_of_max + 1.0;
}

// Build the path-selection QUBO. pools[a] is agent a's candidate menu.
// P is the conflict penalty, P1 the one-hot penalty; pass <= 0 to use the
// safe default for that weight.
inline SelectionQubo build_selection_qubo(const std::vector<std::vector<Candidate>>& pools,
                                          double P = -1.0, double P1 = -1.0) {
    const double penalty = (P > 0.0) ? P : default_penalty(pools);
    const double onehot = (P1 > 0.0) ? P1 : default_penalty(pools);

    SelectionQubo model;
    model.penalty_conflict = penalty;
    model.penalty_onehot = onehot;

    // Assign a global variable index to every candidate.
    std::size_t num_vars = 0;
    model.var_index.resize(pools.size());
    for (std::size_t a = 0; a < pools.size(); ++a) {
        model.var_index[a].resize(pools[a].size());
        for (std::size_t i = 0; i < pools[a].size(); ++i) {
            model.var_index[a][i] = num_vars;
            model.var_owner.push_back({a, i});
            ++num_vars;
        }
    }

    anneal::BQM bqm(num_vars, anneal::Vartype::Binary);

    // (1) travel cost + (3) one-hot linear/quadratic/offset.
    for (std::size_t a = 0; a < pools.size(); ++a) {
        const auto& vars = model.var_index[a];
        for (std::size_t i = 0; i < vars.size(); ++i) {
            bqm.add_linear(vars[i], pools[a][i].cost - onehot);  // cost, plus -P1 from one-hot
            for (std::size_t j = i + 1; j < vars.size(); ++j) {
                bqm.add_interaction(vars[i], vars[j], 2.0 * onehot);
            }
        }
        if (!vars.empty()) bqm.add_offset(onehot);  // +P1 constant per agent
    }

    // (2) conflict penalties across pairs of candidates from different
    // agents. Same-agent pairs are excluded (one-hot already forbids
    // selecting two of them).
    for (std::size_t a = 0; a < pools.size(); ++a) {
        for (std::size_t b = a + 1; b < pools.size(); ++b) {
            for (std::size_t i = 0; i < pools[a].size(); ++i) {
                for (std::size_t j = 0; j < pools[b].size(); ++j) {
                    if (paths_conflict(pools[a][i].path, pools[b][j].path)) {
                        bqm.add_interaction(model.var_index[a][i], model.var_index[b][j], penalty);
                        ++model.num_conflict_edges;
                    }
                }
            }
        }
    }

    model.bqm = std::move(bqm);
    return model;
}

// Result of reading a binary solution back into a per-agent path choice.
struct Decoded {
    std::vector<std::size_t> chosen;  // chosen[a] = candidate index for agent a
    bool onehot_satisfied = true;     // was every agent marked exactly once?
};

// Decode a binary state (0/1 per variable, e.g. from the annealer or from
// brute force on a Binary BQM) into one path per agent. If an agent's
// one-hot is violated (zero or several marked), repair by taking its
// cheapest MARKED candidate; if none is marked, fall back to its cheapest
// candidate overall so the agent always has a path.
inline Decoded decode_selection(const SelectionQubo& model,
                                const std::vector<std::vector<Candidate>>& pools,
                                const std::vector<std::int8_t>& state) {
    Decoded out;
    out.chosen.resize(pools.size());

    for (std::size_t a = 0; a < pools.size(); ++a) {
        const auto& vars = model.var_index[a];
        std::size_t marked_count = 0;
        std::size_t best_marked = 0;
        double best_marked_cost = std::numeric_limits<double>::infinity();
        std::size_t cheapest = 0;
        double cheapest_cost = std::numeric_limits<double>::infinity();

        for (std::size_t i = 0; i < vars.size(); ++i) {
            const double cost = pools[a][i].cost;
            if (cost < cheapest_cost) {
                cheapest_cost = cost;
                cheapest = i;
            }
            if (state[vars[i]] == 1) {
                ++marked_count;
                if (cost < best_marked_cost) {
                    best_marked_cost = cost;
                    best_marked = i;
                }
            }
        }

        if (marked_count != 1) out.onehot_satisfied = false;
        out.chosen[a] = (marked_count >= 1) ? best_marked : cheapest;
    }
    return out;
}

}  // namespace mapf
