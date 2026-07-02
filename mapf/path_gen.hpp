// path_gen.hpp
//
// Time-expanded A* candidate path generation. For each agent we produce a
// MENU of K diverse candidate paths; a later stage (Milestone 10) builds
// a QUBO that picks one path per agent so that the chosen paths do not
// conflict. Diversity in the menu is what gives that selection room to
// find a conflict-free combination.
//
// Why time-expanded (search over (cell, time) instead of just cell):
// waiting is a legal move, and later milestones forbid specific cells at
// specific TIMES (reservation-style, to route around observed conflicts).
// Representing time in the search state is what makes "wait here for one
// step, then go" expressible. In this milestone there are no other
// agents, so waiting never pays off and the first candidate is a genuine
// shortest path; the machinery is built now because M11 needs it.
//
// Diversity comes from two mechanisms, per the spec:
//   1. Randomized tie-breaking: among equally good A* nodes, expansion
//      order is randomized (seeded), so when several shortest paths exist
//      we get different ones on different draws.
//   2. Soft cell penalties: after each candidate, the interior cells it
//      used get a penalty added to their entry cost, so the next A* run
//      is nudged to route around them. Penalties are soft (a bias, not a
//      ban), so a cell is still used if avoiding it is too expensive.
//
// The heuristic is Manhattan distance. It is admissible and consistent
// for 4-connected moves with unit base cost and non-negative penalties
// (h(c) - h(c') <= 1 <= step cost), so A* returns an optimal-cost path
// and the first candidate's cost equals the shortest path length (which
// on an open map equals the scenario's published optimal distance).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <queue>
#include <random>
#include <set>
#include <unordered_map>
#include <vector>

#include "mapf/grid.hpp"

namespace mapf {

struct Candidate {
    std::vector<Cell> path;  // path[0] == start, path.back() == goal
    double cost = 0.0;       // arrival time = number of moves = path.size() - 1
};

namespace detail {

// Time-expanded A* from start to goal. `used[y*W + x]` is how many prior
// candidates passed through cell (x,y); entering a cell costs
// 1 + penalty_weight * used-count + reservation[cell]. `reservation` (if
// non-null, sized W*H) is an externally supplied per-cell extra cost used
// by the solver to route agents around cells that caused conflicts in an
// earlier round (reservation-style replanning). `dynamic_blocked` (if
// non-null) forbids entering cell idx at timestep t when
// dynamic_blocked[t][idx] is set: this is how predicted moving-obstacle
// occupancy is kept out of a path. Because the search is time-expanded,
// an agent can simply wait for a moving obstacle to pass. Returns the
// unpadded cell sequence (ending the timestep it reaches goal) or nullopt
// if goal is unreachable within `horizon` timesteps.
inline std::optional<std::vector<Cell>> astar_timed(
    const Grid& grid, Cell start, Cell goal, const std::vector<int>& used, double penalty_weight,
    std::size_t horizon, std::mt19937_64& rng, const std::vector<double>* reservation,
    const std::vector<std::vector<char>>* dynamic_blocked = nullptr) {
    const int W = grid.width();
    const int H = grid.height();
    const std::uint64_t layer = static_cast<std::uint64_t>(W) * H;  // cells per timestep

    // (cell, t) packed into one 64-bit id. Decoding drops t (id % layer),
    // which is all reconstruction needs.
    auto encode = [&](Cell c, std::size_t t) -> std::uint64_t {
        return static_cast<std::uint64_t>(t) * layer +
               static_cast<std::uint64_t>(c.y) * W + c.x;
    };
    auto heuristic = [&](Cell c) -> double {
        return static_cast<double>(std::abs(c.x - goal.x) + std::abs(c.y - goal.y));
    };

    std::uniform_real_distribution<double> jitter(0.0, 1.0);

    struct Node {
        double f;         // g + heuristic
        double tiebreak;  // random secondary key -> randomized tie-breaking
        double g;
        std::uint64_t id;
        Cell cell;
        std::size_t t;
    };
    auto worse = [](const Node& a, const Node& b) {
        return a.f > b.f || (a.f == b.f && a.tiebreak > b.tiebreak);
    };
    std::priority_queue<Node, std::vector<Node>, decltype(worse)> open(worse);
    std::unordered_map<std::uint64_t, double> best_g;
    std::unordered_map<std::uint64_t, std::uint64_t> came_from;

    const std::uint64_t start_id = encode(start, 0);
    best_g[start_id] = 0.0;
    open.push({heuristic(start), jitter(rng), 0.0, start_id, start, 0});

    while (!open.empty()) {
        Node cur = open.top();
        open.pop();

        // Skip entries superseded by a cheaper path to the same state.
        auto g_it = best_g.find(cur.id);
        if (g_it != best_g.end() && cur.g > g_it->second) continue;

        if (cur.cell == goal) {
            std::vector<Cell> path;
            for (std::uint64_t id = cur.id;;) {
                std::uint64_t within = id % layer;
                path.push_back(Cell{static_cast<int>(within % W), static_cast<int>(within / W)});
                auto p = came_from.find(id);
                if (p == came_from.end()) break;
                id = p->second;
            }
            std::reverse(path.begin(), path.end());
            return path;
        }

        if (cur.t >= horizon) continue;  // out of time to expand further

        for (Cell nb : grid.moves_from(cur.cell)) {
            const std::size_t nt = cur.t + 1;
            const std::size_t idx = static_cast<std::size_t>(nb.y) * W + nb.x;
            // A predicted obstacle occupies this cell at this timestep:
            // treat it as blocked for that instant only.
            if (dynamic_blocked && nt < dynamic_blocked->size() && (*dynamic_blocked)[nt][idx])
                continue;
            const double res = reservation ? (*reservation)[idx] : 0.0;
            const double step = 1.0 + penalty_weight * static_cast<double>(used[idx]) + res;
            const double ng = cur.g + step;
            const std::uint64_t nid = encode(nb, nt);
            auto it = best_g.find(nid);
            if (it == best_g.end() || ng < it->second) {
                best_g[nid] = ng;
                came_from[nid] = cur.id;
                open.push({ng + heuristic(nb), jitter(rng), ng, nid, nb, nt});
            }
        }
    }
    return std::nullopt;
}

}  // namespace detail

// Generate up to K distinct candidate paths from start to goal. The first
// is a shortest path (when there are no reservations); the rest are
// diversified by penalizing cells used by earlier candidates (with
// randomized tie-breaking throughout). Fewer than K are returned if the
// map simply does not offer K distinct routes within the horizon.
// horizon == 0 selects a generous default. `reservations` (if non-null,
// sized W*H) biases every candidate away from the listed cells, used by
// the solver to avoid cells that caused conflicts last round.
// `dynamic_blocked` (if non-null) forbids predicted moving-obstacle cells
// per timestep (see astar_timed).
inline std::vector<Candidate> generate_candidates(
    const Grid& grid, Cell start, Cell goal, std::size_t k, std::uint64_t seed,
    double penalty_weight = 1.0, std::size_t horizon = 0,
    const std::vector<double>* reservations = nullptr,
    const std::vector<std::vector<char>>* dynamic_blocked = nullptr) {
    if (horizon == 0) {
        horizon = 4 * static_cast<std::size_t>(grid.width() + grid.height());
    }
    const int W = grid.width();
    std::mt19937_64 rng(seed);
    std::vector<int> used(static_cast<std::size_t>(W) * grid.height(), 0);
    std::vector<Candidate> out;
    std::set<std::vector<int>> seen;  // path signatures, to dedupe

    // Cap attempts so a map with few distinct routes terminates instead
    // of spinning forever hunting for the K-th path that does not exist.
    const std::size_t max_attempts = k * 8 + 8;
    for (std::size_t attempt = 0; attempt < max_attempts && out.size() < k; ++attempt) {
        const double weight = out.empty() ? 0.0 : penalty_weight;
        auto path = detail::astar_timed(grid, start, goal, used, weight, horizon, rng, reservations,
                                        dynamic_blocked);
        if (!path) break;  // goal unreachable within horizon

        // Penalize interior cells (endpoints are shared by every path, so
        // penalizing them cannot change which path is cheapest). This
        // runs even for duplicates, to push the next attempt away.
        for (std::size_t i = 1; i + 1 < path->size(); ++i) {
            const Cell c = (*path)[i];
            used[static_cast<std::size_t>(c.y) * W + c.x]++;
        }

        std::vector<int> signature;
        signature.reserve(path->size() * 2);
        for (const Cell& c : *path) {
            signature.push_back(c.x);
            signature.push_back(c.y);
        }
        if (seen.insert(std::move(signature)).second) {
            const double cost = path->empty() ? 0.0 : static_cast<double>(path->size() - 1);
            out.push_back(Candidate{std::move(*path), cost});
        }
    }
    return out;
}

// Pad a path to a common makespan by parking on its final cell (repeated
// wait moves). The result has makespan + 1 entries (timesteps 0..makespan).
inline std::vector<Cell> pad_to(const std::vector<Cell>& path, std::size_t makespan) {
    std::vector<Cell> out = path;
    if (out.empty()) return out;
    while (out.size() <= makespan) out.push_back(out.back());
    return out;
}

}  // namespace mapf
