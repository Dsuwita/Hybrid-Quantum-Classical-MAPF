// cbs.hpp
//
// Conflict-Based Search (CBS): the classical, optimal-for-sum-of-costs MAPF
// solver, used in the studio as the comparison target for the hybrid
// annealer. Reference: Sharon, Stern, Felner, Sturtevant, "Conflict-based
// search for optimal multi-agent pathfinding", Artificial Intelligence 2015.
//
// CBS is a two-level search:
//
//   HIGH LEVEL (this file's search loop) explores a binary Constraint Tree.
//   Each node holds a set of constraints and, for every agent, a shortest
//   path that obeys that agent's constraints. We pop the lowest-cost node,
//   look for the first conflict between two agents' paths, and if there is
//   one we SPLIT: create two children, each adding one new constraint that
//   forbids one of the two agents from the conflicting cell/edge at that
//   time. Only the constrained agent is replanned. A node with no conflicts
//   is a valid solution, and because we always expand the cheapest node
//   first, the first conflict-free node found is optimal.
//
//   LOW LEVEL (low_level below) is a space-time A* for ONE agent that finds
//   a minimum-cost path from start to goal while respecting that agent's
//   vertex and edge constraints. Searching over (cell, time) rather than
//   just cell is what lets an agent wait to dodge a constraint.
//
// Two constraint kinds, matching the two conflict kinds the verifier checks:
//   - vertex constraint (a, x, y, t): agent a may not occupy (x,y) at time t.
//   - edge constraint (a, u, v, t): agent a may not move u->v departing at
//     time t (arriving at t+1). This is what resolves a swap conflict.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mapf/grid.hpp"
#include "mapf/plan.hpp"
#include "mapf/scenario.hpp"

namespace mapf {

struct CbsConfig {
    // Cap on high-level node expansions. CBS is exponential in the worst
    // case; on hard/dense instances we stop and report failure honestly
    // rather than running unbounded. Generous by default.
    std::size_t max_expansions = 200000;

    // Optional wall-clock deadline in milliseconds (0 = no limit). CBS can
    // blow up on congested maps, so the studio sets a deadline to stay
    // responsive; on timeout we return the cheapest node found so far (which
    // still has conflicts) with success = false, rather than hanging.
    double time_limit_ms = 0.0;
};

struct CbsResult {
    bool success = false;   // found a conflict-free (hence optimal) plan
    Plan plan;              // padded to a common makespan
    double cost = 0.0;      // sum-of-costs of the returned plan
    std::size_t expansions = 0;  // high-level nodes expanded (search effort)
    double wall_ms = 0.0;
};

namespace cbs_detail {

// A single constraint on one agent. `is_edge` distinguishes the two kinds:
// vertex forbids being at (x,y) at time t; edge forbids the move
// (x,y)->(x2,y2) departing at time t.
struct Constraint {
    bool is_edge = false;
    int x = 0, y = 0;      // vertex cell, or edge "from" cell
    int x2 = 0, y2 = 0;    // edge "to" cell (edge only)
    std::size_t t = 0;
};

// Pack a (cell,time) pair into one key for the constraint hash sets. Time is
// bounded well under 2^24 for any instance we solve, so 20 bits each for
// x/y and the rest for t is comfortable.
inline std::uint64_t vkey(int x, int y, std::size_t t) {
    return (static_cast<std::uint64_t>(t) << 40) |
           (static_cast<std::uint64_t>(x) << 20) | static_cast<std::uint64_t>(y);
}
inline std::uint64_t ekey(int x, int y, int x2, int y2, std::size_t t) {
    // Fold the four coordinates and time; collisions only need to be avoided
    // within one agent's small constraint set, so a simple mix is fine.
    std::uint64_t h = static_cast<std::uint64_t>(t);
    h = h * 1000003 + static_cast<std::uint64_t>(x);
    h = h * 1000003 + static_cast<std::uint64_t>(y);
    h = h * 1000003 + static_cast<std::uint64_t>(x2);
    h = h * 1000003 + static_cast<std::uint64_t>(y2);
    return h;
}

// Space-time A* for one agent. Returns the cell sequence (path[0]==start,
// path.back()==goal) of minimum cost that obeys `vc` (forbidden (cell,time))
// and `ec` (forbidden (from,to,depart-time)) sets, or nullopt if none exists
// within the time bound. `latest` is the latest timestep any constraint
// touches this agent; the agent must be parked at its goal at or after that
// time (otherwise a later constraint could still be violated once it stops
// moving), so goal nodes before `latest` are not accepted as terminal.
inline std::optional<std::vector<Cell>> low_level(
    const Grid& grid, Cell start, Cell goal,
    const std::unordered_set<std::uint64_t>& vc,
    const std::unordered_set<std::uint64_t>& ec, std::size_t latest) {
    const int W = grid.width();
    const int H = grid.height();
    const std::uint64_t layer = static_cast<std::uint64_t>(W) * H;

    // Time bound: an optimal single-agent plan never needs to wait longer
    // than "reach every cell once, plus outlast the last constraint".
    const std::size_t max_t = latest + static_cast<std::size_t>(W) * H + 1;

    auto heuristic = [&](Cell c) {
        return static_cast<double>(std::abs(c.x - goal.x) + std::abs(c.y - goal.y));
    };
    auto encode = [&](Cell c, std::size_t t) {
        return static_cast<std::uint64_t>(t) * layer +
               static_cast<std::uint64_t>(c.y) * W + c.x;
    };

    struct Node {
        double f, g;
        std::uint64_t id;
        Cell cell;
        std::size_t t;
    };
    auto worse = [](const Node& a, const Node& b) {
        // Lower f first; break ties toward larger g (deeper, closer to goal),
        // the standard A* tie-break that speeds convergence.
        return a.f > b.f || (a.f == b.f && a.g < b.g);
    };
    std::priority_queue<Node, std::vector<Node>, decltype(worse)> open(worse);
    std::unordered_map<std::uint64_t, double> best_g;
    std::unordered_map<std::uint64_t, std::uint64_t> came_from;

    const std::uint64_t start_id = encode(start, 0);
    best_g[start_id] = 0.0;
    open.push({heuristic(start), 0.0, start_id, start, 0});

    while (!open.empty()) {
        Node cur = open.top();
        open.pop();
        auto git = best_g.find(cur.id);
        if (git != best_g.end() && cur.g > git->second) continue;

        if (cur.cell == goal && cur.t >= latest) {
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
        if (cur.t >= max_t) continue;

        for (Cell nb : grid.moves_from(cur.cell)) {
            const std::size_t nt = cur.t + 1;
            // Vertex constraint: cannot be at nb at time nt.
            if (vc.count(vkey(nb.x, nb.y, nt))) continue;
            // Edge constraint: cannot traverse cur.cell -> nb departing at cur.t.
            if (ec.count(ekey(cur.cell.x, cur.cell.y, nb.x, nb.y, cur.t))) continue;
            const double ng = cur.g + 1.0;
            const std::uint64_t nid = encode(nb, nt);
            auto it = best_g.find(nid);
            if (it == best_g.end() || ng < it->second) {
                best_g[nid] = ng;
                came_from[nid] = cur.id;
                open.push({ng + heuristic(nb), ng, nid, nb, nt});
            }
        }
    }
    return std::nullopt;
}

// One high-level Constraint Tree node.
struct CTNode {
    std::vector<std::vector<Constraint>> constraints;  // per agent
    std::vector<std::vector<Cell>> paths;              // per agent (unpadded)
    double cost = 0.0;
    std::size_t order = 0;  // insertion order, for a stable tie-break

    // Individual cost = arrival timestep = path length minus the start cell.
    static double path_cost(const std::vector<Cell>& p) {
        return p.empty() ? 0.0 : static_cast<double>(p.size() - 1);
    }
};

// The first conflict (lowest timestep, then lowest agent pair) in a solution,
// or none. Scans the padded solution the same way the verifier does so CBS
// resolves exactly the conflicts the verifier would flag.
struct Conflict {
    bool exists = false;
    bool is_edge = false;
    std::size_t a = 0, b = 0;
    std::size_t t = 0;         // vertex time, or depart time for an edge
    Cell ca, cb;               // a's cell(s): vertex uses ca; edge uses ca->cb
};

inline Conflict first_conflict(const std::vector<std::vector<Cell>>& paths) {
    auto at = [](const std::vector<Cell>& p, std::size_t t) {
        return t < p.size() ? p[t] : p.back();
    };
    std::size_t horizon = 0;
    for (const auto& p : paths) horizon = std::max(horizon, p.size() ? p.size() - 1 : 0);
    const std::size_t n = paths.size();
    for (std::size_t t = 0; t <= horizon; ++t) {
        for (std::size_t a = 0; a < n; ++a) {
            for (std::size_t b = a + 1; b < n; ++b) {
                Cell pa = at(paths[a], t), pb = at(paths[b], t);
                if (pa == pb) return {true, false, a, b, t, pa, pa};  // vertex
                if (t > 0) {
                    Cell pa0 = at(paths[a], t - 1), pb0 = at(paths[b], t - 1);
                    if (pa == pb0 && pb == pa0 && !(pa == pb)) {
                        // Swap across t-1 -> t: forbid each agent's own move.
                        return {true, true, a, b, t - 1, pa0, pa};
                    }
                }
            }
        }
    }
    return {};
}

}  // namespace cbs_detail

inline CbsResult solve_cbs(const Grid& grid, const std::vector<AgentTask>& tasks,
                           const CbsConfig& cfg = {}) {
    using namespace cbs_detail;
    const auto start_time = std::chrono::steady_clock::now();
    const std::size_t n = tasks.size();
    CbsResult result;

    auto finish = [&](CbsResult r) {
        r.wall_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start_time)
                        .count();
        return r;
    };

    // Build the constraint hash sets for one agent from its constraint list,
    // and the latest timestep they touch (for the low-level goal-parking rule).
    auto plan_agent = [&](std::size_t a, const std::vector<Constraint>& cons)
        -> std::optional<std::vector<Cell>> {
        std::unordered_set<std::uint64_t> vc, ec;
        std::size_t latest = 0;
        for (const Constraint& c : cons) {
            if (c.is_edge) {
                ec.insert(ekey(c.x, c.y, c.x2, c.y2, c.t));
                latest = std::max(latest, c.t + 1);
            } else {
                vc.insert(vkey(c.x, c.y, c.t));
                latest = std::max(latest, c.t);
            }
        }
        return low_level(grid, tasks[a].start, tasks[a].goal, vc, ec, latest);
    };

    // Root: every agent's unconstrained shortest path.
    CTNode root;
    root.constraints.assign(n, {});
    root.paths.resize(n);
    for (std::size_t a = 0; a < n; ++a) {
        auto p = plan_agent(a, root.constraints[a]);
        if (!p) return finish(result);  // an agent's goal is unreachable
        root.paths[a] = std::move(*p);
        root.cost += CTNode::path_cost(root.paths[a]);
    }

    // High-level open list, ordered by cost then insertion order.
    auto worse = [](const CTNode& x, const CTNode& y) {
        return x.cost > y.cost || (x.cost == y.cost && x.order > y.order);
    };
    std::priority_queue<CTNode, std::vector<CTNode>, decltype(worse)> open(worse);
    std::size_t counter = 0;
    root.order = counter++;
    open.push(std::move(root));

    // Pad a node's paths into a Plan (used both for the optimal result and
    // for the best-effort plan returned on timeout).
    auto to_plan = [&](const CTNode& node) {
        std::size_t makespan = 0;
        for (const auto& p : node.paths)
            makespan = std::max(makespan, p.size() ? p.size() - 1 : 0);
        Plan plan;
        plan.paths.resize(n);
        for (std::size_t a = 0; a < n; ++a) {
            plan.paths[a] = node.paths[a];
            while (plan.paths[a].size() <= makespan) plan.paths[a].push_back(plan.paths[a].back());
        }
        return plan;
    };

    while (!open.empty() && result.expansions < cfg.max_expansions) {
        // Deadline check: bail out with the cheapest (still-conflicted) node.
        if (cfg.time_limit_ms > 0.0 &&
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time)
                    .count() > cfg.time_limit_ms) {
            result.plan = to_plan(open.top());
            result.cost = open.top().cost;
            return finish(result);
        }

        CTNode node = open.top();
        open.pop();
        ++result.expansions;

        Conflict c = first_conflict(node.paths);
        if (!c.exists) {
            // Conflict-free and cheapest: optimal.
            result.plan = to_plan(node);
            result.success = true;
            result.cost = node.cost;
            return finish(result);
        }

        // Split on the two agents involved. Each child forbids one of them
        // from the conflicting cell (vertex) or move (edge) at that time.
        for (int side = 0; side < 2; ++side) {
            const std::size_t agent = side == 0 ? c.a : c.b;
            Constraint nc;
            nc.is_edge = c.is_edge;
            nc.t = c.t;
            if (!c.is_edge) {
                nc.x = c.ca.x;
                nc.y = c.ca.y;
            } else {
                // Edge: agent a's move was ca->cb; agent b's move was cb->ca.
                Cell from = side == 0 ? c.ca : c.cb;
                Cell to = side == 0 ? c.cb : c.ca;
                nc.x = from.x;
                nc.y = from.y;
                nc.x2 = to.x;
                nc.y2 = to.y;
            }

            CTNode child = node;
            child.constraints[agent].push_back(nc);
            auto p = plan_agent(agent, child.constraints[agent]);
            if (!p) continue;  // this branch is infeasible; drop it
            child.cost -= CTNode::path_cost(child.paths[agent]);
            child.paths[agent] = std::move(*p);
            child.cost += CTNode::path_cost(child.paths[agent]);
            child.order = counter++;
            open.push(std::move(child));
        }
    }

    // Ran out of node budget (or open list) without a conflict-free plan.
    return finish(result);
}

}  // namespace mapf
