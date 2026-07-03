// verifier.hpp
//
// Replay-based plan checker. This is the ground truth for the whole MAPF
// project: solvers may cut corners, the verifier may not. It re-simulates
// the plan timestep by timestep straight from the grid and the scenario,
// trusting nothing the solver computed, and reports every violation:
//
//   - EndpointMismatch: path does not start at the agent's start or end
//     at its goal.
//   - InvalidMove: a step that is not adjacent-or-wait, or enters a
//     blocked/out-of-bounds cell (the start cell itself is checked too).
//   - VertexConflict: two agents in the same cell at the same timestep.
//     Parked agents (whose paths already ended) still occupy their final
//     cell and participate.
//   - SwapConflict: two agents exchange cells across one timestep
//     (a: u -> v while b: v -> u), the classic head-on collision that
//     vertex checks alone miss.
//
// Metrics: sum-of-costs of the plan and its overhead relative to the sum
// of the per-agent optimal distances published in the scenario (a lower
// bound on any conflict-free plan, since ignoring other agents can only
// help).
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "mapf/grid.hpp"
#include "mapf/plan.hpp"
#include "mapf/scenario.hpp"

namespace mapf {

struct Violation {
    enum class Type { EndpointMismatch, InvalidMove, VertexConflict, SwapConflict };
    Type type;
    std::size_t agent_a = 0;
    std::size_t agent_b = 0;  // meaningful for conflicts only
    std::size_t t = 0;        // timestep at which the problem occurs
    Cell cell;                // offending cell (for swaps: agent_a's target)
};

struct VerifyResult {
    std::vector<Violation> violations;
    double sum_of_costs = 0.0;
    double sum_of_optimal = 0.0;

    bool ok() const { return violations.empty(); }
    // Overhead of the plan vs the sum of per-agent lower bounds, in
    // percent; 0 means every agent took a shortest path.
    double overhead_percent() const {
        if (sum_of_optimal <= 0.0) return 0.0;
        return 100.0 * (sum_of_costs - sum_of_optimal) / sum_of_optimal;
    }
};

inline VerifyResult verify(const Grid& grid, const std::vector<AgentTask>& tasks,
                           const Plan& plan) {
    VerifyResult result;
    const std::size_t n = plan.num_agents();

    // Per-agent structural checks: endpoints, then every move.
    for (std::size_t a = 0; a < n && a < tasks.size(); ++a) {
        const auto& path = plan.paths[a];
        if (path.empty() || !(path.front() == tasks[a].start) ||
            !(path.back() == tasks[a].goal)) {
            result.violations.push_back(
                {Violation::Type::EndpointMismatch, a, a, 0,
                 path.empty() ? Cell{} : path.front()});
        }
        for (std::size_t t = 0; t < path.size(); ++t) {
            if (!grid.passable(path[t])) {
                result.violations.push_back({Violation::Type::InvalidMove, a, a, t, path[t]});
            }
            if (t + 1 < path.size()) {
                int dx = path[t + 1].x - path[t].x;
                int dy = path[t + 1].y - path[t].y;
                if (std::abs(dx) + std::abs(dy) > 1) {
                    result.violations.push_back(
                        {Violation::Type::InvalidMove, a, a, t + 1, path[t + 1]});
                }
            }
        }
    }

    // Pairwise replay over the common horizon. Parked agents are handled
    // by position_at clamping to the final cell.
    const std::size_t horizon = plan.makespan();
    for (std::size_t t = 0; t <= horizon; ++t) {
        for (std::size_t a = 0; a < n; ++a) {
            for (std::size_t b = a + 1; b < n; ++b) {
                Cell pa = plan.position_at(a, t);
                Cell pb = plan.position_at(b, t);
                if (pa == pb) {
                    result.violations.push_back({Violation::Type::VertexConflict, a, b, t, pa});
                }
                if (t > 0) {
                    Cell pa_prev = plan.position_at(a, t - 1);
                    Cell pb_prev = plan.position_at(b, t - 1);
                    if (pa == pb_prev && pb == pa_prev && !(pa == pb)) {
                        result.violations.push_back({Violation::Type::SwapConflict, a, b, t, pa});
                    }
                }
            }
        }
    }

    // Metrics.
    std::vector<Cell> goals;
    goals.reserve(tasks.size());
    for (const auto& task : tasks) goals.push_back(task.goal);
    result.sum_of_costs = plan.sum_of_costs(goals);
    for (const auto& task : tasks) result.sum_of_optimal += task.optimal_distance;
    return result;
}

inline const char* to_string(Violation::Type type) {
    switch (type) {
        case Violation::Type::EndpointMismatch: return "endpoint mismatch";
        case Violation::Type::InvalidMove: return "invalid move";
        case Violation::Type::VertexConflict: return "vertex conflict";
        case Violation::Type::SwapConflict: return "swap conflict";
    }
    return "unknown";
}

}  // namespace mapf
