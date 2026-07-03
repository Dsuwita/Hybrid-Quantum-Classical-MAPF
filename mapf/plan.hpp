// plan.hpp
//
// A plan assigns every agent a timestep-indexed sequence of cells:
// paths[a][t] is where agent a stands at time t. Paths may have
// different lengths; an agent whose path has ended is PARKED at its
// final cell and still occupies it (position_at clamps to the last
// entry). Parked agents can collide with moving ones, which is why the
// verifier replays positions through this accessor rather than the raw
// vectors.
//
// Cost model (standard MAPF sum-of-costs): an agent's cost is the last
// timestep at which it is anywhere other than its goal, plus one; an
// agent that sits on its goal for the whole plan costs 0. Waiting at the
// goal AFTER final arrival is free, but leaving the goal and coming back
// is paid (the cost counts up to the LAST time it is off-goal).
#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "mapf/grid.hpp"

namespace mapf {

struct Plan {
    std::vector<std::vector<Cell>> paths;  // paths[agent][t]

    std::size_t num_agents() const { return paths.size(); }

    // Latest timestep any path explicitly covers; the plan's horizon.
    std::size_t makespan() const {
        std::size_t m = 0;
        for (const auto& p : paths) {
            if (!p.empty() && p.size() - 1 > m) m = p.size() - 1;
        }
        return m;
    }

    // Where agent a is at time t: agents park at their final cell after
    // their path ends.
    Cell position_at(std::size_t agent, std::size_t t) const {
        const auto& p = paths[agent];
        if (p.empty()) throw std::runtime_error("Plan::position_at: empty path");
        return t < p.size() ? p[t] : p.back();
    }

    // Sum-of-costs against the agents' goals (see the cost model above).
    double sum_of_costs(const std::vector<Cell>& goals) const {
        double total = 0.0;
        for (std::size_t a = 0; a < paths.size(); ++a) {
            const auto& p = paths[a];
            for (std::size_t t = p.size(); t-- > 0;) {
                if (!(p[t] == goals[a])) {
                    total += static_cast<double>(t) + 1.0;
                    break;
                }
            }
        }
        return total;
    }
};

}  // namespace mapf
