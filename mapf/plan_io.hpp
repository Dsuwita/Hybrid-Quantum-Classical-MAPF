// plan_io.hpp
//
// Writing a plan to the simple text format consumed by the visualizers
// (render_plan.py, serve.py). Shared by the static solver CLI and the
// rolling-horizon demo so both emit exactly the same format.
//
//   # mapf plan
//   map <name>
//   agents <k>
//   makespan <m>
//   <agent> x0,y0 x1,y1 ...      (one line per agent, padded to the horizon)
#pragma once

#include <fstream>
#include <string>

#include "mapf/plan.hpp"

namespace mapf {

inline bool write_plan(const std::string& path, const std::string& map_name, const Plan& plan) {
    std::ofstream out(path);
    if (!out) return false;
    out << "# mapf plan\n";
    out << "map " << map_name << "\n";
    out << "agents " << plan.num_agents() << "\n";
    out << "makespan " << plan.makespan() << "\n";
    for (std::size_t a = 0; a < plan.num_agents(); ++a) {
        out << a;
        for (const Cell& c : plan.paths[a]) out << " " << c.x << "," << c.y;
        out << "\n";
    }
    return true;
}

}  // namespace mapf
