// scenario.hpp
//
// MovingAI .scen scenario parser. A scenario file lists agent tasks, one
// per line, tab- or space-separated:
//
//   version 1                                (optional header line)
//   bucket  map  map_w  map_h  sx  sy  gx  gy  optimal
//
// where (sx,sy) is the agent's start, (gx,gy) its goal (x = column,
// y = row, matching grid.hpp) and `optimal` is the octile-distance
// shortest path length published with the benchmark. For 4-connected
// MAPF the published value is still the correct per-agent lower bound on
// open maps with no diagonal shortcuts; it is carried through so the
// verifier can report sum-of-costs overhead against it.
//
// The standard evaluation protocol solves the first k agents of a
// scenario for growing k, so selection is just "take the first k".
#pragma once

#include <fstream>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mapf/grid.hpp"

namespace mapf {

struct AgentTask {
    int bucket = 0;
    std::string map_name;
    int map_width = 0;
    int map_height = 0;
    Cell start;
    Cell goal;
    double optimal_distance = 0.0;
};

class Scenario {
public:
    static Scenario parse(std::istream& in) {
        Scenario scen;
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            if (line.rfind("version", 0) == 0) continue;  // header
            std::istringstream ls(line);
            AgentTask task;
            if (!(ls >> task.bucket >> task.map_name >> task.map_width >> task.map_height >>
                  task.start.x >> task.start.y >> task.goal.x >> task.goal.y >>
                  task.optimal_distance)) {
                throw std::runtime_error("Scenario::parse: malformed line: " + line);
            }
            scen.tasks_.push_back(std::move(task));
        }
        return scen;
    }

    static Scenario load(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("Scenario::load: cannot open " + path);
        return parse(f);
    }

    const std::vector<AgentTask>& tasks() const { return tasks_; }

    // The standard benchmark protocol: solve the first k agents.
    std::vector<AgentTask> first_k(std::size_t k) const {
        if (k > tasks_.size()) k = tasks_.size();
        return std::vector<AgentTask>(tasks_.begin(), tasks_.begin() + static_cast<long>(k));
    }

private:
    std::vector<AgentTask> tasks_;
};

}  // namespace mapf
