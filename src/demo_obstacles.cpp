// demo_obstacles.cpp
//
// Milestone 14 demo: agents crossing a grid while dodging patrolling
// moving obstacles, planned with perfect prediction. Writes a plan that
// includes the obstacle trajectories (as "obstacle x,y ..." lines) so the
// renderer can draw them, and reports whether the execution stayed clear
// of both other agents and the obstacles.
//
// Usage: demo_obstacles [--out plan.txt]

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "mapf/grid.hpp"
#include "mapf/rolling.hpp"
#include "mapf/verifier.hpp"

using namespace mapf;

namespace {

std::vector<Cell> patrol_vertical(int x, int ymin, int ymax, std::size_t length, int offset = 0) {
    std::vector<Cell> tr;
    tr.reserve(length);
    const int span = ymax - ymin;
    for (std::size_t t = 0; t < length; ++t) {
        const int phase = static_cast<int>((static_cast<long>(t) + offset) % (2 * span));
        tr.push_back(Cell{x, phase <= span ? ymin + phase : ymax - (phase - span)});
    }
    return tr;
}

// Plan file with obstacle trajectory lines appended (truncated to the
// execution length so the renderer and the history line up).
void write_plan_with_obstacles(const std::string& path, const std::string& map_name,
                               const Plan& plan, const ObstacleModel& obs, std::size_t steps) {
    std::ofstream out(path);
    out << "# mapf plan\n";
    out << "map " << map_name << "\n";
    out << "agents " << plan.num_agents() << "\n";
    out << "makespan " << plan.makespan() << "\n";
    out << "obstacles " << obs.count() << "\n";
    for (std::size_t a = 0; a < plan.num_agents(); ++a) {
        out << a;
        for (const Cell& c : plan.paths[a]) out << " " << c.x << "," << c.y;
        out << "\n";
    }
    for (std::size_t o = 0; o < obs.count(); ++o) {
        out << "obstacle";
        for (std::size_t t = 0; t <= steps; ++t) {
            Cell c = obs.at(o, t);
            out << " " << c.x << "," << c.y;
        }
        out << "\n";
    }
}

const char* argv_str(int argc, char** argv, const char* flag, const char* fallback) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
    return fallback;
}

}  // namespace

int main(int argc, char** argv) {
    // 15x11 open grid; three agents cross left-to-right on separate rows.
    std::ostringstream os;
    os << "type octile\nheight 11\nwidth 15\nmap\n";
    for (int y = 0; y < 11; ++y) {
        for (int x = 0; x < 15; ++x) os << '.';
        os << '\n';
    }
    std::istringstream in(os.str());
    Grid grid = Grid::parse(in);

    std::vector<Cell> starts = {{0, 2}, {0, 5}, {0, 8}};
    std::vector<Cell> goals = {{14, 2}, {14, 5}, {14, 8}};

    ObstacleModel obs;
    obs.paths = {patrol_vertical(5, 0, 10, 400, 0), patrol_vertical(10, 0, 10, 400, 6)};
    obs.perfect_prediction = true;

    RollingConfig cfg;
    cfg.window = 8;
    cfg.execute = 2;
    cfg.cycle_deadline_ms = 0.0;
    cfg.sweeps = 4000;
    cfg.replicas = 8;
    cfg.candidates_per_agent = 6;
    cfg.max_steps = 120;

    RollingResult r = simulate_rolling(grid, starts, goals, cfg, {}, &obs);

    // Independent agent-agent check.
    std::vector<AgentTask> tasks;
    for (const auto& p : r.history.paths) {
        AgentTask t;
        t.start = p.front();
        t.goal = p.back();
        tasks.push_back(t);
    }
    bool agent_ok = verify(grid, tasks, r.history).ok();

    std::printf("agents           %zu\n", starts.size());
    std::printf("obstacles        %zu (patrolling, perfect prediction)\n", obs.count());
    std::printf("steps            %zu\n", r.steps);
    std::printf("cycles           %zu\n", r.cycles);
    std::printf("all reached goal %s\n", r.all_done ? "yes" : "no");
    std::printf("agent-agent safe %s\n", agent_ok ? "yes" : "NO");
    std::printf("obstacle hits    %zu\n", r.obstacle_hits);
    std::printf("wall_ms          %.1f\n", r.wall_ms);

    const char* out_path = argv_str(argc, argv, "--out", nullptr);
    if (out_path) {
        write_plan_with_obstacles(out_path, "demo_obstacles.map", r.history, obs, r.steps);
        // Also write the map next to it so the renderer can find it.
        std::string map_out = std::string(out_path);
        auto slash = map_out.find_last_of('/');
        std::string dir = slash == std::string::npos ? "" : map_out.substr(0, slash + 1);
        std::ofstream mf(dir + "demo_obstacles.map");
        mf << "type octile\nheight 11\nwidth 15\nmap\n";
        for (int y = 0; y < 11; ++y) {
            for (int x = 0; x < 15; ++x) mf << '.';
            mf << '\n';
        }
        std::printf("plan written     %s\n", out_path);
    }
    return (agent_ok && r.obstacle_hits == 0) ? 0 : 1;
}
