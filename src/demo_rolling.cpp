// demo_rolling.cpp
//
// Milestone 13 demo: run the rolling-horizon driver in lifelong mode on a
// map and report throughput, then write the execution history as a plan
// the visualizers can play. Agents get a fresh random goal each time they
// arrive, so they never stop moving.
//
// Usage:
//   demo_rolling <map> <k> [options]
// Options:
//   --steps N        global timesteps to simulate (default 60)
//   --window N       plan horizon per cycle W (default 8)
//   --execute N      steps committed per cycle E (default 3)
//   --deadline MS    anytime annealer deadline per cycle (default 20)
//   --seed N         RNG seed (default 1)
//   --out FILE       write the execution history as a plan file

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "mapf/grid.hpp"
#include "mapf/plan_io.hpp"
#include "mapf/rolling.hpp"
#include "mapf/verifier.hpp"

using namespace mapf;

namespace {
long argv_num(int argc, char** argv, const char* flag, long fallback) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return std::atol(argv[i + 1]);
    return fallback;
}
const char* argv_str(int argc, char** argv, const char* flag, const char* fallback) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
    return fallback;
}

// Plan file with per-timestep goal trajectory lines appended ("goal <a>
// x,y ..."), so the renderer can draw each agent's CURRENT goal as it jumps
// on arrival -- the visible signature of lifelong replanning.
bool write_plan_with_goals(const std::string& path, const std::string& map_name,
                           const RollingResult& res) {
    std::ofstream out(path);
    if (!out) return false;
    const Plan& plan = res.history;
    out << "# mapf plan\nmap " << map_name << "\nagents " << plan.num_agents() << "\nmakespan "
        << plan.makespan() << "\n";
    for (std::size_t a = 0; a < plan.num_agents(); ++a) {
        out << a;
        for (const Cell& c : plan.paths[a]) out << " " << c.x << "," << c.y;
        out << "\n";
    }
    for (std::size_t a = 0; a < res.goal_history.size(); ++a) {
        out << "goal " << a;
        for (const Cell& c : res.goal_history[a]) out << " " << c.x << "," << c.y;
        out << "\n";
    }
    return true;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <map> <k> [options]\n", argv[0]);
        return 2;
    }
    const std::string map_path = argv[1];
    const std::size_t k = static_cast<std::size_t>(std::atol(argv[2]));
    Grid grid = Grid::load(map_path);

    std::vector<Cell> free;
    for (int y = 0; y < grid.height(); ++y)
        for (int x = 0; x < grid.width(); ++x)
            if (grid.passable(x, y)) free.push_back(Cell{x, y});
    if (free.size() < 2 * k) {
        std::fprintf(stderr, "map too small for %zu agents\n", k);
        return 2;
    }

    std::mt19937_64 rng(static_cast<std::uint64_t>(argv_num(argc, argv, "--seed", 1)));
    std::shuffle(free.begin(), free.end(), rng);
    std::vector<Cell> starts(free.begin(), free.begin() + k);
    std::vector<Cell> goals(free.begin() + k, free.begin() + 2 * k);

    RollingConfig cfg;
    cfg.max_steps = static_cast<std::size_t>(argv_num(argc, argv, "--steps", 60));
    cfg.window = static_cast<std::size_t>(argv_num(argc, argv, "--window", 8));
    cfg.execute = static_cast<std::size_t>(argv_num(argc, argv, "--execute", 3));
    cfg.cycle_deadline_ms = static_cast<double>(argv_num(argc, argv, "--deadline", 20));
    cfg.seed = static_cast<std::uint64_t>(argv_num(argc, argv, "--seed", 1));

    // Lifelong: on arrival, hand out a random passable cell as the next goal.
    GoalProvider provider = [&grid](std::size_t, std::mt19937_64& r) {
        std::uniform_int_distribution<int> dx(0, grid.width() - 1), dy(0, grid.height() - 1);
        while (true) {
            Cell c{dx(r), dy(r)};
            if (grid.passable(c)) return c;
        }
    };

    RollingResult res = simulate_rolling(grid, starts, goals, cfg, provider);

    // Independently verify the produced history is collision-free.
    std::vector<AgentTask> tasks;
    for (const auto& p : res.history.paths) {
        AgentTask t;
        t.start = p.front();
        t.goal = p.back();
        tasks.push_back(t);
    }
    VerifyResult vr = verify(grid, tasks, res.history);

    std::printf("map              %s\n", map_path.c_str());
    std::printf("agents           %zu\n", k);
    std::printf("mode             lifelong\n");
    std::printf("window/execute   %zu / %zu\n", cfg.window, cfg.execute);
    std::printf("cycle deadline   %.0f ms\n", cfg.cycle_deadline_ms);
    std::printf("steps simulated  %zu\n", res.steps);
    std::printf("goals reached    %zu\n", res.goals_reached);
    std::printf("throughput       %.3f goals/step\n",
                res.steps ? double(res.goals_reached) / res.steps : 0.0);
    std::printf("history valid    %s\n", vr.ok() ? "yes" : "NO");
    std::printf("wall_ms          %.1f\n", res.wall_ms);

    const char* out_path = argv_str(argc, argv, "--out", nullptr);
    if (out_path) {
        if (write_plan_with_goals(out_path, map_path, res))
            std::printf("plan written     %s\n", out_path);
    }
    return vr.ok() ? 0 : 1;
}
