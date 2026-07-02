// bench_mapf.cpp
//
// Milestone 11 evaluation. The standard MAPF protocol: hold a map fixed,
// increase the number of agents k, and watch the success rate fall as the
// instance gets more congested. For each (map, k) we draw several random
// scenarios (distinct passable start/goal cells) and report the fraction
// the solver resolves conflict-free, plus average sum-of-costs overhead
// and wall time over the SUCCESSFUL runs.
//
// The maps live in mapf/bench/maps/ and are synthetic (an empty room,
// scattered obstacles, and four rooms joined by doorways) rather than
// downloaded MovingAI instances, because this environment is offline; the
// download step is deferred with Milestone 5. The evaluation methodology
// is the standard one regardless of where the maps come from.
//
// Output: CSV to stdout (map,agents,success_rate,avg_overhead_pct,
// avg_wall_ms) plus a readable summary to stderr. Run as
//   ./bench_mapf > mapf/bench/results.csv

#include <algorithm>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "mapf/grid.hpp"
#include "mapf/path_gen.hpp"
#include "mapf/scenario.hpp"
#include "mapf/solver.hpp"
#include "mapf/verifier.hpp"

using namespace mapf;

namespace {

std::vector<Cell> passable_cells(const Grid& grid) {
    std::vector<Cell> cells;
    for (int y = 0; y < grid.height(); ++y) {
        for (int x = 0; x < grid.width(); ++x) {
            if (grid.passable(x, y)) cells.push_back(Cell{x, y});
        }
    }
    return cells;
}

// Draw k agents with distinct starts and distinct goals, each with a
// reachable goal (shortest path exists). Returns empty if it cannot.
std::vector<AgentTask> random_scenario(const Grid& grid, const std::vector<Cell>& free,
                                       std::size_t k, std::mt19937_64& rng) {
    std::vector<Cell> starts = free, goals = free;
    std::shuffle(starts.begin(), starts.end(), rng);
    std::shuffle(goals.begin(), goals.end(), rng);
    std::vector<AgentTask> tasks;
    for (std::size_t i = 0; i < k && i < starts.size() && i < goals.size(); ++i) {
        AgentTask t;
        t.start = starts[i];
        t.goal = goals[i];
        auto c = generate_candidates(grid, t.start, t.goal, 1, i);
        if (c.empty()) return {};  // unreachable; give up on this draw
        t.optimal_distance = c[0].cost;
        tasks.push_back(t);
    }
    return tasks.size() == k ? tasks : std::vector<AgentTask>{};
}

}  // namespace

int main() {
    const std::vector<std::string> maps = {
        "mapf/bench/maps/empty16.map",
        "mapf/bench/maps/obstacles16.map",
        "mapf/bench/maps/rooms16.map",
    };
    const std::vector<std::size_t> agent_counts = {2, 4, 6, 8, 12, 16, 20};
    const int scenarios_per_point = 20;

    SolverConfig cfg;
    cfg.sweeps = 3000;
    cfg.replicas = 12;
    cfg.threads = 0;
    cfg.max_iterations = 12;
    cfg.candidates_per_agent = 5;

    std::printf("map,agents,success_rate,avg_overhead_pct,avg_wall_ms\n");

    for (const std::string& map_path : maps) {
        Grid grid = Grid::load(map_path);
        std::vector<Cell> free = passable_cells(grid);
        std::fprintf(stderr, "\n== %s (%d passable cells) ==\n", map_path.c_str(),
                     static_cast<int>(free.size()));

        for (std::size_t k : agent_counts) {
            int successes = 0, valid_draws = 0;
            double overhead_sum = 0.0, wall_sum = 0.0;

            for (int s = 0; s < scenarios_per_point; ++s) {
                std::mt19937_64 rng(1000 * k + s);
                auto tasks = random_scenario(grid, free, k, rng);
                if (tasks.empty()) continue;  // couldn't place k reachable agents
                ++valid_draws;

                cfg.seed = static_cast<std::uint64_t>(7 * s + 1);
                SolverResult r = solve_mapf(grid, tasks, cfg);
                VerifyResult vr = verify(grid, tasks, r.plan);
                wall_sum += r.wall_ms;
                if (r.success && vr.ok()) {
                    ++successes;
                    overhead_sum += vr.overhead_percent();
                }
            }

            if (valid_draws == 0) continue;
            double success_rate = 100.0 * successes / valid_draws;
            double avg_overhead = successes > 0 ? overhead_sum / successes : 0.0;
            double avg_wall = wall_sum / valid_draws;

            std::printf("%s,%zu,%.1f,%.1f,%.1f\n", map_path.c_str(), k, success_rate, avg_overhead,
                        avg_wall);
            std::fflush(stdout);
            std::fprintf(stderr, "  k=%2zu: success %5.1f%%  overhead %5.1f%%  wall %6.1f ms\n", k,
                         success_rate, avg_overhead, avg_wall);
        }
    }
    return 0;
}
