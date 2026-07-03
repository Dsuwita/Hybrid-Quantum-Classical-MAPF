// solve_mapf.cpp
//
// Command-line MAPF solver. Loads a MovingAI map and scenario, solves the
// first k agents with the hybrid anneal loop, prints metrics, and
// optionally writes the plan to a file for the visualizer (Milestone 12).
//
// Usage:
//   solve_mapf <map> <scen> <k> [options]
// Options:
//   --sweeps N       annealer sweeps per replica (default 2000)
//   --replicas N     parallel restarts (default 8)
//   --threads N      worker threads, 0 = hardware (default 0)
//   --iters N        max re-anneal rounds (default 10)
//   --candidates N   candidate paths generated per agent per round (default 4)
//   --seed N         RNG seed (default 1)
//   --out FILE       write the plan to FILE

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "mapf/grid.hpp"
#include "mapf/plan_io.hpp"
#include "mapf/scenario.hpp"
#include "mapf/solver.hpp"
#include "mapf/verifier.hpp"

using namespace mapf;

namespace {

long arg_value(int argc, char** argv, const char* flag, long fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], flag) == 0) return std::atol(argv[i + 1]);
    }
    return fallback;
}

const char* arg_string(int argc, char** argv, const char* flag, const char* fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
    }
    return fallback;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <map> <scen> <k> [options]\n", argv[0]);
        return 2;
    }
    const std::string map_path = argv[1];
    const std::string scen_path = argv[2];
    const std::size_t k = static_cast<std::size_t>(std::atol(argv[3]));

    Grid grid = Grid::load(map_path);
    Scenario scen = Scenario::load(scen_path);
    std::vector<AgentTask> tasks = scen.first_k(k);
    if (tasks.size() < k) {
        std::fprintf(stderr, "scenario has only %zu agents, requested %zu\n", tasks.size(), k);
    }

    SolverConfig cfg;
    cfg.sweeps = static_cast<std::size_t>(arg_value(argc, argv, "--sweeps", 2000));
    cfg.replicas = static_cast<std::size_t>(arg_value(argc, argv, "--replicas", 8));
    cfg.threads = static_cast<std::size_t>(arg_value(argc, argv, "--threads", 0));
    cfg.max_iterations = static_cast<std::size_t>(arg_value(argc, argv, "--iters", 10));
    cfg.candidates_per_agent = static_cast<std::size_t>(arg_value(argc, argv, "--candidates", 4));
    cfg.seed = static_cast<std::uint64_t>(arg_value(argc, argv, "--seed", 1));

    SolverResult r = solve_mapf(grid, tasks, cfg);

    // Re-verify independently for the printed metrics: never trust the
    // solver's own bookkeeping for the reported result.
    VerifyResult vr = verify(grid, tasks, r.plan);

    std::printf("map            %s\n", map_path.c_str());
    std::printf("agents         %zu\n", tasks.size());
    std::printf("success        %s\n", r.success ? "yes" : "no");
    if (r.infeasible_generation) {
        std::printf("note           an agent's goal was unreachable (no plan possible)\n");
    }
    std::printf("iterations     %zu\n", r.iterations);
    std::printf("sum-of-costs   %.0f\n", vr.sum_of_costs);
    std::printf("sum-of-optimal %.0f\n", vr.sum_of_optimal);
    std::printf("overhead       %.1f%%\n", vr.overhead_percent());
    std::printf("conflicts      %zu\n", vr.violations.size());
    std::printf("wall_ms        %.1f\n", r.wall_ms);

    const char* out_path = arg_string(argc, argv, "--out", nullptr);
    if (out_path) {
        if (write_plan(out_path, map_path, r.plan))
            std::printf("plan written   %s\n", out_path);
        else
            std::fprintf(stderr, "warning: cannot open %s for writing\n", out_path);
    }

    return r.success ? 0 : 1;
}
