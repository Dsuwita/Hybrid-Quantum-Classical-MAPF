// solve_cbs.cpp
//
// Command-line CBS solver: the classical, optimal comparison target for the
// hybrid annealer in the studio. Loads a MovingAI map and scenario, solves
// the first k agents with Conflict-Based Search, prints the same key/value
// metrics as solve_mapf (so serve.py parses both the same way), and can
// write the plan for the visualizer.
//
// Usage:
//   solve_cbs <map> <scen> <k> [options]
// Options:
//   --max-expansions N   high-level node cap before giving up (default 200000)
//   --out FILE           write the plan to FILE
//   --json               also print a single-line JSON summary on stdout

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "mapf/cbs.hpp"
#include "mapf/grid.hpp"
#include "mapf/plan_io.hpp"
#include "mapf/scenario.hpp"
#include "mapf/verifier.hpp"

using namespace mapf;

namespace {

long arg_value(int argc, char** argv, const char* flag, long fallback) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return std::atol(argv[i + 1]);
    return fallback;
}

const char* arg_string(int argc, char** argv, const char* flag, const char* fallback) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
    return fallback;
}

bool has_flag(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return true;
    return false;
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

    CbsConfig cfg;
    cfg.max_expansions =
        static_cast<std::size_t>(arg_value(argc, argv, "--max-expansions", 200000));
    cfg.time_limit_ms = static_cast<double>(arg_value(argc, argv, "--time-limit-ms", 0));

    CbsResult r = solve_cbs(grid, tasks, cfg);

    // Re-verify independently; never trust the solver's own bookkeeping.
    VerifyResult vr = verify(grid, tasks, r.plan);

    std::printf("map            %s\n", map_path.c_str());
    std::printf("agents         %zu\n", tasks.size());
    std::printf("solver         cbs\n");
    std::printf("success        %s\n", r.success ? "yes" : "no");
    std::printf("expansions     %zu\n", r.expansions);
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

    if (has_flag(argc, argv, "--json")) {
        // Compact machine-readable summary for the studio backend.
        std::printf(
            "{\"solver\":\"cbs\",\"success\":%s,\"sum_of_costs\":%.0f,"
            "\"sum_of_optimal\":%.0f,\"overhead_pct\":%.2f,\"conflicts\":%zu,"
            "\"expansions\":%zu,\"wall_ms\":%.1f}\n",
            r.success ? "true" : "false", vr.sum_of_costs, vr.sum_of_optimal,
            vr.overhead_percent(), vr.violations.size(), r.expansions, r.wall_ms);
    }

    return r.success ? 0 : 1;
}
