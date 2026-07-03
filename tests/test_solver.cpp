// test_solver.cpp
//
// Milestone 11 tests for the full solve loop. Success is judged by the
// Milestone 8 verifier (the oracle), never by the solver's own report:
// on success the returned plan must verify clean, and on the deliberately
// unsolvable instance the solver must report failure rather than loop
// forever or claim a bogus success.

#include "mapf/grid.hpp"
#include "mapf/scenario.hpp"
#include "mapf/solver.hpp"
#include "mapf/verifier.hpp"

#include <cassert>
#include <cstdio>
#include <sstream>
#include <vector>

using namespace mapf;

namespace {

Grid open_grid(int w, int h) {
    std::ostringstream os;
    os << "type octile\nheight " << h << "\nwidth " << w << "\nmap\n";
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) os << '.';
        os << '\n';
    }
    std::istringstream in(os.str());
    return Grid::parse(in);
}

AgentTask task(const Grid& grid, Cell start, Cell goal) {
    AgentTask t;
    t.start = start;
    t.goal = goal;
    // Optimal distance = length of a shortest path (first candidate).
    auto c = generate_candidates(grid, start, goal, 1, 0);
    t.optimal_distance = c.empty() ? 0.0 : c[0].cost;
    return t;
}

SolverConfig fast_config() {
    SolverConfig cfg;
    cfg.sweeps = 1000;
    cfg.replicas = 8;
    cfg.threads = 4;
    cfg.max_iterations = 12;
    return cfg;
}

void test_two_agents_crossing() {
    Grid grid = open_grid(7, 7);
    // Horizontal and vertical travellers whose shortest paths cross at
    // the center at the same time: a conflict the solver must resolve.
    std::vector<AgentTask> tasks = {task(grid, {0, 3}, {6, 3}), task(grid, {3, 0}, {3, 6})};

    SolverResult r = solve_mapf(grid, tasks, fast_config());
    assert(r.success);
    // Trust the verifier, not the solver's flag.
    VerifyResult check = verify(grid, tasks, r.plan);
    assert(check.ok());
    std::printf("test_two_agents_crossing passed (soc=%.0f, overhead=%.1f%%, iters=%zu)\n",
                check.sum_of_costs, check.overhead_percent(), r.iterations);
}

void test_many_agents_open() {
    Grid grid = open_grid(10, 10);
    std::vector<AgentTask> tasks = {
        task(grid, {0, 0}, {9, 9}), task(grid, {9, 0}, {0, 9}),
        task(grid, {0, 9}, {9, 0}), task(grid, {9, 9}, {0, 0}),
        task(grid, {0, 5}, {9, 5}), task(grid, {5, 0}, {5, 9}),
    };
    SolverResult r = solve_mapf(grid, tasks, fast_config());
    assert(r.success);
    VerifyResult check = verify(grid, tasks, r.plan);
    assert(check.ok());
    std::printf("test_many_agents_open passed (%zu agents, soc=%.0f, overhead=%.1f%%, iters=%zu)\n",
                tasks.size(), check.sum_of_costs, check.overhead_percent(), r.iterations);
}

void test_unsolvable_corridor_reports_failure() {
    // A one-wide horizontal corridor. Two agents must swap ends, which is
    // impossible without a passing place: the solver must give up and say
    // so, not spin forever.
    std::istringstream in("type octile\nheight 1\nwidth 4\nmap\n....\n");
    Grid grid = Grid::parse(in);
    std::vector<AgentTask> tasks = {task(grid, {0, 0}, {3, 0}), task(grid, {3, 0}, {0, 0})};

    SolverResult r = solve_mapf(grid, tasks, fast_config());
    assert(!r.success);
    // And the reported plan honestly still contains a conflict.
    VerifyResult check = verify(grid, tasks, r.plan);
    assert(!check.ok());
    std::printf("test_unsolvable_corridor_reports_failure passed (iters=%zu, %zu violations)\n",
                r.iterations, check.violations.size());
}

void test_unreachable_goal_infeasible() {
    // Goal walled off: generation itself fails.
    std::istringstream in("type octile\nheight 3\nwidth 3\nmap\n...\n.@.\n@.@\n");
    Grid grid = Grid::parse(in);
    std::vector<AgentTask> tasks;
    AgentTask t;
    t.start = {0, 0};
    t.goal = {1, 2};  // (1,2) is surrounded by @ at (0,2),(2,2),(1,1)
    t.optimal_distance = 0;
    tasks.push_back(t);

    SolverResult r = solve_mapf(grid, tasks, fast_config());
    assert(!r.success);
    assert(r.infeasible_generation);
    std::printf("test_unreachable_goal_infeasible passed\n");
}

void test_determinism() {
    Grid grid = open_grid(7, 7);
    std::vector<AgentTask> tasks = {task(grid, {0, 3}, {6, 3}), task(grid, {3, 0}, {3, 6})};
    SolverConfig cfg = fast_config();

    SolverResult a = solve_mapf(grid, tasks, cfg);
    SolverResult b = solve_mapf(grid, tasks, cfg);
    assert(a.success == b.success);
    assert(a.plan.paths == b.plan.paths);
    std::printf("test_determinism passed\n");
}

void test_single_agent_optimal() {
    Grid grid = open_grid(8, 8);
    std::vector<AgentTask> tasks = {task(grid, {0, 0}, {7, 4})};
    SolverResult r = solve_mapf(grid, tasks, fast_config());
    assert(r.success);
    VerifyResult check = verify(grid, tasks, r.plan);
    assert(check.ok());
    // A lone agent should take a shortest path: zero overhead.
    assert(check.overhead_percent() == 0.0);
    std::printf("test_single_agent_optimal passed\n");
}

}  // namespace

int main() {
    test_two_agents_crossing();
    test_many_agents_open();
    test_unsolvable_corridor_reports_failure();
    test_unreachable_goal_infeasible();
    test_determinism();
    test_single_agent_optimal();
    std::printf("All solver tests passed.\n");
    return 0;
}
