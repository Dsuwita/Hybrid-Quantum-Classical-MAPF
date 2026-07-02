// test_verifier.cpp
//
// Milestone 8 verifier tests, all on hand-built plans over a tiny open
// grid so every expected conflict is visible by inspection. The spec
// requires (and this file includes) a swap-conflict case and a
// parked-agent collision case. The verifier is sacred: these tests pin
// its behavior for the whole MAPF project.

#include "mapf/grid.hpp"
#include "mapf/plan.hpp"
#include "mapf/scenario.hpp"
#include "mapf/verifier.hpp"

#include <cassert>
#include <cstdio>
#include <sstream>

using namespace mapf;

namespace {

// 5x5 open grid, built in-memory.
Grid open_grid() {
    std::istringstream in("type octile\nheight 5\nwidth 5\nmap\n.....\n.....\n.....\n.....\n.....\n");
    return Grid::parse(in);
}

AgentTask task(Cell start, Cell goal, double optimal) {
    AgentTask t;
    t.start = start;
    t.goal = goal;
    t.optimal_distance = optimal;
    return t;
}

bool has_violation(const VerifyResult& r, Violation::Type type) {
    for (const auto& v : r.violations) {
        if (v.type == type) return true;
    }
    return false;
}

void test_clean_plan() {
    Grid grid = open_grid();
    // Two agents crossing the grid on different rows: no interaction.
    std::vector<AgentTask> tasks = {task({0, 0}, {2, 0}, 2), task({0, 2}, {2, 2}, 2)};
    Plan plan;
    plan.paths = {
        {{0, 0}, {1, 0}, {2, 0}},
        {{0, 2}, {1, 2}, {2, 2}},
    };
    VerifyResult r = verify(grid, tasks, plan);
    assert(r.ok());
    assert(r.sum_of_costs == 4.0);
    assert(r.sum_of_optimal == 4.0);
    assert(r.overhead_percent() == 0.0);
    std::printf("test_clean_plan passed\n");
}

void test_vertex_conflict() {
    Grid grid = open_grid();
    // Both agents step into (1,1) at t=1.
    std::vector<AgentTask> tasks = {task({0, 1}, {2, 1}, 2), task({1, 0}, {1, 2}, 2)};
    Plan plan;
    plan.paths = {
        {{0, 1}, {1, 1}, {2, 1}},
        {{1, 0}, {1, 1}, {1, 2}},
    };
    VerifyResult r = verify(grid, tasks, plan);
    assert(!r.ok());
    assert(has_violation(r, Violation::Type::VertexConflict));
    assert(r.violations.size() == 1);
    assert(r.violations[0].t == 1);
    assert(r.violations[0].cell == (Cell{1, 1}));
    std::printf("test_vertex_conflict passed\n");
}

void test_swap_conflict() {
    Grid grid = open_grid();
    // Head-on: a goes (0,0)->(1,0), b goes (1,0)->(0,0) across t=0->1.
    // No vertex conflict exists at any single timestep; only the swap
    // check catches this.
    std::vector<AgentTask> tasks = {task({0, 0}, {1, 0}, 1), task({1, 0}, {0, 0}, 1)};
    Plan plan;
    plan.paths = {
        {{0, 0}, {1, 0}},
        {{1, 0}, {0, 0}},
    };
    VerifyResult r = verify(grid, tasks, plan);
    assert(!r.ok());
    assert(has_violation(r, Violation::Type::SwapConflict));
    assert(!has_violation(r, Violation::Type::VertexConflict));
    assert(r.violations[0].t == 1);
    std::printf("test_swap_conflict passed\n");
}

void test_parked_agent_collision() {
    Grid grid = open_grid();
    // Agent 0 reaches (2,0) at t=2 and its path ends there (parked).
    // Agent 1 walks along row 0 and stands on (2,0) at t=3, after agent
    // 0's path has ended: only the parking rule makes this a conflict.
    std::vector<AgentTask> tasks = {task({0, 0}, {2, 0}, 2), task({0, 1}, {4, 0}, 5)};
    Plan plan;
    plan.paths = {
        {{0, 0}, {1, 0}, {2, 0}},
        {{0, 1}, {0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}},
    };
    VerifyResult r = verify(grid, tasks, plan);
    assert(!r.ok());
    assert(has_violation(r, Violation::Type::VertexConflict));
    assert(r.violations.size() == 1);
    assert(r.violations[0].t == 3);
    assert(r.violations[0].cell == (Cell{2, 0}));
    std::printf("test_parked_agent_collision passed\n");
}

void test_invalid_moves() {
    Grid grid = open_grid();
    // Diagonal step (not adjacent-or-wait).
    {
        std::vector<AgentTask> tasks = {task({0, 0}, {1, 1}, 2)};
        Plan plan;
        plan.paths = {{{0, 0}, {1, 1}}};
        VerifyResult r = verify(grid, tasks, plan);
        assert(has_violation(r, Violation::Type::InvalidMove));
    }
    // Step into a blocked cell.
    {
        std::istringstream in("type octile\nheight 2\nwidth 2\nmap\n.@\n..\n");
        Grid walled = Grid::parse(in);
        std::vector<AgentTask> tasks = {task({0, 0}, {1, 0}, 1)};
        Plan plan;
        plan.paths = {{{0, 0}, {1, 0}}};  // (1,0) is '@'
        VerifyResult r = verify(walled, tasks, plan);
        assert(has_violation(r, Violation::Type::InvalidMove));
    }
    // Teleport (distance 2 on one axis).
    {
        std::vector<AgentTask> tasks = {task({0, 0}, {2, 0}, 2)};
        Plan plan;
        plan.paths = {{{0, 0}, {2, 0}}};
        VerifyResult r = verify(grid, tasks, plan);
        assert(has_violation(r, Violation::Type::InvalidMove));
    }
    std::printf("test_invalid_moves passed\n");
}

void test_endpoint_mismatch() {
    Grid grid = open_grid();
    // Ends one cell short of the goal.
    std::vector<AgentTask> tasks = {task({0, 0}, {2, 0}, 2)};
    Plan plan;
    plan.paths = {{{0, 0}, {1, 0}}};
    VerifyResult r = verify(grid, tasks, plan);
    assert(has_violation(r, Violation::Type::EndpointMismatch));
    std::printf("test_endpoint_mismatch passed\n");
}

void test_costs_and_overhead() {
    Grid grid = open_grid();
    // Agent detours: optimal 2, takes 4 (down and back up adds 2).
    std::vector<AgentTask> tasks = {task({0, 0}, {2, 0}, 2)};
    Plan plan;
    plan.paths = {{{0, 0}, {0, 1}, {1, 1}, {1, 0}, {2, 0}}};
    VerifyResult r = verify(grid, tasks, plan);
    assert(r.ok());
    assert(r.sum_of_costs == 4.0);
    assert(r.overhead_percent() == 100.0);

    // Waiting at the goal after arrival is free; leaving and returning
    // is paid up to the LAST time off-goal.
    Plan wait_plan;
    wait_plan.paths = {{{0, 0}, {1, 0}, {2, 0}, {2, 0}, {2, 0}}};
    assert(wait_plan.sum_of_costs({{2, 0}}) == 2.0);

    Plan leave_and_return;
    leave_and_return.paths = {{{0, 0}, {1, 0}, {2, 0}, {1, 0}, {2, 0}}};
    assert(leave_and_return.sum_of_costs({{2, 0}}) == 4.0);

    std::printf("test_costs_and_overhead passed\n");
}

}  // namespace

int main() {
    test_clean_plan();
    test_vertex_conflict();
    test_swap_conflict();
    test_parked_agent_collision();
    test_invalid_moves();
    test_endpoint_mismatch();
    test_costs_and_overhead();
    std::printf("All verifier tests passed.\n");
    return 0;
}
