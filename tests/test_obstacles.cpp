// test_obstacles.cpp
//
// Milestone 14 tests: agents dodging moving obstacles. With PERFECT
// prediction the guarantee is strong -- the execution history has zero
// agent-obstacle overlaps and passes the agent-agent verifier -- because
// the time-expanded A* routes (or waits) around the obstacles' predicted
// cells. With imperfect (ball) prediction the braking safety rule keeps
// agents out of obstacles where it can; any residual overlap is counted
// and reported, and agent-agent safety still holds.

#include "mapf/grid.hpp"
#include "mapf/rolling.hpp"
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

// An obstacle that sweeps up and down a column between ymin and ymax.
std::vector<Cell> patrol_vertical(int x, int ymin, int ymax, std::size_t length, int offset = 0) {
    std::vector<Cell> tr;
    tr.reserve(length);
    const int span = ymax - ymin;
    for (std::size_t t = 0; t < length; ++t) {
        const int phase = static_cast<int>((static_cast<long>(t) + offset) % (2 * span));
        const int y = phase <= span ? ymin + phase : ymax - (phase - span);
        tr.push_back(Cell{x, y});
    }
    return tr;
}

void assert_agent_agent_valid(const Grid& grid, const RollingResult& r) {
    std::vector<AgentTask> tasks;
    for (const auto& p : r.history.paths) {
        AgentTask t;
        t.start = p.front();
        t.goal = p.back();
        tasks.push_back(t);
    }
    assert(verify(grid, tasks, r.history).ok());
}

// Independent check that no agent ever shares a cell with an obstacle.
std::size_t count_obstacle_overlaps(const RollingResult& r, const ObstacleModel& obs) {
    std::size_t hits = 0;
    for (std::size_t t = 0; t <= r.steps; ++t) {
        for (const auto& path : r.history.paths) {
            Cell c = t < path.size() ? path[t] : path.back();
            for (std::size_t o = 0; o < obs.count(); ++o)
                if (obs.at(o, t) == c) ++hits;
        }
    }
    return hits;
}

RollingConfig cfg_det() {
    RollingConfig c;
    c.window = 8;
    c.execute = 2;
    c.cycle_deadline_ms = 0.0;  // deterministic
    c.sweeps = 3000;
    c.replicas = 6;
    c.threads = 4;
    c.candidates_per_agent = 6;
    c.max_steps = 120;
    return c;
}

void test_single_agent_dodges_patrol() {
    Grid grid = open_grid(11, 7);
    std::vector<Cell> starts = {{0, 3}};
    std::vector<Cell> goals = {{10, 3}};
    // Obstacle sweeps column 5, crossing the agent's straight-line row 3.
    ObstacleModel obs;
    obs.paths = {patrol_vertical(5, 0, 6, 200)};
    obs.perfect_prediction = true;

    RollingResult r = simulate_rolling(grid, starts, goals, cfg_det(), {}, &obs);
    assert_agent_agent_valid(grid, r);
    assert(r.obstacle_hits == 0);
    assert(count_obstacle_overlaps(r, obs) == 0);  // independent confirmation
    assert(r.all_done);
    std::printf("test_single_agent_dodges_patrol passed (%zu steps, %zu cycles)\n", r.steps,
                r.cycles);
}

void test_several_agents_dodge_cleanly() {
    // Three agents on separate rows crossing one patrolled column, with room
    // to avoid each other. No gridlock, perfect prediction -> zero overlaps
    // and every agent arrives.
    Grid grid = open_grid(15, 11);
    std::vector<Cell> starts = {{0, 2}, {0, 5}, {0, 8}};
    std::vector<Cell> goals = {{14, 2}, {14, 5}, {14, 8}};
    ObstacleModel obs;
    obs.paths = {patrol_vertical(7, 0, 10, 300)};
    obs.perfect_prediction = true;

    RollingResult r = simulate_rolling(grid, starts, goals, cfg_det(), {}, &obs);
    assert_agent_agent_valid(grid, r);
    assert(count_obstacle_overlaps(r, obs) == 0);
    assert(r.all_done);
    std::printf("test_several_agents_dodge_cleanly passed (%zu steps, %zu cycles)\n", r.steps,
                r.cycles);
}

void test_uncertainty_radius_still_clean() {
    // Perfect prediction with a safety margin: forbid a 1-cell ring around
    // the obstacle too. Still zero overlaps, at the cost of wider detours.
    Grid grid = open_grid(11, 7);
    std::vector<Cell> starts = {{0, 3}};
    std::vector<Cell> goals = {{10, 3}};
    ObstacleModel obs;
    obs.paths = {patrol_vertical(5, 0, 6, 200)};
    obs.perfect_prediction = true;
    obs.prediction_radius = 1;

    RollingResult r = simulate_rolling(grid, starts, goals, cfg_det(), {}, &obs);
    assert_agent_agent_valid(grid, r);
    assert(count_obstacle_overlaps(r, obs) == 0);
    assert(r.all_done);
    std::printf("test_uncertainty_radius_still_clean passed (%zu steps)\n", r.steps);
}

void test_imperfect_prediction_agent_agent_still_safe() {
    // Ball prediction (the planner does NOT see the obstacle's true future,
    // only a growing ball around its current cell). Overlaps may occur if an
    // obstacle actively chases the agent, but the agent-agent history must
    // still be valid and braking should keep overlaps rare.
    Grid grid = open_grid(11, 7);
    std::vector<Cell> starts = {{0, 3}, {10, 3}};
    std::vector<Cell> goals = {{10, 3}, {0, 3}};
    ObstacleModel obs;
    obs.paths = {patrol_vertical(5, 1, 5, 200)};
    obs.perfect_prediction = false;
    obs.prediction_radius = 1;

    RollingResult r = simulate_rolling(grid, starts, goals, cfg_det(), {}, &obs);
    assert_agent_agent_valid(grid, r);  // agent-agent safety is unconditional
    std::printf("test_imperfect_prediction_agent_agent_still_safe passed "
                "(%zu steps, obstacle hits=%zu)\n",
                r.steps, r.obstacle_hits);
}

void test_determinism() {
    Grid grid = open_grid(11, 7);
    std::vector<Cell> starts = {{0, 3}};
    std::vector<Cell> goals = {{10, 3}};
    ObstacleModel obs;
    obs.paths = {patrol_vertical(5, 0, 6, 200)};

    RollingResult a = simulate_rolling(grid, starts, goals, cfg_det(), {}, &obs);
    RollingResult b = simulate_rolling(grid, starts, goals, cfg_det(), {}, &obs);
    assert(a.history.paths == b.history.paths);
    assert(a.obstacle_hits == b.obstacle_hits);
    std::printf("test_determinism passed\n");
}

}  // namespace

int main() {
    test_single_agent_dodges_patrol();
    test_several_agents_dodge_cleanly();
    test_uncertainty_radius_still_clean();
    test_imperfect_prediction_agent_agent_still_safe();
    test_determinism();
    std::printf("All moving-obstacle tests passed.\n");
    return 0;
}
