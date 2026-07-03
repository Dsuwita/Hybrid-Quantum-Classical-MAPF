// test_rolling.cpp
//
// Milestone 13 tests for rolling-horizon replanning. The execution
// HISTORY the driver produces must always pass the Milestone 8 verifier
// (no vertex/swap/invalid moves) -- that safety guarantee holds even when
// the annealer is cut short by its deadline. On easy instances every
// agent should also reach its goal; in lifelong mode goals should keep
// being reached. Also checks the annealer's new anytime deadline directly.

#include "anneal/bqm.hpp"
#include "anneal/fast_annealer.hpp"
#include "anneal/schedule.hpp"
#include "mapf/grid.hpp"
#include "mapf/rolling.hpp"
#include "mapf/scenario.hpp"
#include "mapf/verifier.hpp"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
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

// The execution history must be collision-free and use only legal moves.
// Synthesize tasks whose endpoints match the history so the verifier's
// endpoint check is satisfied and only movement/conflict problems surface.
void assert_history_valid(const Grid& grid, const RollingResult& r) {
    std::vector<AgentTask> tasks;
    for (const auto& path : r.history.paths) {
        AgentTask t;
        t.start = path.front();
        t.goal = path.back();
        tasks.push_back(t);
    }
    VerifyResult vr = verify(grid, tasks, r.history);
    assert(vr.ok());
    // Every agent's history spans the same number of committed steps.
    for (const auto& path : r.history.paths) {
        assert(path.size() == r.steps + 1);
    }
}

RollingConfig fast_cfg() {
    RollingConfig c;
    c.window = 6;
    c.execute = 3;
    c.cycle_deadline_ms = 0.0;  // deterministic: run the fixed sweep budget
    c.sweeps = 3000;
    c.replicas = 6;
    c.threads = 4;
    c.candidates_per_agent = 5;
    c.max_steps = 120;
    return c;
}

void test_two_agents_reach_goals() {
    Grid grid = open_grid(7, 7);
    std::vector<Cell> starts = {{0, 3}, {3, 0}};
    std::vector<Cell> goals = {{6, 3}, {3, 6}};
    RollingResult r = simulate_rolling(grid, starts, goals, fast_cfg());
    assert_history_valid(grid, r);
    assert(r.all_done);
    for (std::size_t a = 0; a < starts.size(); ++a) {
        assert(r.history.paths[a].back() == goals[a]);
    }
    std::printf("test_two_agents_reach_goals passed (%zu steps, %zu goals)\n", r.steps,
                r.goals_reached);
}

void test_four_agents_reach_goals() {
    Grid grid = open_grid(9, 9);
    std::vector<Cell> starts = {{0, 0}, {8, 0}, {0, 8}, {8, 8}};
    std::vector<Cell> goals = {{8, 8}, {0, 8}, {8, 0}, {0, 0}};
    RollingResult r = simulate_rolling(grid, starts, goals, fast_cfg());
    assert_history_valid(grid, r);
    assert(r.all_done);
    std::printf("test_four_agents_reach_goals passed (%zu steps)\n", r.steps);
}

void test_history_valid_under_tiny_deadline() {
    // A 1 ms deadline gives the annealer almost no time; the committed
    // history must STILL be collision-free (safety does not depend on
    // annealer quality -- only progress does).
    Grid grid = open_grid(8, 8);
    std::vector<Cell> starts = {{0, 4}, {7, 4}, {4, 0}, {4, 7}};
    std::vector<Cell> goals = {{7, 4}, {0, 4}, {4, 7}, {4, 0}};
    RollingConfig c = fast_cfg();
    c.cycle_deadline_ms = 1.0;
    c.sweeps = 1000000;  // huge; the deadline is what stops each cycle
    RollingResult r = simulate_rolling(grid, starts, goals, c);
    assert_history_valid(grid, r);
    std::printf("test_history_valid_under_tiny_deadline passed (%zu steps, done=%d)\n", r.steps,
                r.all_done);
}

void test_lifelong_keeps_reaching_goals() {
    Grid grid = open_grid(10, 10);
    std::vector<Cell> starts = {{0, 0}, {9, 9}, {0, 9}, {9, 0}};
    std::vector<Cell> goals = {{9, 9}, {0, 0}, {9, 0}, {0, 9}};

    // Provider: on arrival, hand out a random passable cell as the next goal.
    GoalProvider provider = [&grid](std::size_t, std::mt19937_64& rng) {
        std::uniform_int_distribution<int> dx(0, grid.width() - 1), dy(0, grid.height() - 1);
        while (true) {
            Cell c{dx(rng), dy(rng)};
            if (grid.passable(c)) return c;
        }
    };

    RollingConfig c = fast_cfg();
    c.max_steps = 120;
    RollingResult r = simulate_rolling(grid, starts, goals, c, provider);
    assert_history_valid(grid, r);
    // Over 120 steps with new goals on arrival, several goals get reached.
    assert(r.goals_reached >= 4);
    std::printf("test_lifelong_keeps_reaching_goals passed (%zu goals in %zu steps)\n",
                r.goals_reached, r.steps);
}

void test_determinism() {
    Grid grid = open_grid(7, 7);
    std::vector<Cell> starts = {{0, 3}, {3, 0}};
    std::vector<Cell> goals = {{6, 3}, {3, 6}};
    RollingConfig c = fast_cfg();  // deadline 0 -> deterministic
    RollingResult a = simulate_rolling(grid, starts, goals, c);
    RollingResult b = simulate_rolling(grid, starts, goals, c);
    assert(a.steps == b.steps);
    assert(a.history.paths == b.history.paths);
    std::printf("test_determinism passed\n");
}

// The anytime deadline at the annealer level: a huge sweep budget with a
// small deadline returns promptly with a correct (audited) best state.
void test_annealer_deadline() {
    // Integer couplings so incremental energy tracking is exact and the
    // audit can use ==. (Float couplings accumulate rounding over many
    // sweeps; that is expected and is why the M3 audit uses integers for
    // exact checks.)
    std::mt19937_64 rng(7);
    anneal::BQM bqm(200, anneal::Vartype::Spin);
    std::uniform_int_distribution<int> bias(-3, 3);
    for (std::size_t i = 0; i < 200; ++i) bqm.add_linear(i, bias(rng));
    for (std::size_t i = 0; i < 200; ++i)
        for (std::size_t j = i + 1; j < 200; ++j)
            if (std::uniform_real_distribution<double>(0, 1)(rng) < 0.05) {
                int c = bias(rng);
                if (c != 0) bqm.add_interaction(i, j, c);
            }

    anneal::GeometricSchedule schedule(5.0, 0.999);
    anneal::FastAnnealer<anneal::GeometricSchedule> annealer(bqm, schedule, 100000000, 1,
                                                             /*max_wall_ms=*/5.0);
    auto t0 = std::chrono::steady_clock::now();
    anneal::SolveResult res = annealer.solve();
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                    .count();

    // Stopped near the deadline, not after 100M sweeps.
    assert(ms < 500.0);
    // Best-so-far is a real, audited state.
    assert(bqm.energy(res.best_state) == res.best_energy);
    std::printf("test_annealer_deadline passed (%.1f ms, E=%.3f)\n", ms, res.best_energy);
}

}  // namespace

int main() {
    test_two_agents_reach_goals();
    test_four_agents_reach_goals();
    test_history_valid_under_tiny_deadline();
    test_lifelong_keeps_reaching_goals();
    test_determinism();
    test_annealer_deadline();
    std::printf("All rolling-horizon tests passed.\n");
    return 0;
}
