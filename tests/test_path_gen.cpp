// test_path_gen.cpp
//
// Milestone 9 tests for candidate path generation. The verifier from
// Milestone 8 is the oracle: every generated candidate, padded or not,
// must pass single-agent validity (correct endpoints, adjacent-or-wait
// moves, no blocked cells). The first candidate's cost must equal the
// shortest path length, which on an open map is the Manhattan distance.

#include "mapf/grid.hpp"
#include "mapf/path_gen.hpp"
#include "mapf/plan.hpp"
#include "mapf/verifier.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <set>
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

AgentTask task(Cell start, Cell goal, double optimal) {
    AgentTask t;
    t.start = start;
    t.goal = goal;
    t.optimal_distance = optimal;
    return t;
}

// A candidate is valid if a single-agent plan built from it verifies
// clean against the grid and the agent's task.
void assert_valid_candidate(const Grid& grid, Cell start, Cell goal, const std::vector<Cell>& path,
                            double optimal) {
    std::vector<AgentTask> tasks = {task(start, goal, optimal)};
    Plan plan;
    plan.paths = {path};
    VerifyResult r = verify(grid, tasks, plan);
    assert(r.ok());
}

void test_first_candidate_is_shortest_open_map() {
    Grid grid = open_grid(10, 10);
    Cell start{0, 0}, goal{7, 5};
    double manhattan = 12.0;  // 7 + 5

    auto candidates = generate_candidates(grid, start, goal, 5, /*seed=*/1);
    assert(!candidates.empty());
    // First candidate is a shortest path: cost == Manhattan on an open map.
    assert(candidates[0].cost == manhattan);
    assert(candidates[0].path.front() == start);
    assert(candidates[0].path.back() == goal);
    assert_valid_candidate(grid, start, goal, candidates[0].path, manhattan);

    // Every candidate is valid and never cheaper than optimal.
    for (const auto& c : candidates) {
        assert(c.cost >= manhattan);
        assert_valid_candidate(grid, start, goal, c.path, manhattan);
    }
    std::printf("test_first_candidate_is_shortest_open_map passed (%zu candidates)\n",
                candidates.size());
}

void test_candidates_are_distinct_and_diverse() {
    Grid grid = open_grid(10, 10);
    Cell start{0, 0}, goal{9, 9};

    auto candidates = generate_candidates(grid, start, goal, 6, /*seed=*/7);
    // An open 10x10 with a diagonal task has many shortest paths, so we
    // expect several distinct candidates.
    assert(candidates.size() >= 3);

    std::set<std::vector<int>> sigs;
    for (const auto& c : candidates) {
        std::vector<int> sig;
        for (const Cell& cell : c.path) {
            sig.push_back(cell.x);
            sig.push_back(cell.y);
        }
        assert(sigs.insert(sig).second);  // all distinct
        assert_valid_candidate(grid, start, goal, c.path, 18.0);
    }
    std::printf("test_candidates_are_distinct_and_diverse passed (%zu distinct)\n",
                candidates.size());
}

void test_obstacle_map() {
    // A wall with a single gap forces every path through one cell.
    //   . . . . .
    //   . . @ . .
    //   . . @ . .
    //   . . . . .   <- gap at (2,3)
    //   . . @ . .
    std::istringstream in(
        "type octile\nheight 5\nwidth 5\nmap\n.....\n..@..\n..@..\n.....\n..@..\n");
    Grid grid = Grid::parse(in);
    Cell start{0, 0}, goal{4, 0};

    auto candidates = generate_candidates(grid, start, goal, 4, /*seed=*/3);
    assert(!candidates.empty());
    for (const auto& c : candidates) {
        assert_valid_candidate(grid, start, goal, c.path, 4.0);
        // No candidate may pass through a blocked cell (validity already
        // guarantees this; assert directly as a second check).
        for (const Cell& cell : c.path) assert(grid.passable(cell));
    }
    std::printf("test_obstacle_map passed (%zu candidates)\n", candidates.size());
}

void test_unreachable_goal() {
    // Goal walled off entirely: (2,2) surrounded by blocked cells.
    std::istringstream in(
        "type octile\nheight 5\nwidth 5\nmap\n.....\n..@..\n.@.@.\n..@..\n.....\n");
    Grid grid = Grid::parse(in);
    auto candidates = generate_candidates(grid, Cell{0, 0}, Cell{2, 2}, 3, /*seed=*/1);
    assert(candidates.empty());
    std::printf("test_unreachable_goal passed\n");
}

void test_start_equals_goal() {
    Grid grid = open_grid(5, 5);
    auto candidates = generate_candidates(grid, Cell{2, 2}, Cell{2, 2}, 3, /*seed=*/1);
    assert(candidates.size() == 1);
    assert(candidates[0].cost == 0.0);
    assert(candidates[0].path.size() == 1);
    std::printf("test_start_equals_goal passed\n");
}

void test_determinism() {
    Grid grid = open_grid(8, 8);
    auto a = generate_candidates(grid, Cell{0, 0}, Cell{7, 7}, 5, /*seed=*/99);
    auto b = generate_candidates(grid, Cell{0, 0}, Cell{7, 7}, 5, /*seed=*/99);
    assert(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        assert(a[i].path == b[i].path);
        assert(a[i].cost == b[i].cost);
    }
    std::printf("test_determinism passed\n");
}

void test_padding_preserves_validity() {
    Grid grid = open_grid(8, 8);
    Cell start{0, 0}, goal{5, 3};
    auto candidates = generate_candidates(grid, start, goal, 3, /*seed=*/2);
    assert(!candidates.empty());

    // Pad all candidates to a common makespan and re-verify: parking at
    // the goal is a run of wait moves, which must stay valid.
    std::size_t makespan = 0;
    for (const auto& c : candidates) makespan = std::max(makespan, c.path.size() - 1);
    makespan += 3;  // extra parking

    for (const auto& c : candidates) {
        auto padded = pad_to(c.path, makespan);
        assert(padded.size() == makespan + 1);
        assert(padded.back() == goal);
        assert_valid_candidate(grid, start, goal, padded, 8.0);
    }
    std::printf("test_padding_preserves_validity passed\n");
}

}  // namespace

int main() {
    test_first_candidate_is_shortest_open_map();
    test_candidates_are_distinct_and_diverse();
    test_obstacle_map();
    test_unreachable_goal();
    test_start_equals_goal();
    test_determinism();
    test_padding_preserves_validity();
    std::printf("All path generation tests passed.\n");
    return 0;
}
