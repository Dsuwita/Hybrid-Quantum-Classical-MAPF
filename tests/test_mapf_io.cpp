// test_mapf_io.cpp
//
// Milestone 8 parser tests: grid (.map) and scenario (.scen) loading,
// both from the committed sample files and from in-memory strings for
// the error cases.

#include "mapf/grid.hpp"
#include "mapf/scenario.hpp"

#include <cassert>
#include <cstdio>
#include <sstream>
#include <string>

using namespace mapf;

#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "tests/data"
#endif

namespace {

void test_grid_parse_sample() {
    Grid grid = Grid::load(std::string(TEST_DATA_DIR) + "/sample.map");
    assert(grid.width() == 8);
    assert(grid.height() == 8);

    // Spot checks against the committed map.
    assert(grid.passable(0, 0));
    assert(!grid.passable(2, 1));  // '@'
    assert(!grid.passable(3, 2));  // '@'
    assert(!grid.passable(6, 2));  // 'T'
    assert(!grid.passable(1, 5));  // 'T'
    assert(grid.passable(7, 7));

    // Out of bounds is not passable and does not crash.
    assert(!grid.passable(-1, 0));
    assert(!grid.passable(0, 8));

    // Count blocked cells: 4 '@' in rows 1-2, 2 '@' in rows 4-5, 2 'T'.
    int blocked = 0;
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            if (!grid.passable(x, y)) ++blocked;
        }
    }
    assert(blocked == 8);

    std::printf("test_grid_parse_sample passed\n");
}

void test_grid_moves() {
    Grid grid = Grid::load(std::string(TEST_DATA_DIR) + "/sample.map");

    // Corner cell: wait + right + down (up/left out of bounds).
    auto corner = grid.moves_from(Cell{0, 0});
    assert(corner.size() == 3);
    assert(corner[0] == (Cell{0, 0}));  // wait is always first

    // (1,1) has blocked (2,1) to its right: wait, left, down, up = 4.
    auto near_wall = grid.moves_from(Cell{1, 1});
    assert(near_wall.size() == 4);
    for (const auto& c : near_wall) assert(!(c == Cell{2, 1}));

    std::printf("test_grid_moves passed\n");
}

void test_grid_rejects_garbage() {
    bool threw = false;
    try {
        std::istringstream bad("type octile\nheight 2\nwidth 2\nmap\n..\n.X\n");
        Grid::parse(bad);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        std::istringstream truncated("type octile\nheight 3\nwidth 3\nmap\n...\n...\n");
        Grid::parse(truncated);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);

    std::printf("test_grid_rejects_garbage passed\n");
}

void test_scenario_parse_sample() {
    Scenario scen = Scenario::load(std::string(TEST_DATA_DIR) + "/sample.scen");
    assert(scen.tasks().size() == 5);

    const AgentTask& first = scen.tasks()[0];
    assert(first.bucket == 0);
    assert(first.map_name == "sample.map");
    assert(first.map_width == 8 && first.map_height == 8);
    assert(first.start == (Cell{0, 0}));
    assert(first.goal == (Cell{7, 0}));
    assert(first.optimal_distance == 7.0);

    const AgentTask& last = scen.tasks()[4];
    assert(last.bucket == 1);
    assert(last.start == (Cell{5, 7}));
    assert(last.goal == (Cell{5, 0}));

    // first_k selection, including k past the end.
    assert(scen.first_k(3).size() == 3);
    assert(scen.first_k(3)[2].start == (Cell{7, 3}));
    assert(scen.first_k(99).size() == 5);

    // Every task's endpoints must be passable on the matching grid.
    Grid grid = Grid::load(std::string(TEST_DATA_DIR) + "/sample.map");
    for (const auto& task : scen.tasks()) {
        assert(grid.passable(task.start));
        assert(grid.passable(task.goal));
    }

    std::printf("test_scenario_parse_sample passed\n");
}

void test_scenario_rejects_garbage() {
    bool threw = false;
    try {
        std::istringstream bad("version 1\n0 map.map 8 8 0 0 7\n");  // too few fields
        Scenario::parse(bad);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
    std::printf("test_scenario_rejects_garbage passed\n");
}

}  // namespace

int main() {
    test_grid_parse_sample();
    test_grid_moves();
    test_grid_rejects_garbage();
    test_scenario_parse_sample();
    test_scenario_rejects_garbage();
    std::printf("All MAPF I/O tests passed.\n");
    return 0;
}
