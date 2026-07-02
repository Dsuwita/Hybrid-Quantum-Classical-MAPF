// bench_rolling.cpp
//
// Milestone 14 metric: replan cycle time vs number of agents, with moving
// obstacles present. Each cycle the driver plans a window, solves a QUBO,
// and commits a few steps; as agents are added the per-agent candidate
// menu grows the QUBO, so the cycle gets more expensive. We run lifelong
// mode (agents get new goals on arrival) so the measurement reflects
// sustained operation, with a fixed sweep budget (deadline off) so the
// number measures the ALGORITHM's cost, not an imposed deadline.
//
// Output: CSV to stdout (agents,avg_cycle_ms,throughput_goals_per_step,
// obstacle_hits) and a readable summary to stderr. Run as
//   ./bench_rolling > mapf/bench/rolling.csv

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <random>
#include <vector>

#include "mapf/grid.hpp"
#include "mapf/rolling.hpp"
#include "mapf/verifier.hpp"

using namespace mapf;

namespace {
std::vector<Cell> patrol_vertical(int x, int ymin, int ymax, std::size_t length, int offset) {
    std::vector<Cell> tr;
    tr.reserve(length);
    const int span = ymax - ymin;
    for (std::size_t t = 0; t < length; ++t) {
        const int phase = static_cast<int>((static_cast<long>(t) + offset) % (2 * span));
        tr.push_back(Cell{x, phase <= span ? ymin + phase : ymax - (phase - span)});
    }
    return tr;
}
}  // namespace

int main() {
    const int N = 20;
    std::ostringstream os;
    os << "type octile\nheight " << N << "\nwidth " << N << "\nmap\n";
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) os << '.';
        os << '\n';
    }
    std::istringstream in(os.str());
    Grid grid = Grid::parse(in);

    // Four patrolling obstacles spread across the field.
    ObstacleModel obs;
    obs.paths = {patrol_vertical(4, 0, N - 1, 600, 0), patrol_vertical(9, 0, N - 1, 600, 7),
                 patrol_vertical(14, 0, N - 1, 600, 3), patrol_vertical(18, 0, N - 1, 600, 11)};
    obs.perfect_prediction = true;

    std::vector<Cell> free;
    for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) free.push_back(Cell{x, y});

    GoalProvider provider = [&grid](std::size_t, std::mt19937_64& r) {
        std::uniform_int_distribution<int> d(0, grid.width() - 1);
        return Cell{d(r), d(r)};
    };

    std::printf("agents,avg_cycle_ms,throughput_goals_per_step,obstacle_hits\n");
    std::fprintf(stderr, "Rolling-horizon cycle time vs agents (20x20, 4 obstacles, lifelong)\n\n");

    for (std::size_t k : {2u, 4u, 8u, 16u, 24u, 32u}) {
        std::mt19937_64 rng(100 + k);
        std::shuffle(free.begin(), free.end(), rng);
        std::vector<Cell> starts(free.begin(), free.begin() + k);
        std::vector<Cell> goals(free.begin() + k, free.begin() + 2 * k);

        RollingConfig cfg;
        cfg.window = 8;
        cfg.execute = 3;
        cfg.cycle_deadline_ms = 0.0;  // measure algorithm cost, not a deadline
        cfg.sweeps = 3000;
        cfg.replicas = 12;
        cfg.candidates_per_agent = 5;
        cfg.max_steps = 90;
        cfg.seed = 7;

        RollingResult r = simulate_rolling(grid, starts, goals, cfg, provider, &obs);
        double avg_cycle = r.cycles ? r.wall_ms / r.cycles : 0.0;
        double thru = r.steps ? double(r.goals_reached) / r.steps : 0.0;

        std::printf("%zu,%.2f,%.3f,%zu\n", k, avg_cycle, thru, r.obstacle_hits);
        std::fflush(stdout);
        std::fprintf(stderr, "  k=%2zu: %6.2f ms/cycle  throughput %.3f goals/step  hits %zu\n", k,
                     avg_cycle, thru, r.obstacle_hits);
    }
    return 0;
}
