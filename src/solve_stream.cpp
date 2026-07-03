// solve_stream.cpp
//
// Streaming hybrid MAPF solver for the interactive studio. It runs the
// rolling-horizon annealer (mapf/rolling.hpp) in one-shot mode -- replan a
// window, commit E steps, repeat until every agent reaches its goal -- and
// emits one line of newline-delimited JSON per committed cycle to stdout,
// flushing each line. serve.py reads these lines and forwards them to the
// browser as Server-Sent Events, so the canvas animates the agents moving as
// the solver commits each window rather than waiting for a finished plan.
//
// Output protocol (one JSON object per line):
//   {"event":"meta", grid, agents:[{sx,sy,gx,gy}], obstacles:[[x,y]...]}
//   {"event":"cycle", cycle, timestep, cycle_ms, active, goals_reached,
//                     overhead_pct, frames:[[[x,y]...] per step], obstacles}
//   {"event":"done", success, steps, cycles, sum_of_costs, sum_of_optimal,
//                    overhead_pct, obstacle_hits, wall_ms, paths, obstacle_paths}
//
// Usage:
//   solve_stream <map> <scen> <k> [options]
// Options mirror the rolling config: --window --execute --sweeps --replicas
//   --threads --deadline-ms --candidates --seed --max-steps
//   --obstacles N --motion scripted|random  (moving obstacles, optional)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "mapf/grid.hpp"
#include "mapf/rolling.hpp"
#include "mapf/scenario.hpp"
#include "mapf/verifier.hpp"

using namespace mapf;

namespace {

long argl(int argc, char** argv, const char* flag, long fallback) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return std::atol(argv[i + 1]);
    return fallback;
}
double argd(int argc, char** argv, const char* flag, double fallback) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return std::atof(argv[i + 1]);
    return fallback;
}
const char* args(int argc, char** argv, const char* flag, const char* fallback) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
    return fallback;
}

// Minimal JSON array-of-positions printers (no dependency).
void print_frame(const std::vector<Cell>& f) {
    std::printf("[");
    for (std::size_t a = 0; a < f.size(); ++a)
        std::printf("%s[%d,%d]", a ? "," : "", f[a].x, f[a].y);
    std::printf("]");
}
void print_path(const std::vector<Cell>& p) { print_frame(p); }

// A patrolling vertical obstacle, bouncing between ymin and ymax.
std::vector<Cell> patrol_vertical(int x, int ymin, int ymax, std::size_t length, int offset) {
    std::vector<Cell> tr;
    tr.reserve(length);
    const int span = std::max(1, ymax - ymin);
    for (std::size_t t = 0; t < length; ++t) {
        const int phase = static_cast<int>((static_cast<long>(t) + offset) % (2 * span));
        tr.push_back(Cell{x, phase <= span ? ymin + phase : ymax - (phase - span)});
    }
    return tr;
}

// A random-walk obstacle over passable cells.
std::vector<Cell> random_walk(const Grid& grid, Cell start, std::size_t length,
                              std::mt19937_64& rng) {
    std::vector<Cell> tr;
    tr.reserve(length);
    Cell c = start;
    for (std::size_t t = 0; t < length; ++t) {
        tr.push_back(c);
        auto moves = grid.moves_from(c);
        c = moves[rng() % moves.size()];
    }
    return tr;
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
    const std::size_t n = tasks.size();

    std::vector<Cell> starts(n), goals(n);
    double sum_optimal = 0.0;
    for (std::size_t a = 0; a < n; ++a) {
        starts[a] = tasks[a].start;
        goals[a] = tasks[a].goal;
        sum_optimal += tasks[a].optimal_distance;
    }

    RollingConfig cfg;
    cfg.window = static_cast<std::size_t>(argl(argc, argv, "--window", 8));
    cfg.execute = static_cast<std::size_t>(argl(argc, argv, "--execute", 3));
    cfg.sweeps = static_cast<std::size_t>(argl(argc, argv, "--sweeps", 4000));
    cfg.replicas = static_cast<std::size_t>(argl(argc, argv, "--replicas", 8));
    cfg.threads = static_cast<std::size_t>(argl(argc, argv, "--threads", 0));
    cfg.cycle_deadline_ms = argd(argc, argv, "--deadline-ms", 0.0);
    cfg.candidates_per_agent = static_cast<std::size_t>(argl(argc, argv, "--candidates", 5));
    cfg.max_steps = static_cast<std::size_t>(argl(argc, argv, "--max-steps", 256));
    cfg.seed = static_cast<std::uint64_t>(argl(argc, argv, "--seed", 1));

    // Optional moving obstacles.
    ObstacleModel obs;
    const long n_obs = argl(argc, argv, "--obstacles", 0);
    const std::string motion = args(argc, argv, "--motion", "scripted");
    if (n_obs > 0) {
        std::mt19937_64 orng(cfg.seed ^ 0x9e3779b97f4a7c15ULL);
        const std::size_t traj_len = cfg.max_steps + cfg.window + 4;
        for (long o = 0; o < n_obs; ++o) {
            if (motion == "random") {
                Cell s{static_cast<int>(orng() % grid.width()),
                       static_cast<int>(orng() % grid.height())};
                if (!grid.passable(s)) s = starts.empty() ? Cell{0, 0} : starts[0];
                obs.paths.push_back(random_walk(grid, s, traj_len, orng));
            } else {
                const int col = 1 + static_cast<int>((o + 1) * grid.width() / (n_obs + 1));
                obs.paths.push_back(
                    patrol_vertical(std::min(col, grid.width() - 1), 0, grid.height() - 1,
                                    traj_len, static_cast<int>(o) * 3));
            }
        }
        obs.perfect_prediction = (motion != "random");
        obs.prediction_radius = (motion == "random") ? 1 : 0;
    }
    const ObstacleModel* obs_ptr = n_obs > 0 ? &obs : nullptr;

    // Emit the meta line up front so the browser can draw the initial state.
    std::printf("{\"event\":\"meta\",\"grid\":{\"w\":%d,\"h\":%d,\"blocked\":[",
                grid.width(), grid.height());
    bool first = true;
    for (int y = 0; y < grid.height(); ++y)
        for (int x = 0; x < grid.width(); ++x)
            if (!grid.passable(x, y)) {
                std::printf("%s[%d,%d]", first ? "" : ",", x, y);
                first = false;
            }
    std::printf("]},\"solver\":\"hybrid\",\"agents\":[");
    for (std::size_t a = 0; a < n; ++a)
        std::printf("%s{\"sx\":%d,\"sy\":%d,\"gx\":%d,\"gy\":%d}", a ? "," : "", starts[a].x,
                    starts[a].y, goals[a].x, goals[a].y);
    std::printf("],\"obstacles\":[");
    for (std::size_t o = 0; o < obs.count(); ++o)
        std::printf("%s[%d,%d]", o ? "," : "", obs.at(o, 0).x, obs.at(o, 0).y);
    std::printf("]}\n");
    std::fflush(stdout);

    // Running cost-so-far bookkeeping for a live overhead estimate: track the
    // last global timestep at which each agent was off its goal.
    std::vector<long> last_off(n, -1);
    std::size_t global_t = 0;

    ProgressCallback cb = [&](const RollingProgress& p) {
        for (const auto& frame : p.frames) {
            for (std::size_t a = 0; a < n; ++a)
                if (!(frame[a] == goals[a])) last_off[a] = static_cast<long>(global_t);
            ++global_t;
        }
        double cost_so_far = 0.0;
        for (std::size_t a = 0; a < n; ++a) cost_so_far += static_cast<double>(last_off[a] + 1);
        double overhead = sum_optimal > 0.0 ? 100.0 * (cost_so_far - sum_optimal) / sum_optimal : 0.0;

        std::printf(
            "{\"event\":\"cycle\",\"cycle\":%zu,\"timestep\":%zu,\"cycle_ms\":%.2f,"
            "\"active\":%zu,\"goals_reached\":%zu,\"overhead_pct\":%.2f,\"frames\":[",
            p.cycle, p.step, p.cycle_ms, p.active, p.goals_reached, overhead);
        for (std::size_t f = 0; f < p.frames.size(); ++f) {
            if (f) std::printf(",");
            print_frame(p.frames[f]);
        }
        std::printf("],\"obstacles\":[");
        for (std::size_t o = 0; o < p.obstacles.size(); ++o)
            std::printf("%s[%d,%d]", o ? "," : "", p.obstacles[o].x, p.obstacles[o].y);
        std::printf("]}\n");
        std::fflush(stdout);
    };

    RollingResult r = simulate_rolling(grid, starts, goals, cfg, {}, obs_ptr, cb);

    // Final verification against the original tasks (agent-agent only; the
    // rolling driver guarantees obstacle safety separately).
    VerifyResult vr = verify(grid, tasks, r.history);

    std::printf(
        "{\"event\":\"done\",\"success\":%s,\"steps\":%zu,\"cycles\":%zu,"
        "\"sum_of_costs\":%.0f,\"sum_of_optimal\":%.0f,\"overhead_pct\":%.2f,"
        "\"conflicts\":%zu,\"obstacle_hits\":%zu,\"wall_ms\":%.1f,\"paths\":[",
        r.all_done ? "true" : "false", r.steps, r.cycles, vr.sum_of_costs, vr.sum_of_optimal,
        vr.overhead_percent(), vr.violations.size(), r.obstacle_hits, r.wall_ms);
    for (std::size_t a = 0; a < n; ++a) {
        if (a) std::printf(",");
        print_path(r.history.paths[a]);
    }
    std::printf("],\"obstacle_paths\":[");
    for (std::size_t o = 0; o < obs.count(); ++o) {
        if (o) std::printf(",");
        std::printf("[");
        for (std::size_t t = 0; t <= r.steps; ++t)
            std::printf("%s[%d,%d]", t ? "," : "", obs.at(o, t).x, obs.at(o, t).y);
        std::printf("]");
    }
    std::printf("]}\n");
    std::fflush(stdout);

    return r.all_done ? 0 : 1;
}
