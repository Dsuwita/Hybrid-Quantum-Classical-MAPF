// rolling.hpp
//
// Rolling-horizon MAPF (RHCR-style: Li et al., "Lifelong Multi-Agent Path
// Finding in Large-Scale Warehouses", AAAI 2021). Project 2 planned one
// full set of paths up front; here we replan continuously under a global
// clock, which is what real deployments need:
//
//   each cycle, from the agents' CURRENT positions:
//     - generate candidate paths to their current goals (classical A*)
//     - resolve conflicts only within a WINDOW of W timesteps (a small
//       QUBO, solved by the annealer under a per-cycle wall-clock
//       DEADLINE -- anytime: best-so-far when the deadline hits)
//     - COMMIT the first E < W steps of the chosen paths to the execution
//       history, advance the clock by E, and replan from the new positions
//
// Only committing E of the W planned steps is the key idea: the far future
// is uncertain (other agents, and in Milestone 14 moving obstacles), so we
// plan far enough ahead to avoid imminent trouble but commit only a little,
// then look again. Planning a bounded window each cycle keeps the QUBO
// small no matter how long the agents actually travel.
//
// Lifelong mode: when an agent reaches its goal it is handed a new one by a
// caller-supplied provider and keeps moving, so the simulation runs
// indefinitely (until a step cap) rather than ending when goals are met.
//
// Safety of the committed history: each cycle we commit only the longest
// conflict-free prefix of the planned window (re-checked directly here,
// never trusting the annealer). If even the first step would collide, all
// agents wait one step instead -- always collision-free because they were
// non-overlapping at the cycle start and nobody moves. The produced
// history therefore always passes the verifier; progress (not just safety)
// is what can degrade under heavy congestion, and we report that honestly.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <vector>

#include "anneal/parallel.hpp"
#include "anneal/schedule.hpp"
#include "mapf/conflict_qubo.hpp"
#include "mapf/grid.hpp"
#include "mapf/path_gen.hpp"
#include "mapf/plan.hpp"

namespace mapf {

struct RollingConfig {
    std::size_t window = 8;             // W: timesteps planned/resolved per cycle
    std::size_t execute = 3;            // E: timesteps committed per cycle (E < W)
    double cycle_deadline_ms = 20.0;    // anytime annealer deadline per cycle
    std::size_t max_steps = 256;        // global clock cap
    std::size_t sweeps = 50000;         // annealer sweep cap (deadline usually bites first)
    std::size_t candidates_per_agent = 5;
    std::size_t replicas = 8;
    std::size_t threads = 0;            // 0 = hardware_concurrency
    std::uint64_t seed = 1;
    double schedule_t0 = 5.0;
    double schedule_alpha = 0.99;
};

struct RollingResult {
    Plan history;                 // execution history: history.paths[a][t]
    std::size_t steps = 0;        // global timesteps simulated
    std::size_t cycles = 0;       // replan cycles run (for cycle-time metrics)
    std::size_t goals_reached = 0;
    std::size_t obstacle_hits = 0;  // agent-on-obstacle overlaps (0 with perfect prediction)
    bool all_done = false;        // one-shot: every agent ended on its goal
    double wall_ms = 0.0;
};

// Moving obstacles. Each obstacle has a true trajectory (its actual cell at
// each global timestep); the planner sees a PREDICTION of that, which may be
// exact (perfect_prediction) or, for genuinely unpredictable movers, a ball
// around the current position that grows with look-ahead (prediction_radius
// as the base uncertainty). Predicted cells are forbidden in candidate
// generation; when a prediction is nonetheless violated at execution time
// the affected agent brakes (see the driver). This models the robot-soccer
// setting: teammates and the ball move on their own and must be dodged.
struct ObstacleModel {
    std::vector<std::vector<Cell>> paths;  // paths[o][t] = obstacle o's true cell at time t
    int prediction_radius = 0;             // safety margin / base uncertainty (cells)
    bool perfect_prediction = true;        // predict the true future vs a growing ball

    std::size_t count() const { return paths.size(); }
    Cell at(std::size_t o, std::size_t t) const {
        const auto& p = paths[o];
        return p.empty() ? Cell{-1, -1} : (t < p.size() ? p[t] : p.back());
    }
};

// Lifelong goal source: called when agent `a` reaches its goal, returns its
// next goal. Leave empty for one-shot mode (agents stop once they arrive).
using GoalProvider = std::function<Cell(std::size_t a, std::mt19937_64& rng)>;

// Per-cycle progress, reported to a caller-supplied callback right after each
// replan cycle commits its steps. This is what the studio's Server-Sent
// Events endpoint forwards to the browser so the canvas animates the solve
// live (agents move as each window is committed) instead of waiting for the
// whole run to finish. `frames` are the cells committed THIS cycle, one entry
// per committed timestep, each a per-agent position vector, so the frontend
// can simply append them to what it is already drawing.
struct RollingProgress {
    std::size_t cycle = 0;          // replan cycle index (0-based)
    std::size_t step = 0;           // global timestep after this cycle
    double cycle_ms = 0.0;          // wall time spent in this cycle
    std::size_t goals_reached = 0;  // cumulative goal arrivals so far
    std::size_t active = 0;         // agents not yet on their goal
    std::vector<std::vector<Cell>> frames;   // committed frames this cycle
    std::vector<Cell> obstacles;    // obstacle cells at the latest timestep
};
using ProgressCallback = std::function<void(const RollingProgress&)>;

namespace detail {

inline Cell path_at(const std::vector<Cell>& p, std::size_t t) {
    return t < p.size() ? p[t] : p.back();
}

// Is the transition from `prev` to `cur` collision-free across all agents?
// Vertex: two agents on the same cell. Swap: two agents trading cells.
inline bool step_ok(const std::vector<Cell>& prev, const std::vector<Cell>& cur) {
    const std::size_t n = cur.size();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            if (cur[i] == cur[j]) return false;
            if (cur[i] == prev[j] && cur[j] == prev[i]) return false;
        }
    }
    return true;
}

// Predicted obstacle occupancy for relative timesteps 0..window, as a
// [k][cell] boolean grid the time-expanded A* can forbid. With perfect
// prediction each obstacle's cell is read from its true future; otherwise
// it is a ball around the obstacle's current cell whose radius grows with
// look-ahead k (uncertainty compounds the further we predict).
inline std::vector<std::vector<char>> predict_occupancy(const Grid& grid,
                                                        const ObstacleModel& obs,
                                                        std::size_t global_t,
                                                        std::size_t window) {
    const int W = grid.width();
    std::vector<std::vector<char>> occ(window + 1,
                                       std::vector<char>(static_cast<std::size_t>(W) * grid.height(), 0));
    for (std::size_t o = 0; o < obs.count(); ++o) {
        for (std::size_t k = 0; k <= window; ++k) {
            const Cell c = obs.perfect_prediction ? obs.at(o, global_t + k) : obs.at(o, global_t);
            const int r = obs.perfect_prediction ? obs.prediction_radius
                                                 : obs.prediction_radius + static_cast<int>(k);
            for (int dy = -r; dy <= r; ++dy) {
                for (int dx = -r; dx <= r; ++dx) {
                    if (grid.in_bounds(c.x + dx, c.y + dy)) {
                        occ[k][static_cast<std::size_t>(c.y + dy) * W + (c.x + dx)] = 1;
                    }
                }
            }
        }
    }
    return occ;
}

inline bool obstacle_on(const ObstacleModel& obs, std::size_t global_t, Cell c) {
    for (std::size_t o = 0; o < obs.count(); ++o)
        if (obs.at(o, global_t) == c) return true;
    return false;
}

}  // namespace detail

inline RollingResult simulate_rolling(const Grid& grid, const std::vector<Cell>& starts,
                                      const std::vector<Cell>& goals_in,
                                      const RollingConfig& cfg,
                                      const GoalProvider& provider = {},
                                      const ObstacleModel* obstacles = nullptr,
                                      const ProgressCallback& on_cycle = {}) {
    const auto start_time = std::chrono::steady_clock::now();
    const std::size_t n = starts.size();
    const bool lifelong = static_cast<bool>(provider);

    RollingResult result;
    result.history.paths.resize(n);
    std::vector<Cell> cur = starts;
    std::vector<Cell> goals = goals_in;
    std::vector<bool> done(n, false);  // one-shot: reached goal and staying
    for (std::size_t a = 0; a < n; ++a) result.history.paths[a].push_back(starts[a]);

    std::mt19937_64 rng(cfg.seed);
    std::size_t cycle = 0;

    while (result.steps < cfg.max_steps) {
        // One-shot termination: everyone parked on their goal.
        if (!lifelong && std::all_of(done.begin(), done.end(), [](bool d) { return d; })) {
            result.all_done = true;
            break;
        }

        const auto cycle_start = std::chrono::steady_clock::now();
        const std::size_t hist_len_before =
            result.history.paths.empty() ? 0 : result.history.paths[0].size();
        const std::size_t clock = result.steps;  // global time at cycle start

        // Predicted obstacle occupancy over this cycle's window, forbidden
        // during candidate generation so paths route (or wait) around it.
        std::vector<std::vector<char>> occ;
        const std::vector<std::vector<char>>* occ_ptr = nullptr;
        if (obstacles && obstacles->count() > 0) {
            occ = detail::predict_occupancy(grid, *obstacles, clock, cfg.window);
            occ_ptr = &occ;
        }

        // 1. Candidate menu from current positions to current goals.
        std::vector<std::vector<Candidate>> pools(n);
        for (std::size_t a = 0; a < n; ++a) {
            pools[a] = generate_candidates(grid, cur[a], goals[a], cfg.candidates_per_agent,
                                           cfg.seed + a + cycle * 100003, 1.0, 0, nullptr, occ_ptr);
            if (pools[a].empty()) pools[a].push_back(Candidate{{cur[a]}, 0.0});  // stuck: wait
        }

        // 2. Windowed QUBO, annealed under the per-cycle deadline (anytime).
        SelectionQubo model = build_selection_qubo(pools, -1.0, -1.0, cfg.window);
        anneal::GeometricSchedule schedule(cfg.schedule_t0, cfg.schedule_alpha);
        anneal::ParallelAnnealer<anneal::GeometricSchedule> annealer(
            model.bqm, schedule, cfg.sweeps, cfg.seed + 1 + cycle, cfg.replicas, cfg.threads,
            cfg.cycle_deadline_ms);
        anneal::ParallelResult ar = annealer.solve();
        Decoded decoded = decode_selection(model, pools, ar.best.best_state);

        std::vector<std::vector<Cell>> chosen(n);
        for (std::size_t a = 0; a < n; ++a) chosen[a] = pools[a][decoded.chosen[a]].path;

        // 3. Commit the longest conflict-free prefix of the window, up to E.
        // Each committed step is validated directly (never trusting the
        // annealer) against BOTH other agents and the obstacles' true
        // positions. Safety rule: if a prediction was wrong and an agent's
        // planned move would enter (or swap through) an obstacle, that agent
        // brakes -- it holds its previous cell for this step.
        std::size_t committed = 0;
        std::vector<Cell> prev = cur;
        for (std::size_t e = 1; e <= cfg.execute; ++e) {
            const std::size_t gt = clock + e;  // global time of this step
            std::vector<Cell> next(n);
            for (std::size_t a = 0; a < n; ++a) next[a] = detail::path_at(chosen[a], e);

            if (obstacles && obstacles->count() > 0) {
                for (std::size_t a = 0; a < n; ++a) {
                    bool vertex = detail::obstacle_on(*obstacles, gt, next[a]);
                    bool swap = false;
                    for (std::size_t o = 0; o < obstacles->count(); ++o) {
                        if (obstacles->at(o, gt) == prev[a] && obstacles->at(o, gt - 1) == next[a]) {
                            swap = true;
                            break;
                        }
                    }
                    if (vertex || swap) next[a] = prev[a];  // brake: hold position
                }
            }

            if (!detail::step_ok(prev, next)) break;  // residual agent-agent conflict
            for (std::size_t a = 0; a < n; ++a) {
                result.history.paths[a].push_back(next[a]);
                // Count an overlap only an obstacle could still cause (it
                // moved onto a braked/held agent). Zero under perfect
                // prediction; a metric of prediction quality otherwise.
                if (obstacles && obstacles->count() > 0 &&
                    detail::obstacle_on(*obstacles, gt, next[a])) {
                    result.obstacle_hits += 1;
                }
            }
            prev = next;
            ++committed;
        }
        if (committed == 0) {
            // Even the first planned step collides: hold position one step
            // (always safe against other agents -- nobody moves) so the
            // clock still advances.
            for (std::size_t a = 0; a < n; ++a) {
                result.history.paths[a].push_back(cur[a]);
                if (obstacles && obstacles->count() > 0 &&
                    detail::obstacle_on(*obstacles, clock + 1, cur[a])) {
                    result.obstacle_hits += 1;
                }
            }
            result.steps += 1;
        } else {
            cur = prev;
            result.steps += committed;
        }

        // 4. Goal arrivals.
        for (std::size_t a = 0; a < n; ++a) {
            if (!(cur[a] == goals[a])) continue;
            if (lifelong) {
                result.goals_reached += 1;
                goals[a] = provider(a, rng);
            } else if (!done[a]) {
                done[a] = true;
                result.goals_reached += 1;
            }
        }
        ++cycle;
        result.cycles = cycle;

        // Report this cycle's committed frames to a live listener (SSE).
        if (on_cycle) {
            RollingProgress prog;
            prog.cycle = cycle - 1;
            prog.step = result.steps;
            prog.cycle_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - cycle_start)
                                .count();
            prog.goals_reached = result.goals_reached;
            prog.active = 0;
            for (std::size_t a = 0; a < n; ++a)
                if (!(cur[a] == goals[a])) ++prog.active;
            // Slice the newly appended timesteps into per-step position frames.
            const std::size_t hist_len_after = result.history.paths[0].size();
            for (std::size_t t = hist_len_before; t < hist_len_after; ++t) {
                std::vector<Cell> frame(n);
                for (std::size_t a = 0; a < n; ++a) frame[a] = result.history.paths[a][t];
                prog.frames.push_back(std::move(frame));
            }
            if (obstacles && obstacles->count() > 0) {
                for (std::size_t o = 0; o < obstacles->count(); ++o)
                    prog.obstacles.push_back(obstacles->at(o, result.steps));
            }
            on_cycle(prog);
        }
    }

    result.wall_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time)
            .count();
    return result;
}

}  // namespace mapf
