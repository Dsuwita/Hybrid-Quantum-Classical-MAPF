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
    std::size_t goals_reached = 0;
    bool all_done = false;        // one-shot: every agent ended on its goal
    double wall_ms = 0.0;
};

// Lifelong goal source: called when agent `a` reaches its goal, returns its
// next goal. Leave empty for one-shot mode (agents stop once they arrive).
using GoalProvider = std::function<Cell(std::size_t a, std::mt19937_64& rng)>;

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

}  // namespace detail

inline RollingResult simulate_rolling(const Grid& grid, const std::vector<Cell>& starts,
                                      const std::vector<Cell>& goals_in,
                                      const RollingConfig& cfg,
                                      const GoalProvider& provider = {}) {
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

        // 1. Candidate menu from current positions to current goals.
        std::vector<std::vector<Candidate>> pools(n);
        for (std::size_t a = 0; a < n; ++a) {
            pools[a] = generate_candidates(grid, cur[a], goals[a], cfg.candidates_per_agent,
                                           cfg.seed + a + cycle * 100003);
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
        std::size_t committed = 0;
        std::vector<Cell> prev = cur;
        for (std::size_t e = 1; e <= cfg.execute; ++e) {
            std::vector<Cell> next(n);
            for (std::size_t a = 0; a < n; ++a) next[a] = detail::path_at(chosen[a], e);
            if (!detail::step_ok(prev, next)) break;
            for (std::size_t a = 0; a < n; ++a) result.history.paths[a].push_back(next[a]);
            prev = next;
            ++committed;
        }
        if (committed == 0) {
            // Even the first planned step collides: hold position one step
            // (always safe -- nobody moves) so the clock still advances.
            for (std::size_t a = 0; a < n; ++a) result.history.paths[a].push_back(cur[a]);
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
    }

    result.wall_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time)
            .count();
    return result;
}

}  // namespace mapf
