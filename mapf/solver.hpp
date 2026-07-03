// solver.hpp
//
// The full hybrid MAPF loop, tying together everything in project 2:
//
//   generate candidates (A*) -> build conflict QUBO -> anneal -> decode
//     -> verify. If conflicts remain, add reservation penalties on the
//     conflicting cells, generate MORE candidates that route around them,
//     and re-anneal on the enlarged menu. Repeat up to a cap.
//
// Why enlarge the menu instead of replacing it: the QUBO can only pick a
// conflict-free combination if one exists among the candidates offered.
// Each round adds conflict-avoiding alternatives for the agents that were
// actually in a collision, so the menu monotonically gains options and
// the annealer gets progressively more room, while paths that were fine
// are never thrown away.
//
// Honest failure: on a genuinely unsolvable instance (e.g. two agents
// that must swap through a one-wide corridor) no new candidates can be
// generated, the menu stops growing, and the loop reports success=false
// with the least-conflicted plan it found rather than looping forever or
// pretending it succeeded.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <set>
#include <vector>

#include "anneal/parallel.hpp"
#include "anneal/schedule.hpp"
#include "mapf/conflict_qubo.hpp"
#include "mapf/grid.hpp"
#include "mapf/path_gen.hpp"
#include "mapf/plan.hpp"
#include "mapf/scenario.hpp"
#include "mapf/verifier.hpp"

namespace mapf {

struct SolverConfig {
    std::size_t candidates_per_agent = 4;   // K generated per agent per round
    std::size_t max_candidates_per_agent = 16;  // cap on menu growth
    std::size_t max_iterations = 10;        // re-anneal rounds before giving up
    std::size_t sweeps = 2000;              // annealer budget per replica
    std::size_t replicas = 8;               // parallel restarts
    std::size_t threads = 0;                // 0 = hardware_concurrency
    std::uint64_t seed = 1;
    double reservation_penalty = 4.0;       // A* cost added per conflict at a cell
    double schedule_t0 = 5.0;
    double schedule_alpha = 0.99;
};

struct SolverResult {
    bool success = false;
    Plan plan;
    VerifyResult verification;
    std::size_t iterations = 0;   // rounds actually run
    double wall_ms = 0.0;
    bool infeasible_generation = false;  // an agent had no path at all
};

namespace detail {

// Signature of a path, for deduping a growing candidate menu.
inline std::vector<int> path_signature(const std::vector<Cell>& path) {
    std::vector<int> sig;
    sig.reserve(path.size() * 2);
    for (const Cell& c : path) {
        sig.push_back(c.x);
        sig.push_back(c.y);
    }
    return sig;
}

// Build a padded plan from the chosen candidate per agent.
inline Plan build_plan(const std::vector<std::vector<Candidate>>& pools,
                       const std::vector<std::size_t>& chosen) {
    std::size_t makespan = 0;
    for (std::size_t a = 0; a < pools.size(); ++a) {
        makespan = std::max(makespan, pools[a][chosen[a]].path.size() - 1);
    }
    Plan plan;
    plan.paths.resize(pools.size());
    for (std::size_t a = 0; a < pools.size(); ++a) {
        plan.paths[a] = pad_to(pools[a][chosen[a]].path, makespan);
    }
    return plan;
}

}  // namespace detail

inline SolverResult solve_mapf(const Grid& grid, const std::vector<AgentTask>& tasks,
                               const SolverConfig& cfg) {
    const auto start_time = std::chrono::steady_clock::now();
    const std::size_t n = tasks.size();
    const std::size_t cell_count = static_cast<std::size_t>(grid.width()) * grid.height();

    SolverResult result;

    // Round 0: an initial diverse menu per agent, no reservations.
    std::vector<std::vector<Candidate>> pools(n);
    std::vector<std::set<std::vector<int>>> seen(n);
    std::vector<std::vector<double>> reservations(n, std::vector<double>(cell_count, 0.0));

    for (std::size_t a = 0; a < n; ++a) {
        pools[a] = generate_candidates(grid, tasks[a].start, tasks[a].goal,
                                       cfg.candidates_per_agent, cfg.seed + a);
        if (pools[a].empty()) {
            // Goal unreachable for this agent: no plan is possible.
            result.infeasible_generation = true;
            result.wall_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - start_time)
                                 .count();
            return result;
        }
        for (const auto& c : pools[a]) seen[a].insert(detail::path_signature(c.path));
    }

    std::size_t best_violation_count = static_cast<std::size_t>(-1);

    for (std::size_t iter = 0; iter < cfg.max_iterations; ++iter) {
        result.iterations = iter + 1;

        // Anneal the current selection QUBO.
        SelectionQubo model = build_selection_qubo(pools);
        anneal::GeometricSchedule schedule(cfg.schedule_t0, cfg.schedule_alpha);
        anneal::ParallelAnnealer<anneal::GeometricSchedule> annealer(
            model.bqm, schedule, cfg.sweeps, cfg.seed + 1 + iter, cfg.replicas, cfg.threads);
        anneal::ParallelResult ar = annealer.solve();

        Decoded decoded = decode_selection(model, pools, ar.best.best_state);
        Plan plan = detail::build_plan(pools, decoded.chosen);
        VerifyResult vr = verify(grid, tasks, plan);

        // Keep the least-conflicted plan seen, so even a failure reports
        // the best attempt.
        if (vr.violations.size() < best_violation_count) {
            best_violation_count = vr.violations.size();
            result.plan = plan;
            result.verification = vr;
        }

        if (vr.ok()) {
            result.success = true;
            break;
        }

        // Add reservation penalties on the cells where conflicts happened,
        // for exactly the agents involved, then generate conflict-avoiding
        // candidates for those agents.
        std::set<std::size_t> touched;
        for (const Violation& v : vr.violations) {
            if (v.type != Violation::Type::VertexConflict &&
                v.type != Violation::Type::SwapConflict) {
                continue;  // candidates are valid single-agent paths, so
                           // only agent-agent conflicts should appear
            }
            const std::size_t idx = static_cast<std::size_t>(v.cell.y) * grid.width() + v.cell.x;
            reservations[v.agent_a][idx] += cfg.reservation_penalty;
            reservations[v.agent_b][idx] += cfg.reservation_penalty;
            touched.insert(v.agent_a);
            touched.insert(v.agent_b);
        }

        bool menu_grew = false;
        for (std::size_t a : touched) {
            if (pools[a].size() >= cfg.max_candidates_per_agent) continue;
            // A fresh seed each round so we do not regenerate the same set.
            auto fresh = generate_candidates(grid, tasks[a].start, tasks[a].goal,
                                             cfg.candidates_per_agent,
                                             cfg.seed + a + (iter + 1) * 100003, 1.0, 0,
                                             &reservations[a]);
            for (auto& c : fresh) {
                if (pools[a].size() >= cfg.max_candidates_per_agent) break;
                if (seen[a].insert(detail::path_signature(c.path)).second) {
                    pools[a].push_back(std::move(c));
                    menu_grew = true;
                }
            }
        }

        // No new options anywhere: the instance is unsolvable with these
        // candidates. Stop and report honestly.
        if (!menu_grew) break;
    }

    result.wall_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time)
            .count();
    return result;
}

}  // namespace mapf
