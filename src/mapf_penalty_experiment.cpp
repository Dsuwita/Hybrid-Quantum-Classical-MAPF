// mapf_penalty_experiment.cpp
//
// Milestone 10 dynamic-range experiment. The path-selection QUBO needs
// penalty weights large enough that a constraint violation never pays for
// itself, but LARGER penalties make the same problem harder to anneal:
// the energy landscape grows tall walls between feasible configurations,
// and the acceptance-probability range widens, so a fixed sweep budget
// finds the optimum less often. This program makes that tradeoff visible.
//
// Instance: four agents crossing a small open grid, three candidate paths
// each (12 binary variables), so the QUBO ground truth is available by
// exhaustive enumeration. For a range of penalty scales we report:
//   - feasible_optimal_gs: does the QUBO's exact ground state decode to a
//     conflict-free, one-hot-valid, minimum-cost selection? (formulation
//     correctness -- fails when penalties are too SMALL)
//   - anneal_success: fraction of annealer runs (over many seeds, fixed
//     modest budget) that recover that optimum (drops when penalties are
//     too LARGE)
//
// Output is a markdown table on stdout; committed as
// mapf/bench/penalty_experiment.md.

#include "anneal/bruteforce.hpp"
#include "anneal/fast_annealer.hpp"
#include "anneal/schedule.hpp"
#include "mapf/conflict_qubo.hpp"
#include "mapf/path_gen.hpp"

#include <cstdio>
#include <optional>
#include <sstream>
#include <vector>

using namespace mapf;

namespace {

std::optional<double> best_conflict_free_cost(const std::vector<std::vector<Candidate>>& pools) {
    std::vector<std::size_t> choice(pools.size(), 0);
    std::optional<double> best;
    while (true) {
        bool conflict = false;
        double cost = 0.0;
        for (std::size_t a = 0; a < pools.size() && !conflict; ++a) {
            cost += pools[a][choice[a]].cost;
            for (std::size_t b = a + 1; b < pools.size(); ++b) {
                if (paths_conflict(pools[a][choice[a]].path, pools[b][choice[b]].path)) {
                    conflict = true;
                    break;
                }
            }
        }
        if (!conflict && (!best || cost < *best)) best = cost;
        std::size_t pos = 0;
        for (; pos < pools.size(); ++pos) {
            if (++choice[pos] < pools[pos].size()) break;
            choice[pos] = 0;
        }
        if (pos == pools.size()) break;
    }
    return best;
}

bool decoded_is_optimal(const SelectionQubo& model,
                        const std::vector<std::vector<Candidate>>& pools,
                        const std::vector<std::int8_t>& state, double optimal_cost) {
    Decoded d = decode_selection(model, pools, state);
    if (!d.onehot_satisfied) return false;
    double cost = 0.0;
    for (std::size_t a = 0; a < pools.size(); ++a) {
        for (std::size_t b = a + 1; b < pools.size(); ++b) {
            if (paths_conflict(pools[a][d.chosen[a]].path, pools[b][d.chosen[b]].path)) return false;
        }
        cost += pools[a][d.chosen[a]].cost;
    }
    return cost == optimal_cost;
}

}  // namespace

int main() {
    // Congested 7x7 open grid: four agents whose shortest paths all pass
    // near the center.
    std::istringstream in(
        "type octile\nheight 7\nwidth 7\nmap\n.......\n.......\n.......\n.......\n.......\n"
        ".......\n.......\n");
    Grid grid = Grid::parse(in);

    std::vector<std::vector<Candidate>> pools = {
        generate_candidates(grid, Cell{0, 3}, Cell{6, 3}, 3, 11),
        generate_candidates(grid, Cell{3, 0}, Cell{3, 6}, 3, 22),
        generate_candidates(grid, Cell{0, 0}, Cell{6, 6}, 3, 33),
        generate_candidates(grid, Cell{6, 0}, Cell{0, 6}, 3, 44),
    };

    std::size_t nvars = 0;
    for (const auto& p : pools) nvars += p.size();

    auto optimal = best_conflict_free_cost(pools);
    if (!optimal) {
        std::printf("no conflict-free selection exists for this instance\n");
        return 1;
    }

    // The safe default penalty for reference; we sweep multiples of the
    // maximum single-path cost so the "too small" regime is reachable.
    double max_cost = 0.0;
    for (const auto& p : pools)
        for (const auto& c : p) max_cost = std::max(max_cost, c.cost);

    const int num_seeds = 200;
    const std::size_t sweeps = 300;  // deliberately modest, to expose the tradeoff

    std::printf("# Milestone 10 penalty dynamic-range experiment\n\n");
    std::printf("Instance: 4 agents x 3 candidates (%zu binary vars) on a 7x7 open grid.\n",
                nvars);
    std::printf("Max single-path cost = %.0f. Optimal feasible selection cost = %.0f.\n",
                max_cost, *optimal);
    std::printf("Annealer: geometric T0=5 alpha=0.99, %zu sweeps, %d seeds.\n\n", sweeps,
                num_seeds);
    std::printf("| penalty P=P1 | scale (xMaxCost) | exact GS feasible+optimal | anneal success |\n");
    std::printf("|---|---|---|---|\n");

    for (double scale : {0.5, 1.0, 2.0, 5.0, 20.0, 100.0, 1000.0}) {
        double penalty = scale * max_cost;
        SelectionQubo model = build_selection_qubo(pools, penalty, penalty);

        anneal::BruteForceResult gs = anneal::brute_force_ground_state(model.bqm);
        bool gs_ok = decoded_is_optimal(model, pools, gs.ground_state, *optimal);

        int hits = 0;
        for (int s = 0; s < num_seeds; ++s) {
            anneal::GeometricSchedule schedule(5.0, 0.99);
            anneal::FastAnnealer<anneal::GeometricSchedule> annealer(
                model.bqm, schedule, sweeps, static_cast<std::uint64_t>(1000 + s));
            anneal::SolveResult r = annealer.solve();
            if (decoded_is_optimal(model, pools, r.best_state, *optimal)) ++hits;
        }

        std::printf("| %.0f | %.1f | %s | %.1f%% |\n", penalty, scale, gs_ok ? "yes" : "NO",
                    100.0 * hits / num_seeds);
    }
    return 0;
}
