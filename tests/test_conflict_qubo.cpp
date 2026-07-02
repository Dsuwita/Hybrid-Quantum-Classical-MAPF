// test_conflict_qubo.cpp
//
// Milestone 10 tests. The oracle is an independent brute force over
// SELECTIONS (one candidate per agent): enumerate every combination,
// keep those with no pairwise conflict, take the cheapest. The QUBO's
// ground state (found by anneal's brute_force_ground_state over the 2^n
// binary assignments) must decode to a selection of exactly that cost,
// whenever a conflict-free selection exists. This pins the QUBO
// formulation, penalty weights included, to the ground truth.

#include "anneal/bruteforce.hpp"
#include "anneal/fast_annealer.hpp"
#include "anneal/schedule.hpp"
#include "mapf/conflict_qubo.hpp"
#include "mapf/path_gen.hpp"

#include <cassert>
#include <cstdio>
#include <limits>
#include <optional>
#include <sstream>
#include <vector>

using namespace mapf;

namespace {

Candidate cand(std::vector<Cell> path, double cost) {
    return Candidate{std::move(path), cost};
}

// Brute force over selections: the cheapest conflict-free combination,
// or nullopt if every combination has a conflict.
struct BestSelection {
    std::vector<std::size_t> choice;
    double cost;
};

std::optional<BestSelection> best_conflict_free(
    const std::vector<std::vector<Candidate>>& pools) {
    std::vector<std::size_t> choice(pools.size(), 0);
    std::optional<BestSelection> best;

    // Odometer over the cartesian product of candidate indices.
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
        if (!conflict && (!best || cost < best->cost)) {
            best = BestSelection{choice, cost};
        }

        // increment odometer
        std::size_t pos = 0;
        for (; pos < pools.size(); ++pos) {
            if (++choice[pos] < pools[pos].size()) break;
            choice[pos] = 0;
        }
        if (pos == pools.size()) break;
    }
    return best;
}

double selection_cost(const std::vector<std::vector<Candidate>>& pools,
                      const std::vector<std::size_t>& choice) {
    double c = 0.0;
    for (std::size_t a = 0; a < pools.size(); ++a) c += pools[a][choice[a]].cost;
    return c;
}

bool selection_conflict_free(const std::vector<std::vector<Candidate>>& pools,
                             const std::vector<std::size_t>& choice) {
    for (std::size_t a = 0; a < pools.size(); ++a) {
        for (std::size_t b = a + 1; b < pools.size(); ++b) {
            if (paths_conflict(pools[a][choice[a]].path, pools[b][choice[b]].path)) return false;
        }
    }
    return true;
}

// The central test: QUBO ground state decodes to the true best selection.
void check_qubo_matches_truth(const std::vector<std::vector<Candidate>>& pools) {
    auto truth = best_conflict_free(pools);
    assert(truth.has_value() && "test instances must have a conflict-free selection");

    SelectionQubo model = build_selection_qubo(pools);
    anneal::BruteForceResult gs = anneal::brute_force_ground_state(model.bqm);

    // With the safe default penalties, the ground energy equals the min
    // feasible travel cost exactly (feasible optimum pays zero penalty).
    assert(gs.ground_energy == truth->cost);

    Decoded decoded = decode_selection(model, pools, gs.ground_state);
    assert(decoded.onehot_satisfied);
    assert(selection_conflict_free(pools, decoded.chosen));
    assert(selection_cost(pools, decoded.chosen) == truth->cost);
}

void test_paths_conflict_primitive() {
    // Vertex: same cell same time.
    assert(paths_conflict({{0, 0}, {1, 0}}, {{1, 1}, {1, 0}}));
    // Swap: exchange across one step.
    assert(paths_conflict({{0, 0}, {1, 0}}, {{1, 0}, {0, 0}}));
    // Disjoint columns: no conflict.
    assert(!paths_conflict({{0, 0}, {0, 1}}, {{2, 0}, {2, 1}}));
    // Parked agent: short path ends on a cell the other enters later.
    assert(paths_conflict({{5, 5}}, {{4, 5}, {5, 5}}));  // agent A parked at (5,5)
    std::printf("test_paths_conflict_primitive passed\n");
}

void test_cheapest_conflicts_must_detour() {
    // Two agents, two candidates each. The cheap candidates share a
    // column (conflict); the only conflict-free selections cost 15.
    std::vector<std::vector<Candidate>> pools = {
        {cand({{0, 0}, {0, 1}, {0, 2}}, 5), cand({{1, 0}, {1, 1}, {1, 2}}, 10)},
        {cand({{0, 0}, {0, 1}, {0, 2}}, 5), cand({{2, 0}, {2, 1}, {2, 2}}, 10)},
    };
    auto truth = best_conflict_free(pools);
    assert(truth->cost == 15.0);  // cheap+cheap (=10) conflicts, so must be 15
    check_qubo_matches_truth(pools);
    std::printf("test_cheapest_conflicts_must_detour passed\n");
}

void test_trivial_no_conflicts() {
    // All candidates on disjoint columns: cheapest of each wins.
    std::vector<std::vector<Candidate>> pools = {
        {cand({{0, 0}, {0, 1}}, 1), cand({{0, 0}, {0, 1}, {0, 2}}, 2)},
        {cand({{3, 0}, {3, 1}}, 1), cand({{3, 0}, {3, 1}, {3, 2}}, 2)},
    };
    auto truth = best_conflict_free(pools);
    assert(truth->cost == 2.0);
    check_qubo_matches_truth(pools);
    std::printf("test_trivial_no_conflicts passed\n");
}

void test_three_agents() {
    // Three agents, three candidates each (9 variables). Column 1 is
    // contested by the cheap candidates of agents 0 and 1; agent 2's
    // cheap path crosses agent 0's cheap path too.
    std::vector<std::vector<Candidate>> pools = {
        {cand({{1, 0}, {1, 1}, {1, 2}}, 3), cand({{0, 0}, {0, 1}, {0, 2}}, 5),
         cand({{4, 0}, {4, 1}, {4, 2}}, 7)},
        {cand({{1, 2}, {1, 1}, {1, 0}}, 3), cand({{5, 0}, {5, 1}, {5, 2}}, 6),
         cand({{6, 0}, {6, 1}, {6, 2}}, 8)},
        {cand({{1, 1}, {1, 0}}, 2), cand({{8, 0}, {8, 1}}, 4), cand({{9, 0}, {9, 1}}, 5)},
    };
    auto truth = best_conflict_free(pools);
    assert(truth.has_value());
    check_qubo_matches_truth(pools);
    std::printf("test_three_agents passed (best feasible cost = %.0f)\n", truth->cost);
}

void test_from_generated_candidates() {
    // Real candidates from A* on a small congested grid: two agents whose
    // shortest paths cross.
    std::ostringstream os;
    os << "type octile\nheight 5\nwidth 5\nmap\n.....\n.....\n.....\n.....\n.....\n";
    std::istringstream in(os.str());
    Grid grid = Grid::parse(in);

    std::vector<std::vector<Candidate>> pools = {
        generate_candidates(grid, Cell{0, 2}, Cell{4, 2}, 3, /*seed=*/1),
        generate_candidates(grid, Cell{2, 0}, Cell{2, 4}, 3, /*seed=*/2),
    };
    assert(!pools[0].empty() && !pools[1].empty());
    auto truth = best_conflict_free(pools);
    assert(truth.has_value());
    check_qubo_matches_truth(pools);
    std::printf("test_from_generated_candidates passed (%zu x %zu candidates)\n", pools[0].size(),
                pools[1].size());
}

void test_decode_repairs_onehot_violation() {
    std::vector<std::vector<Candidate>> pools = {
        {cand({{0, 0}}, 5), cand({{1, 0}}, 3), cand({{2, 0}}, 9)},
    };
    SelectionQubo model = build_selection_qubo(pools);

    // Mark candidates 0 and 2 (violates one-hot). Repair picks the
    // cheaper marked one: candidate 0 (cost 5) over candidate 2 (cost 9).
    std::vector<std::int8_t> state(model.var_owner.size(), 0);
    state[model.var_index[0][0]] = 1;
    state[model.var_index[0][2]] = 1;
    Decoded d = decode_selection(model, pools, state);
    assert(!d.onehot_satisfied);
    assert(d.chosen[0] == 0);

    // Nothing marked: fall back to the cheapest candidate (index 1, cost 3).
    std::vector<std::int8_t> empty(model.var_owner.size(), 0);
    Decoded d2 = decode_selection(model, pools, empty);
    assert(!d2.onehot_satisfied);
    assert(d2.chosen[0] == 1);
    std::printf("test_decode_repairs_onehot_violation passed\n");
}

// The annealer (not just brute force) should recover the optimal feasible
// selection on these small instances.
void test_annealer_finds_optimum() {
    std::vector<std::vector<Candidate>> pools = {
        {cand({{0, 0}, {0, 1}, {0, 2}}, 5), cand({{1, 0}, {1, 1}, {1, 2}}, 10)},
        {cand({{0, 0}, {0, 1}, {0, 2}}, 5), cand({{2, 0}, {2, 1}, {2, 2}}, 10)},
    };
    auto truth = best_conflict_free(pools);
    SelectionQubo model = build_selection_qubo(pools);

    anneal::GeometricSchedule schedule(5.0, 0.99);
    anneal::FastAnnealer<anneal::GeometricSchedule> annealer(model.bqm, schedule, 2000, 42);
    anneal::SolveResult r = annealer.solve();

    Decoded d = decode_selection(model, pools, r.best_state);
    assert(d.onehot_satisfied);
    assert(selection_conflict_free(pools, d.chosen));
    assert(selection_cost(pools, d.chosen) == truth->cost);
    std::printf("test_annealer_finds_optimum passed\n");
}

}  // namespace

int main() {
    test_paths_conflict_primitive();
    test_cheapest_conflicts_must_detour();
    test_trivial_no_conflicts();
    test_three_agents();
    test_from_generated_candidates();
    test_decode_repairs_onehot_violation();
    test_annealer_finds_optimum();
    std::printf("All conflict QUBO tests passed.\n");
    return 0;
}
