// test_cbs.cpp
//
// Tests for the CBS classical solver. CBS is optimal for sum-of-costs, so
// the gold-standard test is an exhaustive joint-space search that finds the
// true optimum on tiny instances (the same brute-force discipline used
// throughout the project). On every instance the verifier -- never CBS's own
// report -- is the judge of a clean plan.
//
// The joint brute force uses the "reach and stay" model: an agent that
// arrives at its goal waits there for the rest of the plan. On the small
// open grids used here the optimum never requires an agent to vacate its
// goal to let another pass, so this model contains the true optimum and
// gives an exact SOC lower/upper bound for CBS to match.

#include "mapf/cbs.hpp"
#include "mapf/grid.hpp"
#include "mapf/scenario.hpp"
#include "mapf/solver.hpp"
#include "mapf/verifier.hpp"

#include <cassert>
#include <cstdio>
#include <functional>
#include <map>
#include <queue>
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

AgentTask task(const Grid& grid, Cell start, Cell goal) {
    AgentTask t;
    t.start = start;
    t.goal = goal;
    auto c = generate_candidates(grid, start, goal, 1, 0);
    t.optimal_distance = c.empty() ? 0.0 : c[0].cost;
    return t;
}

// Exhaustive optimal sum-of-costs by joint-space Dijkstra (reach-and-stay
// model). State = all agents' positions; an agent already on its goal is
// forced to wait, so absolute time drops out of the state key and remaining
// optimal cost depends only on positions. Returns -1 if no conflict-free
// plan exists. Only used on tiny instances.
double brute_force_soc(const Grid& grid, const std::vector<AgentTask>& tasks) {
    const std::size_t n = tasks.size();
    auto pack = [&](const std::vector<Cell>& pos) {
        std::uint64_t key = 0;
        for (const Cell& c : pos)
            key = key * 1000 + static_cast<std::uint64_t>(c.y * grid.width() + c.x);
        return key;
    };
    auto all_at_goal = [&](const std::vector<Cell>& pos) {
        for (std::size_t a = 0; a < n; ++a)
            if (!(pos[a] == tasks[a].goal)) return false;
        return true;
    };

    std::vector<Cell> start(n), goal(n);
    for (std::size_t a = 0; a < n; ++a) {
        start[a] = tasks[a].start;
        goal[a] = tasks[a].goal;
    }

    using Item = std::pair<double, std::vector<Cell>>;
    auto worse = [](const Item& x, const Item& y) { return x.first > y.first; };
    std::priority_queue<Item, std::vector<Item>, decltype(worse)> pq(worse);
    std::map<std::uint64_t, double> best;
    pq.push({0.0, start});
    best[pack(start)] = 0.0;

    // Enumerate all joint moves (Cartesian product of each agent's legal
    // moves; goal-parked agents may only wait).
    std::function<void(std::size_t, const std::vector<Cell>&, std::vector<Cell>&,
                       const std::vector<Cell>&, std::vector<std::vector<Cell>>&)>
        expand = [&](std::size_t a, const std::vector<Cell>& cur, std::vector<Cell>& acc,
                     const std::vector<Cell>& g, std::vector<std::vector<Cell>>& out) {
            if (a == n) {
                out.push_back(acc);
                return;
            }
            std::vector<Cell> moves;
            if (cur[a] == g[a]) {
                moves = {cur[a]};  // parked: wait only
            } else {
                moves = grid.moves_from(cur[a]);
            }
            for (Cell m : moves) {
                acc.push_back(m);
                expand(a + 1, cur, acc, g, out);
                acc.pop_back();
            }
        };

    while (!pq.empty()) {
        auto [cost, pos] = pq.top();
        pq.pop();
        std::uint64_t key = pack(pos);
        auto it = best.find(key);
        if (it != best.end() && cost > it->second) continue;
        if (all_at_goal(pos)) return cost;

        std::vector<std::vector<Cell>> nexts;
        std::vector<Cell> acc;
        expand(0, pos, acc, goal, nexts);
        for (const auto& np : nexts) {
            bool bad = false;
            for (std::size_t i = 0; i < n && !bad; ++i)
                for (std::size_t j = i + 1; j < n && !bad; ++j) {
                    if (np[i] == np[j]) bad = true;                       // vertex
                    if (np[i] == pos[j] && np[j] == pos[i]) bad = true;   // swap
                }
            if (bad) continue;
            double add = 0.0;  // each agent not yet at goal costs 1 this step
            for (std::size_t a = 0; a < n; ++a)
                if (!(pos[a] == goal[a])) add += 1.0;
            double nc = cost + add;
            std::uint64_t nk = pack(np);
            auto f = best.find(nk);
            if (f == best.end() || nc < f->second) {
                best[nk] = nc;
                pq.push({nc, np});
            }
        }
    }
    return -1.0;
}

void test_cbs_matches_bruteforce() {
    std::mt19937_64 rng(20250702);
    int checked = 0;
    for (int trial = 0; trial < 40; ++trial) {
        const int W = 4 + static_cast<int>(rng() % 2);  // 4x4 or 5x5
        Grid grid = open_grid(W, W);
        std::uniform_int_distribution<int> coord(0, W - 1);
        auto rc = [&]() { return Cell{coord(rng), coord(rng)}; };
        // Two agents with distinct starts and distinct goals.
        Cell s0 = rc(), s1 = rc();
        Cell g0 = rc(), g1 = rc();
        if (s0 == s1 || g0 == g1) continue;
        std::vector<AgentTask> tasks = {task(grid, s0, g0), task(grid, s1, g1)};

        double bf = brute_force_soc(grid, tasks);
        CbsResult r = solve_cbs(grid, tasks);
        if (bf < 0.0) {
            // Brute force found no conflict-free plan: CBS must agree.
            assert(!r.success);
            continue;
        }
        assert(r.success);
        VerifyResult v = verify(grid, tasks, r.plan);
        assert(v.ok());
        // The plan the verifier scores must equal the exact optimum.
        assert(v.sum_of_costs == bf);
        ++checked;
    }
    assert(checked >= 25);
    std::printf("test_cbs_matches_bruteforce passed (%d instances vs joint brute force)\n", checked);
}

void test_cbs_resolves_crossing() {
    Grid grid = open_grid(7, 7);
    std::vector<AgentTask> tasks = {task(grid, {0, 3}, {6, 3}), task(grid, {3, 0}, {3, 6})};
    CbsResult r = solve_cbs(grid, tasks);
    assert(r.success);
    VerifyResult v = verify(grid, tasks, r.plan);
    assert(v.ok());
    // Optimal cost is never below the sum of per-agent shortest paths.
    assert(v.sum_of_costs >= v.sum_of_optimal);
    std::printf("test_cbs_resolves_crossing passed (soc=%.0f, overhead=%.1f%%, expansions=%zu)\n",
                v.sum_of_costs, v.overhead_percent(), r.expansions);
}

void test_cbs_no_conflict_is_optimal() {
    // Two agents on parallel rows never interact: optimal SOC == sum of
    // shortest paths, so CBS overhead must be exactly zero.
    Grid grid = open_grid(8, 8);
    std::vector<AgentTask> tasks = {task(grid, {0, 0}, {7, 0}), task(grid, {0, 7}, {7, 7})};
    CbsResult r = solve_cbs(grid, tasks);
    assert(r.success);
    VerifyResult v = verify(grid, tasks, r.plan);
    assert(v.ok());
    assert(v.overhead_percent() == 0.0);
    std::printf("test_cbs_no_conflict_is_optimal passed (soc=%.0f)\n", v.sum_of_costs);
}

void test_cbs_swap_pocket() {
    // Head-on agents in a corridor with a single passing pocket. Solvable,
    // and CBS must find it. Layout (5 wide, pocket at (2,0) opening up):
    //   row0: . . . . .
    //   row1: @ @ . @ @
    std::istringstream in("type octile\nheight 2\nwidth 5\nmap\n.....\n@@.@@\n");
    Grid grid = Grid::parse(in);
    std::vector<AgentTask> tasks = {task(grid, {0, 0}, {4, 0}), task(grid, {4, 0}, {0, 0})};
    CbsResult r = solve_cbs(grid, tasks);
    assert(r.success);
    VerifyResult v = verify(grid, tasks, r.plan);
    assert(v.ok());
    double bf = brute_force_soc(grid, tasks);
    assert(v.sum_of_costs == bf);
    std::printf("test_cbs_swap_pocket passed (soc=%.0f == optimal %.0f)\n", v.sum_of_costs, bf);
}

void test_cbs_unsolvable_corridor() {
    // One-wide corridor, two agents must swap ends: impossible. CBS exhausts
    // its search and reports failure honestly.
    std::istringstream in("type octile\nheight 1\nwidth 4\nmap\n....\n");
    Grid grid = Grid::parse(in);
    std::vector<AgentTask> tasks = {task(grid, {0, 0}, {3, 0}), task(grid, {3, 0}, {0, 0})};
    CbsResult r = solve_cbs(grid, tasks);
    assert(!r.success);
    std::printf("test_cbs_unsolvable_corridor passed (expansions=%zu)\n", r.expansions);
}

void test_cbs_deterministic() {
    Grid grid = open_grid(6, 6);
    std::vector<AgentTask> tasks = {task(grid, {0, 2}, {5, 2}), task(grid, {2, 0}, {2, 5}),
                                    task(grid, {5, 5}, {0, 0})};
    CbsResult a = solve_cbs(grid, tasks);
    CbsResult b = solve_cbs(grid, tasks);
    assert(a.success && b.success);
    assert(a.plan.paths == b.plan.paths);
    std::printf("test_cbs_deterministic passed\n");
}

void test_cbs_never_worse_than_hybrid() {
    // CBS is optimal, so whenever the hybrid solver also succeeds, CBS's cost
    // is <= the hybrid's on the same instance.
    std::mt19937_64 rng(99);
    Grid grid = open_grid(8, 8);
    int compared = 0;
    for (int trial = 0; trial < 8; ++trial) {
        std::uniform_int_distribution<int> coord(0, 7);
        auto rc = [&]() { return Cell{coord(rng), coord(rng)}; };
        std::vector<AgentTask> tasks = {task(grid, rc(), rc()), task(grid, rc(), rc()),
                                        task(grid, rc(), rc())};
        CbsResult c = solve_cbs(grid, tasks);
        SolverConfig cfg;
        cfg.sweeps = 1000;
        cfg.replicas = 8;
        cfg.threads = 4;
        SolverResult h = solve_mapf(grid, tasks, cfg);
        if (!c.success || !h.success) continue;
        VerifyResult vc = verify(grid, tasks, c.plan);
        VerifyResult vh = verify(grid, tasks, h.plan);
        assert(vc.ok() && vh.ok());
        assert(vc.sum_of_costs <= vh.sum_of_costs);
        ++compared;
    }
    std::printf("test_cbs_never_worse_than_hybrid passed (%d comparisons)\n", compared);
}

}  // namespace

int main() {
    test_cbs_matches_bruteforce();
    test_cbs_resolves_crossing();
    test_cbs_no_conflict_is_optimal();
    test_cbs_swap_pocket();
    test_cbs_unsolvable_corridor();
    test_cbs_deterministic();
    test_cbs_never_worse_than_hybrid();
    std::printf("All CBS tests passed.\n");
    return 0;
}
