// bench_tempering.cpp
//
// Milestone 7(a) comparison: parallel tempering versus equal-budget
// independent restarts on a hard Max-Cut instance. Both methods get the same
// total sweep budget (R replicas x S sweeps): tempering runs R fixed-
// temperature replicas with adjacent swaps; restarts run R independent
// cooled anneals and keep the best. We repeat over several seeds and report
// the mean and best cut each method reaches, so the comparison is not a
// single lucky run.
//
// The point of tempering is exactly the hard, rugged landscapes where a
// single cooled run gets trapped; G39 (a Gset instance with +/-1 weights) is
// that kind of instance. On easy instances the two are indistinguishable.
//
// Usage:
//   bench_tempering [gset-file] [--rungs R] [--sweeps S] [--seeds N]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "anneal/parallel.hpp"
#include "anneal/parallel_tempering.hpp"
#include "anneal/problems/maxcut.hpp"
#include "anneal/schedule.hpp"

using namespace anneal;

namespace {
long argl(int argc, char** argv, const char* f, long d) {
    for (int i = 1; i + 1 < argc; ++i)
        if (!std::strcmp(argv[i], f)) return std::atol(argv[i + 1]);
    return d;
}
}  // namespace

int main(int argc, char** argv) {
    Graph g;
    std::string name;
    if (argc > 1 && argv[1][0] != '-') {
        std::ifstream in(argv[1]);
        if (!in) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
        g = parse_gset(in);
        name = argv[1];
    } else {
        // Frustrated random instance: +/-1 weights make a rugged landscape.
        g = random_d_regular(200, 3, 777);
        for (auto& e : g.edges) e.w = (e.u + e.v) % 2 ? 1.0 : -1.0;
        name = "random-3-regular-n200-pm1";
    }
    const BQM bqm = maxcut_to_bqm(g);
    const std::size_t rungs = static_cast<std::size_t>(argl(argc, argv, "--rungs", 16));
    const std::size_t sweeps = static_cast<std::size_t>(argl(argc, argv, "--sweeps", 2000));
    const std::size_t seeds = static_cast<std::size_t>(argl(argc, argv, "--seeds", 10));

    std::fprintf(stderr, "instance %s (n=%zu, m=%zu), budget %zu replicas x %zu sweeps\n",
                 name.c_str(), g.n, g.edges.size(), rungs, sweeps);

    auto timer = [] { return std::chrono::steady_clock::now(); };
    auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    std::vector<double> pt_cuts, rs_cuts;
    double pt_ms = 0.0, rs_ms = 0.0;
    for (std::size_t s = 0; s < seeds; ++s) {
        // Parallel tempering.
        auto ladder = geometric_ladder(0.15, 8.0, rungs);
        auto t0 = timer();
        ParallelTempering<> pt(bqm, ladder, sweeps, 1000 + s);
        auto ptr = pt.solve();
        pt_ms += ms(t0, timer());
        pt_cuts.push_back(cut_value(g, ptr.best.best_state));

        // Equal-budget independent restarts (R replicas, S sweeps each).
        GeometricSchedule sched(4.0, std::pow(0.05 / 4.0, 1.0 / static_cast<double>(sweeps)));
        auto t1 = timer();
        ParallelAnnealer<GeometricSchedule> pa(bqm, sched, sweeps, 2000 + s * rungs, rungs, 1);
        auto par = pa.solve();
        rs_ms += ms(t1, timer());
        rs_cuts.push_back(cut_value(g, par.best.best_state));
    }

    auto mean = [](const std::vector<double>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };
    auto best = [](const std::vector<double>& v) {
        return *std::max_element(v.begin(), v.end());
    };

    std::printf("method              mean_cut   best_cut   total_ms\n");
    std::printf("parallel_tempering  %8.1f   %8.0f   %8.1f\n", mean(pt_cuts), best(pt_cuts), pt_ms);
    std::printf("independent_restart %8.1f   %8.0f   %8.1f\n", mean(rs_cuts), best(rs_cuts), rs_ms);
    return 0;
}
