// anneal_stream.cpp
//
// Streaming annealer for the studio's parameter lab. It runs a single-thread
// simulated-annealing trajectory on one problem and emits the best energy
// seen so far at regular sweep checkpoints as newline-delimited JSON, which
// serve.py forwards to the browser as Server-Sent Events. Watching the
// energy fall live -- and comparing trajectories at different T0 / cooling /
// sweep budgets -- is what builds intuition for the hot/cold tradeoff.
//
// The Metropolis loop here is the same naive single-flip update as the M2
// annealer (recompute the local field each proposal); it is reproduced
// rather than reused so the M2 annealer stays untouched and this binary can
// emit intermediate checkpoints. The schedule classes and BQM come straight
// from the library.
//
// Output (one JSON object per line):
//   {"event":"meta", problem, n, threads}
//   {"event":"sweep", sweep, best_energy, temperature, objective[, assign]}
//   {"event":"done", best_energy, objective, sweeps, wall_ms[, speedup]}
//
// `objective` is the human-facing quantity: cut value for Max-Cut, partition
// difference for number partitioning. `assign` (partition only) is the +1/-1
// group of each number at the current best, so the frontend can animate the
// two piles.
//
// Usage:
//   anneal_stream --problem partition|maxcut-random|maxcut-gset [options]
// Options:
//   --n N            problem size (partition count / graph vertices)
//   --p P            edge probability for maxcut-random (default 0.5)
//   --gset FILE      instance file for maxcut-gset
//   --t0 T           initial temperature (default 10)
//   --alpha A        geometric cooling factor (default 0.99)
//   --schedule geometric|linear   (default geometric)
//   --t-end T        final temperature for linear schedule (default 0.01)
//   --sweeps N       sweep budget (default 2000)
//   --seed N         RNG seed (default 1)
//   --points N       max checkpoints to emit (default 200)
//   --threads N      if >1, also time a parallel run for a speedup number

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "anneal/bqm.hpp"
#include "anneal/parallel.hpp"
#include "anneal/problems/maxcut.hpp"
#include "anneal/problems/partition.hpp"
#include "anneal/schedule.hpp"

using namespace anneal;

namespace {

long argl(int argc, char** argv, const char* f, long d) {
    for (int i = 1; i + 1 < argc; ++i)
        if (!std::strcmp(argv[i], f)) return std::atol(argv[i + 1]);
    return d;
}
double argd(int argc, char** argv, const char* f, double d) {
    for (int i = 1; i + 1 < argc; ++i)
        if (!std::strcmp(argv[i], f)) return std::atof(argv[i + 1]);
    return d;
}
const char* args(int argc, char** argv, const char* f, const char* d) {
    for (int i = 1; i + 1 < argc; ++i)
        if (!std::strcmp(argv[i], f)) return argv[i + 1];
    return d;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string problem = args(argc, argv, "--problem", "partition");
    const std::size_t sweeps = static_cast<std::size_t>(argl(argc, argv, "--sweeps", 2000));
    const std::uint64_t seed = static_cast<std::uint64_t>(argl(argc, argv, "--seed", 1));
    const double t0 = argd(argc, argv, "--t0", 10.0);
    const double alpha = argd(argc, argv, "--alpha", 0.99);
    const std::string sched = args(argc, argv, "--schedule", "geometric");
    const double t_end = argd(argc, argv, "--t-end", 0.01);
    const std::size_t max_points = static_cast<std::size_t>(argl(argc, argv, "--points", 200));
    const std::size_t threads = static_cast<std::size_t>(argl(argc, argv, "--threads", 1));

    // Build the problem BQM (kept in spin space, the annealer's native form).
    BQM bqm(0, Vartype::Spin);
    std::vector<double> numbers;   // partition only
    Graph graph;                   // maxcut only
    bool is_partition = false;

    if (problem == "partition") {
        is_partition = true;
        const std::size_t n = static_cast<std::size_t>(argl(argc, argv, "--n", 24));
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<int> pick(1, 1000);
        numbers.resize(n);
        for (auto& v : numbers) v = pick(rng);
        bqm = partition_to_bqm(numbers);
    } else if (problem == "maxcut-random") {
        const std::size_t n = static_cast<std::size_t>(argl(argc, argv, "--n", 60));
        const double p = argd(argc, argv, "--p", 0.5);
        graph = erdos_renyi(n, p, seed);
        bqm = maxcut_to_bqm(graph);
    } else if (problem == "maxcut-gset") {
        const char* file = args(argc, argv, "--gset", nullptr);
        if (!file) {
            std::fprintf(stderr, "maxcut-gset requires --gset FILE\n");
            return 2;
        }
        std::ifstream in(file);
        if (!in) {
            std::fprintf(stderr, "cannot open %s\n", file);
            return 2;
        }
        graph = parse_gset(in);
        bqm = maxcut_to_bqm(graph);
    } else {
        std::fprintf(stderr, "unknown problem %s\n", problem.c_str());
        return 2;
    }

    const std::size_t n = bqm.num_variables();
    GeometricSchedule geo(t0, alpha);
    LinearSchedule lin(t0, t_end, sweeps);
    const bool use_linear = (sched == "linear");
    auto temperature = [&](std::size_t s) {
        return use_linear ? lin.temperature(s) : geo.temperature(s);
    };

    auto objective = [&](const std::vector<std::int8_t>& st) {
        return is_partition ? partition_difference(numbers, st) : cut_value(graph, st);
    };

    std::printf("{\"event\":\"meta\",\"problem\":\"%s\",\"n\":%zu,\"threads\":%zu}\n",
                problem.c_str(), n, threads);
    std::fflush(stdout);

    // --- the streaming single-thread trajectory ---
    std::mt19937_64 rng(seed ^ 0xabcdefULL);
    std::vector<std::int8_t> state(n);
    std::uniform_int_distribution<int> coin(0, 1);
    for (auto& s : state) s = coin(rng) ? 1 : -1;

    double energy = bqm.energy(state);
    std::vector<std::int8_t> best = state;
    double best_energy = energy;
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    const std::size_t interval = std::max<std::size_t>(1, sweeps / std::max<std::size_t>(1, max_points));
    const auto wall_start = std::chrono::steady_clock::now();

    auto emit_point = [&](std::size_t sweep, double temp) {
        std::printf("{\"event\":\"sweep\",\"sweep\":%zu,\"best_energy\":%.4f,\"temperature\":%.4f,"
                    "\"objective\":%.4f",
                    sweep, best_energy, temp, objective(best));
        if (is_partition) {
            std::printf(",\"assign\":[");
            for (std::size_t i = 0; i < n; ++i) std::printf("%s%d", i ? "," : "", best[i]);
            std::printf("]");
        }
        std::printf("}\n");
        std::fflush(stdout);
    };

    for (std::size_t sweep = 0; sweep < sweeps; ++sweep) {
        const double t = temperature(sweep);
        for (std::size_t i = 0; i < n; ++i) {
            double field = bqm.linear(i);
            for (const auto& [j, coupling] : bqm.neighbors(i))
                field += coupling * static_cast<double>(state[j]);
            const double dE = -2.0 * static_cast<double>(state[i]) * field;
            if (dE <= 0.0 || u01(rng) < std::exp(-dE / t)) {
                state[i] = static_cast<std::int8_t>(-state[i]);
                energy += dE;
                if (energy < best_energy) {
                    best_energy = energy;
                    best = state;
                }
            }
        }
        if (sweep % interval == 0) emit_point(sweep, t);
    }
    emit_point(sweeps ? sweeps - 1 : 0, temperature(sweeps ? sweeps - 1 : 0));

    const double wall_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall_start)
            .count();

    // Optional speedup number: time a 1-thread and an N-thread parallel run
    // of the same budget so the lab can show a real speedup bar.
    double speedup = 0.0;
    if (threads > 1) {
        GeometricSchedule s(t0, alpha);
        auto time_run = [&](std::size_t th) {
            const auto a = std::chrono::steady_clock::now();
            ParallelAnnealer<GeometricSchedule> pa(bqm, s, sweeps, seed, th, th);
            pa.solve();
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a)
                .count();
        };
        const double t1 = time_run(1);
        const double tn = time_run(threads);
        if (tn > 0.0) speedup = t1 / tn;
    }

    std::printf("{\"event\":\"done\",\"best_energy\":%.4f,\"objective\":%.4f,\"sweeps\":%zu,"
                "\"wall_ms\":%.1f,\"speedup\":%.3f}\n",
                best_energy, objective(best), sweeps, wall_ms, speedup);
    std::fflush(stdout);
    return 0;
}
