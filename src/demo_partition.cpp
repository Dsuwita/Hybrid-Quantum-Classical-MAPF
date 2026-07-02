// demo_partition.cpp
//
// Small demo: solve a 30-number partitioning instance with the annealer
// and print the resulting two-group split.
//
// Number partitioning: given numbers a_1..a_n, split them into two groups
// so the difference between group sums is as small as possible. Framed as
// minimizing (sum a_i s_i)^2 over spins s_i in {-1,+1} (s_i encodes which
// group a_i lands in). Expanding the square gives the Ising couplings
// J_ij = 2 a_i a_j and constant offset sum a_i^2 (see project spec
// section 3).

#include "anneal/annealer.hpp"
#include "anneal/bqm.hpp"
#include "anneal/schedule.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

using namespace anneal;

int main() {
    std::vector<double> numbers = {
        23, 45, 12, 67, 34, 89, 21, 56, 78, 90,
        11, 33, 44, 55, 66, 77, 88, 99, 10, 20,
        30, 40, 50, 60, 70, 80, 15, 25, 35, 45,
    };

    std::size_t n = numbers.size();
    double total = 0.0;
    for (double a : numbers) total += a;

    BQM bqm(n, Vartype::Spin);
    double offset = 0.0;
    for (double a : numbers) offset += a * a;
    bqm.add_offset(offset);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            bqm.add_interaction(i, j, 2.0 * numbers[i] * numbers[j]);
        }
    }

    SolveResult best;
    best.best_energy = std::numeric_limits<double>::infinity();
    const int num_seeds = 5;
    for (int s = 0; s < num_seeds; ++s) {
        GeometricSchedule schedule(50.0, 0.995);
        Annealer<GeometricSchedule> annealer(bqm, schedule, 4000, static_cast<std::uint64_t>(s));
        SolveResult r = annealer.solve();
        if (r.best_energy < best.best_energy) best = r;
    }

    double group_a = 0.0, group_b = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        if (best.best_state[i] == 1) {
            group_a += numbers[i];
        } else {
            group_b += numbers[i];
        }
    }

    std::printf("Number partitioning demo: %zu numbers, total = %.1f\n", n, total);
    std::printf("Group A sum: %.1f\n", group_a);
    std::printf("Group B sum: %.1f\n", group_b);
    std::printf("Difference: %.1f\n", std::fabs(group_a - group_b));
    std::printf("Best energy (should equal difference^2): %.6f\n", best.best_energy);

    return 0;
}
