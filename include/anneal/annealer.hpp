// annealer.hpp
//
// Sequential simulated annealing solver for Ising-vartype BQMs.
//
// Simulated annealing (Kirkpatrick, Gelatt, Vecchi 1983) searches for a
// low-energy state by repeatedly proposing single-spin flips and accepting
// or rejecting them with the Metropolis rule:
//
//   accept if dE <= 0 (downhill move: always take it)
//   else accept with probability exp(-dE / T)
//
// At high temperature T, uphill moves are accepted often, so the search
// explores broadly. As T is cooled toward zero (per the Schedule), uphill
// moves become rare and the state settles into a local (ideally global)
// energy minimum. One "sweep" here means proposing a flip for every
// variable once, in index order.
//
// This is the DELIBERATELY NAIVE version: for every proposed flip we
// recompute the local field (the sum of neighbor couplings) from scratch
// by walking the BQM's adjacency list, rather than maintaining a running
// cache. That is O(degree) per proposal, same asymptotic cost as the fast
// version in Milestone 3 will have, but the fast version updates a cached
// field incrementally after each *accepted* flip instead of recomputing
// it after every *proposed* flip. Having this slow-but-obviously-correct
// baseline first means Milestone 3's optimizations have something to be
// checked against (see the differential test there).
#pragma once

#include "anneal/bqm.hpp"
#include "anneal/rng.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace anneal {

struct SolveResult {
    std::vector<std::int8_t> best_state;
    double best_energy;
    std::vector<std::int8_t> final_state;
    double final_energy;
    std::size_t num_sweeps;
};

template <typename Schedule>
class Annealer {
public:
    // bqm may be Spin or Binary vartype; if Binary, a converted Spin copy
    // is made internally (the annealer always works in spin space, since
    // the Metropolis single-flip move is naturally s_i -> -s_i).
    Annealer(const BQM& bqm, Schedule schedule, std::size_t num_sweeps, std::uint64_t seed)
        : bqm_(bqm),
          schedule_(schedule),
          num_sweeps_(num_sweeps),
          rng_(seed) {
        if (bqm_.vartype() != Vartype::Spin) {
            bqm_.change_vartype(Vartype::Spin);
        }
    }

    // decision_log, if non-null, records one byte per proposed flip
    // (1 = accepted, 0 = rejected) in proposal order. Used by the
    // Milestone 3 differential test to check that the optimized annealer
    // makes bit-identical accept/reject decisions to this naive one.
    SolveResult solve(std::vector<std::uint8_t>* decision_log = nullptr) {
        std::size_t n = bqm_.num_variables();
        std::vector<std::int8_t> state(n);

        std::uniform_int_distribution<int> spin_dist(0, 1);
        for (std::size_t i = 0; i < n; ++i) {
            state[i] = spin_dist(rng_) ? std::int8_t(1) : std::int8_t(-1);
        }

        double current_energy = bqm_.energy(state);
        std::vector<std::int8_t> best_state = state;
        double best_energy = current_energy;

        for (std::size_t sweep = 0; sweep < num_sweeps_; ++sweep) {
            double t = schedule_.temperature(sweep);

            for (std::size_t i = 0; i < n; ++i) {
                // Local field: how much the neighborhood "wants" s_i to be
                // +1 vs -1. dE_i = -2 s_i (h_i + sum_j J_ij s_j) is the
                // energy change from flipping s_i to -s_i (see section 3
                // of the project spec for the derivation).
                double field = bqm_.linear(i);
                for (const auto& [j, coupling] : bqm_.neighbors(i)) {
                    field += coupling * static_cast<double>(state[j]);
                }
                double d_energy = -2.0 * static_cast<double>(state[i]) * field;

                bool accept = (d_energy <= 0.0) || (uniform01(rng_) < std::exp(-d_energy / t));
                if (decision_log) decision_log->push_back(accept ? 1 : 0);
                if (accept) {
                    state[i] = static_cast<std::int8_t>(-state[i]);
                    current_energy += d_energy;
                    if (current_energy < best_energy) {
                        best_energy = current_energy;
                        best_state = state;
                    }
                }
            }
        }

        return SolveResult{best_state, best_energy, state, current_energy, num_sweeps_};
    }

private:
    BQM bqm_;
    Schedule schedule_;
    std::size_t num_sweeps_;
    std::mt19937_64 rng_;
};

}  // namespace anneal
