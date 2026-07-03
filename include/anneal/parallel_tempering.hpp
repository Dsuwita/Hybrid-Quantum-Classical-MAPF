// parallel_tempering.hpp
//
// Parallel tempering (replica exchange Monte Carlo). Reference: Hukushima &
// Nemoto, "Exchange Monte Carlo method and application to spin glass
// simulations", J. Phys. Soc. Japan 1996.
//
// Plain simulated annealing cools one replica through a schedule and can get
// stuck: once the temperature is low, a wrong early decision is hard to
// undo. Parallel tempering instead runs R replicas simultaneously, each held
// at a FIXED temperature on a ladder from hot to cold, and periodically lets
// adjacent replicas SWAP configurations. A configuration that gets trapped
// in a poor basin at low temperature can ride up the ladder to a hot replica
// (which explores freely), escape, and ride back down. The hot replicas act
// as a source of fresh, uncorrelated states for the cold ones.
//
// The swap between replicas i and j (temperatures T_i, T_j, energies E_i,
// E_j) is accepted with the Metropolis-like probability
//
//     min(1, exp( (beta_i - beta_j) * (E_i - E_j) )),   beta = 1/T
//
// which exactly preserves the joint Boltzmann distribution over the ladder,
// so every replica stays a correct sample of its own temperature. We only
// ever attempt swaps between ADJACENT rungs, where acceptance is high.
//
// Within a temperature, each replica does ordinary Metropolis single-spin
// sweeps with a local-field cache (the same trick as the fast annealer):
// f_i = h_i + sum_j J_ij s_j, dE from flipping s_i is -2 s_i f_i, and an
// accepted flip updates only the neighbors' fields.
#pragma once

#include "anneal/annealer.hpp"       // SolveResult
#include "anneal/fast_annealer.hpp"  // CompactBQM
#include "anneal/rng.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace anneal {

// A geometric temperature ladder from t_min (coldest) to t_max (hottest)
// with `rungs` entries, ascending. Geometric spacing keeps the swap
// acceptance roughly constant along the ladder, which is the usual choice.
inline std::vector<double> geometric_ladder(double t_min, double t_max, std::size_t rungs) {
    std::vector<double> temps(rungs);
    if (rungs == 1) {
        temps[0] = t_min;
        return temps;
    }
    const double ratio = std::pow(t_max / t_min, 1.0 / static_cast<double>(rungs - 1));
    double t = t_min;
    for (std::size_t r = 0; r < rungs; ++r) {
        temps[r] = t;
        t *= ratio;
    }
    return temps;
}

struct TemperingResult {
    SolveResult best;                    // best state/energy over all replicas & sweeps
    std::vector<double> swap_rate;       // accepted-swap fraction per adjacent pair
    std::size_t swap_attempts = 0;
};

template <typename Rng = std::mt19937_64>
class ParallelTempering {
public:
    // temps: the temperature ladder (any order; sorted ascending internally).
    // num_sweeps: SA sweeps per replica. swap_interval: attempt neighbor
    // swaps every this many sweeps. seed: RNG seed.
    ParallelTempering(const BQM& bqm, std::vector<double> temps, std::size_t num_sweeps,
                      std::uint64_t seed, std::size_t swap_interval = 1)
        : view_(bqm),
          temps_(std::move(temps)),
          num_sweeps_(num_sweeps),
          swap_interval_(swap_interval == 0 ? 1 : swap_interval),
          rng_(seed) {
        std::sort(temps_.begin(), temps_.end());  // ascending: rung 0 = coldest
    }

    TemperingResult solve() {
        const std::size_t n = view_.num_variables;
        const std::size_t R = temps_.size();
        TemperingResult out;
        out.swap_rate.assign(R > 0 ? R - 1 : 0, 0.0);
        std::vector<std::uint64_t> swap_accepts(R > 0 ? R - 1 : 0, 0);

        // Per-replica working state: spins (as doubles, matching CompactBQM),
        // local fields, and current energy.
        std::vector<std::vector<double>> spin(R, std::vector<double>(n));
        std::vector<std::vector<double>> field(R, std::vector<double>(n));
        std::vector<double> energy(R);
        std::uniform_real_distribution<double> u01(0.0, 1.0);

        // Random independent starts, fields and energies initialized once.
        for (std::size_t r = 0; r < R; ++r) {
            for (std::size_t i = 0; i < n; ++i) spin[r][i] = (rng_() & 1u) ? 1.0 : -1.0;
            for (std::size_t i = 0; i < n; ++i) field[r][i] = view_.compute_field(i, spin[r]);
            energy[r] = view_.energy(spin[r]);
        }

        // Track the global best across every replica and every sweep.
        std::size_t best_r = 0;
        for (std::size_t r = 1; r < R; ++r)
            if (energy[r] < energy[best_r]) best_r = r;
        double best_energy = energy[best_r];
        std::vector<std::int8_t> best_state = to_int8(spin[best_r]);

        for (std::size_t sweep = 0; sweep < num_sweeps_; ++sweep) {
            // 1. One Metropolis sweep for every replica at its own temperature.
            for (std::size_t r = 0; r < R; ++r) {
                const double T = temps_[r];
                for (std::size_t i = 0; i < n; ++i) {
                    const double dE = -2.0 * spin[r][i] * field[r][i];
                    if (dE <= 0.0 || u01(rng_) < std::exp(-dE / T)) {
                        const double s_old = spin[r][i];
                        spin[r][i] = -s_old;
                        energy[r] += dE;
                        // Update neighbor fields: field_j changes by
                        // J_ij * (s_i_new - s_i_old) = J_ij * (-2 s_old).
                        const double delta = -2.0 * s_old;
                        for (std::size_t e = view_.row_start[i]; e < view_.row_start[i + 1]; ++e)
                            field[r][view_.nbr_index[e]] += view_.nbr_coupling[e] * delta;
                        if (energy[r] < best_energy) {
                            best_energy = energy[r];
                            best_state = to_int8(spin[r]);
                        }
                    }
                }
            }

            // 2. Periodically attempt swaps between adjacent rungs.
            if ((sweep + 1) % swap_interval_ == 0) {
                for (std::size_t r = 0; r + 1 < R; ++r) {
                    const double beta_lo = 1.0 / temps_[r];      // colder
                    const double beta_hi = 1.0 / temps_[r + 1];  // hotter
                    const double arg = (beta_lo - beta_hi) * (energy[r] - energy[r + 1]);
                    ++out.swap_attempts;
                    if (arg >= 0.0 || u01(rng_) < std::exp(arg)) {
                        // Swap the two configurations (spins, fields, energy).
                        std::swap(spin[r], spin[r + 1]);
                        std::swap(field[r], field[r + 1]);
                        std::swap(energy[r], energy[r + 1]);
                        ++swap_accepts[r];
                    }
                }
            }
        }

        for (std::size_t r = 0; r + 1 < R; ++r) {
            const double attempts = static_cast<double>(num_sweeps_ / swap_interval_);
            out.swap_rate[r] = attempts > 0 ? swap_accepts[r] / attempts : 0.0;
        }
        out.best.best_state = best_state;
        out.best.best_energy = best_energy;
        out.best.final_state = best_state;
        out.best.final_energy = best_energy;
        out.best.num_sweeps = num_sweeps_;
        return out;
    }

private:
    static std::vector<std::int8_t> to_int8(const std::vector<double>& spin) {
        std::vector<std::int8_t> out(spin.size());
        for (std::size_t i = 0; i < spin.size(); ++i) out[i] = spin[i] > 0 ? 1 : -1;
        return out;
    }

    CompactBQM view_;
    std::vector<double> temps_;
    std::size_t num_sweeps_;
    std::size_t swap_interval_;
    Rng rng_;
};

}  // namespace anneal
