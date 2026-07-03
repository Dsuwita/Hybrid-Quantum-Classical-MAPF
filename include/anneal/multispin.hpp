// multispin.hpp
//
// Multi-spin coding: run 64 independent annealing replicas at once by packing
// one bit per replica into a uint64_t word per spin, then updating all 64
// with bitwise operations. Reference: Isakov et al. 2015 (arXiv:1401.1084),
// the "an_ms" codes. Scoped to the case the technique is built for: Ising
// models with +/-1 couplings and zero fields (h = 0), i.e. unweighted
// Max-Cut, which is where the arithmetic below stays exact and cheap.
//
// Layout. word[i] holds site i for all 64 replicas: bit r is replica r's
// spin at site i, with 1 <-> +1 and 0 <-> -1. Two spins are equal exactly
// when their bits match, which is what makes the whole update bitwise.
//
// The flip decision for site i, per replica r. Ising energy is
// E = sum_edges J s_u s_v. Let k_r = the number of neighbours j of i whose
// edge is currently UNSATISFIED (J s_i s_j = +1, i.e. it raises the energy).
// Flipping s_i satisfies those k_r edges and unsatisfies the other d - k_r,
// so the energy change is dE_r = 2d - 4 k_r (d = degree of i). Downhill
// (dE <= 0) exactly when k_r >= d/2.
//
// Computing k_r for all 64 replicas at once. For each neighbour j with
// coupling c, the per-replica "unsatisfied" bit is XNOR(word[i], word[j]) if
// c = +1, or XOR if c = -1. Summing those 0/1 bits across the neighbours is
// a bit-sliced integer addition (a small ripple-carry adder over bit-planes),
// giving k_r in binary, one lane per replica.
//
// Acceptance for all 64 replicas at once (an_ms). For each possible value v
// of k there is one dE and one probability p_v = exp(-dE/T). We build the
// lane mask {k_r == v} by bit-sliced equality, and for uphill v we AND it
// with {rand_r < floor(p_v * 2^B)} computed by a bit-sliced magnitude
// compare of one B-bit per-lane random number against the constant
// threshold. The accepted lanes' bits in word[i] are then flipped in one XOR.
#pragma once

#include "anneal/bqm.hpp"
#include "anneal/fast_annealer.hpp"  // CompactBQM
#include "anneal/rng.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace anneal {

struct MultiSpinResult {
    std::vector<std::int8_t> best_state;  // best single-replica state found
    double best_energy = 0.0;
    std::size_t replicas = 64;
    std::size_t num_sweeps = 0;
};

// B = fixed-point bits for the acceptance probability (2^-B resolution).
template <typename Schedule, typename Rng = Xoshiro256pp, unsigned B = 16>
class MultiSpinAnnealer {
public:
    static constexpr std::size_t W = 64;  // replicas per word

    MultiSpinAnnealer(const BQM& bqm, Schedule schedule, std::size_t num_sweeps,
                      std::uint64_t seed, std::size_t sample_interval = 32)
        : view_(bqm),
          schedule_(schedule),
          num_sweeps_(num_sweeps),
          rng_(seed),
          sample_interval_(sample_interval == 0 ? 1 : sample_interval) {
        validate_pm1_zero_field();
        // Enough bit-planes to hold the largest degree exactly.
        std::size_t max_deg = 0;
        for (std::size_t i = 0; i < view_.num_variables; ++i)
            max_deg = std::max(max_deg, view_.row_start[i + 1] - view_.row_start[i]);
        planes_ = static_cast<unsigned>(std::bit_width(max_deg)) + 1;
    }

    MultiSpinResult solve() {
        const std::size_t n = view_.num_variables;
        std::vector<std::uint64_t> word(n);
        for (std::size_t i = 0; i < n; ++i) word[i] = rng_();  // random 64 replicas

        MultiSpinResult out;
        out.num_sweeps = num_sweeps_;
        out.best_energy = sample_best(word, out.best_state);  // initial best

        std::vector<std::uint64_t> kplane(planes_);
        std::array<std::uint64_t, B> rbits{};

        for (std::size_t sweep = 0; sweep < num_sweeps_; ++sweep) {
            const double T = schedule_.temperature(sweep);
            for (std::size_t i = 0; i < n; ++i) {
                const std::size_t deg = view_.row_start[i + 1] - view_.row_start[i];
                if (deg == 0) continue;

                // --- bit-sliced k_r = number of unsatisfied incident edges ---
                for (auto& p : kplane) p = 0;
                for (std::size_t e = view_.row_start[i]; e < view_.row_start[i + 1]; ++e) {
                    const std::uint64_t x = word[i] ^ word[view_.nbr_index[e]];
                    std::uint64_t carry = (view_.nbr_coupling[e] > 0.0) ? ~x : x;
                    for (unsigned b = 0; b < planes_ && carry; ++b) {
                        const std::uint64_t t = kplane[b];
                        kplane[b] = t ^ carry;
                        carry = t & carry;
                    }
                }

                // --- accept mask over all k-values ---
                // The B-bit per-lane random word is drawn lazily: only the
                // first time an uphill k-value actually has lanes (many sites
                // are all-downhill and need no randomness at all). Once drawn
                // it is reused across the remaining k-values, since each lane
                // has exactly one k.
                std::uint64_t accept = 0;
                bool have_rand = false;
                for (std::size_t v = 0; v <= deg; ++v) {
                    const std::uint64_t mask = equals(kplane, v);
                    if (!mask) continue;
                    const double dE = 2.0 * deg - 4.0 * static_cast<double>(v);
                    if (dE <= 0.0) {
                        accept |= mask;  // downhill: all these lanes flip
                    } else {
                        if (!have_rand) {
                            for (unsigned b = 0; b < B; ++b) rbits[b] = rng_();
                            have_rand = true;
                        }
                        const double p = std::exp(-dE / T);
                        const std::uint64_t K =
                            static_cast<std::uint64_t>(p * static_cast<double>(kMax));
                        accept |= mask & rand_lt(rbits, K);
                    }
                }
                word[i] ^= accept;  // flip the accepted lanes' bits at site i
            }

            if ((sweep + 1) % sample_interval_ == 0) {
                std::vector<std::int8_t> cand;
                double e = sample_best(word, cand);
                if (e < out.best_energy) {
                    out.best_energy = e;
                    out.best_state = std::move(cand);
                }
            }
        }
        // Final sample too, so the last sweep's state is not missed.
        std::vector<std::int8_t> cand;
        double e = sample_best(word, cand);
        if (e < out.best_energy) {
            out.best_energy = e;
            out.best_state = std::move(cand);
        }
        return out;
    }

    // Exposed for the correctness test: per-lane k_r at site i for a given
    // packed state, computed by the bit-sliced path.
    std::vector<std::size_t> lane_k(std::size_t i, const std::vector<std::uint64_t>& word) const {
        std::vector<std::uint64_t> kplane(planes_, 0);
        for (std::size_t e = view_.row_start[i]; e < view_.row_start[i + 1]; ++e) {
            const std::uint64_t x = word[i] ^ word[view_.nbr_index[e]];
            std::uint64_t carry = (view_.nbr_coupling[e] > 0.0) ? ~x : x;
            for (unsigned b = 0; b < planes_ && carry; ++b) {
                const std::uint64_t t = kplane[b];
                kplane[b] = t ^ carry;
                carry = t & carry;
            }
        }
        std::vector<std::size_t> out(W, 0);
        for (std::size_t r = 0; r < W; ++r)
            for (unsigned b = 0; b < planes_; ++b)
                out[r] |= static_cast<std::size_t>((kplane[b] >> r) & 1u) << b;
        return out;
    }

    const CompactBQM& view() const { return view_; }

private:
    static constexpr std::uint64_t kMax = std::uint64_t(1) << B;

    // Lane mask where the bit-sliced integer in `planes` equals constant v.
    std::uint64_t equals(const std::vector<std::uint64_t>& plane, std::size_t v) const {
        std::uint64_t match = ~std::uint64_t(0);
        for (unsigned b = 0; b < planes_; ++b) {
            const std::uint64_t bit = (v >> b) & 1u ? plane[b] : ~plane[b];
            match &= bit;
        }
        return match;
    }

    // Lane mask where the per-lane B-bit random number (bit-planes r[0..B-1],
    // r[b] the b-th bit of every lane) is strictly less than constant K.
    // Walk MSB->LSB tracking "equal so far"; a lane goes "less" the first
    // time it has a 0 where K has a 1.
    static std::uint64_t rand_lt(const std::array<std::uint64_t, B>& r, std::uint64_t K) {
        std::uint64_t lt = 0, eq = ~std::uint64_t(0);
        for (int b = static_cast<int>(B) - 1; b >= 0; --b) {
            const std::uint64_t Rb = r[b];
            if ((K >> b) & 1u) {
                lt |= eq & ~Rb;  // K bit 1, R bit 0 -> less
                eq &= Rb;        // stay equal only where R bit is 1
            } else {
                eq &= ~Rb;       // K bit 0, R bit 1 -> greater (drop from equal)
            }
        }
        return lt;
    }

    // Reconstruct all 64 lane states, return the lowest energy and its state.
    double sample_best(const std::vector<std::uint64_t>& word, std::vector<std::int8_t>& best) const {
        const std::size_t n = view_.num_variables;
        std::vector<double> spin(n);
        double best_e = 1e300;
        std::vector<std::int8_t> tmp(n);
        for (std::size_t r = 0; r < W; ++r) {
            for (std::size_t i = 0; i < n; ++i) spin[i] = ((word[i] >> r) & 1u) ? 1.0 : -1.0;
            const double e = view_.energy(spin);
            if (e < best_e) {
                best_e = e;
                for (std::size_t i = 0; i < n; ++i) tmp[i] = spin[i] > 0 ? 1 : -1;
                best = tmp;
            }
        }
        return best_e;
    }

    void validate_pm1_zero_field() const {
        for (std::size_t i = 0; i < view_.num_variables; ++i) {
            if (view_.linear[i] != 0.0)
                throw std::runtime_error("MultiSpinAnnealer: requires zero fields (h = 0)");
            for (std::size_t e = view_.row_start[i]; e < view_.row_start[i + 1]; ++e) {
                const double c = view_.nbr_coupling[e];
                if (c != 1.0 && c != -1.0)
                    throw std::runtime_error("MultiSpinAnnealer: requires +/-1 couplings");
            }
        }
    }

    CompactBQM view_;
    Schedule schedule_;
    std::size_t num_sweeps_;
    Rng rng_;
    std::size_t sample_interval_;
    unsigned planes_ = 1;
};

}  // namespace anneal
