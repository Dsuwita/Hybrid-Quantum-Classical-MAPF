// fast_annealer.hpp
//
// Optimized single-threaded simulated annealer, following the techniques
// in Isakov, Zintchenko, Ronnow, Troyer, "Optimised simulated annealing
// for Ising spin glasses" (arXiv:1401.1084). Algorithmically identical to
// the naive Annealer in annealer.hpp (same Metropolis rule, same sweep
// order); every change below is purely about doing the same work faster.
//
// The optimizations are gated behind a compile-time OptLevel so each one
// can be benchmarked in isolation (see src/bench_throughput.cpp and
// bench/opt_log.md). Levels are cumulative:
//
//   Level 1 — flat storage. The BQM's std::map adjacency lists are copied
//     once into contiguous CSR-style arrays (a neighbor index array + a
//     parallel coupling array + per-variable offsets into both). Walking
//     a variable's neighbors becomes a linear scan of a few consecutive
//     doubles instead of chasing red-black tree nodes scattered across
//     the heap; the CPU prefetcher loves this.
//
//   Level 2 — local field cache. Instead of recomputing the local field
//     f_i = h_i + sum_j J_ij s_j from scratch for every proposal, keep
//     all n fields cached. A proposal is then one multiply
//     (dE = -2 s_i f_i). The price: when a flip IS accepted, walk i's
//     neighbors once and adjust their cached fields. Proposals vastly
//     outnumber accepts at low temperature, so this trades cheap-accept
//     for cheap-propose.
//
//   Level 3 — downhill early-exit. dE <= 0 is always accepted, so skip
//     both the exp() call and the RNG draw for downhill proposals.
//     (Correctness is unchanged: for dE <= 0, exp(-dE/T) >= 1 and a
//     uniform draw from [0,1) is always below it.)
//
//   Level 4 — acceptance lookup table. If every bias and coupling is an
//     integer (detected once when the compact view is built), every
//     possible dE is an even integer with |dE| <= 2 * max_i (|h_i| +
//     sum_j |J_ij|). So each time the temperature changes (once per
//     sweep), precompute the acceptance threshold for every positive
//     integer dE up to that bound; the hot loop replaces exp() with an
//     array lookup. Non-integer problems fall back to calling exp().
//
// The RNG is a template parameter (std::mt19937_64 by default, Xoshiro256pp
// from rng.hpp as the fast alternative — that swap is optimization 5).
//
// Equivalence guarantee, used by the differential test: with the same
// seed and the same RNG type, levels 3 and 4 consume the random stream
// exactly as the naive Annealer does (one draw per UPHILL proposal only),
// so on integer-coupling problems they reproduce its accept/reject
// decisions bit for bit. Levels 1-2 deliberately draw a uniform for
// every proposal (the spec's step ordering puts early-exit third), so
// their random streams differ from naive; they are still the same
// Metropolis algorithm.
#pragma once

#include "anneal/annealer.hpp"  // SolveResult
#include "anneal/bqm.hpp"
#include "anneal/rng.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace anneal {

// Read-only CSR-style snapshot of a (spin-vartype) BQM, built once from
// the std::map-based BQM and then never modified. Separated from the
// annealer itself so that parallel restarts (parallel.hpp) can build the
// view a single time and share it across worker threads: sharing is safe
// precisely because nothing in a solve ever writes to it.
struct CompactBQM {
    std::size_t num_variables = 0;
    double offset = 0.0;
    std::vector<double> linear;
    // Variable i's neighbors live at [row_start[i], row_start[i+1]) of
    // nbr_index / nbr_coupling.
    std::vector<std::size_t> row_start;
    std::vector<std::size_t> nbr_index;
    std::vector<double> nbr_coupling;
    // Level 4 metadata: whether every bias/coupling is an integer, and
    // if so how large the per-temperature acceptance table must be
    // (max possible dE + 1). The table itself lives in the annealer
    // (it changes with temperature); only its size is decided here.
    bool integer_mode = false;
    std::size_t accept_table_size = 0;

    explicit CompactBQM(const BQM& input) {
        // Work in spin space, like the naive annealer.
        const BQM* source = &input;
        BQM converted(0, Vartype::Spin);
        if (input.vartype() != Vartype::Spin) {
            converted = input;
            converted.change_vartype(Vartype::Spin);
            source = &converted;
        }
        const BQM& bqm = *source;

        num_variables = bqm.num_variables();
        offset = bqm.offset();
        linear.resize(num_variables);
        row_start.resize(num_variables + 1, 0);
        for (std::size_t i = 0; i < num_variables; ++i) {
            linear[i] = bqm.linear(i);
            row_start[i + 1] = row_start[i] + bqm.neighbors(i).size();
        }
        nbr_index.resize(row_start.back());
        nbr_coupling.resize(row_start.back());
        for (std::size_t i = 0; i < num_variables; ++i) {
            std::size_t e = row_start[i];
            for (const auto& [j, coupling] : bqm.neighbors(i)) {
                nbr_index[e] = j;
                nbr_coupling[e] = coupling;
                ++e;
            }
        }
        detect_integer_mode();
    }

    double compute_field(std::size_t i, const std::vector<double>& spin) const {
        double field = linear[i];
        for (std::size_t e = row_start[i]; e < row_start[i + 1]; ++e) {
            field += nbr_coupling[e] * spin[nbr_index[e]];
        }
        return field;
    }

    double energy(const std::vector<double>& spin) const {
        double e = offset;
        for (std::size_t i = 0; i < num_variables; ++i) {
            e += linear[i] * spin[i];
            double local = 0.0;
            for (std::size_t k = row_start[i]; k < row_start[i + 1]; ++k) {
                local += nbr_coupling[k] * spin[nbr_index[k]];
            }
            e += 0.5 * spin[i] * local;  // each edge visited from both endpoints
        }
        return e;
    }

private:
    // Check whether every bias/coupling is an integer. If so, the largest
    // possible uphill dE is 2 * max_i (|h_i| + sum |J_ij|), which bounds
    // the acceptance table. Bail out to the exp() fallback if the table
    // would be unreasonably large (huge integer couplings give no benefit
    // over exp).
    void detect_integer_mode() {
        double max_field_magnitude = 0.0;
        for (std::size_t i = 0; i < num_variables; ++i) {
            if (std::floor(linear[i]) != linear[i]) return;
            double mag = std::fabs(linear[i]);
            for (std::size_t e = row_start[i]; e < row_start[i + 1]; ++e) {
                if (std::floor(nbr_coupling[e]) != nbr_coupling[e]) return;
                mag += std::fabs(nbr_coupling[e]);
            }
            max_field_magnitude = std::max(max_field_magnitude, mag);
        }
        const double max_de = 2.0 * max_field_magnitude;
        if (max_de > 65536.0) return;  // table too big to be worth it
        integer_mode = true;
        accept_table_size = static_cast<std::size_t>(max_de) + 1;
    }
};

template <typename Schedule, typename Rng = std::mt19937_64, int OptLevel = 4>
class FastAnnealer {
    static_assert(OptLevel >= 1 && OptLevel <= 4, "OptLevel must be 1..4");

public:
    // Standalone use: build (and own) the compact view from a BQM.
    FastAnnealer(const BQM& bqm, Schedule schedule, std::size_t num_sweeps, std::uint64_t seed)
        : owned_view_(std::make_shared<const CompactBQM>(bqm)),
          view_(owned_view_.get()),
          schedule_(schedule),
          num_sweeps_(num_sweeps),
          rng_(seed) {
        allocate_accept_table();
    }

    // Shared-view use (parallel restarts): the caller guarantees the view
    // outlives this annealer. The view is strictly read-only here, so any
    // number of annealers on any threads may share one.
    FastAnnealer(const CompactBQM& view, Schedule schedule, std::size_t num_sweeps,
                 std::uint64_t seed)
        : view_(&view), schedule_(schedule), num_sweeps_(num_sweeps), rng_(seed) {
        allocate_accept_table();
    }

    // decision_log: same contract as Annealer::solve, for differential
    // tests. Dispatching on it at compile time keeps the log-recording
    // branch entirely out of the benchmarked hot loop.
    SolveResult solve(std::vector<std::uint8_t>* decision_log = nullptr) {
        return decision_log ? solve_impl<true>(decision_log) : solve_impl<false>(nullptr);
    }

private:
    // One cache entry per variable, interleaved so a proposal touches a
    // single cache line (see the comment at the point of use).
    struct SpinField {
        double minus_two_spin;  // -2 * s_i, i.e. -2.0 or +2.0
        double field;           // f_i = h_i + sum_j J_ij s_j
    };

    void allocate_accept_table() {
        if constexpr (OptLevel >= 4) {
            accept_table_.assign(view_->accept_table_size, 0.0);
        }
    }

    template <bool kRecordDecisions>
    SolveResult solve_impl(std::vector<std::uint8_t>* decision_log) {
        const CompactBQM& view = *view_;
        const std::size_t n = view.num_variables;
        const bool integer_mode = view.integer_mode;

        // Spins are stored as doubles (+1.0 / -1.0) in the hot loop: dE
        // and the field updates are double arithmetic, and keeping the
        // spin already in double form removes an int8->double conversion
        // from every single proposal. The values are exactly +/-1.0, so
        // results are bit-identical to an int8 representation; we convert
        // back to int8 only when exporting states in the SolveResult.
        std::vector<double> spin(n);

        // Local working copy of the RNG (see the aliasing comment below);
        // written back to the member before returning.
        Rng rng = rng_;

        // Identical initial-state generation to the naive annealer, so a
        // same-seed run starts from the same configuration.
        std::uniform_int_distribution<int> spin_dist(0, 1);
        for (std::size_t i = 0; i < n; ++i) {
            spin[i] = spin_dist(rng) ? 1.0 : -1.0;
        }

        // Level 2+: the local field cache, initialized once and then only
        // updated incrementally on accepted flips. Each variable's cache
        // entry holds TWO values, interleaved in one struct:
        //   minus_two_spin = -2 * s_i   (so dE is a single multiply:
        //                                dE = -2 s_i f_i = ms_i * f_i)
        //   field          = f_i
        // Interleaving matters: a proposal reads both values, and packing
        // them side by side means one cache line touched per proposal
        // instead of two lines from two separate arrays. The plain spin
        // array is still maintained on accepts (one extra store) so state
        // export and best-state copies stay simple.
        // Exactness: -2*s is a power-of-two multiple of +/-1, so ms*f
        // rounds identically to (-2.0*s)*f, and the field update
        // f_j -= J_ij * ms_i equals f_j += 2 J_ij s_i exactly; decisions
        // stay bit-identical to the naive annealer.
        std::vector<SpinField> pairs;
        if constexpr (OptLevel >= 2) {
            pairs.resize(n);
            for (std::size_t i = 0; i < n; ++i) {
                pairs[i] = SpinField{-2.0 * spin[i], view.compute_field(i, spin)};
            }
        }

        double current_energy = view.energy(spin);
        // Best-so-far is kept as a preallocated double buffer: assigning
        // `best_spin = spin` on an improvement is a straight memcpy with
        // no allocation (capacity is already there), where converting to
        // int8 on every improvement would allocate in the hot phase.
        std::vector<double> best_spin = spin;
        double best_energy = current_energy;

        // Hoist everything the hot loop touches into locals. The RNG and
        // the arrays are class members; accessed through `this`, the
        // compiler must assume any store into spin/pairs could alias
        // them, forcing it to spill and reload the RNG state around every
        // draw (8 extra memory ops per proposal). A local RNG copy whose
        // address is never taken, plus raw local pointers, let it keep
        // the RNG state and pointers in registers across the whole loop.
        double* const spin_ptr = spin.data();
        [[maybe_unused]] SpinField* const pairs_ptr = pairs.data();
        [[maybe_unused]] const std::size_t* const row_start = view.row_start.data();
        [[maybe_unused]] const std::size_t* const nbr_index = view.nbr_index.data();
        [[maybe_unused]] const double* const nbr_coupling = view.nbr_coupling.data();
        [[maybe_unused]] double* const table = accept_table_.data();
        [[maybe_unused]] const std::size_t table_size = accept_table_.size();

        for (std::size_t sweep = 0; sweep < num_sweeps_; ++sweep) {
            const double t = schedule_.temperature(sweep);

            // Level 4: refresh the acceptance table for this sweep's
            // temperature. Entries are exp(-k/t) PREMULTIPLIED by 2^53:
            // the Metropolis test "(u >> 11) * 2^-53 < p" (u a 64-bit
            // draw) is exactly equivalent to "double(u >> 11) < p * 2^53"
            // because both sides of either comparison are exactly
            // representable doubles (u >> 11 < 2^53 fits a mantissa, and
            // scaling p by a power of two is exact). Folding the scale
            // into the table removes a multiply from every uphill draw
            // while preserving bit-identical decisions vs the naive path.
            if constexpr (OptLevel >= 4) {
                if (integer_mode) {
                    constexpr double kTwoPow53 = 9007199254740992.0;
                    for (std::size_t k = 1; k < table_size; ++k) {
                        table[k] = std::exp(-static_cast<double>(k) / t) * kTwoPow53;
                    }
                }
            }

            for (std::size_t i = 0; i < n; ++i) {
                double d_energy;
                if constexpr (OptLevel >= 2) {
                    d_energy = pairs_ptr[i].minus_two_spin * pairs_ptr[i].field;
                } else {
                    d_energy = -2.0 * spin_ptr[i] * view.compute_field(i, spin);
                }

                bool accept;
                if constexpr (OptLevel >= 3) {
                    if (d_energy <= 0.0) {
                        accept = true;
                    } else if constexpr (OptLevel >= 4) {
                        accept = integer_mode
                                     ? static_cast<double>(rng() >> 11) <
                                           table[static_cast<std::size_t>(d_energy)]
                                     : uniform01(rng) < std::exp(-d_energy / t);
                    } else {
                        accept = uniform01(rng) < std::exp(-d_energy / t);
                    }
                } else {
                    // Levels 1-2: always draw. Correct because for
                    // dE <= 0, exp(-dE/T) >= 1 > any draw from [0,1).
                    accept = uniform01(rng) < std::exp(-d_energy / t);
                }

                if constexpr (kRecordDecisions) decision_log->push_back(accept ? 1 : 0);

                if (accept) {
                    spin_ptr[i] = -spin_ptr[i];
                    if constexpr (OptLevel >= 2) {
                        // s_i changed sign; each neighbor j's field
                        // f_j = h_j + sum_k J_jk s_k shifts by 2 J_ij s_i_new,
                        // and 2 J_ij s_i_new = -J_ij * minus_two_spin_i_new.
                        const double ms_new = -pairs_ptr[i].minus_two_spin;
                        pairs_ptr[i].minus_two_spin = ms_new;
                        for (std::size_t e = row_start[i]; e < row_start[i + 1]; ++e) {
                            pairs_ptr[nbr_index[e]].field -= nbr_coupling[e] * ms_new;
                        }
                    }
                    current_energy += d_energy;
                    if (current_energy < best_energy) {
                        best_energy = current_energy;
                        best_spin = spin;
                    }
                }
            }
        }

        rng_ = rng;  // preserve "the member RNG advanced" semantics
        return SolveResult{to_int8(best_spin), best_energy, to_int8(spin), current_energy,
                           num_sweeps_};
    }

    static std::vector<std::int8_t> to_int8(const std::vector<double>& spin) {
        std::vector<std::int8_t> out(spin.size());
        for (std::size_t i = 0; i < spin.size(); ++i) {
            out[i] = spin[i] > 0.0 ? std::int8_t(1) : std::int8_t(-1);
        }
        return out;
    }

    // Present only when this annealer built its own view; empty when a
    // caller-owned shared view is used.
    std::shared_ptr<const CompactBQM> owned_view_;
    const CompactBQM* view_;

    // Level 4 acceptance table, rebuilt at each temperature change.
    std::vector<double> accept_table_;

    Schedule schedule_;
    std::size_t num_sweeps_;
    Rng rng_;
};

}  // namespace anneal
