// bqm.hpp
//
// Binary Quadratic Model (BQM): the shared problem container for the whole
// anneal library. A BQM represents an energy function over n variables that
// are either spins in {-1,+1} (Ising model) or bits in {0,1} (QUBO model):
//
//   Ising:  E(s) = offset + sum_i h_i s_i + sum_{i<j} J_ij s_i s_j
//   QUBO:   E(x) = offset + sum_i Q_ii x_i + sum_{i<j} Q_ij x_i x_j
//
// The two forms describe the same family of problems under the affine
// substitution x = (s+1)/2, and this class can convert between them while
// preserving the energy of every corresponding state exactly. See
// change_vartype() for the conversion formulas.
//
// Storage is adjacency-list: each variable holds a map from neighbor index
// to coupling strength, and every undirected edge (i,j) is stored once in
// i's list and once in j's list. This keeps energy() and per-variable
// neighbor iteration at O(degree) instead of scanning all pairs.
#pragma once

#include <cassert>
#include <cstdint>
#include <map>
#include <span>
#include <vector>

namespace anneal {

enum class Vartype { Spin, Binary };

class BQM {
public:
    explicit BQM(std::size_t num_variables, Vartype vartype)
        : vartype_(vartype),
          linear_(num_variables, 0.0),
          neighbors_(num_variables),
          offset_(0.0) {}

    std::size_t num_variables() const { return linear_.size(); }
    Vartype vartype() const { return vartype_; }
    double offset() const { return offset_; }

    // Accumulating semantics: repeated calls add to any existing bias
    // rather than overwrite it. This matches how problem-mapping code
    // (e.g. Max-Cut, one-hot penalties) naturally builds up a BQM term
    // by term without needing to check "have I touched this variable yet".
    void add_linear(std::size_t i, double bias) {
        assert(i < num_variables());
        linear_[i] += bias;
    }

    void add_interaction(std::size_t i, std::size_t j, double coupling) {
        assert(i < num_variables() && j < num_variables());
        assert(i != j && "no self-loops: use add_linear for diagonal terms");
        neighbors_[i][j] += coupling;
        neighbors_[j][i] += coupling;
    }

    void add_offset(double delta) { offset_ += delta; }

    double linear(std::size_t i) const { return linear_[i]; }

    const std::map<std::size_t, double>& neighbors(std::size_t i) const {
        return neighbors_[i];
    }

    // Evaluate the energy of a full assignment. state[i] must be in
    // {-1,+1} for Spin vartype or {0,1} for Binary vartype.
    // O(V + E): every variable's linear term once, every edge counted
    // from both endpoints but halved to avoid double-counting.
    double energy(std::span<const std::int8_t> state) const {
        assert(state.size() == num_variables());
        double e = offset_;
        for (std::size_t i = 0; i < num_variables(); ++i) {
            e += linear_[i] * static_cast<double>(state[i]);
            double local = 0.0;
            for (const auto& [j, coupling] : neighbors_[i]) {
                local += coupling * static_cast<double>(state[j]);
            }
            // Each edge (i,j) contributes coupling * s_i * s_j once to the
            // energy, but is stored in both i's and j's neighbor maps, so
            // summing "i's view" over all i double counts it. Halve it.
            e += 0.5 * static_cast<double>(state[i]) * local;
        }
        return e;
    }

    // Convert this BQM to the given vartype in place, rewriting all
    // biases so that energy(state) is unchanged for every corresponding
    // state under x = (s+1)/2. If already the target vartype, no-op.
    //
    // QUBO -> Ising (x = (s+1)/2, x_i in {0,1} -> s_i in {-1,+1}):
    //   h_i    = Q_ii/2 + (1/4) sum_{j != i} Q_ij
    //   J_ij   = Q_ij / 4
    //   offset += sum_i Q_ii/2 + sum_{i<j} Q_ij/4
    //
    // Ising -> QUBO (s = 2x - 1):
    //   Q_ii   = 2 h_i - 2 sum_{j != i} J_ij
    //   Q_ij   = 4 J_ij
    //   offset += -sum_i h_i + sum_{i<j} J_ij
    void change_vartype(Vartype target) {
        if (target == vartype_) return;

        if (vartype_ == Vartype::Binary && target == Vartype::Spin) {
            std::vector<double> new_linear(num_variables(), 0.0);
            double new_offset = offset_;
            for (std::size_t i = 0; i < num_variables(); ++i) {
                double neighbor_sum = 0.0;
                for (const auto& [j, q] : neighbors_[i]) {
                    (void)j;
                    neighbor_sum += q;
                }
                new_linear[i] = linear_[i] / 2.0 + neighbor_sum / 4.0;
                new_offset += linear_[i] / 2.0;
            }
            for (std::size_t i = 0; i < num_variables(); ++i) {
                for (auto& [j, q] : neighbors_[i]) {
                    if (j > i) new_offset += q / 4.0;
                    q = q / 4.0;
                }
            }
            linear_ = std::move(new_linear);
            offset_ = new_offset;
            vartype_ = Vartype::Spin;
        } else {
            // Spin -> Binary
            std::vector<double> new_linear(num_variables(), 0.0);
            double new_offset = offset_;
            for (std::size_t i = 0; i < num_variables(); ++i) {
                double neighbor_sum = 0.0;
                for (const auto& [j, jc] : neighbors_[i]) {
                    (void)j;
                    neighbor_sum += jc;
                }
                new_linear[i] = 2.0 * linear_[i] - 2.0 * neighbor_sum;
                new_offset -= linear_[i];
            }
            for (std::size_t i = 0; i < num_variables(); ++i) {
                for (auto& [j, jc] : neighbors_[i]) {
                    if (j > i) new_offset += jc;
                    jc = 4.0 * jc;
                }
            }
            linear_ = std::move(new_linear);
            offset_ = new_offset;
            vartype_ = Vartype::Binary;
        }
    }

private:
    Vartype vartype_;
    std::vector<double> linear_;
    std::vector<std::map<std::size_t, double>> neighbors_;
    double offset_;
};

}  // namespace anneal
