// rng.hpp
//
// xoshiro256++ pseudo-random generator (Blackman & Vigna, public domain,
// https://prng.di.unimi.it/). Much cheaper per draw than std::mt19937_64
// (a handful of shifts/xors vs the Mersenne Twister's large state churn)
// while still passing the standard statistical test batteries. Not
// cryptographic, which is fine: an annealer only needs statistically
// unbiased noise, fast.
//
// The class satisfies the C++ UniformRandomBitGenerator concept
// (result_type, min(), max(), operator()), so it drops into
// std::uniform_real_distribution and friends exactly like mt19937_64.
// mt19937_64 remains the library default for reproducibility
// documentation; this is the opt-in fast alternative.
#pragma once

#include <array>
#include <cstdint>

namespace anneal {

// Draw a uniform double in [0, 1) from a 64-bit generator: keep the top
// 53 bits (a double mantissa holds exactly 53 bits) and scale by 2^-53.
//
// Used by BOTH the naive and the optimized annealer instead of
// std::uniform_real_distribution, for two reasons:
//  1. The differential test demands that naive and optimized paths make
//     bit-identical accept/reject decisions from the same seed, which
//     requires drawing uniforms in exactly the same way.
//  2. libstdc++'s generate_canonical machinery costs several times more
//     than this one shift and multiply (measured in bench/opt_log.md);
//     the uniform draw sits in the annealer's hot loop.
template <typename Rng>
inline double uniform01(Rng& rng) {
    static_assert(Rng::max() == ~std::uint64_t{0} && Rng::min() == 0,
                  "uniform01 assumes a full-range 64-bit generator");
    return static_cast<double>(rng() >> 11) * 0x1.0p-53;
}

class Xoshiro256pp {
public:
    using result_type = std::uint64_t;

    // A xoshiro generator needs 256 bits of well-mixed initial state; a
    // raw 64-bit seed (especially a small one like 0, 1, 2...) would
    // leave most of the state zero and correlate nearby seeds. splitmix64
    // is the seeding mixer the xoshiro authors recommend for exactly this.
    explicit Xoshiro256pp(std::uint64_t seed) {
        std::uint64_t x = seed;
        for (auto& word : state_) {
            x += 0x9e3779b97f4a7c15ULL;
            std::uint64_t z = x;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            word = z ^ (z >> 31);
        }
    }

    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return ~std::uint64_t{0}; }

    result_type operator()() {
        const std::uint64_t result = rotl(state_[0] + state_[3], 23) + state_[0];
        const std::uint64_t t = state_[1] << 17;
        state_[2] ^= state_[0];
        state_[3] ^= state_[1];
        state_[1] ^= state_[2];
        state_[0] ^= state_[3];
        state_[2] ^= t;
        state_[3] = rotl(state_[3], 45);
        return result;
    }

private:
    static constexpr std::uint64_t rotl(std::uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    std::array<std::uint64_t, 4> state_;
};

}  // namespace anneal
