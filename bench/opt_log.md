# Milestone 3 optimization log

Single-threaded annealer throughput, measured after each optimization.

## Setup

- Problem: random 3-regular graph, n = 2000, couplings +/-1, h = 0
  (built by three random perfect matchings, seed 12345).
- Schedule: geometric, T0 = 3.0, alpha = 0.999, 10000 sweeps. Every
  config runs the same schedule and budget, so all configs face the same
  mix of hot (accept-heavy) and frozen (reject-heavy) work.
- Metric: proposals per second = n * sweeps / wall time. Each config is
  the minimum of 5 timed runs, interleaved round-robin across configs so
  background-load waves cannot eat all of one config's runs. One
  untimed warmup run precedes everything.
- Machine: AMD Ryzen 5 7600, g++ 13, -std=c++20 -O2, WSL2. Runs pinned
  with `taskset -c 4`. Reproduce with:
  `cmake --build build && taskset -c 4 ./build/bench_throughput`
- Run-to-run variance on this machine is about +/-15 percent for the
  absolute numbers (WSL2 frequency and load noise). The final speedup
  ratio measured 9.6x to 10.6x over four pinned runs, median 10.2x.

## Results (one representative run)

| config | flips/s | flips/ns | speedup |
|---|---|---|---|
| M2 baseline (verbatim copy)  | 2.72e7 | 0.027 | 1.0x |
| naive + fast uniform01       | 3.56e7 | 0.036 | 1.3x |
| L1 flat CSR storage          | 4.85e7 | 0.048 | 1.8x |
| L2 + local field cache       | 6.25e7 | 0.063 | 2.3x |
| L3 + downhill early-exit     | 6.46e7 | 0.065 | 2.4x |
| L4 + acceptance lookup table | 1.74e8 | 0.174 | 6.4x |
| L4 + xoshiro256++            | 2.80e8 | 0.280 | 10.3x |

All optimized configs pass the differential test: with the same seed and
mt19937_64, level 4 reproduces the naive annealer's accept/reject
decisions bit for bit (5 integer + 2 float instances, tests/test_fast_annealer.cpp).

## What each step did

1. **Fast uniform draw (uniform01)**: replaced
   std::uniform_real_distribution with `(rng() >> 11) * 2^-53` in BOTH
   the naive and fast annealers. Required anyway so the differential
   test can compare decision streams; also removes libstdc++'s
   generate_canonical overhead. This is why the in-tree naive annealer
   is 1.3x the frozen M2 baseline copy kept in bench_throughput.cpp.
2. **L1 flat storage**: the BQM's std::map adjacency lists are copied
   once into CSR arrays (neighbor indices + couplings + per-variable
   offsets). Walking neighbors becomes a contiguous scan instead of
   pointer chasing. Levels 1-2 draw a uniform on every proposal (the
   early exit is step 3), so their random streams differ from naive;
   they are still exact Metropolis (for dE <= 0, exp(-dE/T) >= 1 always
   beats a [0,1) draw).
3. **L2 local field cache**: keep f_i = h_i + sum_j J_ij s_j for all i;
   a proposal is one multiply (dE = -2 s_i f_i), an accepted flip
   updates the neighbors' cached fields. The spin and field are stored
   interleaved in one struct (and the spin premultiplied as -2s), so a
   proposal reads one cache line and does one multiply. Exactness is
   preserved: -2s is a power-of-two multiple of +/-1.
4. **L3 downhill early-exit**: accept dE <= 0 without drawing. Small
   effect here because most of the budget is spent at low temperature
   where proposals are uphill anyway; it also restores the naive
   annealer's draw-only-on-uphill RNG consumption, which is what makes
   the level 3+ decision streams comparable to naive.
5. **L4 acceptance lookup table**: with integer couplings (detected at
   solve start) every dE is a small integer, so exp(-dE/T) is
   precomputed per temperature change (once a sweep) and the hot loop
   does a table lookup instead of calling exp(). Entries are
   premultiplied by 2^53 so the test is `double(rng() >> 11) < table[dE]`
   with no scaling multiply; the comparison is exactly equivalent
   (both sides are exactly representable doubles). Biggest single win:
   2.7x over L3. Non-integer problems fall back to exp().
6. **xoshiro256++**: swaps mt19937_64 (2.5 KB of state, expensive
   output function) for xoshiro256++ (32 bytes, a handful of xors and
   rotates). 1.6x on top of L4. mt19937_64 stays the default;
   the RNG is a template parameter.

Two cross-cutting fixes found while measuring, folded into all
FastAnnealer levels:

- **Aliasing**: the RNG and arrays are class members; accessed through
  `this`, every store into the spin/field arrays forced the compiler to
  spill and reload RNG state. Copying the RNG into a local and hoisting
  raw data pointers before the loop was worth ~30 percent on the final
  config.
- **No allocation on improvement**: the best-so-far state is kept in a
  preallocated buffer (plain memcpy on improvement), converting to the
  int8 result format once at the end.

Not adopted: -O3 and -march=native both measured SLOWER than -O2 on
this loop (about -15 percent); the frozen-phase ceiling probe showed the
loop is latency-bound, not throughput-bound, so wider vectorization has
nothing to feed on.

## Acceptance

Final config vs the verbatim Milestone 2 baseline on the same schedule:
**10.3x in the run above; 9.6x-10.6x (median 10.2x) across repeated
pinned runs.** The >= 10x criterion is met at the median but is within
this machine's noise band; the per-step contributions above are stable
across runs even when the absolute numbers shift.
