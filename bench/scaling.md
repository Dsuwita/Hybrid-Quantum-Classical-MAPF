# Milestone 4 scaling results

Parallel restarts: 24 replicas x 10000 sweeps on the n=2000 3-regular
+/-1 benchmark graph, xoshiro256++, geometric schedule T0=3.0
alpha=0.999. Wall time is the best of 3 runs per thread count.
Machine: AMD Ryzen 5 7600 (6 physical cores, 12 SMT threads), WSL2.

Reproduce: `./build/bench_scaling > bench/scaling.csv` then
`python3 bench/plot_scaling.py`.

| threads | wall ms | speedup | efficiency | best energy |
|---|---|---|---|---|
| 1  | 2692.0 | 1.00x | 100%  | -2520 |
| 2  | 1485.7 | 1.81x | 90.6% | -2520 |
| 4  |  834.2 | 3.23x | 80.7% | -2520 |
| 6  |  651.9 | 4.13x | 68.8% | -2520 |
| 8  |  603.4 | 4.46x | 55.8% | -2520 |
| 12 |  479.2 | 5.62x | 46.8% | -2520 |

Best energy is identical in every row, as required: thread count only
changes which worker runs which replica, never any replica's result
(tests/test_parallel.cpp asserts this per replica against a sequential
reference).

## Where the curve bends, and why

Two bends, both explained by the hardware rather than the code:

1. **1 to 6 threads (physical cores): frequency, not contention.** The
   speedup at 6 threads is 4.13x rather than 6x. To separate our
   threading from the machine, a control experiment ran the same
   single-replica workload as 6 fully independent PROCESSES (no shared
   process state at all): each took 73-82 ms vs 47.6 ms alone, i.e.
   independent processes lose slightly MORE than our 6 worker threads
   do. The loss is therefore the CPU dropping from single-core boost to
   all-core clocks (plus WSL2 host noise), not lock contention, false
   sharing, or memory bandwidth in the annealer. The working set per
   replica is about 50 KB, so DRAM bandwidth is not a factor at this
   problem size.
2. **6 to 12 threads: SMT.** Beyond 6 threads the two hardware threads
   of each core share one out-of-order backend. The annealer's hot loop
   is latency-bound (serial RNG chain, dependent loads), which is the
   favorable case for SMT, and it indeed keeps scaling (4.13x -> 5.62x)
   but at roughly half the per-thread efficiency.
