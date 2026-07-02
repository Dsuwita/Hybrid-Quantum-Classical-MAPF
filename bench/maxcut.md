# Max-Cut on Gset benchmarks

The annealer solved as a Max-Cut solver on standard Gset instances. Each
graph is mapped to an Ising model (edge (u,v,w) -> interaction J_uv = +w,
no fields); the cut value is recounted independently from the edge list,
never trusted from the energy. Best-known values are the published bests
for these instances (Benlic & Hao, "Breakout Local Search for the Max-Cut
problem", Engineering Applications of Artificial Intelligence 26 (2013),
1162-1173; consistent with the standard Gset tables).

Budget: 16 parallel replicas x 20000 sweeps, seed 1, geometric schedule
T0 = 3, alpha = 0.9995. Machine: AMD Ryzen 5 7600, WSL2. Reproduce with
`./bench/run_maxcut.sh` (downloads the instances first if needed).

| instance | nodes | edges | best-known | our cut | percent | wall |
|---|---|---|---|---|---|---|
| G1  |  800 | 19176 | 11624 | 11624 | 100.00% | 0.5 s |
| G22 | 2000 | 19990 | 13359 | 13358 |  99.99% | 1.2 s |
| G39 | 2000 | 11778 |  2408 |  2399 |  99.63% | 1.7 s |
| G55 | 5000 | 12498 | 10294 | 10291 |  99.97% | 3.2 s |

All four are at or above 99.6% of best-known, comfortably clearing the
98% target, on four instances spanning 800 to 5000 nodes (G39 additionally
has signed +/-1 edge weights). Larger sweep budgets close the small
remaining gaps; these are the numbers from the committed fixed budget.

Correctness is pinned separately by tests/test_problems.cpp: on 20 random
brute-forceable graphs the annealer's cut equals the exact maximum cut,
the independent verifier catches a corrupted state, and the Gset parser
round-trips edge count and total weight.
