# dwave-neal comparison (Milestone 6)

Both solvers run 16 restarts of 2000 single-spin-flip sweeps on the
same Gset instance. `neal` is D-Wave's SimulatedAnnealingSampler
(`pip install dwave-neal`). Cut percentages are of the published best-known
value. Reproduce with `python3 bench/neal_compare.py`.

| instance | n | best-known | ours cut | ours % | ours ms | neal cut | neal % | neal ms |
|---|---|---|---|---|---|---|---|---|
| G1 | 800 | 11624 | 11624 | 100.00% | 34.3 | 11624 | 100.00% | 786.1 |
| G22 | 2000 | 13359 | 13348 | 99.92% | 81.8 | 13358 | 99.99% | 1382.4 |
| G39 | 2000 | 2408 | 2357 | 97.88% | 102.2 | 2386 | 99.09% | 1261.1 |
| G55 | 5000 | 10294 | 10056 | 97.69% | 281.2 | 10270 | 99.77% | 2532.6 |
