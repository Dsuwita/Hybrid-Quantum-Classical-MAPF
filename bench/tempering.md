# Parallel tempering vs independent restarts (Milestone 7a)

Parallel tempering (Hukushima & Nemoto 1996) runs R replicas at fixed
temperatures on a hot-to-cold ladder and periodically swaps adjacent
replicas, so a configuration trapped in a poor low-temperature basin can
ride up to a hot replica, escape, and come back. It is compared here
against equal-budget independent restarts: the same R replicas and the same
per-replica sweep count S, but cooled independently and reduced to their
best.

## Setup

- Instance: Gset G39 (n=2000, m=11778, +/-1 edge weights; a hard,
  frustrated Max-Cut instance). Best-known cut 2408.
- Budget: R=24 replicas x S=3000 sweeps for both methods (same total work).
- Tempering ladder: geometric, 24 rungs from T=0.15 to T=8.0, swap attempts
  every sweep. Restart schedule: geometric cool from T=4.0 to ~0.05.
- 6 seeds; mean and best cut reported. Reproduce with
  `./build/bench_tempering data/gset/G39 --rungs 24 --sweeps 3000 --seeds 6`.

## Result (representative run)

| method | mean cut | best cut | total ms |
|---|---|---|---|
| parallel tempering | 2383.8 | 2395 | 30820 |
| independent restarts | 2398.0 | 2401 | 13178 |

## Reading it honestly

At this budget independent restarts are the stronger baseline: they reach a
slightly higher cut (2401 vs 2395 best; 2398 vs 2384 mean) in less than half
the wall time. Two things drive that:

1. **Quality.** Parallel tempering pays off mainly on very rugged spin-glass
   landscapes and with careful ladder tuning and long runs, where a single
   cooled chain reliably gets stuck. At a few thousand sweeps on G39, a
   pool of 24 independent cooled restarts already samples enough basins that
   the exchange machinery does not add much; both land within ~0.5% of the
   best-known cut.
2. **Wall time.** The restart path runs the fully optimized `FastAnnealer`
   (flat storage, field cache, downhill early-exit, acceptance lookup
   table). The tempering loop here is a straightforward field-cached
   Metropolis without those inner-loop optimizations, and it additionally
   does R-1 swap tests per interval, so it is about 2x slower per sweep.

So on standard Gset instances at a practical budget, independent restarts
win. Parallel tempering is retained as a correct, tested building block
(20/20 exact ground states on brute-forceable instances, energy audited);
its advantage would show on harder, larger spin glasses with tuned ladders
and much longer runs, which is beyond this project's benchmark budget.
