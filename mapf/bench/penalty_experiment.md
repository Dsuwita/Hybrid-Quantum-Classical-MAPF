# Milestone 10 penalty dynamic-range experiment

Instance: 4 agents x 3 candidates (12 binary vars) on a 7x7 open grid.
Max single-path cost = 12. Optimal feasible selection cost = 38.
Annealer: geometric T0=5 alpha=0.99, 300 sweeps, 200 seeds.

| penalty P=P1 | scale (xMaxCost) | exact GS feasible+optimal | anneal success |
|---|---|---|---|
| 6 | 0.5 | NO | 0.0% |
| 12 | 1.0 | NO | 17.0% |
| 24 | 2.0 | yes | 91.0% |
| 60 | 5.0 | yes | 78.5% |
| 240 | 20.0 | yes | 78.5% |
| 1200 | 100.0 | yes | 78.5% |
| 12000 | 1000.0 | yes | 78.5% |

## Interpretation

Three regimes, exactly as the theory predicts:

- **Too small (scale 0.5-1.0): the formulation is wrong.** The exact
  ground state is no longer the optimal feasible selection ("exact GS
  feasible+optimal = NO"): the penalty is small enough that dropping an
  agent or accepting a collision saves more travel cost than it costs, so
  the QUBO's true minimum is infeasible. No annealer can fix a QUBO whose
  own ground state is the wrong answer. This is why the safe default
  penalty is (sum of per-agent max costs) + 1, comfortably above this
  region.

- **Just right (scale 2.0): best annealing.** Penalties are large enough
  to make feasible selections win but small enough to keep the energy
  landscape gentle. Highest success rate (91%).

- **Too large (scale >= 5): degraded, then saturated.** Success drops to a
  78.5% plateau. The walls between feasible configurations grow so tall
  that, relative to the schedule's temperatures, the annealer can almost
  never climb them, so it gets stuck more often. The plateau (identical
  across scales 5x-1000x) is a property of this annealer specifically:
  it detects the integer couplings and uses an acceptance lookup table,
  and once a penalty exceeds what any temperature in the schedule can
  ever accept, exp(-penalty/T) has already underflowed to zero in the
  table, so raising the penalty further changes nothing. A float-RNG
  annealer without the table would keep degrading rather than flatten.

Practical rule (used as the default in conflict_qubo.hpp): pick the
smallest penalty that still makes every feasible selection beat every
infeasible one, not the largest number you can write down.
