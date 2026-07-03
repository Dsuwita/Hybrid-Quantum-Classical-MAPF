# Project status

| Milestone | Description | Status |
|---|---|---|
| 1-5 | BQM, annealer, fast annealer, parallel, Max-Cut | Done |
| 6 | dwave-neal comparison + portfolio README | Done |
| 7 | Stretch: parallel tempering AND multi-spin coding | Done (both) |
| 8-12 | Static MAPF + basic GUI | Done |
| 13-14 | Rolling horizon + moving obstacles | Done |
| 15 | Interactive studio | Done |

## Milestone 15 classical-solver decision

The studio compares the hybrid annealer against a classical MAPF solver.
Decision (2026-07-02): **Conflict-Based Search (CBS)**, implemented in
`mapf/cbs.hpp`. It is optimal for sum-of-costs, so "overhead vs CBS optimal"
is a meaningful quality metric, and it is a recognized algorithm that adds
independent value. Implemented from scratch (Sharon et al., "Conflict-based
search for optimal multi-agent pathfinding", AIJ 2015).
