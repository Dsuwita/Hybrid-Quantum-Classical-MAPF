# Description

A multithreaded simulated annealing library in C++20 for hard combinatorial optimization problems, using the Ising/QUBO model format of quantum annealers. Includes a planned downstream application: a hybrid Multi-Agent Path Finding solver built on the library.
 
Zero dependencies. Header-only core. Work in progress, built milestone by milestone (see PROJECT_SPEC.md for the full roadmap).
 
## Status
 
Done so far: the BQM problem container with exact Ising/QUBO conversion, exhaustively tested by enumerating all 2^n states of small random instances.
 
Next up: the annealing engine, single-thread performance work, parallel restarts, and Max-Cut on standard Gset benchmark instances.
 
## Quickstart
 
Requires a C++20 compiler, nothing else.
 
```
g++ -std=c++20 -Wall -Wextra -O2 -I include tests/test_bqm.cpp -o test_bqm
./test_bqm
```
 
## Example
 
```cpp
#include "anneal/bqm.hpp"
using namespace anneal;
 
// Max-Cut on a triangle: every edge wants its endpoints in different groups.
BQM bqm(3, Vartype::Spin);
bqm.add_interaction(0, 1, 1.0);
bqm.add_interaction(1, 2, 1.0);
bqm.add_interaction(0, 2, 1.0);
 
std::vector<std::int8_t> state = {+1, -1, +1};
double e = bqm.energy(state);  // -1.0, the ground state: 2 of 3 edges cut
```

## Max-Cut on Gset benchmarks

The annealer's quality is measured on the standard Gset Max-Cut instances.
Each graph maps to an Ising model (edge (u,v,w) becomes interaction
J_uv = +w); the cut is recounted independently from the edge list, never
trusted from the energy. Best-known values are the published bests
(Benlic & Hao, Engineering Applications of AI, 2013).

| instance | nodes | best-known | our cut | percent | wall |
|---|---|---|---|---|---|
| G1  |  800 | 11624 | 11624 | 100.00% | 0.5 s |
| G22 | 2000 | 13359 | 13358 |  99.99% | 1.2 s |
| G39 | 2000 |  2408 |  2399 |  99.63% | 1.7 s |
| G55 | 5000 | 10294 | 10291 |  99.97% | 3.2 s |

16 replicas x 20000 sweeps, seed 1, on a Ryzen 5 7600. Reproduce with
`./data/download_gset.sh` then `./bench/run_maxcut.sh`; details and the
brute-force correctness tests are in `bench/maxcut.md`.

```
./build/solve_maxcut data/gset/G1 --best 11624 --replicas 16 --sweeps 20000
```

## Multi-Agent Path Finding (Project 2)

The downstream application: route many agents across a grid to their
goals without collisions, using the annealer as the combinatorial core.
The approach is a hybrid decomposition (inspired by Gerlach et al., ICML
2025). Classical A* proposes a menu of candidate paths per agent; a QUBO
selects one path per agent, penalizing pairwise conflicts and enforcing
one path per agent; the annealer solves it; the plan is verified, and any
remaining conflicts trigger reservation-guided replanning and another
anneal.

### Interactive GUI

`mapf/viz/serve.py` is a small local web app (standard-library HTTP
server, no dependencies) for building and solving MAPF instances by hand:

```
cmake -S . -B build && cmake --build build   # so the server can call solve_mapf
python3 mapf/viz/serve.py                     # opens http://localhost:8000
```

In the browser you can load a bundled map or a demo scenario, or draw
your own: click to place agent start/goal pairs, toggle obstacles, and
set the annealer budget. Press Solve and the server runs the compiled
`solve_mapf` on your instance and streams the plan back; the canvas then
animates it with play, pause, step, scrub, speed, and loop controls, and
shows sum-of-costs, overhead, conflicts, and wall time. A Download GIF
button exports the current plan (that step needs matplotlib on the
server; everything else is dependency-free).

### Static GIFs

For a fixed result, render a plan to an animated GIF directly:

```
./build/solve_mapf mapf/viz/demo_crossing.map mapf/viz/demo_crossing.scen 6 --out plan.txt
python3 mapf/viz/render_plan.py plan.txt --out plan.gif
```

The two demos below are six agents crossing an open grid (head-on and
diagonal) and five agents moving between four rooms through single-cell
doorways. Colored dots are agents; the matching X marks each agent's
goal.

![agents crossing an open grid](mapf/viz/demo_crossing.gif)
![agents moving between rooms](mapf/viz/demo_rooms.gif)

Regenerate the demo plans and GIFs with `./mapf/viz/make_demos.sh` (needs
the project built and Python with matplotlib).

## Real-time replanning (Project 3)

The static solver plans every path once up front. Real deployments cannot:
agents keep getting new tasks and the world keeps changing. The
rolling-horizon driver (`mapf/rolling.hpp`, RHCR-style after Li et al.,
AAAI 2021) replans continuously under a global clock. Each cycle it plans
a window of W timesteps from the agents' current positions, resolves
conflicts only within that window (a small QUBO), commits the first E < W
steps, and looks again. The annealer runs under a per-cycle wall-clock
deadline and returns its best-so-far when time is up (anytime use). In
lifelong mode an agent that reaches its goal is immediately handed a new
one, so the agents never stop.

The committed history is always collision-free (each cycle commits only
the longest conflict-free prefix it verified, and falls back to holding
position if even the first step would collide); throughput is what
degrades under congestion, and it is reported honestly. This GIF is eight
agents running lifelong on a 16x16 grid, produced by `demo_rolling`:

```
./build/demo_rolling mapf/bench/maps/empty16.map 8 --steps 80 --out plan.txt
python3 mapf/viz/render_plan.py plan.txt --out lifelong.gif
```

![eight agents replanning continuously in lifelong mode](mapf/viz/demo_lifelong.gif)

### Moving obstacles

The last piece is other things that move on their own. This is the
robot-soccer setting that motivates the whole project: a robot has to get
to a spot on the field while teammates, opponents, and the ball move
independently. Obstacles carry a predicted occupancy over the planning
window (exact for scripted movers, or a ball around the current cell that
grows with look-ahead when motion is uncertain), and those predicted
cells are forbidden in candidate generation, so the time-expanded A*
either routes around a mover or waits for it to pass. If a prediction is
violated at execution time, the affected agent brakes (holds) for that
step. `demo_obstacles` sends three agents across a field past two
patrolling obstacles; with perfect prediction and room to move they reach
their goals with zero collisions of either kind:

![agents dodging two patrolling obstacles](mapf/viz/demo_obstacles.gif)

Replan cycle time and throughput as the field fills up are in
`mapf/bench/rolling.md` (about 5 ms per cycle at 2 agents, 40 ms at 32).
Agent-agent safety is unconditional; agent-obstacle avoidance is
guaranteed with perfect prediction as long as agents are free to move, and
degrades gracefully (reported, not hidden) under heavy congestion.

Solve a scenario yourself:

```
cmake -S . -B build && cmake --build build
./build/solve_mapf mapf/viz/demo_crossing.map mapf/viz/demo_crossing.scen 6 --out plan.txt
```

The solver prints success, sum-of-costs, overhead versus the sum of
per-agent shortest paths, and wall time. Every reported result is
re-checked by an independent verifier. Success rate versus number of
agents on three maps is in `mapf/bench/results.md`.
