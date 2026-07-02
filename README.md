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
