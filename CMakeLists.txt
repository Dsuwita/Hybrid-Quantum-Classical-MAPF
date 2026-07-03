# anneal: header-only annealing library + test/bench executables.
cmake_minimum_required(VERSION 3.16)
project(anneal CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# The tests are assert-based, and the default Release flags define NDEBUG
# which would compile every assert away, turning the test suite into a
# no-op that always passes. Strip -DNDEBUG so asserts stay live even in
# optimized builds.
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

add_compile_options(-Wall -Wextra)

add_library(anneal INTERFACE)
target_include_directories(anneal INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# std::thread needs the platform thread library on older glibc/other libcs
# (glibc >= 2.34 folds pthreads into libc, but don't rely on that).
find_package(Threads REQUIRED)
target_link_libraries(anneal INTERFACE Threads::Threads)

enable_testing()

add_executable(test_bqm tests/test_bqm.cpp)
target_link_libraries(test_bqm PRIVATE anneal)
add_test(NAME bqm COMMAND test_bqm)

add_executable(test_annealer tests/test_annealer.cpp)
target_link_libraries(test_annealer PRIVATE anneal)
add_test(NAME annealer COMMAND test_annealer)

add_executable(test_fast_annealer tests/test_fast_annealer.cpp)
target_link_libraries(test_fast_annealer PRIVATE anneal)
add_test(NAME fast_annealer COMMAND test_fast_annealer)

add_executable(test_parallel tests/test_parallel.cpp)
target_link_libraries(test_parallel PRIVATE anneal)
add_test(NAME parallel COMMAND test_parallel)

# MAPF (project 2): header-only, lives under mapf/, includes as "mapf/...".
add_library(mapf INTERFACE)
target_include_directories(mapf INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_mapf_io tests/test_mapf_io.cpp)
target_link_libraries(test_mapf_io PRIVATE mapf)
target_compile_definitions(test_mapf_io PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
add_test(NAME mapf_io COMMAND test_mapf_io)

add_executable(test_verifier tests/test_verifier.cpp)
target_link_libraries(test_verifier PRIVATE mapf)
add_test(NAME verifier COMMAND test_verifier)

add_executable(test_path_gen tests/test_path_gen.cpp)
target_link_libraries(test_path_gen PRIVATE mapf)
add_test(NAME path_gen COMMAND test_path_gen)

# conflict_qubo consumes the anneal library (brute force + annealer), so
# this test links both.
add_executable(test_conflict_qubo tests/test_conflict_qubo.cpp)
target_link_libraries(test_conflict_qubo PRIVATE mapf anneal)
add_test(NAME conflict_qubo COMMAND test_conflict_qubo)

add_executable(test_solver tests/test_solver.cpp)
target_link_libraries(test_solver PRIVATE mapf anneal)
add_test(NAME solver COMMAND test_solver)

add_executable(test_rolling tests/test_rolling.cpp)
target_link_libraries(test_rolling PRIVATE mapf anneal)
add_test(NAME rolling COMMAND test_rolling)

add_executable(test_obstacles tests/test_obstacles.cpp)
target_link_libraries(test_obstacles PRIVATE mapf anneal)
add_test(NAME obstacles COMMAND test_obstacles)

add_executable(mapf_penalty_experiment src/mapf_penalty_experiment.cpp)
target_link_libraries(mapf_penalty_experiment PRIVATE mapf anneal)

add_executable(solve_mapf src/solve_mapf.cpp)
target_link_libraries(solve_mapf PRIVATE mapf anneal)

add_executable(bench_mapf src/bench_mapf.cpp)
target_link_libraries(bench_mapf PRIVATE mapf anneal)

add_executable(demo_rolling src/demo_rolling.cpp)
target_link_libraries(demo_rolling PRIVATE mapf anneal)

add_executable(demo_obstacles src/demo_obstacles.cpp)
target_link_libraries(demo_obstacles PRIVATE mapf anneal)

add_executable(test_problems tests/test_problems.cpp)
target_link_libraries(test_problems PRIVATE anneal)
add_test(NAME problems COMMAND test_problems)

add_executable(demo_partition src/demo_partition.cpp)
target_link_libraries(demo_partition PRIVATE anneal)

add_executable(bench_throughput src/bench_throughput.cpp)
target_link_libraries(bench_throughput PRIVATE anneal)

add_executable(bench_scaling src/bench_scaling.cpp)
target_link_libraries(bench_scaling PRIVATE anneal)

add_executable(bench_rolling src/bench_rolling.cpp)
target_link_libraries(bench_rolling PRIVATE mapf anneal)

add_executable(solve_maxcut src/solve_maxcut.cpp)
target_link_libraries(solve_maxcut PRIVATE anneal)

add_executable(compare_maxcut src/compare_maxcut.cpp)
target_link_libraries(compare_maxcut PRIVATE anneal)
