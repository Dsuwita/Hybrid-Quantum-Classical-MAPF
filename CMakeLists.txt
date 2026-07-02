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

add_executable(demo_partition src/demo_partition.cpp)
target_link_libraries(demo_partition PRIVATE anneal)

add_executable(bench_throughput src/bench_throughput.cpp)
target_link_libraries(bench_throughput PRIVATE anneal)
