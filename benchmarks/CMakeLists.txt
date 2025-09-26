cmake_minimum_required(VERSION 3.14)
project(stepanov_benchmarks)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the include directory
include_directories(${CMAKE_SOURCE_DIR}/../include)

# Enable optimizations for Release builds
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

# Find Google Benchmark (optional but recommended)
find_package(benchmark QUIET)

# Find Eigen for comparisons (optional)
find_package(Eigen3 QUIET)

# Core benchmarks that should compile
add_executable(bench_fundamentals bench_fundamentals.cpp)
add_executable(bench_realistic bench_realistic.cpp)

# If Google Benchmark is found, use it
if(benchmark_FOUND)
    message(STATUS "Google Benchmark found - building advanced benchmarks")
    add_executable(bench_google bench_google.cpp)
    target_link_libraries(bench_google benchmark::benchmark)
endif()

# If Eigen is found, build comparison benchmarks
if(Eigen3_FOUND)
    message(STATUS "Eigen3 found - building comparison benchmarks")
    add_executable(bench_eigen_comparison bench_eigen_comparison.cpp)
    target_link_libraries(bench_eigen_comparison Eigen3::Eigen)
endif()

# Add custom targets for easy running
add_custom_target(run_benchmarks
    COMMAND bench_fundamentals
    COMMAND bench_realistic
    DEPENDS bench_fundamentals bench_realistic
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all benchmarks"
)

# Add target for generating reports
add_custom_target(generate_report
    COMMAND ${CMAKE_COMMAND} -E echo "Generating benchmark report..."
    COMMAND bench_fundamentals > fundamentals_report.txt 2>&1
    COMMAND bench_realistic > realistic_report.txt 2>&1
    DEPENDS bench_fundamentals bench_realistic
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generating benchmark reports"
)