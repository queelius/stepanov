cmake_minimum_required(VERSION 3.14)

# Check for Google Benchmark
find_package(benchmark QUIET)

if(benchmark_FOUND)
    # Helper function to add a benchmark
    function(add_benchmark_test BENCH_NAME)
        add_executable(${BENCH_NAME} ${BENCH_NAME}.cpp)
        target_link_libraries(${BENCH_NAME}
            PRIVATE
            generic_math
            benchmark::benchmark
        )
        # Benchmarks are not added as tests
    endfunction()

    add_benchmark_test(bench_math_operations)
    add_benchmark_test(bench_gcd_algorithms)
    add_benchmark_test(bench_polynomial)
    add_benchmark_test(bench_fft)
else()
    message(STATUS "Google Benchmark not found. Benchmarks will use custom timing.")

    # Simple benchmark without Google Benchmark
    add_executable(bench_simple bench_simple.cpp)
    target_link_libraries(bench_simple PRIVATE generic_math)
endif()