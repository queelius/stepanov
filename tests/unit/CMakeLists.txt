cmake_minimum_required(VERSION 3.14)

# Helper function to add a unit test
function(add_unit_test TEST_NAME)
    add_executable(${TEST_NAME} ${TEST_NAME}.cpp)
    target_link_libraries(${TEST_NAME}
        PRIVATE
        stepanov
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
    )
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})

    # Add to coverage if enabled
    if(STEPANOV_ENABLE_COVERAGE)
        target_compile_options(${TEST_NAME} PRIVATE --coverage -O0 -g)
        target_link_options(${TEST_NAME} PRIVATE --coverage)
    endif()
endfunction()

# Core mathematical operations tests
add_unit_test(test_math)
add_unit_test(test_concepts)
add_unit_test(test_builtin_adaptors)

# Number theory tests
add_unit_test(test_gcd)
add_unit_test(test_mod)
add_unit_test(test_expmod)
add_unit_test(test_primality)

# Data structure tests
add_unit_test(test_polynomial)
add_unit_test(test_bounded_nat)
add_unit_test(test_interval)
add_unit_test(test_continued_fraction)
add_unit_test(test_fenwick_tree)
add_unit_test(test_sparse_matrix)

# Advanced algorithm tests
add_unit_test(test_fft)
add_unit_test(test_padic)
add_unit_test(test_cache_oblivious)

# Group theory tests
add_unit_test(test_groups)

# Graph algorithm tests
add_unit_test(test_graph)

# Matrix tests
add_unit_test(test_matrix_expressions)
add_unit_test(test_matrix_algorithms)

# Type system tests
add_unit_test(test_type_erasure)

# Tool tests
add_unit_test(test_math_tools)