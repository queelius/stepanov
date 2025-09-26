cmake_minimum_required(VERSION 3.14)

# Optimization tests
set(OPTIMIZATION_TEST_SOURCES
    test_all.cpp
    test_golden_section.cpp
    test_gradient_descent.cpp
    test_integration.cpp
    test_newton.cpp
    test_root_finding.cpp
    test_simulated_annealing.cpp
)

foreach(TEST_SOURCE ${OPTIMIZATION_TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    set(TEST_NAME "opt_${TEST_NAME}")
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} stepanov GTest::gtest_main)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    set_tests_properties(${TEST_NAME} PROPERTIES
        TIMEOUT 30
        LABELS "optimization"
    )
endforeach()