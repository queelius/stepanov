cmake_minimum_required(VERSION 3.14)

# Unit tests subdirectory
add_subdirectory(unit)

# Integration tests subdirectory
add_subdirectory(integration)

# Benchmarks subdirectory
if(STEPANOV_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Helper function to add a test executable
function(add_stepanov_test TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} stepanov GTest::gtest_main)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    set_tests_properties(${TEST_NAME} PROPERTIES
        TIMEOUT 30
        LABELS "stepanov"
    )
endfunction()

# Add tests from the tests/ directory
# These are the test files we moved from root
file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp")

foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    if(EXISTS ${TEST_SOURCE})
        add_stepanov_test(${TEST_NAME} ${TEST_SOURCE})
    endif()
endforeach()

# Add optimization tests subdirectory
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/optimization)
    add_subdirectory(optimization)
endif()

# Add a custom target to run all tests with verbose output
add_custom_target(test_verbose
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose --output-on-failure
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all tests with verbose output"
)

# Add a custom target for coverage with tests
if(STEPANOV_ENABLE_COVERAGE)
    add_custom_target(test_coverage
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target coverage
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running tests and generating coverage report"
    )
endif()