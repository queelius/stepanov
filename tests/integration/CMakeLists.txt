cmake_minimum_required(VERSION 3.14)

# Helper function to add an integration test
function(add_integration_test TEST_NAME)
    add_executable(${TEST_NAME} ${TEST_NAME}.cpp)
    target_link_libraries(${TEST_NAME}
        PRIVATE
        generic_math
        GTest::gtest
        GTest::gtest_main
    )
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})

    if(GENERIC_MATH_ENABLE_COVERAGE)
        target_compile_options(${TEST_NAME} PRIVATE --coverage -O0 -g)
        target_link_options(${TEST_NAME} PRIVATE --coverage)
    endif()
endfunction()

# Integration tests
add_integration_test(test_algebraic_structures)
add_integration_test(test_numerical_algorithms)
add_integration_test(test_polynomial_systems)
add_integration_test(test_template_instantiation)