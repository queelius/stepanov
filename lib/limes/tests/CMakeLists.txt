# Find Google Test
find_package(GTest REQUIRED)

# Enable testing
enable_testing()
include(GoogleTest)

# Coverage setup for GCC/Clang
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Coverage enabled for tests")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -fprofile-arcs -ftest-coverage")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
endif()

# Keep basic tests for simple validation
add_executable(basic_tests basic_tests.cpp)
target_link_libraries(basic_tests PRIVATE limes)
add_test(NAME BasicTests COMMAND basic_tests)
set_tests_properties(BasicTests PROPERTIES
    TIMEOUT 30
    LABELS "unit"
)

# Helper function to reduce repetition for GTest targets
function(limes_add_test name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE limes GTest::gtest GTest::gtest_main pthread)
    gtest_discover_tests(${name})
endfunction()

# Google Test-based test executables
limes_add_test(test_accumulators test_accumulators.cpp)
limes_add_test(test_quadrature   test_quadrature.cpp)
limes_add_test(test_integrators  test_integrators.cpp)
limes_add_test(test_result       test_result.cpp)
limes_add_test(test_concepts     test_concepts.cpp)
limes_add_test(test_expr         test_expr.cpp)
limes_add_test(test_expr_domain  test_expr_domain.cpp)

# Add custom target for running tests with coverage
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_custom_target(coverage
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/coverage
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/coverage
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        COMMAND lcov --capture --directory ${CMAKE_BINARY_DIR} --output-file ${CMAKE_BINARY_DIR}/coverage/coverage.info --no-external
        COMMAND lcov --remove ${CMAKE_BINARY_DIR}/coverage/coverage.info '*/tests/*' '*/build/*' '/usr/*' --output-file ${CMAKE_BINARY_DIR}/coverage/coverage.info
        COMMAND genhtml ${CMAKE_BINARY_DIR}/coverage/coverage.info --output-directory ${CMAKE_BINARY_DIR}/coverage/html
        COMMAND ${CMAKE_COMMAND} -E echo "Coverage report generated in ${CMAKE_BINARY_DIR}/coverage/html/index.html"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        DEPENDS test_accumulators test_quadrature test_integrators test_result test_concepts test_expr test_expr_domain
    )
endif()
