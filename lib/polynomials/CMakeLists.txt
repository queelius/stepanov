add_library(polynomials INTERFACE)
target_include_directories(polynomials INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if(STEPANOV_BUILD_TESTS)
    add_executable(test_poly_arithmetic tests/test_arithmetic.cpp)
    target_link_libraries(test_poly_arithmetic PRIVATE polynomials GTest::gtest_main)
    gtest_discover_tests(test_poly_arithmetic)

    add_executable(test_poly_gcd tests/test_gcd.cpp)
    target_link_libraries(test_poly_gcd PRIVATE polynomials GTest::gtest_main)
    gtest_discover_tests(test_poly_gcd)

    add_executable(test_poly_evaluation tests/test_evaluation.cpp)
    target_link_libraries(test_poly_evaluation PRIVATE polynomials GTest::gtest_main)
    gtest_discover_tests(test_poly_evaluation)
endif()

if(STEPANOV_BUILD_EXAMPLES)
    add_executable(poly_basic examples/basic.cpp)
    target_link_libraries(poly_basic PRIVATE polynomials)

    add_executable(poly_euclidean_gcd examples/euclidean_gcd.cpp)
    target_link_libraries(poly_euclidean_gcd PRIVATE polynomials)
endif()
