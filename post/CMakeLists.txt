cmake_minimum_required(VERSION 3.16)
project(cpp-blog LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Fetch Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

enable_testing()

# =============================================================================
# Peasant: One Algorithm, Infinite Powers
# =============================================================================
add_subdirectory(2019-03-peasant-stepanov/tests)

# =============================================================================
# Miller-Rabin: Is It Prime?
# =============================================================================
add_executable(test_primality 2019-09-miller-rabin-stepanov/test_primality.cpp)
target_link_libraries(test_primality GTest::gtest_main)
target_include_directories(test_primality PRIVATE 2019-09-miller-rabin-stepanov)
add_test(NAME test_primality COMMAND test_primality)

# =============================================================================
# Rational: Exact Arithmetic
# =============================================================================
add_executable(test_rational 2020-02-rational-stepanov/test_rational.cpp)
target_link_libraries(test_rational GTest::gtest_main)
target_include_directories(test_rational PRIVATE 2020-02-rational-stepanov)
add_test(NAME test_rational COMMAND test_rational)

# =============================================================================
# Type Erasure: Runtime Polymorphism Without Inheritance
# =============================================================================
add_executable(test_type_erasure 2024-02-type-erasure-stepanov/test_type_erasure.cpp)
target_link_libraries(test_type_erasure GTest::gtest_main)
target_include_directories(test_type_erasure PRIVATE 2024-02-type-erasure-stepanov)
add_test(NAME test_type_erasure COMMAND test_type_erasure)

# =============================================================================
# Modular: Finite Rings
# =============================================================================
add_executable(test_mod 2019-06-modular-stepanov/test_mod.cpp)
target_link_libraries(test_mod GTest::gtest_main)
target_include_directories(test_mod PRIVATE 2019-06-modular-stepanov)
add_test(NAME test_mod COMMAND test_mod)

# =============================================================================
# Elementa: Pedagogical Linear Algebra
# =============================================================================
add_executable(test_elementa 2021-03-elementa-stepanov/test_elementa.cpp)
target_link_libraries(test_elementa GTest::gtest_main)
target_include_directories(test_elementa PRIVATE 2021-03-elementa-stepanov)
add_test(NAME test_elementa COMMAND test_elementa)

# =============================================================================
# Autodiff: Reverse-Mode Automatic Differentiation
# (Depends on elementa)
# =============================================================================
add_executable(test_autodiff 2023-01-autodiff-stepanov/test_gradator.cpp)
target_link_libraries(test_autodiff GTest::gtest_main)
target_include_directories(test_autodiff PRIVATE 2023-01-autodiff-stepanov 2021-03-elementa-stepanov)
add_test(NAME test_autodiff COMMAND test_autodiff)

# =============================================================================
# Dual: Forward-Mode Automatic Differentiation via Dual Numbers
# =============================================================================
add_executable(test_dual
    2021-09-dual-stepanov/test_dual.cpp
    2021-09-dual-stepanov/test_functions.cpp
    2021-09-dual-stepanov/test_gradient.cpp
    2021-09-dual-stepanov/test_higher_order.cpp
    2021-09-dual-stepanov/test_numerical.cpp
)
target_link_libraries(test_dual GTest::gtest_main)
target_include_directories(test_dual PRIVATE 2021-09-dual-stepanov)
add_test(NAME test_dual COMMAND test_dual)

# =============================================================================
# Polynomials: Polynomial Arithmetic and Algebra
# =============================================================================
add_executable(test_polynomials
    2020-07-polynomials-stepanov/test_arithmetic.cpp
    2020-07-polynomials-stepanov/test_evaluation.cpp
    2020-07-polynomials-stepanov/test_gcd.cpp
)
target_link_libraries(test_polynomials GTest::gtest_main)
target_include_directories(test_polynomials PRIVATE 2020-07-polynomials-stepanov)
add_test(NAME test_polynomials COMMAND test_polynomials)

# =============================================================================
# Finite Differences: Numerical Differentiation
# =============================================================================
add_executable(test_finite_diff 2022-04-finite-diff-stepanov/test_finite_diff.cpp)
target_link_libraries(test_finite_diff GTest::gtest_main)
target_include_directories(test_finite_diff PRIVATE 2022-04-finite-diff-stepanov)
add_test(NAME test_finite_diff COMMAND test_finite_diff)

# =============================================================================
# Integration: Numerical Quadrature
# (Now generic - uses ordered_field concept, supports dual<T> for autodiff)
# =============================================================================
add_executable(test_integrate 2023-08-integration-stepanov/test_integrate.cpp)
target_link_libraries(test_integrate GTest::gtest_main)
target_include_directories(test_integrate PRIVATE 2023-08-integration-stepanov 2021-09-dual-stepanov)
add_test(NAME test_integrate COMMAND test_integrate)
