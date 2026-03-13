include(FetchContent)

# Fetch Google Test
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Test executables
add_executable(test_arithmetic test_arithmetic.cpp)
target_link_libraries(test_arithmetic PRIVATE poly::polynomials GTest::gtest_main)

add_executable(test_gcd test_gcd.cpp)
target_link_libraries(test_gcd PRIVATE poly::polynomials GTest::gtest_main)

add_executable(test_evaluation test_evaluation.cpp)
target_link_libraries(test_evaluation PRIVATE poly::polynomials GTest::gtest_main)

# Enable compiler warnings
foreach(target test_arithmetic test_gcd test_evaluation)
    target_compile_options(${target} PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
    )
endforeach()

# Register tests
include(GoogleTest)
gtest_discover_tests(test_arithmetic)
gtest_discover_tests(test_gcd)
gtest_discover_tests(test_evaluation)
