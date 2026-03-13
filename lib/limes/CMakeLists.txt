add_library(limes INTERFACE)
add_library(limes::limes ALIAS limes)

target_include_directories(limes INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# OpenMP is optional
if(STEPANOV_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(limes INTERFACE OpenMP::OpenMP_CXX)
    endif()
endif()

find_package(Threads REQUIRED)
target_link_libraries(limes INTERFACE Threads::Threads)

if(STEPANOV_BUILD_TESTS)
    # basic_tests doesn't use GTest
    add_executable(limes_basic_tests tests/basic_tests.cpp)
    target_link_libraries(limes_basic_tests PRIVATE limes)
    add_test(NAME LimesBasicTests COMMAND limes_basic_tests)

    # GTest-based tests
    foreach(test accumulators quadrature integrators result concepts expr expr_domain)
        add_executable(limes_test_${test} tests/test_${test}.cpp)
        target_link_libraries(limes_test_${test} PRIVATE limes GTest::gtest GTest::gtest_main)
        gtest_discover_tests(limes_test_${test})
    endforeach()
endif()

if(STEPANOV_BUILD_EXAMPLES)
    add_executable(limes_demo examples/demo.cpp)
    target_link_libraries(limes_demo PRIVATE limes)
endif()
