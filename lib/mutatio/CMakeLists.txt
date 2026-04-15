add_library(mutatio INTERFACE)
target_include_directories(mutatio INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if(STEPANOV_BUILD_TESTS)
    add_executable(test_mutatio tests/test_mutatio.cpp)
    target_link_libraries(test_mutatio PRIVATE mutatio Catch2::Catch2WithMain)
    catch_discover_tests(test_mutatio)
endif()
