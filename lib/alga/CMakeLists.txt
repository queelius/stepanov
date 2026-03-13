add_library(alga INTERFACE)
target_include_directories(alga INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if(STEPANOV_BUILD_TESTS)
    add_executable(test_alga tests/test_alga.cpp)
    target_link_libraries(test_alga PRIVATE alga Catch2::Catch2WithMain)
    catch_discover_tests(test_alga)
endif()
