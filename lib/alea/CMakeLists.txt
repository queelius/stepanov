add_library(alea INTERFACE)
target_include_directories(alea INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(alea INTERFACE elementa)

if(STEPANOV_BUILD_TESTS)
    add_executable(test_alea tests/test_alea.cpp)
    target_link_libraries(test_alea PRIVATE alea Catch2::Catch2WithMain)
    catch_discover_tests(test_alea)
endif()
