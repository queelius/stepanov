add_library(elementa INTERFACE)
target_include_directories(elementa INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if(STEPANOV_BUILD_TESTS)
    add_executable(test_elementa tests/test_elementa.cpp)
    target_link_libraries(test_elementa PRIVATE elementa Catch2::Catch2WithMain)
    catch_discover_tests(test_elementa)
endif()
