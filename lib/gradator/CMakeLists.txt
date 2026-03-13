add_library(gradator INTERFACE)
target_include_directories(gradator INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(gradator INTERFACE elementa)

option(GRADATOR_BUILD_EXAMPLES "Build gradator examples" ON)

if(STEPANOV_BUILD_TESTS)
    add_executable(test_gradator tests/test_gradator.cpp)
    target_link_libraries(test_gradator PRIVATE gradator Catch2::Catch2WithMain)
    catch_discover_tests(test_gradator)
endif()

if(GRADATOR_BUILD_EXAMPLES)
    foreach(example mle_normal logistic_regression optimization)
        add_executable(${example} examples/${example}.cpp)
        target_link_libraries(${example} PRIVATE gradator)
    endforeach()
endif()
