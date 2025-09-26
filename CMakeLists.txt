cmake_minimum_required(VERSION 3.14)
project(Stepanov VERSION 1.0.0 LANGUAGES CXX)

# Set C++23 standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable testing
enable_testing()

# Options
option(STEPANOV_BUILD_TESTS "Build the tests" ON)
option(STEPANOV_BUILD_BENCHMARKS "Build the benchmarks" ON)
option(STEPANOV_ENABLE_COVERAGE "Enable code coverage" OFF)

# Interface library for stepanov (header-only)
add_library(stepanov INTERFACE)
target_include_directories(stepanov INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Set compile features
target_compile_features(stepanov INTERFACE cxx_std_23)

# Add warning flags for the interface
target_compile_options(stepanov INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# Coverage configuration
if(STEPANOV_ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(STATUS "Enabling code coverage")
    add_compile_options(--coverage -O0 -g)
    add_link_options(--coverage)
endif()

# Testing
if(STEPANOV_BUILD_TESTS)
    # Find Google Test
    find_package(GTest)

    if(NOT GTest_FOUND)
        message(STATUS "Google Test not found. Fetching from GitHub...")
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.14.0
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()

    add_subdirectory(tests)
endif()

# Installation
include(GNUInstallDirs)

install(TARGETS stepanov
    EXPORT stepanov-targets
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT stepanov-targets
    FILE stepanov-targets.cmake
    NAMESPACE stepanov::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stepanov
)

# Create package configuration file
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/stepanov-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/stepanov-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stepanov
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/stepanov-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/stepanov-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/stepanov-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stepanov
)

# Export from build tree
export(EXPORT stepanov-targets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/stepanov-targets.cmake
    NAMESPACE stepanov::
)

# Add custom target for coverage report
if(STEPANOV_ENABLE_COVERAGE)
    find_program(LCOV lcov)
    find_program(GENHTML genhtml)

    if(LCOV AND GENHTML)
        add_custom_target(coverage
            COMMAND ${LCOV} --capture --directory . --output-file coverage.info
            COMMAND ${LCOV} --remove coverage.info '/usr/*' '*/test/*' '*/googletest/*' --output-file coverage.info
            COMMAND ${GENHTML} coverage.info --output-directory coverage_report
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating coverage report"
        )
    else()
        message(WARNING "lcov and/or genhtml not found. Coverage report generation disabled.")
    endif()
endif()