cmake_minimum_required(VERSION 3.20)
project(stepanov LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(STEPANOV_BUILD_TESTS "Build all library tests" ON)
option(STEPANOV_BUILD_EXAMPLES "Build example programs" ON)
option(STEPANOV_ENABLE_OPENMP "Enable OpenMP for limes" OFF)

# Concepts layer (header-only, no target — just include path)
include_directories(${CMAKE_SOURCE_DIR})

# Test frameworks (shared across all libraries)
if(STEPANOV_BUILD_TESTS)
    enable_testing()

    include(FetchContent)

    # Catch2 v3 — used by elementa, gradator, alea, alga
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.5.0
    )

    # Google Test — used by polynomials, limes
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )

    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(Catch2 googletest)

    include(CTest)
    include(Catch)
    include(GoogleTest)
endif()

# Libraries
add_subdirectory(lib/elementa)
add_subdirectory(lib/gradator)
add_subdirectory(lib/polynomials)
add_subdirectory(lib/limes)
add_subdirectory(lib/alea)
add_subdirectory(lib/alga)
add_subdirectory(lib/mutatio)
