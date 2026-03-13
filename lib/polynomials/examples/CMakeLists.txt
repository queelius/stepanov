# Example executables
add_executable(basic basic.cpp)
target_link_libraries(basic PRIVATE poly::polynomials)

add_executable(euclidean_gcd euclidean_gcd.cpp)
target_link_libraries(euclidean_gcd PRIVATE poly::polynomials)

# Enable compiler warnings
foreach(target basic euclidean_gcd)
    target_compile_options(${target} PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
    )
endforeach()
