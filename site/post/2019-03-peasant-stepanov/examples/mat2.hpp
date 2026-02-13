#pragma once

/**
 * @file mat2.hpp
 * @brief 2x2 matrices - compute Fibonacci in O(log n)
 *
 * The insight: [F(n+1)]   [1 1]^n   [1]
 *              [F(n)  ] = [1 0]   × [0]
 *
 * So power(mat2{1,1,1,0}, n) gives Fibonacci instantly!
 * This generalizes to ANY linear recurrence.
 */

#include <cstdint>
#include <compare>

namespace peasant::examples {

struct mat2 {
    int64_t a, b, c, d;  // [[a,b],[c,d]]

    constexpr bool operator==(mat2 const&) const = default;
    constexpr auto operator<=>(mat2 const&) const = default;

    constexpr mat2 operator+(mat2 const& m) const {
        return {a + m.a, b + m.b, c + m.c, d + m.d};
    }

    constexpr mat2 operator-() const {
        return {-a, -b, -c, -d};
    }

    // Matrix multiplication
    constexpr mat2 operator*(mat2 const& m) const {
        return {
            a * m.a + b * m.c,  a * m.b + b * m.d,
            c * m.a + d * m.c,  c * m.b + d * m.d
        };
    }
};

// ADL functions for peasant algorithms
constexpr mat2 zero(mat2) { return {0, 0, 0, 0}; }
constexpr mat2 one(mat2)  { return {1, 0, 0, 1}; }  // Identity matrix

constexpr mat2 twice(mat2 const& m) { return m * mat2{2,0,0,2}; }
constexpr mat2 half(mat2 const& m)  { return {m.a/2, m.b/2, m.c/2, m.d/2}; }
constexpr bool even(mat2 const& m)  { return (m.a & 1) == 0; }

constexpr mat2 increment(mat2 const& m) { return {m.a+1, m.b, m.c, m.d+1}; }
constexpr mat2 decrement(mat2 const& m) { return {m.a-1, m.b, m.c, m.d-1}; }

// The Fibonacci matrix
constexpr mat2 fib_matrix{1, 1, 1, 0};

// Extract F(n) from the result of power(fib_matrix, n)
constexpr int64_t fib_from_matrix(mat2 const& m) { return m.b; }

} // namespace peasant::examples
