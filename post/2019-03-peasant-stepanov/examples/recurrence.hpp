#pragma once

/**
 * @file recurrence.hpp
 * @brief Companion matrices - ANY linear recurrence via matrix power
 *
 * A k-th order linear recurrence:
 *   a(n) = c₁*a(n-1) + c₂*a(n-2) + ... + cₖ*a(n-k)
 *
 * Can be computed via matrix exponentiation using the companion matrix:
 *   [a(n)  ]   [c₁ c₂ ... cₖ₋₁ cₖ]^(n-k+1)   [a(k-1)]
 *   [a(n-1)] = [1  0  ... 0    0 ]         × [a(k-2)]
 *   [...]     [0  1  ... 0    0 ]           [...]
 *   [a(n-k+1)][0  0  ... 1    0 ]           [a(0)  ]
 *
 * Examples:
 * - Fibonacci: a(n) = a(n-1) + a(n-2), coeffs = {1, 1}
 * - Tribonacci: a(n) = a(n-1) + a(n-2) + a(n-3), coeffs = {1, 1, 1}
 * - Pell: a(n) = 2*a(n-1) + a(n-2), coeffs = {2, 1}
 * - Lucas: a(n) = a(n-1) + a(n-2), coeffs = {1, 1} (different initial values)
 */

#include <array>
#include <compare>
#include <cstdint>

namespace peasant::examples {

template<size_t K, typename T = int64_t>
struct companion {
    std::array<std::array<T, K>, K> m;

    constexpr bool operator==(companion const&) const = default;
    constexpr auto operator<=>(companion const&) const = default;

    // Matrix multiplication
    constexpr companion operator*(companion const& o) const {
        companion result{};
        for (size_t i = 0; i < K; ++i) {
            for (size_t j = 0; j < K; ++j) {
                T sum{0};
                for (size_t k = 0; k < K; ++k) {
                    sum += m[i][k] * o.m[k][j];
                }
                result.m[i][j] = sum;
            }
        }
        return result;
    }

    // Required for algebraic concept
    constexpr companion operator+(companion const& o) const { return *this * o; }
    constexpr companion operator-() const { return *this; }

    // Apply to a state vector, return a(n) (the top element)
    constexpr T apply(std::array<T, K> const& state) const {
        T sum{0};
        for (size_t j = 0; j < K; ++j) {
            sum += m[0][j] * state[j];
        }
        return sum;
    }
};

// Identity matrix
template<size_t K, typename T = int64_t>
constexpr companion<K, T> identity_companion() {
    companion<K, T> result{};
    for (size_t i = 0; i < K; ++i) {
        result.m[i][i] = T{1};
    }
    return result;
}

// ADL functions
template<size_t K, typename T>
constexpr companion<K, T> zero(companion<K, T>) { return identity_companion<K, T>(); }

template<size_t K, typename T>
constexpr companion<K, T> one(companion<K, T>) { return identity_companion<K, T>(); }

template<size_t K, typename T>
constexpr companion<K, T> twice(companion<K, T> const& c) { return c * c; }

template<size_t K, typename T>
constexpr companion<K, T> half(companion<K, T> const& c) { return c; }

template<size_t K, typename T>
constexpr bool even(companion<K, T> const&) { return true; }

template<size_t K, typename T>
constexpr companion<K, T> increment(companion<K, T> const& c) { return c; }

template<size_t K, typename T>
constexpr companion<K, T> decrement(companion<K, T> const& c) { return c; }

// Factory: Create companion matrix from recurrence coefficients
// For a(n) = c₁*a(n-1) + c₂*a(n-2) + ... + cₖ*a(n-k)
// Pass coeffs = {c₁, c₂, ..., cₖ}
template<size_t K, typename T = int64_t>
constexpr companion<K, T> make_companion(std::array<T, K> coeffs) {
    companion<K, T> result{};
    // First row: coefficients
    for (size_t j = 0; j < K; ++j) {
        result.m[0][j] = coeffs[j];
    }
    // Subdiagonal: ones
    for (size_t i = 1; i < K; ++i) {
        result.m[i][i-1] = T{1};
    }
    return result;
}

// Predefined recurrence matrices
template<typename T = int64_t>
constexpr companion<2, T> fibonacci_companion() {
    return make_companion<2, T>({T{1}, T{1}});
}

template<typename T = int64_t>
constexpr companion<3, T> tribonacci_companion() {
    return make_companion<3, T>({T{1}, T{1}, T{1}});
}

template<typename T = int64_t>
constexpr companion<2, T> pell_companion() {
    return make_companion<2, T>({T{2}, T{1}});
}

template<typename T = int64_t>
constexpr companion<2, T> lucas_companion() {
    // Same matrix as Fibonacci, different initial values
    return make_companion<2, T>({T{1}, T{1}});
}

// Compute the nth term of a recurrence given companion matrix and initial values
// initial[0] = a(K-1), initial[1] = a(K-2), ..., initial[K-1] = a(0)
template<size_t K, typename T>
T compute_term(companion<K, T> const& mat, std::array<T, K> const& initial, int64_t n) {
    if (n < 0) return T{0};
    if (n < static_cast<int64_t>(K)) return initial[K - 1 - static_cast<size_t>(n)];

    // Compute mat^(n-K+1) using repeated squaring
    companion<K, T> result = identity_companion<K, T>();
    companion<K, T> base = mat;
    int64_t exp = n - static_cast<int64_t>(K) + 1;

    while (exp > 0) {
        if (exp & 1) result = result * base;
        base = base * base;
        exp >>= 1;
    }

    return result.apply(initial);
}

} // namespace peasant::examples
