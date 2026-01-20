#pragma once

/**
 * @file mod.hpp
 * @brief Modular Arithmetic as Rings
 *
 * A minimal implementation of integers modulo N, forming a ring (Z/NZ).
 * When N is prime, this is actually a field—every non-zero element has
 * a multiplicative inverse.
 *
 * Key algebraic properties:
 * - (Z/NZ, +) is a cyclic group of order N
 * - (Z/NZ, ×) is a monoid with identity 1
 * - × distributes over +
 * - When N is prime: (Z/NZ)* is a cyclic group of order N-1
 *
 * The implementation uses compile-time modulus for efficiency.
 */

#include <concepts>
#include <cstdint>
#include <iostream>

namespace modular {

template<int64_t N>
    requires (N > 0)
struct mod_int {
    int64_t v;

    // Normalize to [0, N)
    static constexpr int64_t normalize(int64_t x) {
        x %= N;
        return x < 0 ? x + N : x;
    }

    // Constructors
    constexpr mod_int() : v(0) {}
    constexpr mod_int(int64_t x) : v(normalize(x)) {}

    // Arithmetic
    constexpr mod_int operator+(mod_int rhs) const { return mod_int(v + rhs.v); }
    constexpr mod_int operator-(mod_int rhs) const { return mod_int(v - rhs.v); }
    constexpr mod_int operator*(mod_int rhs) const { return mod_int(v * rhs.v); }
    constexpr mod_int operator-() const { return mod_int(-v); }

    constexpr mod_int& operator+=(mod_int rhs) { return *this = *this + rhs; }
    constexpr mod_int& operator-=(mod_int rhs) { return *this = *this - rhs; }
    constexpr mod_int& operator*=(mod_int rhs) { return *this = *this * rhs; }

    // Comparison
    constexpr bool operator==(mod_int rhs) const { return v == rhs.v; }
    constexpr auto operator<=>(mod_int rhs) const { return v <=> rhs.v; }

    // Power (using repeated squaring)
    constexpr mod_int pow(int64_t exp) const {
        if (exp < 0) return inverse().pow(-exp);
        mod_int result(1), base = *this;
        while (exp > 0) {
            if (exp & 1) result *= base;
            base *= base;
            exp >>= 1;
        }
        return result;
    }

    // Multiplicative inverse (requires gcd(v, N) = 1)
    // Uses Fermat's little theorem when N is prime: a^(-1) = a^(N-2)
    constexpr mod_int inverse() const {
        return pow(N - 2);
    }

    // Division (only valid when N is prime and rhs != 0)
    constexpr mod_int operator/(mod_int rhs) const {
        return *this * rhs.inverse();
    }

    constexpr mod_int& operator/=(mod_int rhs) { return *this = *this / rhs; }

    // Conversion
    explicit constexpr operator int64_t() const { return v; }
};

// Stream output
template<int64_t N>
std::ostream& operator<<(std::ostream& os, mod_int<N> m) {
    return os << m.v << " (mod " << N << ")";
}

// =============================================================================
// Algebraic operations for generic algorithms
// =============================================================================

template<int64_t N>
constexpr mod_int<N> zero(mod_int<N>) { return mod_int<N>(0); }

template<int64_t N>
constexpr mod_int<N> one(mod_int<N>) { return mod_int<N>(1); }

template<int64_t N>
constexpr mod_int<N> twice(mod_int<N> x) { return x + x; }

template<int64_t N>
constexpr mod_int<N> half(mod_int<N> x) { return x * mod_int<N>(2).inverse(); }

template<int64_t N>
constexpr bool even(mod_int<N> x) { return x.v % 2 == 0; }

// =============================================================================
// Common moduli
// =============================================================================

// Popular prime moduli
using mod_1e9_7 = mod_int<1000000007>;  // 10^9 + 7 (common in competitive programming)
using mod_1e9_9 = mod_int<1000000009>;  // 10^9 + 9
using mod_998244353 = mod_int<998244353>;  // 2^23 × 7 × 17 + 1 (NTT-friendly)

// Small primes for testing
using mod_7 = mod_int<7>;
using mod_11 = mod_int<11>;
using mod_13 = mod_int<13>;

} // namespace modular
