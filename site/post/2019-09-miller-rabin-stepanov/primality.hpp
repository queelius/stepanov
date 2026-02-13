#pragma once

/**
 * @file primality.hpp
 * @brief Miller-Rabin Primality Testing
 *
 * A minimal, elegant implementation of the Miller-Rabin probabilistic
 * primality test. This algorithm answers "Is n prime?" in O(k log³ n)
 * time, where k is the number of witness tests.
 *
 * Key insight: For any odd prime p, the only square roots of 1 (mod p)
 * are 1 and -1. The Miller-Rabin test exploits this using Fermat's
 * Little Theorem: a^(p-1) ≡ 1 (mod p) for prime p.
 *
 * Error bounds: Each witness test has ≤ 1/4 chance of false positive.
 * With k witnesses, error ≤ (1/4)^k.
 */

#include <concepts>
#include <cstdint>
#include <cmath>
#include <random>

namespace primality {

// =============================================================================
// Modular exponentiation: the workhorse
// =============================================================================

/// Compute (base^exp) mod m using repeated squaring
template<std::integral T>
constexpr T mod_pow(T base, T exp, T m) {
    T result = 1;
    base %= m;
    while (exp > 0) {
        if (exp & 1)
            result = static_cast<T>((static_cast<__int128>(result) * base) % m);
        exp >>= 1;
        base = static_cast<T>((static_cast<__int128>(base) * base) % m);
    }
    return result;
}

// =============================================================================
// Miller-Rabin witness test
// =============================================================================

/**
 * Single witness test for Miller-Rabin
 *
 * Write n-1 = 2^r * d (where d is odd).
 * Compute x = a^d mod n. If x = 1 or x = n-1, probably prime.
 * Otherwise, square x up to r-1 times. If we ever get n-1, probably prime.
 * If we never see 1 or n-1, definitely composite.
 */
template<std::integral T>
constexpr bool witness_test(T n, T a) {
    // Decompose n-1 = 2^r * d
    T d = n - 1;
    int r = 0;
    while ((d & 1) == 0) {
        d >>= 1;
        ++r;
    }

    // Compute x = a^d mod n
    T x = mod_pow(a, d, n);

    if (x == 1 || x == n - 1)
        return true;  // Probably prime

    // Square repeatedly
    for (int i = 1; i < r; ++i) {
        x = static_cast<T>((static_cast<__int128>(x) * x) % n);
        if (x == n - 1)
            return true;  // Probably prime
    }

    return false;  // Definitely composite
}

// =============================================================================
// Result type with probability information
// =============================================================================

struct primality_result {
    bool probably_prime;    // true if passed all witness tests
    double error_bound;     // upper bound on P(false positive)
    int witnesses_tested;   // number of witnesses used

    explicit operator bool() const { return probably_prime; }
};

// =============================================================================
// Main primality test
// =============================================================================

/**
 * Compute number of witnesses needed for desired error bound
 *
 * Since each witness has ≤ 1/4 false positive rate:
 *   k witnesses → error ≤ (1/4)^k
 *   For error ≤ ε, we need k ≥ -log₄(ε) = -ln(ε) / ln(4)
 */
inline int witnesses_for_error(double max_error) {
    if (max_error >= 0.25) return 1;
    if (max_error <= 0) return 40;  // Practical maximum
    return static_cast<int>(std::ceil(-std::log(max_error) / std::log(4.0)));
}

/**
 * Compute error bound for k witnesses
 */
inline double error_bound(int k) {
    return std::pow(0.25, k);
}

/**
 * Miller-Rabin primality test with error bound
 *
 * @param n         Number to test
 * @param max_error Maximum acceptable false positive probability (default 10^-12)
 * @return          Result with prime status and actual error bound
 */
template<std::integral T>
primality_result is_prime_with_error(T n, double max_error = 1e-12) {
    // Handle trivial cases
    if (n < 2) return {false, 0.0, 0};
    if (n == 2 || n == 3) return {true, 0.0, 0};
    if ((n & 1) == 0) return {false, 0.0, 0};

    int k = witnesses_for_error(max_error);

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<T> dist(2, n - 2);

    for (int i = 0; i < k; ++i) {
        T a = dist(gen);
        if (!witness_test(n, a))
            return {false, 0.0, i + 1};  // Definitely composite
    }

    return {true, error_bound(k), k};  // Probably prime
}

/**
 * Simple boolean interface
 *
 * @param n         Number to test
 * @param max_error Maximum acceptable false positive probability (default 10^-12)
 */
template<std::integral T>
bool is_prime(T n, double max_error = 1e-12) {
    return is_prime_with_error(n, max_error).probably_prime;
}

} // namespace primality
