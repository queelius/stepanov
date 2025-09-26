#pragma once

#include <utility>
#include <concepts>
#include <type_traits>
#include <optional>
#include <vector>
#include "concepts.hpp"

namespace stepanov {

/**
 * Generic GCD algorithms for Euclidean domains
 *
 * A Euclidean domain is an integral domain equipped with a Euclidean function
 * (norm) that allows the division algorithm. This includes integers, polynomials
 * over a field, and Gaussian integers.
 *
 * Requirements for type T:
 * - Models integral_domain (has +, -, *, 0, 1, no zero divisors)
 * - Has quotient(a, b) and remainder(a, b) operations
 * - Has norm(a) function with properties:
 *     (i) norm(a) == 0 iff a == 0
 *    (ii) norm(a*b) >= norm(a) for b != 0
 *   (iii) norm(remainder(a,b)) < norm(b) for b != 0
 */

// Basic Euclidean algorithm
template <typename T>
    requires euclidean_domain<T> || std::integral<T>
constexpr T gcd(T a, T b) {
    while (b != T(0)) {
        a = remainder(a, b);
        std::swap(a, b);
    }
    // GCD is always non-negative by convention
    if constexpr (std::is_signed_v<T>) {
        return a < T(0) ? -a : a;
    } else {
        return a;
    }
}

// Binary GCD (Stein's algorithm) for types that support bit operations
template <typename T>
    requires std::integral<T>
constexpr T binary_gcd(T a, T b) {
    if (a == 0) return b;
    if (b == 0) return a;

    // Find common factor of 2
    unsigned shift = 0;
    while (((a | b) & 1) == 0) {
        a >>= 1;
        b >>= 1;
        ++shift;
    }

    // Remove remaining factors of 2 from a
    while ((a & 1) == 0) {
        a >>= 1;
    }

    do {
        // Remove remaining factors of 2 from b
        while ((b & 1) == 0) {
            b >>= 1;
        }

        // Ensure a <= b
        if (a > b) {
            std::swap(a, b);
        }

        b -= a;
    } while (b != 0);

    return a << shift;
}

// Extended Euclidean algorithm
// Returns gcd(a, b) and finds x, y such that ax + by = gcd(a, b)
template <typename T>
    requires euclidean_domain<T> || std::integral<T>
struct extended_gcd_result {
    T gcd;
    T x;  // Bézout coefficient for a
    T y;  // Bézout coefficient for b
};

template <typename T>
    requires euclidean_domain<T> || std::integral<T>
constexpr extended_gcd_result<T> extended_gcd(T a, T b) {
    T old_r = a, r = b;
    T old_s = T(1), s = T(0);
    T old_t = T(0), t = T(1);

    while (r != T(0)) {
        T q = quotient(old_r, r);

        T temp = r;
        r = old_r - q * r;
        old_r = temp;

        temp = s;
        s = old_s - q * s;
        old_s = temp;

        temp = t;
        t = old_t - q * t;
        old_t = temp;
    }

    return {old_r, old_s, old_t};
}

// Least common multiple
template <typename T>
    requires euclidean_domain<T> || std::integral<T>
constexpr T lcm(T a, T b) {
    if (a == T(0) || b == T(0)) {
        return T(0);
    }
    return quotient(a * b, gcd(a, b));
}

// GCD of multiple values using fold expression
template <typename T, typename... Args>
    requires euclidean_domain<T> && (std::same_as<T, Args> && ...)
constexpr T gcd(T first, T second, Args... args) {
    if constexpr (sizeof...(args) == 0) {
        return gcd(first, second);
    } else {
        return gcd(gcd(first, second), args...);
    }
}

// LCM of multiple values
template <typename T, typename... Args>
    requires euclidean_domain<T> && (std::same_as<T, Args> && ...)
constexpr T lcm(T first, T second, Args... args) {
    if constexpr (sizeof...(args) == 0) {
        return lcm(first, second);
    } else {
        return lcm(lcm(first, second), args...);
    }
}

// Check if two values are coprime (relatively prime)
template <typename T>
    requires euclidean_domain<T>
constexpr bool coprime(T a, T b) {
    return gcd(a, b) == T(1);
}

// Generic GCD algorithm for ranges
template <typename InputIt>
    requires std::input_iterator<InputIt> &&
             euclidean_domain<std::iter_value_t<InputIt>>
constexpr auto gcd_range(InputIt first, InputIt last) {
    using T = std::iter_value_t<InputIt>;

    if (first == last) {
        return T(0);
    }

    T result = *first++;
    while (first != last) {
        result = gcd(result, *first++);
        if (result == T(1)) {
            break;  // Early exit if GCD becomes 1
        }
    }

    return result;
}

// Modular inverse using extended GCD
// Returns x such that (a * x) mod m = 1
template <typename T>
    requires euclidean_domain<T>
constexpr std::optional<T> mod_inverse(T a, T m) {
    auto [g, x, y] = extended_gcd(a, m);

    if (g != T(1)) {
        return std::nullopt;  // No inverse exists
    }

    // Make sure x is positive
    return ((remainder(x, m) + m) % m);
}

// Chinese Remainder Theorem solver
// Solves system: x ≡ a₁ (mod m₁), x ≡ a₂ (mod m₂), ...
template <typename T>
    requires euclidean_domain<T>
struct crt_result {
    T solution;     // x mod M where M = lcm(m₁, m₂, ...)
    T modulus;      // M
    bool has_solution;
};

template <typename T>
    requires euclidean_domain<T>
constexpr crt_result<T> chinese_remainder(
    const std::vector<T>& remainders,
    const std::vector<T>& moduli)
{
    if (remainders.size() != moduli.size() || remainders.empty()) {
        return {T(0), T(0), false};
    }

    T x = remainders[0];
    T m = moduli[0];

    for (size_t i = 1; i < remainders.size(); ++i) {
        auto [g, p, q] = extended_gcd(m, moduli[i]);

        if (remainder(remainders[i] - x, g) != T(0)) {
            return {T(0), T(0), false};  // No solution exists
        }

        x = x + m * quotient(p * (remainders[i] - x), g);
        m = quotient(m * moduli[i], g);  // lcm(m, moduli[i])
        x = remainder(x, m);
        // Ensure x is positive
        if (x < T(0)) {
            x = x + m;
        }
    }

    // Final normalization
    if (x < T(0)) {
        x = x + m;
    }

    return {x, m, true};
}

} // namespace stepanov