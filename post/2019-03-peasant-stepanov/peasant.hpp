#pragma once

/**
 * @file peasant.hpp
 * @brief Generic algorithms from minimal operations
 *
 * This module teaches Stepanov's key insight: algorithms arise from the
 * *minimal* operations a type provides. The Russian peasant algorithm
 * computes multiplication using only:
 *
 *     twice(x), half(x), even(x), zero(x), one(x)
 *
 * The same pattern gives us power() via repeated squaring, and gcd()
 * for any type with quotient() and remainder().
 *
 * To adapt your own type, provide these as free functions findable via ADL.
 * See examples/big_int_simple.hpp for a complete example.
 *
 * Reference: "From Mathematics to Generic Programming" by Stepanov & Rose
 */

#include <concepts>
#include <utility>

namespace peasant {

// =============================================================================
// Built-in integer adaptors
// These must come FIRST so they're visible to concept definitions.
// For custom types, provide your own via ADL.
// =============================================================================

template<std::integral T> constexpr T zero(T) { return T{0}; }
template<std::integral T> constexpr T one(T)  { return T{1}; }

template<std::integral T> constexpr T twice(T x)     { return x << 1; }
template<std::integral T> constexpr T half(T x)      { return x >> 1; }
template<std::integral T> constexpr bool even(T x)   { return (x & 1) == 0; }

template<std::integral T> constexpr T increment(T x) { return x + 1; }
template<std::integral T> constexpr T decrement(T x) { return x - 1; }

template<std::integral T> constexpr T quotient(T a, T b)  { return a / b; }
template<std::integral T> constexpr T remainder(T a, T b) { return a % b; }

// =============================================================================
// Concepts: what operations must a type provide?
// =============================================================================

template<typename T>
concept has_zero = requires(T a) {
    { zero(a) } -> std::convertible_to<T>;
};

template<typename T>
concept has_one = requires(T a) {
    { one(a) } -> std::convertible_to<T>;
};

template<typename T>
concept has_twice = requires(T a) {
    { twice(a) } -> std::convertible_to<T>;
};

template<typename T>
concept has_half = requires(T a) {
    { half(a) } -> std::convertible_to<T>;
};

template<typename T>
concept has_even = requires(T a) {
    { even(a) } -> std::convertible_to<bool>;
};

template<typename T>
concept has_increment = requires(T a) {
    { increment(a) } -> std::convertible_to<T>;
};

template<typename T>
concept has_decrement = requires(T a) {
    { decrement(a) } -> std::convertible_to<T>;
};

/// A type satisfying `algebraic` can be used with product() and power()
template<typename T>
concept algebraic =
    std::regular<T> &&
    has_zero<T> && has_one<T> &&
    has_twice<T> && has_half<T> && has_even<T> &&
    has_increment<T> && has_decrement<T> &&
    requires(T a) {
        { a + a } -> std::convertible_to<T>;
        { -a } -> std::convertible_to<T>;
        { a < a } -> std::convertible_to<bool>;
    };

// =============================================================================
// The Algorithms
// =============================================================================

/**
 * Russian peasant multiplication
 *
 * Computes a * n using only: twice, half, even, zero, one, +, -
 *
 * The insight: a * n = 2*(a * n/2)      if n is even
 *              a * n = a + a*(n-1)      if n is odd
 *
 * This is O(log n) doublings and additions.
 */
template<typename T>
    requires algebraic<T>
constexpr T product(T const& a, T n)
{
    if (n == zero(n)) return zero(n);
    if (n == one(n))  return a;

    if (n < zero(n)) return -product(a, -n);

    return even(n)
        ? twice(product(a, half(n)))
        : a + product(a, decrement(n));
}

/// Square: just product(x, x)
template<typename T>
    requires algebraic<T>
constexpr T square(T const& x)
{
    return product(x, x);
}

/**
 * Power by repeated squaring
 *
 * Computes base^exp using O(log exp) multiplications.
 *
 * Same structure as product(): square when even, multiply when odd.
 */
template<typename T>
    requires algebraic<T>
constexpr T power(T const& base, T exp)
{
    if (exp == zero(exp)) return one(exp);
    if (exp == one(exp))  return base;
    if (exp < zero(exp))  return zero(exp);  // No inverse for integers

    return even(exp)
        ? square(power(base, half(exp)))
        : product(base, power(base, decrement(exp)));
}

// =============================================================================
// GCD and friends
// =============================================================================

/**
 * Euclidean GCD
 *
 * Works for any integral type with remainder().
 */
template<std::integral T>
constexpr T gcd(T a, T b)
{
    while (b != zero(b)) {
        a = remainder(a, b);
        std::swap(a, b);
    }
    return a < zero(a) ? -a : a;
}

/// Extended GCD: returns {gcd, x, y} where ax + by = gcd
template<std::integral T>
struct egcd_result { T gcd, x, y; };

template<std::integral T>
constexpr egcd_result<T> extended_gcd(T a, T b)
{
    T old_r = a, r = b;
    T old_s = one(a), s = zero(a);
    T old_t = zero(a), t = one(a);

    while (r != zero(r)) {
        T q = quotient(old_r, r);
        T tmp = r; r = old_r - q * r; old_r = tmp;
        tmp = s; s = old_s - q * s; old_s = tmp;
        tmp = t; t = old_t - q * t; old_t = tmp;
    }
    return {old_r, old_s, old_t};
}

/// Least common multiple
template<std::integral T>
constexpr T lcm(T a, T b)
{
    if (a == zero(a) || b == zero(b)) return zero(a);
    return quotient(a * b, gcd(a, b));
}

/// Test if two values are coprime
template<std::integral T>
constexpr bool coprime(T a, T b)
{
    return gcd(a, b) == one(a);
}

} // namespace peasant
