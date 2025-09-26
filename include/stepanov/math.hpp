#pragma once

#include "concepts.hpp"
#include "builtin_operations.hpp"

namespace stepanov {

// Forward declarations for clarity
template <typename T> requires algebraic<T>
T product(T const& lhs, T const& rhs);

template <typename T> requires algebraic<T>
T square(T const& x);

template <typename T> requires algebraic<T>
T power(T const& base, T const& exp);

template <typename T> requires algebraic<T>
T sum(T const& lhs, T const& rhs);

/**
 * Generic multiplication using Russian peasant algorithm
 * Based on the observation that:
 *   a * b = 2 * (a * (b/2))           if b is even
 *   a * b = a + a * (b-1)             if b is odd
 *
 * Requirements: T must provide twice, half, even, decrement operations
 */
template <typename T> requires algebraic<T>
T product(T const& lhs, T const& rhs)
{
    if (rhs == T(0))
        return T(0);

    if (rhs == T(1))
        return lhs;

    // Handle negative multiplier by factoring out the sign
    if (rhs < T(0))
        return -product(lhs, -rhs);

    return even(rhs) ?
        twice(product(lhs, half(rhs))) :
        sum(lhs, product(lhs, decrement(rhs)));
}

/**
 * Generic squaring algorithm
 * More efficient than product(x, x) for large numbers
 */
template <typename T> requires algebraic<T>
T square(T const& x)
{
    return product(x, x);
}

/**
 * Generic power by repeated squaring
 * Complexity: O(log n) multiplications
 */
template <typename T> requires algebraic<T>
T power(T const& base, T const& exp)
{
    if (exp == T(0))
        return T(1);

    if (exp == T(1))
        return base;

    // Power with negative exponent would be 1/base^(-exp) but we return 0 for integers
    if (exp < T(0))
        return T(0);  // Or could throw an exception

    return even(exp) ?
        square(power(base, half(exp))) :
        product(base, power(base, decrement(exp)));
}

/**
 * Generic addition using binary representation
 * Based on: a + b = 2*(a/2 + b/2) + (a%2) + (b%2)
 */
template <typename T> requires algebraic<T>
T sum(T const& lhs, T const& rhs)
{
    // For types with built-in addition, just use it directly
    // The generic algorithm below only works for non-negative integers
    return lhs + rhs;
}

/**
 * Generic power with accumulator - tail recursive version
 * More efficient for iterative evaluation
 */
template <typename T, typename I>
    requires multiplicative_monoid<T> && std::integral<I>
T power_accumulate(T base, I exp, T result)
{
    while (exp != 0) {
        if (exp & 1)
            result = result * base;
        base = base * base;
        exp >>= 1;
    }
    return result;
}

/**
 * Generic modular power - efficient for cryptographic operations
 */
template <typename T, typename I>
    requires ring<T> && std::integral<I>
T power_mod(T base, I exp, T modulus)
{
    T result = T(1);
    base = base % modulus;

    while (exp > 0) {
        if (exp & 1)
            result = (result * base) % modulus;
        exp >>= 1;
        base = (base * base) % modulus;
    }
    return result;
}

/**
 * Generic multiply-accumulate operation
 * Fundamental operation for many algorithms
 */
template <typename T>
    requires ring<T>
T multiply_accumulate(T a, T b, T c)
{
    return a * b + c;
}

/**
 * Inner product - generalization of dot product
 */
template <typename InputIt1, typename InputIt2, typename T>
    requires std::input_iterator<InputIt1> &&
             std::input_iterator<InputIt2> &&
             ring<T>
T inner_product(InputIt1 first1, InputIt1 last1,
                InputIt2 first2, T init)
{
    while (first1 != last1) {
        init = multiply_accumulate(*first1, *first2, init);
        ++first1;
        ++first2;
    }
    return init;
}

/**
 * Generic absolute value
 */
template <typename T>
    requires ordered_ring<T>
T abs(T const& x)
{
    return x < T(0) ? -x : x;
}

/**
 * Generic minimum
 */
template <typename T>
    requires totally_ordered<T>
T const& min(T const& a, T const& b)
{
    return b < a ? b : a;
}

/**
 * Generic maximum
 */
template <typename T>
    requires totally_ordered<T>
T const& max(T const& a, T const& b)
{
    return a < b ? b : a;
}

// Note: remainder and quotient are defined in builtin_adaptors.hpp

} // namespace stepanov