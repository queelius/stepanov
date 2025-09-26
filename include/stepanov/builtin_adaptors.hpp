#pragma once

#include <concepts>
#include <cmath>
#include <type_traits>
// Note: concepts.hpp not included here to avoid circular dependency

namespace stepanov {

/**
 * Adaptors to make built-in types work with generic algorithms
 * These provide the required operations for built-in arithmetic types
 *
 * Note: These functions are designed to be found via ADL (Argument-Dependent Lookup)
 * when used with built-in types in generic algorithms.
 */

// Forward declarations for better ADL
template <typename T> constexpr bool even(T x);
template <typename T> constexpr T twice(T x);
template <typename T> constexpr T half(T x);
template <typename T> constexpr T increment(T x);
template <typename T> constexpr T decrement(T x);
template <typename T> constexpr T quotient(T a, T b);
template <typename T> constexpr T remainder(T a, T b);
template <typename T> constexpr auto norm(T x);

// Fundamental operations for integral types
template <std::integral T>
constexpr bool even(T x) {
    return (x & 1) == 0;
}

template <std::integral T>
constexpr T twice(T x) {
    return x << 1;
}

template <std::integral T>
constexpr T half(T x) {
    return x >> 1;
}

template <std::integral T>
constexpr T increment(T x) {
    return x + 1;
}

template <std::integral T>
constexpr T decrement(T x) {
    return x - 1;
}

// Fundamental operations for floating-point types
template <std::floating_point T>
constexpr bool even(T x) {
    return std::fmod(x, T(2)) == T(0);
}

template <std::floating_point T>
constexpr T twice(T x) {
    return x * T(2);
}

template <std::floating_point T>
constexpr T half(T x) {
    return x / T(2);
}

template <std::floating_point T>
constexpr T increment(T x) {
    return x + T(1);
}

template <std::floating_point T>
constexpr T decrement(T x) {
    return x - T(1);
}

// Euclidean domain operations for integral types
template <std::integral T>
constexpr T quotient(T a, T b) {
    return a / b;
}

template <std::integral T>
constexpr T remainder(T a, T b) {
    return a % b;
}

template <std::integral T>
constexpr T floor_half(T x) {
    return x >> 1;
}

// Note: square is defined in math.hpp using the generic product function

// Norm function for integral types (absolute value)
template <std::integral T>
constexpr auto norm(T x) {
    return x < 0 ? -x : x;
}

// Norm function for floating-point types
template <std::floating_point T>
constexpr auto norm(T x) {
    return std::abs(x);
}

// Group operations for arithmetic types
template <typename T>
    requires std::is_arithmetic_v<T>
constexpr T compose(T a, T b) {
    return a + b;  // Addition as group operation
}

template <typename T>
    requires std::is_arithmetic_v<T>
constexpr T inverse(T a) {
    return -a;  // Negation as inverse
}

// For subtraction in Fenwick tree
template <typename T>
    requires std::is_arithmetic_v<T>
constexpr T inverse(T a, T b) {
    return a - b;  // Subtraction as inverse operation
}

template <typename T>
    requires std::is_arithmetic_v<T>
constexpr T identity(T) {
    return T(0);  // Zero as identity
}

// Helper to check if a type has the necessary operations
template <typename T>
concept has_builtin_operations =
    requires(T a, T b) {
        { even(a) } -> std::convertible_to<bool>;
        { twice(a) } -> std::convertible_to<T>;
        { half(a) } -> std::convertible_to<T>;
        { increment(a) } -> std::convertible_to<T>;
        { decrement(a) } -> std::convertible_to<T>;
    };

// Specialization for checking if integral types are euclidean domains
template <typename T>
concept builtin_euclidean = std::integral<T> && requires(T a, T b) {
    { a / b } -> std::convertible_to<T>;
    { a % b } -> std::convertible_to<T>;
    { norm(a) } -> std::integral;
};

} // namespace stepanov