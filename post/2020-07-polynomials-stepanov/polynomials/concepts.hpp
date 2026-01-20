#pragma once

/**
 * @file concepts.hpp
 * @brief Algebraic concepts for polynomial arithmetic
 *
 * Polynomials form a ring over any coefficient ring, and a Euclidean domain
 * when coefficients are from a field. These concepts capture those requirements.
 */

#include <concepts>
#include <type_traits>

namespace poly {

/**
 * Ring concept: an algebraic structure with +, -, * operations.
 *
 * Polynomials with coefficients from a ring form a ring themselves.
 */
template<typename T>
concept ring = requires(T a, T b) {
    { T(0) } -> std::convertible_to<T>;
    { T(1) } -> std::convertible_to<T>;
    { a + b } -> std::convertible_to<T>;
    { a - b } -> std::convertible_to<T>;
    { -a } -> std::convertible_to<T>;
    { a * b } -> std::convertible_to<T>;
    { a == b } -> std::convertible_to<bool>;
    { a != b } -> std::convertible_to<bool>;
};

/**
 * Field concept: a ring where division is always possible.
 *
 * When coefficients are from a field, polynomial division is well-defined
 * and polynomials form a Euclidean domain.
 */
template<typename T>
concept field = ring<T> && requires(T a, T b) {
    { a / b } -> std::convertible_to<T>;
};

/**
 * Ordered field: a field with a total ordering.
 *
 * Required for root-finding algorithms that compare values.
 */
template<typename T>
concept ordered_field = field<T> && requires(T a, T b) {
    { a < b } -> std::convertible_to<bool>;
    { a <= b } -> std::convertible_to<bool>;
    { a > b } -> std::convertible_to<bool>;
    { a >= b } -> std::convertible_to<bool>;
};

/**
 * Euclidean domain concept: has norm and division with remainder.
 *
 * Key property: for any a, b with b != 0, there exist q, r such that
 * a = b*q + r and either r = 0 or norm(r) < norm(b).
 */
template<typename T>
concept euclidean_domain = ring<T> && requires(T a, T b) {
    { quotient(a, b) } -> std::convertible_to<T>;
    { remainder(a, b) } -> std::convertible_to<T>;
    { norm(a) } -> std::convertible_to<int>;
};

} // namespace poly
