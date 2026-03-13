// algebra.hpp - Algebraic structure concepts
//
// Pure concept definitions for algebraic structures. Zero implementation.
// These concepts capture the mathematical structures that Stepanov's
// algorithms depend upon: monoids, groups, rings, fields, Euclidean domains.
//
// Conventions:
//   - Operations are ADL-discoverable free functions: zero(x), one(x), etc.
//   - This follows the peasant.hpp pattern from the Stepanov blog series.
//   - Types satisfy a concept by providing the required operations.

#pragma once

#include <concepts>
#include <type_traits>

namespace stepanov {

// =============================================================================
// Additive structure
// =============================================================================

/// A type with an additive identity: zero(x) -> T
template <typename T>
concept has_zero = requires(T a) {
    { zero(a) } -> std::convertible_to<T>;
};

/// A type with a multiplicative identity: one(x) -> T
template <typename T>
concept has_one = requires(T a) {
    { one(a) } -> std::convertible_to<T>;
};

// =============================================================================
// Monoid: identity + associative binary operation
// =============================================================================

/// Additive monoid: zero(x) + (a + b) is associative
template <typename T>
concept AdditiveMonoid = std::regular<T> && has_zero<T> && requires(T a, T b) {
    { a + b } -> std::convertible_to<T>;
};

/// Multiplicative monoid: one(x) * (a * b) is associative
template <typename T>
concept MultiplicativeMonoid = std::regular<T> && has_one<T> && requires(T a, T b) {
    { a * b } -> std::convertible_to<T>;
};

/// General monoid: default-constructible identity + operator*
template <typename T>
concept Monoid = std::regular<T> && requires(T a, T b) {
    { T{} };              // identity element
    { a * b } -> std::convertible_to<T>;
};

// =============================================================================
// Group: monoid + inverse
// =============================================================================

/// Additive group: monoid with negation (additive inverse)
template <typename T>
concept AdditiveGroup = AdditiveMonoid<T> && requires(T a, T b) {
    { -a } -> std::convertible_to<T>;
    { a - b } -> std::convertible_to<T>;
};

/// Group: monoid with inverse(x) -> T
template <typename T>
concept Group = Monoid<T> && requires(T a) {
    { inverse(a) } -> std::convertible_to<T>;
};

// =============================================================================
// Ring: additive group + multiplicative monoid + distributivity
// =============================================================================

/// A ring has both additive and multiplicative structure.
/// Polynomials form a ring over any coefficient ring.
template <typename T>
concept Ring = AdditiveGroup<T> && has_one<T> && requires(T a, T b) {
    { a * b } -> std::convertible_to<T>;
    { a == b } -> std::convertible_to<bool>;
};

// =============================================================================
// Field: ring where every nonzero element has a multiplicative inverse
// =============================================================================

/// A field supports division (by nonzero elements).
/// When coefficients are from a field, polynomial division is well-defined.
template <typename T>
concept Field = Ring<T> && requires(T a, T b) {
    { a / b } -> std::convertible_to<T>;
};

/// Ordered field: field with a total ordering (e.g., reals).
template <typename T>
concept OrderedField = Field<T> && std::totally_ordered<T>;

// =============================================================================
// Euclidean domain: ring with division algorithm
// =============================================================================

/// A Euclidean domain has quotient and remainder satisfying:
///   a = b * quotient(a, b) + remainder(a, b)
/// with norm(remainder) < norm(b) or remainder = zero.
///
/// This is the structure that makes GCD algorithms work — the same
/// algorithm applies to integers, polynomials, and Gaussian integers.
template <typename T>
concept EuclideanDomain = Ring<T> && requires(T a, T b) {
    { quotient(a, b) } -> std::convertible_to<T>;
    { remainder(a, b) } -> std::convertible_to<T>;
};

// =============================================================================
// Arithmetic: minimal requirements for scalar types
// =============================================================================

/// Scalar types supporting basic arithmetic.
/// This is what matrix elements must satisfy.
template <typename T>
concept Arithmetic = requires(T a, T b) {
    { a + b } -> std::convertible_to<T>;
    { a - b } -> std::convertible_to<T>;
    { a * b } -> std::convertible_to<T>;
    { a / b } -> std::convertible_to<T>;
    { -a } -> std::convertible_to<T>;
};

// =============================================================================
// Peasant-style operations (for power/product algorithms)
// =============================================================================

template <typename T>
concept has_twice = requires(T a) {
    { twice(a) } -> std::convertible_to<T>;
};

template <typename T>
concept has_half = requires(T a) {
    { half(a) } -> std::convertible_to<T>;
};

template <typename T>
concept has_even = requires(T a) {
    { even(a) } -> std::convertible_to<bool>;
};

/// Algebraic type usable with power() and product() from peasant.hpp
template <typename T>
concept Algebraic =
    std::regular<T> &&
    has_zero<T> && has_one<T> &&
    has_twice<T> && has_half<T> && has_even<T> &&
    requires(T a) {
        { a + a } -> std::convertible_to<T>;
        { -a } -> std::convertible_to<T>;
        { a < a } -> std::convertible_to<bool>;
    };

} // namespace stepanov
