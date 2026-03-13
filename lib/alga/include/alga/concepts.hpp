// concepts.hpp - Algebraic concepts for parser combinators
//
// Defines the algebraic structure that parsers inhabit:
// parsers form a monoid under sequential composition, with
// choice providing an alternative (semiring-like) structure.

#pragma once

#include <concepts>
#include <optional>
#include <type_traits>
#include <utility>

namespace alga {

// =============================================================================
// AlgebraicType — Monoid-like structure
// =============================================================================

/// An AlgebraicType has:
///   - Default construction (identity element)
///   - operator* (associative binary operation)
///
/// Law: associativity — (a * b) * c == a * (b * c)
/// Law: identity — T{} * a == a == a * T{}
template <typename T>
concept AlgebraicType = std::default_initializable<T> && requires(T a, T b) {
    { a * b } -> std::convertible_to<T>;
};

// =============================================================================
// Parse result type
// =============================================================================

/// The universal return type for parsers.
/// Success: pair of (remaining input position, parsed value)
/// Failure: empty optional
template <typename T, typename Iter>
using parse_result = std::optional<std::pair<Iter, T>>;

// =============================================================================
// Parser concept
// =============================================================================

/// A Parser is a callable that:
///   - Takes (begin, end) iterators over input
///   - Returns parse_result<output_type, Iter>
///
/// Parsers form a monoid under sequential composition:
///   identity: the parser that consumes nothing and succeeds
///   compose:  run first parser, then run second on remaining input
///
/// They also support alternation (choice), forming a semiring-like
/// structure when combined with sequential composition.
template <typename P, typename Iter = std::string::const_iterator>
concept Parser = requires(const P p, Iter begin, Iter end) {
    typename P::output_type;
    { p.parse(begin, end) } -> std::same_as<parse_result<typename P::output_type, Iter>>;
};

} // namespace alga
