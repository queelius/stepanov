// algebraic_operators.hpp - Operator overloads for parser composition
//
// Maps algebraic operations to C++ operators:
//   p1 >> p2   — sequential composition (monoid operation)
//   p1 | p2    — choice / alternation
//
// These operators make parser expressions read like grammars.

#pragma once

#include <alga/combinators.hpp>
#include <alga/concepts.hpp>

namespace alga {

// =============================================================================
// operator>> — Sequential Composition (Monoid Operation)
// =============================================================================

/// p1 >> p2 runs p1 then p2, collecting both results.
/// This is the monoid operation for parsers.
template <typename P1, typename P2>
    requires requires { typename P1::output_type; typename P2::output_type; }
auto operator>>(P1 p1, P2 p2) -> sequence_parser<P1, P2> {
    return sequence(std::move(p1), std::move(p2));
}

// =============================================================================
// operator| — Choice / Alternation
// =============================================================================

/// p1 | p2 tries p1 first; if it fails, tries p2.
template <typename P1, typename P2>
    requires requires { typename P1::output_type; typename P2::output_type; }
auto operator|(P1 p1, P2 p2) -> choice_parser<P1, P2> {
    return choice(std::move(p1), std::move(p2));
}

} // namespace alga
