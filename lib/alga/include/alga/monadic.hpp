// monadic.hpp - Functorial and monadic parser combinators
//
// Parsers form a functor (fmap), applicative (pure, apply), and monad (bind).
//
// Functor laws:
//   fmap(id, p) == p                         [identity]
//   fmap(f . g, p) == fmap(f, fmap(g, p))    [composition]
//
// Monad laws:
//   bind(pure(x), f) == f(x)                 [left identity]
//   bind(m, pure) == m                        [right identity]
//   bind(bind(m, f), g) == bind(m, λx. bind(f(x), g))  [associativity]

#pragma once

#include <alga/concepts.hpp>

#include <functional>
#include <type_traits>
#include <utility>

namespace alga {

// =============================================================================
// fmap — Functor
// =============================================================================

/// Transforms a parser's output: if P parses an A, fmap(f, P) parses f(A).
/// This is the functorial action on parsers.
template <typename F, typename P>
class fmap_parser {
    F func_;
    P parser_;

public:
    using output_type = std::invoke_result_t<F, typename P::output_type>;

    fmap_parser(F f, P p) : func_(std::move(f)), parser_(std::move(p)) {}

    template <typename Iter>
    auto parse(Iter begin, Iter end) const -> parse_result<output_type, Iter> {
        auto r = parser_.parse(begin, end);
        if (!r) return std::nullopt;
        return std::pair{r->first, std::invoke(func_, std::move(r->second))};
    }
};

template <typename F, typename P>
auto fmap(F f, P p) -> fmap_parser<F, P> {
    return {std::move(f), std::move(p)};
}

// =============================================================================
// pure — Monadic Lift
// =============================================================================

/// Lifts a value into the parser context without consuming input.
/// This is the unit/return of the parser monad.
template <typename T>
class pure_parser {
    T value_;

public:
    using output_type = T;

    explicit pure_parser(T v) : value_(std::move(v)) {}

    template <typename Iter>
    auto parse(Iter begin, [[maybe_unused]] Iter end) const
        -> parse_result<output_type, Iter> {
        return std::pair{begin, value_};
    }
};

template <typename T>
auto pure(T value) -> pure_parser<T> {
    return pure_parser<T>(std::move(value));
}

// =============================================================================
// bind — Monad
// =============================================================================

/// Monadic bind: runs parser P, then feeds its result to F which returns
/// a new parser, and runs that parser on the remaining input.
///
/// This is the >>= operation: m >>= f
template <typename P, typename F>
class bind_parser {
    P parser_;
    F func_;

    // F takes P::output_type and returns a parser
    using inner_parser_type = std::invoke_result_t<F, typename P::output_type>;

public:
    using output_type = typename inner_parser_type::output_type;

    bind_parser(P p, F f) : parser_(std::move(p)), func_(std::move(f)) {}

    template <typename Iter>
    auto parse(Iter begin, Iter end) const -> parse_result<output_type, Iter> {
        auto r = parser_.parse(begin, end);
        if (!r) return std::nullopt;
        auto next_parser = std::invoke(func_, std::move(r->second));
        return next_parser.parse(r->first, end);
    }
};

template <typename P, typename F>
auto bind(P p, F f) -> bind_parser<P, F> {
    return {std::move(p), std::move(f)};
}

// =============================================================================
// apply — Applicative
// =============================================================================

/// Applicative apply: if PF parses a function and PA parses a value,
/// apply(PF, PA) parses the function applied to the value.
template <typename PF, typename PA>
class apply_parser {
    PF pf_;
    PA pa_;

    using func_type = typename PF::output_type;
    using arg_type = typename PA::output_type;

public:
    using output_type = std::invoke_result_t<func_type, arg_type>;

    apply_parser(PF pf, PA pa) : pf_(std::move(pf)), pa_(std::move(pa)) {}

    template <typename Iter>
    auto parse(Iter begin, Iter end) const -> parse_result<output_type, Iter> {
        auto rf = pf_.parse(begin, end);
        if (!rf) return std::nullopt;

        auto ra = pa_.parse(rf->first, end);
        if (!ra) return std::nullopt;

        return std::pair{ra->first, std::invoke(rf->second, std::move(ra->second))};
    }
};

template <typename PF, typename PA>
auto apply(PF pf, PA pa) -> apply_parser<PF, PA> {
    return {std::move(pf), std::move(pa)};
}

} // namespace alga
