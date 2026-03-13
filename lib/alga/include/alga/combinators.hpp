// combinators.hpp - Core parser combinators
//
// Parser combinators are higher-order functions that compose parsers.
// They form the algebraic core of the library:
//
//   sequence(p1, p2) — monoid composition (run p1, then p2)
//   choice(p1, p2)   — alternation (try p1, if fails try p2)
//   many(p)          — Kleene star (zero or more)
//   many1(p)         — one or more
//
// Each combinator takes parsers and returns a new parser.
// Parsers are concrete types composed via templates — no virtual dispatch.

#pragma once

#include <alga/concepts.hpp>

#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace alga {

// =============================================================================
// Primitive Parsers
// =============================================================================

/// Parser that matches a single character satisfying a predicate.
template <typename Pred>
class char_parser_t {
    Pred pred_;

public:
    using output_type = char;

    explicit char_parser_t(Pred pred) : pred_(std::move(pred)) {}

    template <typename Iter>
    auto parse(Iter begin, Iter end) const -> parse_result<char, Iter> {
        if (begin != end && pred_(*begin)) {
            return std::pair{std::next(begin), *begin};
        }
        return std::nullopt;
    }
};

/// Factory for char_parser_t
template <typename Pred>
auto char_parser(Pred pred) -> char_parser_t<Pred> {
    return char_parser_t<Pred>(std::move(pred));
}

/// Parser that matches a literal string.
class literal_parser {
    std::string expected_;

public:
    using output_type = std::string;

    explicit literal_parser(std::string s) : expected_(std::move(s)) {}

    template <typename Iter>
    auto parse(Iter begin, Iter end) const -> parse_result<std::string, Iter> {
        auto it = begin;
        for (char c : expected_) {
            if (it == end || *it != c) return std::nullopt;
            ++it;
        }
        return std::pair{it, expected_};
    }
};

/// Factory for literal parser
inline auto literal(std::string s) -> literal_parser {
    return literal_parser(std::move(s));
}

// =============================================================================
// Sequence Combinator — Monoid Composition
// =============================================================================

/// Runs P1 then P2 on the remaining input.
/// This is the monoid operation for parsers.
///
/// Law: sequence(sequence(a, b), c) ≡ sequence(a, sequence(b, c))  [associativity]
/// Law: sequence(pure(x), p) ≡ p  [left identity — roughly]
template <typename P1, typename P2>
class sequence_parser {
    P1 first_;
    P2 second_;

public:
    // Output is a pair of both results
    using output_type = std::pair<typename P1::output_type, typename P2::output_type>;

    sequence_parser(P1 p1, P2 p2) : first_(std::move(p1)), second_(std::move(p2)) {}

    template <typename Iter>
    auto parse(Iter begin, Iter end) const -> parse_result<output_type, Iter> {
        auto r1 = first_.parse(begin, end);
        if (!r1) return std::nullopt;

        auto r2 = second_.parse(r1->first, end);
        if (!r2) return std::nullopt;

        return std::pair{r2->first, output_type{r1->second, r2->second}};
    }
};

template <typename P1, typename P2>
auto sequence(P1 p1, P2 p2) -> sequence_parser<P1, P2> {
    return {std::move(p1), std::move(p2)};
}

// =============================================================================
// Choice Combinator — Alternation
// =============================================================================

/// Tries P1 first; if it fails, tries P2 on the original input.
///
/// This gives parsers a semiring-like structure when combined with sequence:
///   sequence distributes over choice (from the left).
template <typename P1, typename P2>
class choice_parser {
    P1 first_;
    P2 second_;

    // Both must have the same output type
    static_assert(std::is_same_v<typename P1::output_type, typename P2::output_type>,
                  "Choice combinator requires parsers with the same output type");

public:
    using output_type = typename P1::output_type;

    choice_parser(P1 p1, P2 p2) : first_(std::move(p1)), second_(std::move(p2)) {}

    template <typename Iter>
    auto parse(Iter begin, Iter end) const -> parse_result<output_type, Iter> {
        auto r1 = first_.parse(begin, end);
        if (r1) return r1;
        return second_.parse(begin, end);
    }
};

template <typename P1, typename P2>
auto choice(P1 p1, P2 p2) -> choice_parser<P1, P2> {
    return {std::move(p1), std::move(p2)};
}

// =============================================================================
// Many Combinator — Kleene Star
// =============================================================================

/// Applies parser P zero or more times, collecting results into a vector.
/// Always succeeds (with an empty vector if P fails immediately).
///
/// This is the free monoid construction applied to parse results.
template <typename P>
class many_parser {
    P parser_;

public:
    using output_type = std::vector<typename P::output_type>;

    explicit many_parser(P p) : parser_(std::move(p)) {}

    template <typename Iter>
    auto parse(Iter begin, Iter end) const -> parse_result<output_type, Iter> {
        output_type results;
        auto pos = begin;

        while (true) {
            auto r = parser_.parse(pos, end);
            if (!r) break;
            if (r->first == pos) break;  // prevent infinite loop on empty match
            results.push_back(std::move(r->second));
            pos = r->first;
        }

        return std::pair{pos, std::move(results)};
    }
};

template <typename P>
auto many(P p) -> many_parser<P> {
    return many_parser<P>(std::move(p));
}

// =============================================================================
// Many1 Combinator — One or More
// =============================================================================

/// Like many, but requires at least one successful parse.
template <typename P>
class many1_parser {
    P parser_;

public:
    using output_type = std::vector<typename P::output_type>;

    explicit many1_parser(P p) : parser_(std::move(p)) {}

    template <typename Iter>
    auto parse(Iter begin, Iter end) const -> parse_result<output_type, Iter> {
        auto r = many_parser<P>(parser_).parse(begin, end);
        if (!r || r->second.empty()) return std::nullopt;
        return r;
    }
};

template <typename P>
auto many1(P p) -> many1_parser<P> {
    return many1_parser<P>(std::move(p));
}

// =============================================================================
// Optional Parser
// =============================================================================

/// Makes a parser optional: always succeeds, returning std::optional.
template <typename P>
class optional_parser {
    P parser_;

public:
    using output_type = std::optional<typename P::output_type>;

    explicit optional_parser(P p) : parser_(std::move(p)) {}

    template <typename Iter>
    auto parse(Iter begin, Iter end) const -> parse_result<output_type, Iter> {
        auto r = parser_.parse(begin, end);
        if (r) return std::pair{r->first, output_type{std::move(r->second)}};
        return std::pair{begin, output_type{std::nullopt}};
    }
};

template <typename P>
auto optional_parse(P p) -> optional_parser<P> {
    return optional_parser<P>(std::move(p));
}

// =============================================================================
// SepBy Combinator — Separated Lists
// =============================================================================

/// Parses zero or more occurrences of P separated by Sep.
/// Useful for CSV-like formats: "1,2,3"
template <typename P, typename Sep>
class sep_by_parser {
    P elem_;
    Sep sep_;

public:
    using output_type = std::vector<typename P::output_type>;

    sep_by_parser(P p, Sep s) : elem_(std::move(p)), sep_(std::move(s)) {}

    template <typename Iter>
    auto parse(Iter begin, Iter end) const -> parse_result<output_type, Iter> {
        output_type results;

        // Try to parse the first element
        auto r = elem_.parse(begin, end);
        if (!r) return std::pair{begin, output_type{}};

        results.push_back(std::move(r->second));
        auto pos = r->first;

        // Parse (sep, elem) pairs
        while (true) {
            auto sep_r = sep_.parse(pos, end);
            if (!sep_r) break;

            auto elem_r = elem_.parse(sep_r->first, end);
            if (!elem_r) break;

            results.push_back(std::move(elem_r->second));
            pos = elem_r->first;
        }

        return std::pair{pos, std::move(results)};
    }
};

template <typename P, typename Sep>
auto sep_by(P p, Sep s) -> sep_by_parser<P, Sep> {
    return {std::move(p), std::move(s)};
}

// =============================================================================
// Count Combinator — Exact Repetition
// =============================================================================

/// Parses exactly N occurrences of P.
template <typename P>
class count_parser {
    P parser_;
    std::size_t n_;

public:
    using output_type = std::vector<typename P::output_type>;

    count_parser(std::size_t n, P p) : parser_(std::move(p)), n_(n) {}

    template <typename Iter>
    auto parse(Iter begin, Iter end) const -> parse_result<output_type, Iter> {
        output_type results;
        auto pos = begin;

        for (std::size_t i = 0; i < n_; ++i) {
            auto r = parser_.parse(pos, end);
            if (!r) return std::nullopt;
            results.push_back(std::move(r->second));
            pos = r->first;
        }

        return std::pair{pos, std::move(results)};
    }
};

template <typename P>
auto count(std::size_t n, P p) -> count_parser<P> {
    return {n, std::move(p)};
}

// =============================================================================
// Between Combinator
// =============================================================================

/// Parses: open, then p, then close. Returns only p's result.
template <typename Open, typename P, typename Close>
class between_parser {
    Open open_;
    P parser_;
    Close close_;

public:
    using output_type = typename P::output_type;

    between_parser(Open o, P p, Close c)
        : open_(std::move(o)), parser_(std::move(p)), close_(std::move(c)) {}

    template <typename Iter>
    auto parse(Iter begin, Iter end) const -> parse_result<output_type, Iter> {
        auto r1 = open_.parse(begin, end);
        if (!r1) return std::nullopt;

        auto r2 = parser_.parse(r1->first, end);
        if (!r2) return std::nullopt;

        auto r3 = close_.parse(r2->first, end);
        if (!r3) return std::nullopt;

        return std::pair{r3->first, std::move(r2->second)};
    }
};

template <typename Open, typename P, typename Close>
auto between(Open o, P p, Close c) -> between_parser<Open, P, Close> {
    return {std::move(o), std::move(p), std::move(c)};
}

} // namespace alga
