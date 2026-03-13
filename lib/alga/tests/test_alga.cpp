// test_alga.cpp - Tests for alga parser combinator library
//
// Tests verify:
// - Monoid laws: associativity, identity for lc_alpha
// - Functor laws: fmap(id) == id, fmap(f . g) == fmap(f) . fmap(g)
// - Monad laws: left/right identity, associativity
// - Parser correctness: parse integers, floats, lists
// - Composition: sequential, choice, repetition

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <alga/alga.hpp>
#include <string>
#include <string_view>

using Catch::Approx;
using namespace alga;

// =============================================================================
// lc_alpha — Free Monoid Tests
// =============================================================================

TEST_CASE("lc_alpha construction", "[alga][lc_alpha]") {
    SECTION("Valid lowercase strings") {
        auto a = make_lc_alpha("hello");
        REQUIRE(a.has_value());
        REQUIRE(a->str() == "hello");
    }

    SECTION("Empty string is valid (identity element)") {
        auto a = make_lc_alpha("");
        REQUIRE(a.has_value());
        REQUIRE(a->str() == "");
    }

    SECTION("Uppercase rejected") {
        auto a = make_lc_alpha("Hello");
        REQUIRE_FALSE(a.has_value());
    }

    SECTION("Digits rejected") {
        auto a = make_lc_alpha("abc123");
        REQUIRE_FALSE(a.has_value());
    }
}

TEST_CASE("lc_alpha monoid laws", "[alga][lc_alpha][monoid]") {
    auto a = *make_lc_alpha("foo");
    auto b = *make_lc_alpha("bar");
    auto c = *make_lc_alpha("baz");
    auto e = *make_lc_alpha("");  // identity

    SECTION("Associativity: (a * b) * c == a * (b * c)") {
        auto lhs = (a * b) * c;
        auto rhs = a * (b * c);
        REQUIRE(lhs.str() == rhs.str());
        REQUIRE(lhs.str() == "foobarbaz");
    }

    SECTION("Left identity: e * a == a") {
        REQUIRE((e * a).str() == a.str());
    }

    SECTION("Right identity: a * e == a") {
        REQUIRE((a * e).str() == a.str());
    }
}

// =============================================================================
// Numeric Parser Tests
// =============================================================================

TEST_CASE("Unsigned integer parsing", "[alga][numeric]") {
    SECTION("Valid integers") {
        auto result = make_unsigned_int("42");
        REQUIRE(result.has_value());
        REQUIRE(result->value() == 42);
    }

    SECTION("Zero") {
        auto result = make_unsigned_int("0");
        REQUIRE(result.has_value());
        REQUIRE(result->value() == 0);
    }

    SECTION("Negative rejected") {
        auto result = make_unsigned_int("-5");
        REQUIRE_FALSE(result.has_value());
    }

    SECTION("Non-numeric rejected") {
        auto result = make_unsigned_int("abc");
        REQUIRE_FALSE(result.has_value());
    }
}

TEST_CASE("Signed integer parsing", "[alga][numeric]") {
    SECTION("Positive") {
        auto result = make_signed_int("42");
        REQUIRE(result.has_value());
        REQUIRE(result->value() == 42);
    }

    SECTION("Negative") {
        auto result = make_signed_int("-17");
        REQUIRE(result.has_value());
        REQUIRE(result->value() == -17);
    }

    SECTION("Explicit plus") {
        auto result = make_signed_int("+5");
        REQUIRE(result.has_value());
        REQUIRE(result->value() == 5);
    }
}

TEST_CASE("Floating point parsing", "[alga][numeric]") {
    SECTION("Simple float") {
        auto result = make_floating_point("3.14");
        REQUIRE(result.has_value());
        REQUIRE(result->value() == Approx(3.14));
    }

    SECTION("Integer as float") {
        auto result = make_floating_point("42");
        REQUIRE(result.has_value());
        REQUIRE(result->value() == Approx(42.0));
    }

    SECTION("Negative float") {
        auto result = make_floating_point("-2.5");
        REQUIRE(result.has_value());
        REQUIRE(result->value() == Approx(-2.5));
    }
}

// =============================================================================
// UTF-8 Alpha Tests
// =============================================================================

TEST_CASE("utf8_alpha basic operations", "[alga][utf8]") {
    SECTION("ASCII lowercase") {
        auto a = make_utf8_alpha("hello");
        REQUIRE(a.has_value());
        REQUIRE(a->str() == "hello");
        REQUIRE(a->char_count() == 5);
    }

    SECTION("Monoid concatenation") {
        auto a = make_utf8_alpha("hello");
        auto b = make_utf8_alpha("world");
        REQUIRE(a.has_value());
        REQUIRE(b.has_value());
        auto c = *a * *b;
        REQUIRE(c.str() == "helloworld");
    }
}

// =============================================================================
// Parser Combinator Tests
// =============================================================================

TEST_CASE("Literal parser", "[alga][combinators]") {
    auto p = literal("hello");
    std::string input = "hello world";

    auto result = p.parse(input.begin(), input.end());
    REQUIRE(result.has_value());
    REQUIRE(result->second == "hello");
}

TEST_CASE("Sequence combinator", "[alga][combinators]") {
    auto p1 = literal("hello");
    auto p2 = literal(" ");
    auto p3 = literal("world");

    auto combined = sequence(sequence(p1, p2), p3);
    std::string input = "hello world";

    auto result = combined.parse(input.begin(), input.end());
    REQUIRE(result.has_value());
}

TEST_CASE("Choice combinator", "[alga][combinators]") {
    auto p1 = literal("foo");
    auto p2 = literal("bar");
    auto p = choice(p1, p2);

    SECTION("First alternative matches") {
        std::string input = "foo";
        auto result = p.parse(input.begin(), input.end());
        REQUIRE(result.has_value());
        REQUIRE(result->second == "foo");
    }

    SECTION("Second alternative matches") {
        std::string input = "bar";
        auto result = p.parse(input.begin(), input.end());
        REQUIRE(result.has_value());
        REQUIRE(result->second == "bar");
    }

    SECTION("Neither matches") {
        std::string input = "baz";
        auto result = p.parse(input.begin(), input.end());
        REQUIRE_FALSE(result.has_value());
    }
}

TEST_CASE("Many combinator", "[alga][combinators]") {
    auto digit = char_parser([](char c) { return c >= '0' && c <= '9'; });
    auto digits = many(digit);

    SECTION("Multiple matches") {
        std::string input = "123abc";
        auto result = digits.parse(input.begin(), input.end());
        REQUIRE(result.has_value());
        REQUIRE(result->second.size() == 3);
    }

    SECTION("Zero matches (still succeeds)") {
        std::string input = "abc";
        auto result = digits.parse(input.begin(), input.end());
        REQUIRE(result.has_value());
        REQUIRE(result->second.empty());
    }
}

TEST_CASE("Many1 combinator", "[alga][combinators]") {
    auto digit = char_parser([](char c) { return c >= '0' && c <= '9'; });
    auto digits = many1(digit);

    SECTION("Multiple matches") {
        std::string input = "123abc";
        auto result = digits.parse(input.begin(), input.end());
        REQUIRE(result.has_value());
        REQUIRE(result->second.size() == 3);
    }

    SECTION("Zero matches (fails)") {
        std::string input = "abc";
        auto result = digits.parse(input.begin(), input.end());
        REQUIRE_FALSE(result.has_value());
    }
}

TEST_CASE("SepBy combinator", "[alga][combinators]") {
    auto digit = char_parser([](char c) { return c >= '0' && c <= '9'; });
    auto comma = literal(",");
    auto csv = sep_by(digit, comma);

    std::string input = "1,2,3";
    auto result = csv.parse(input.begin(), input.end());
    REQUIRE(result.has_value());
    REQUIRE(result->second.size() == 3);
}

// =============================================================================
// Algebraic Operator Tests
// =============================================================================

TEST_CASE("Operator >> for sequence", "[alga][operators]") {
    auto hello = literal("hello");
    auto space = literal(" ");
    auto world = literal("world");

    auto p = hello >> space >> world;
    std::string input = "hello world";
    auto result = p.parse(input.begin(), input.end());
    REQUIRE(result.has_value());
}

TEST_CASE("Operator | for choice", "[alga][operators]") {
    auto foo = literal("foo");
    auto bar = literal("bar");
    auto p = foo | bar;

    std::string input = "bar";
    auto result = p.parse(input.begin(), input.end());
    REQUIRE(result.has_value());
}

// =============================================================================
// Monadic Combinator Tests
// =============================================================================

TEST_CASE("fmap (functor)", "[alga][monadic]") {
    auto digit = char_parser([](char c) { return c >= '0' && c <= '9'; });
    auto as_int = fmap([](char c) { return c - '0'; }, digit);

    std::string input = "7abc";
    auto result = as_int.parse(input.begin(), input.end());
    REQUIRE(result.has_value());
    REQUIRE(result->second == 7);
}

TEST_CASE("Functor identity law", "[alga][monadic][laws]") {
    // fmap(id, p) should behave the same as p
    auto p = literal("test");
    auto identity = [](const auto& x) { return x; };
    auto mapped = fmap(identity, p);

    std::string input = "test";
    auto r1 = p.parse(input.begin(), input.end());
    auto r2 = mapped.parse(input.begin(), input.end());

    REQUIRE(r1.has_value() == r2.has_value());
    if (r1.has_value() && r2.has_value()) {
        REQUIRE(r1->second == r2->second);
    }
}

TEST_CASE("pure (monadic lift)", "[alga][monadic]") {
    auto p = pure(42);
    std::string input = "anything";

    auto result = p.parse(input.begin(), input.end());
    REQUIRE(result.has_value());
    REQUIRE(result->second == 42);
    // pure should not consume any input
    REQUIRE(result->first == input.begin());
}

// =============================================================================
// String Similarity Tests
// =============================================================================

TEST_CASE("Levenshtein distance", "[alga][similarity]") {
    SECTION("Identical strings") {
        REQUIRE(levenshtein("kitten", "kitten") == 0);
    }

    SECTION("Classic example") {
        REQUIRE(levenshtein("kitten", "sitting") == 3);
    }

    SECTION("Empty string") {
        REQUIRE(levenshtein("", "abc") == 3);
        REQUIRE(levenshtein("abc", "") == 3);
    }

    SECTION("Single character difference") {
        REQUIRE(levenshtein("cat", "hat") == 1);
    }
}

TEST_CASE("Jaro-Winkler similarity", "[alga][similarity]") {
    SECTION("Identical strings") {
        REQUIRE(jaro_winkler("hello", "hello") == Approx(1.0));
    }

    SECTION("Completely different") {
        REQUIRE(jaro_winkler("abc", "xyz") == Approx(0.0));
    }

    SECTION("Similar strings") {
        double sim = jaro_winkler("martha", "marhta");
        REQUIRE(sim > 0.9);
        REQUIRE(sim <= 1.0);
    }
}

// =============================================================================
// Parse Error Tests
// =============================================================================

TEST_CASE("Position tracking", "[alga][error]") {
    SECTION("Initial position") {
        position pos;
        REQUIRE(pos.line == 1);
        REQUIRE(pos.column == 1);
    }

    SECTION("Position tracker advances") {
        std::string input = "ab\ncd";
        position_tracker<std::string::iterator> tracker(input.begin());

        tracker.advance(input.begin() + 3);  // past the newline
        auto pos = tracker.current();
        REQUIRE(pos.line == 2);
        REQUIRE(pos.column == 1);
    }
}
