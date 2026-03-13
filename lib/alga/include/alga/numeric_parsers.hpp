// numeric_parsers.hpp - Type-safe numeric parsing
//
// Numeric types that form monoids under addition with 0 as identity.
// Each type validates its invariant at construction time via factory functions.

#pragma once

#include <charconv>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

namespace alga {

// =============================================================================
// unsigned_int — Non-negative integers (monoid under addition)
// =============================================================================

class unsigned_int {
    unsigned long long value_;

    explicit unsigned_int(unsigned long long v) : value_(v) {}
    friend auto make_unsigned_int(std::string_view) -> std::optional<unsigned_int>;

public:
    unsigned_int() : value_(0) {}

    [[nodiscard]] auto value() const -> unsigned long long { return value_; }

    /// Monoid operation: addition
    [[nodiscard]] auto operator*(const unsigned_int& other) const -> unsigned_int {
        return unsigned_int(value_ + other.value_);
    }

    auto operator<=>(const unsigned_int&) const = default;
    bool operator==(const unsigned_int&) const = default;
};

inline auto make_unsigned_int(std::string_view sv) -> std::optional<unsigned_int> {
    if (sv.empty()) return std::nullopt;

    unsigned long long val = 0;
    auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), val);
    if (ec != std::errc{} || ptr != sv.data() + sv.size()) {
        return std::nullopt;
    }

    return unsigned_int(val);
}

// =============================================================================
// signed_int — Signed integers (group under addition)
// =============================================================================

class signed_int {
    long long value_;

    explicit signed_int(long long v) : value_(v) {}
    friend auto make_signed_int(std::string_view) -> std::optional<signed_int>;

public:
    signed_int() : value_(0) {}

    [[nodiscard]] auto value() const -> long long { return value_; }

    [[nodiscard]] auto operator*(const signed_int& other) const -> signed_int {
        return signed_int(value_ + other.value_);
    }

    auto operator<=>(const signed_int&) const = default;
    bool operator==(const signed_int&) const = default;
};

inline auto make_signed_int(std::string_view sv) -> std::optional<signed_int> {
    if (sv.empty()) return std::nullopt;

    // std::from_chars doesn't accept leading '+', so skip it
    auto start = sv.data();
    auto stop = sv.data() + sv.size();
    if (*start == '+') ++start;
    if (start == stop) return std::nullopt;

    long long val = 0;
    auto [ptr, ec] = std::from_chars(start, stop, val);
    if (ec != std::errc{} || ptr != stop) {
        return std::nullopt;
    }

    return signed_int(val);
}

// =============================================================================
// floating_point — IEEE 754 doubles (approximate field)
// =============================================================================

class floating_point {
    double value_;

    explicit floating_point(double v) : value_(v) {}
    friend auto make_floating_point(std::string_view) -> std::optional<floating_point>;

public:
    floating_point() : value_(0.0) {}

    [[nodiscard]] auto value() const -> double { return value_; }

    [[nodiscard]] auto operator*(const floating_point& other) const -> floating_point {
        return floating_point(value_ + other.value_);
    }

    bool operator==(const floating_point& other) const {
        return std::abs(value_ - other.value_) < std::numeric_limits<double>::epsilon() * 100;
    }
};

inline auto make_floating_point(std::string_view sv) -> std::optional<floating_point> {
    if (sv.empty()) return std::nullopt;

    // std::from_chars for doubles may not be available on all platforms,
    // fall back to stod
    try {
        std::size_t pos = 0;
        double val = std::stod(std::string(sv), &pos);
        if (pos != sv.size()) return std::nullopt;
        return floating_point(val);
    } catch (...) {
        return std::nullopt;
    }
}

} // namespace alga
