// lc_alpha.hpp - Lowercase alphabetic strings as a free monoid
//
// The free monoid over the alphabet {a..z}: strings under concatenation.
// This is the foundational algebraic type in alga.
//
// Monoid laws:
//   Identity:      "" * s == s == s * ""
//   Associativity: (a * b) * c == a * (b * c)
//
// The type enforces its invariant (lowercase-only) via a factory function.

#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace alga {

/// Lowercase alphabetic string — a free monoid under concatenation.
///
/// Invariant: contains only lowercase ASCII letters [a-z] or is empty.
/// Construction is via make_lc_alpha(), which validates the invariant.
class lc_alpha {
    std::string data_;

    // Private constructor — use make_lc_alpha() factory
    explicit lc_alpha(std::string s) : data_(std::move(s)) {}

    friend auto make_lc_alpha(std::string_view sv) -> std::optional<lc_alpha>;

public:
    /// Default constructor creates the identity element (empty string)
    lc_alpha() = default;

    /// Monoid operation: concatenation
    [[nodiscard]] auto operator*(const lc_alpha& other) const -> lc_alpha {
        return lc_alpha(data_ + other.data_);
    }

    /// Access the underlying string
    [[nodiscard]] auto str() const -> const std::string& { return data_; }

    /// Length in characters
    [[nodiscard]] auto size() const -> std::size_t { return data_.size(); }

    /// Is this the identity element?
    [[nodiscard]] auto empty() const -> bool { return data_.empty(); }

    // Comparison operators
    auto operator<=>(const lc_alpha&) const = default;
    bool operator==(const lc_alpha&) const = default;
};

/// Factory function: validates that all characters are lowercase ASCII.
/// Returns std::nullopt if validation fails.
inline auto make_lc_alpha(std::string_view sv) -> std::optional<lc_alpha> {
    for (char c : sv) {
        if (c < 'a' || c > 'z') {
            return std::nullopt;
        }
    }
    return lc_alpha(std::string(sv));
}

} // namespace alga
