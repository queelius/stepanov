#pragma once

/**
 * @file str_repeat.hpp
 * @brief String repetition via the peasant algorithm
 *
 * power("ab", 1000) builds "abab...ab" (2000 chars) in O(log n) concatenations.
 *
 * Instead of 1000 concatenations, we do ~10:
 *   "ab" → "abab" → "abababab" → ... → "ab"×1000
 *
 * This is the same algorithm, just with string concatenation as the operation.
 */

#include <string>
#include <compare>

namespace peasant::examples {

// Wrapper to give std::string the right algebraic interface
struct str_rep {
    std::string s;

    str_rep() : s() {}
    str_rep(std::string str) : s(std::move(str)) {}
    str_rep(const char* str) : s(str) {}

    bool operator==(str_rep const& o) const { return s == o.s; }
    auto operator<=>(str_rep const& o) const { return s <=> o.s; }

    // "Multiplication" is concatenation
    str_rep operator*(str_rep const& o) const { return str_rep{s + o.s}; }

    // Required for algebraic concept
    str_rep operator+(str_rep const& o) const { return *this * o; }
    str_rep operator-() const { return *this; }

    size_t size() const { return s.size(); }
    const std::string& str() const { return s; }
};

// ADL functions
inline str_rep zero(str_rep) { return str_rep{""}; }
inline str_rep one(str_rep)  { return str_rep{""}; }  // Empty string is identity for concat

inline str_rep twice(str_rep const& x) { return x * x; }
inline str_rep half(str_rep const& x)  { return x; }
inline bool even(str_rep const&)       { return true; }  // Force multiply path

inline str_rep increment(str_rep const& x) { return x; }
inline str_rep decrement(str_rep const& x) { return x; }

} // namespace peasant::examples
