#pragma once

/**
 * @file rational.hpp
 * @brief Exact Rational Arithmetic
 *
 * A minimal implementation of rational numbers (fractions) that maintains
 * exact precision. Unlike floating-point, rationals never lose precision
 * through representation—1/3 stays exactly 1/3.
 *
 * Key design decisions:
 * - Always reduced to lowest terms (coprime numerator/denominator)
 * - Denominator always positive (sign carried by numerator)
 * - Zero uniquely represented as 0/1
 *
 * The reduction uses Euclid's GCD algorithm—the same algorithm that
 * appears in peasant.hpp for integer types.
 */

#include <concepts>
#include <stdexcept>
#include <iostream>
#include <numeric>  // for std::gcd

namespace rational {

template<std::integral T>
class rat {
    T num_;  // numerator (carries sign)
    T den_;  // denominator (always positive)

    // Reduce to lowest terms
    constexpr void reduce() {
        if (den_ == 0)
            throw std::domain_error("Division by zero");

        if (num_ == 0) {
            den_ = 1;
            return;
        }

        // Ensure denominator is positive
        if (den_ < 0) {
            num_ = -num_;
            den_ = -den_;
        }

        // Reduce using GCD
        T g = std::gcd(num_ < 0 ? -num_ : num_, den_);
        num_ /= g;
        den_ /= g;
    }

public:
    // Constructors
    constexpr rat() : num_(0), den_(1) {}
    constexpr rat(T n) : num_(n), den_(1) {}
    constexpr rat(T n, T d) : num_(n), den_(d) { reduce(); }

    // Accessors
    constexpr T numerator() const { return num_; }
    constexpr T denominator() const { return den_; }

    // Predicates
    constexpr bool is_zero() const { return num_ == 0; }
    constexpr bool is_integer() const { return den_ == 1; }
    constexpr bool is_positive() const { return num_ > 0; }
    constexpr bool is_negative() const { return num_ < 0; }

    // Conversion
    explicit constexpr operator double() const {
        return static_cast<double>(num_) / static_cast<double>(den_);
    }

    // Arithmetic operators
    constexpr rat operator-() const { return rat(-num_, den_); }

    constexpr rat operator+(rat const& rhs) const {
        return rat(num_ * rhs.den_ + rhs.num_ * den_, den_ * rhs.den_);
    }

    constexpr rat operator-(rat const& rhs) const {
        return rat(num_ * rhs.den_ - rhs.num_ * den_, den_ * rhs.den_);
    }

    constexpr rat operator*(rat const& rhs) const {
        return rat(num_ * rhs.num_, den_ * rhs.den_);
    }

    constexpr rat operator/(rat const& rhs) const {
        if (rhs.num_ == 0)
            throw std::domain_error("Division by zero");
        return rat(num_ * rhs.den_, den_ * rhs.num_);
    }

    // Compound assignment
    constexpr rat& operator+=(rat const& rhs) { return *this = *this + rhs; }
    constexpr rat& operator-=(rat const& rhs) { return *this = *this - rhs; }
    constexpr rat& operator*=(rat const& rhs) { return *this = *this * rhs; }
    constexpr rat& operator/=(rat const& rhs) { return *this = *this / rhs; }

    // Comparison (exact, no floating-point!)
    constexpr bool operator==(rat const& rhs) const {
        return num_ == rhs.num_ && den_ == rhs.den_;
    }

    constexpr auto operator<=>(rat const& rhs) const {
        // a/b <=> c/d  iff  a*d <=> c*b (when b,d > 0)
        return num_ * rhs.den_ <=> rhs.num_ * den_;
    }

    // Reciprocal
    constexpr rat reciprocal() const {
        if (num_ == 0)
            throw std::domain_error("Reciprocal of zero");
        return rat(den_, num_);
    }
};

// Stream output
template<std::integral T>
std::ostream& operator<<(std::ostream& os, rat<T> const& r) {
    os << r.numerator();
    if (r.denominator() != 1)
        os << "/" << r.denominator();
    return os;
}

// Absolute value
template<std::integral T>
constexpr rat<T> abs(rat<T> const& r) {
    return r.is_negative() ? -r : r;
}

// Mediant: (a/b) ⊕ (c/d) = (a+c)/(b+d)
// Important in Stern-Brocot tree and Farey sequences
template<std::integral T>
constexpr rat<T> mediant(rat<T> const& lhs, rat<T> const& rhs) {
    return rat<T>(lhs.numerator() + rhs.numerator(),
                  lhs.denominator() + rhs.denominator());
}

// Common type aliases
using rational = rat<int>;
using rational64 = rat<int64_t>;

} // namespace rational
