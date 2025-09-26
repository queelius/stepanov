#pragma once

#include <concepts>
#include <limits>
#include <stdexcept>
#include <utility>
#include <cmath>
#include <iostream>
#include "concepts.hpp"
#include "gcd.hpp"
#include "math.hpp"
#include "continued_fraction.hpp"
#include "builtin_adaptors.hpp"

namespace stepanov {

/**
 * rational<T> - Generic rational number implementation
 *
 * Represents exact rational numbers as numerator/denominator pairs.
 * Following Alex Stepanov's principles:
 * - Works with any Euclidean domain type
 * - Maintains reduced form (coprime numerator and denominator)
 * - Supports special values: infinities (0 denominator) and epsilon (1/max)
 * - Integrates with continued fraction algorithms
 *
 * Design decisions:
 * - Always maintains reduced form for uniqueness of representation
 * - Sign is always in the numerator (denominator is non-negative)
 * - Zero has unique representation: 0/1
 * - Infinity represented as n/0 where sign(n) determines +/- infinity
 * - Epsilon is the smallest positive rational: 1/largest_value
 */
template <typename T>
    requires integral_domain<T> || std::integral<T>
class rational {
private:
    T num_;   // numerator (carries sign)
    T den_;   // denominator (always non-negative, 0 for infinity)

    // Helper: Reduce to lowest terms
    constexpr void reduce() {
        if (den_ == T(0)) {
            // Infinity: normalize to +/-1/0
            if (num_ != T(0)) {
                num_ = (num_ > T(0)) ? T(1) : T(-1);
            }
            return;
        }

        if (num_ == T(0)) {
            // Zero: normalize to 0/1
            den_ = T(1);
            return;
        }

        // Ensure denominator is positive
        if constexpr (ordered_ring<T>) {
            if (den_ < T(0)) {
                num_ = -num_;
                den_ = -den_;
            }
        }

        // Reduce to lowest terms
        using stepanov::gcd;
        using stepanov::abs;
        using stepanov::quotient;
        T g = gcd(abs(num_), abs(den_));
        num_ = quotient(num_, g);
        den_ = quotient(den_, g);
    }

public:
    using value_type = T;

    // Constructors
    constexpr rational() : num_(T(0)), den_(T(1)) {}

    constexpr rational(T n) : num_(n), den_(T(1)) {}

    constexpr rational(T n, T d) : num_(n), den_(d) {
        if (den_ == T(0) && num_ == T(0)) {
            throw std::invalid_argument("Indeterminate form 0/0");
        }
        reduce();
    }

    // Copy and move constructors
    constexpr rational(const rational&) = default;
    constexpr rational(rational&&) = default;

    constexpr rational& operator=(const rational&) = default;
    constexpr rational& operator=(rational&&) = default;

    // Construct from continued fraction
    template <typename Container>
    explicit constexpr rational(const Container& cf) {
        if (cf.empty()) {
            num_ = T(0);
            den_ = T(1);
            return;
        }

        // Build rational from continued fraction using convergents
        auto convergents = compute_convergents(cf);
        if (!convergents.empty()) {
            auto& last = convergents.back();
            num_ = last.p;
            den_ = last.q;
        } else {
            num_ = T(0);
            den_ = T(1);
        }
    }

    // Accessors
    constexpr T numerator() const { return num_; }
    constexpr T denominator() const { return den_; }

    // Special value checks
    constexpr bool is_zero() const { return num_ == T(0) && den_ != T(0); }
    constexpr bool is_infinity() const { return den_ == T(0); }
    constexpr bool is_positive_infinity() const { return den_ == T(0) && num_ > T(0); }
    constexpr bool is_negative_infinity() const { return den_ == T(0) && num_ < T(0); }
    constexpr bool is_finite() const { return den_ != T(0); }
    constexpr bool is_integer() const { return den_ == T(1) || den_ == T(-1); }

    // Sign operations
    constexpr int sign() const {
        if (num_ == T(0)) return 0;
        if constexpr (ordered_ring<T>) {
            return (num_ > T(0)) ? 1 : -1;
        } else {
            return 1;  // Cannot determine sign without ordering
        }
    }

    // Conversion to continued fraction
    std::vector<T> to_continued_fraction() const {
        if (is_infinity()) {
            throw std::domain_error("Cannot convert infinity to continued fraction");
        }
        return stepanov::to_continued_fraction(num_, den_);
    }

    // Conversion to double (approximate)
    explicit operator double() const {
        if (is_positive_infinity()) return std::numeric_limits<double>::infinity();
        if (is_negative_infinity()) return -std::numeric_limits<double>::infinity();
        return static_cast<double>(num_) / static_cast<double>(den_);
    }

    // Static factory methods for special values
    static constexpr rational zero() { return rational(T(0), T(1)); }
    static constexpr rational one() { return rational(T(1), T(1)); }
    static constexpr rational infinity() { return rational(T(1), T(0)); }
    static constexpr rational negative_infinity() { return rational(T(-1), T(0)); }

    // Epsilon: smallest positive rational (1/max_value if T has max)
    static constexpr rational epsilon() {
        if constexpr (std::numeric_limits<T>::is_specialized) {
            return rational(T(1), std::numeric_limits<T>::max());
        } else {
            return rational(T(1), T(1000000));  // Default large denominator
        }
    }
};

// Comparison operators

template <typename T>
    requires integral_domain<T> || std::integral<T>
constexpr bool operator==(const rational<T>& lhs, const rational<T>& rhs) {
    // Rationals are in reduced form, so direct comparison works
    return lhs.numerator() == rhs.numerator() && lhs.denominator() == rhs.denominator();
}

template <typename T>
    requires integral_domain<T> || std::integral<T>
constexpr bool operator!=(const rational<T>& lhs, const rational<T>& rhs) {
    return !(lhs == rhs);
}

template <typename T>
    requires (integral_domain<T> || std::integral<T>) && totally_ordered<T>
constexpr bool operator<(const rational<T>& lhs, const rational<T>& rhs) {
    // Handle infinities
    if (lhs.is_negative_infinity()) return !rhs.is_negative_infinity();
    if (rhs.is_negative_infinity()) return false;
    if (lhs.is_positive_infinity()) return false;
    if (rhs.is_positive_infinity()) return !lhs.is_positive_infinity();

    // Both finite: compare cross products
    // a/b < c/d iff a*d < b*c (when b, d > 0)
    return lhs.numerator() * rhs.denominator() < rhs.numerator() * lhs.denominator();
}

template <typename T>
    requires (integral_domain<T> || std::integral<T>) && totally_ordered<T>
constexpr bool operator<=(const rational<T>& lhs, const rational<T>& rhs) {
    return !(rhs < lhs);
}

template <typename T>
    requires (integral_domain<T> || std::integral<T>) && totally_ordered<T>
constexpr bool operator>(const rational<T>& lhs, const rational<T>& rhs) {
    return rhs < lhs;
}

template <typename T>
    requires (integral_domain<T> || std::integral<T>) && totally_ordered<T>
constexpr bool operator>=(const rational<T>& lhs, const rational<T>& rhs) {
    return !(lhs < rhs);
}

// Arithmetic operations

template <typename T>
    requires integral_domain<T> || std::integral<T>
constexpr rational<T> operator+(const rational<T>& lhs, const rational<T>& rhs) {
    // Handle infinities
    if (lhs.is_infinity() || rhs.is_infinity()) {
        if (lhs.is_positive_infinity() && rhs.is_negative_infinity()) {
            throw std::domain_error("Indeterminate form: inf + (-inf)");
        }
        if (lhs.is_negative_infinity() && rhs.is_positive_infinity()) {
            throw std::domain_error("Indeterminate form: (-inf) + inf");
        }
        return lhs.is_infinity() ? lhs : rhs;
    }

    // a/b + c/d = (a*d + b*c) / (b*d)
    T num = lhs.numerator() * rhs.denominator() + rhs.numerator() * lhs.denominator();
    T den = lhs.denominator() * rhs.denominator();
    return rational<T>(num, den);
}

template <typename T>
    requires integral_domain<T> || std::integral<T>
constexpr rational<T> operator-(const rational<T>& lhs, const rational<T>& rhs) {
    // Handle infinities
    if (lhs.is_infinity() || rhs.is_infinity()) {
        if (lhs.is_positive_infinity() && rhs.is_positive_infinity()) {
            throw std::domain_error("Indeterminate form: inf - inf");
        }
        if (lhs.is_negative_infinity() && rhs.is_negative_infinity()) {
            throw std::domain_error("Indeterminate form: (-inf) - (-inf)");
        }
        if (lhs.is_infinity()) return lhs;
        return rational<T>(-rhs.numerator(), rhs.denominator());
    }

    // a/b - c/d = (a*d - b*c) / (b*d)
    T num = lhs.numerator() * rhs.denominator() - rhs.numerator() * lhs.denominator();
    T den = lhs.denominator() * rhs.denominator();
    return rational<T>(num, den);
}

template <typename T>
    requires integral_domain<T> || std::integral<T>
constexpr rational<T> operator*(const rational<T>& lhs, const rational<T>& rhs) {
    // Handle infinities
    if (lhs.is_infinity() || rhs.is_infinity()) {
        if (lhs.is_zero() || rhs.is_zero()) {
            throw std::domain_error("Indeterminate form: 0 * inf");
        }
        T sign_num = T(1);
        if constexpr (ordered_ring<T>) {
            bool neg = (lhs.sign() < 0) != (rhs.sign() < 0);
            sign_num = neg ? T(-1) : T(1);
        }
        return rational<T>(sign_num, T(0));
    }

    // (a/b) * (c/d) = (a*c) / (b*d)
    T num = lhs.numerator() * rhs.numerator();
    T den = lhs.denominator() * rhs.denominator();
    return rational<T>(num, den);
}

template <typename T>
    requires integral_domain<T> || std::integral<T>
constexpr rational<T> operator/(const rational<T>& lhs, const rational<T>& rhs) {
    if (rhs.is_zero()) {
        if (lhs.is_zero()) {
            throw std::domain_error("Indeterminate form: 0/0");
        }
        return lhs.sign() > 0 ? rational<T>::infinity() : rational<T>::negative_infinity();
    }

    // Handle infinities
    if (lhs.is_infinity() && rhs.is_infinity()) {
        throw std::domain_error("Indeterminate form: inf/inf");
    }
    if (rhs.is_infinity()) {
        return rational<T>::zero();
    }

    // (a/b) / (c/d) = (a*d) / (b*c)
    T num = lhs.numerator() * rhs.denominator();
    T den = lhs.denominator() * rhs.numerator();
    return rational<T>(num, den);
}

template <typename T>
    requires integral_domain<T> || std::integral<T>
constexpr rational<T> operator-(const rational<T>& r) {
    return rational<T>(-r.numerator(), r.denominator());
}

// Absolute value
template <typename T>
    requires (integral_domain<T> || std::integral<T>) && totally_ordered<T>
constexpr rational<T> abs(const rational<T>& r) {
    return rational<T>(stepanov::abs(r.numerator()), r.denominator());
}

// Min/max operations
template <typename T>
    requires (integral_domain<T> || std::integral<T>) && totally_ordered<T>
constexpr rational<T> min(const rational<T>& a, const rational<T>& b) {
    return (b < a) ? b : a;
}

template <typename T>
    requires (integral_domain<T> || std::integral<T>) && totally_ordered<T>
constexpr rational<T> max(const rational<T>& a, const rational<T>& b) {
    return (a < b) ? b : a;
}

// Infimum and supremum for sets (represented as initializer lists)
template <typename T>
    requires (integral_domain<T> || std::integral<T>) && totally_ordered<T>
constexpr rational<T> infimum(std::initializer_list<rational<T>> values) {
    if (values.size() == 0) {
        return rational<T>::infinity();  // Empty set infimum is +inf
    }
    auto it = values.begin();
    rational<T> result = *it++;
    while (it != values.end()) {
        result = min(result, *it++);
    }
    return result;
}

template <typename T>
    requires (integral_domain<T> || std::integral<T>) && totally_ordered<T>
constexpr rational<T> supremum(std::initializer_list<rational<T>> values) {
    if (values.size() == 0) {
        return rational<T>::negative_infinity();  // Empty set supremum is -inf
    }
    auto it = values.begin();
    rational<T> result = *it++;
    while (it != values.end()) {
        result = max(result, *it++);
    }
    return result;
}

// Fundamental operations for generic algorithms

template <typename T>
    requires integral_domain<T> || std::integral<T>
constexpr bool even(const rational<T>& r) {
    // A rational is "even" if it's an even integer
    return r.is_integer() && even(r.numerator());
}

template <typename T>
    requires integral_domain<T> || std::integral<T>
constexpr rational<T> twice(const rational<T>& r) {
    return rational<T>(twice(r.numerator()), r.denominator());
}

template <typename T>
    requires integral_domain<T> || std::integral<T>
constexpr rational<T> half(const rational<T>& r) {
    return rational<T>(r.numerator(), twice(r.denominator()));
}

template <typename T>
    requires integral_domain<T> || std::integral<T>
constexpr rational<T> increment(const rational<T>& r) {
    return r + rational<T>::one();
}

template <typename T>
    requires integral_domain<T> || std::integral<T>
constexpr rational<T> decrement(const rational<T>& r) {
    return r - rational<T>::one();
}

// Field operations (rational numbers form a field)

template <typename T>
    requires integral_domain<T> || std::integral<T>
constexpr rational<T> reciprocal(const rational<T>& r) {
    if (r.is_zero()) {
        throw std::domain_error("Reciprocal of zero");
    }
    return rational<T>(r.denominator(), r.numerator());
}

// Power operation for rational exponents (returns double for now)
template <typename T>
    requires (integral_domain<T> || std::integral<T>) && totally_ordered<T>
double power(const rational<T>& base, const rational<T>& exp) {
    if (base.is_zero() && exp.is_zero()) {
        throw std::domain_error("Indeterminate form: 0^0");
    }
    if (base.is_zero()) return 0.0;
    if (exp.is_zero()) return 1.0;

    // For rational exponents, we need to use floating point
    // base^(p/q) = (base^p)^(1/q) = qth root of (base^p)
    double b = static_cast<double>(base);
    double e = static_cast<double>(exp);
    return std::pow(b, e);
}

// Mediant operation (important for Farey sequences and continued fractions)
template <typename T>
    requires integral_domain<T> || std::integral<T>
constexpr rational<T> mediant(const rational<T>& lhs, const rational<T>& rhs) {
    if (lhs.is_infinity() || rhs.is_infinity()) {
        throw std::domain_error("Mediant with infinity");
    }
    // Mediant of a/b and c/d is (a+c)/(b+d)
    T num = lhs.numerator() + rhs.numerator();
    T den = lhs.denominator() + rhs.denominator();
    return rational<T>(num, den);
}

// Best rational approximation with bounded denominator
template <typename T>
    requires (integral_domain<T> || std::integral<T>) && totally_ordered<T>
constexpr rational<T> best_approximation(const rational<T>& r, T max_denominator) {
    if (r.is_infinity()) {
        return r;
    }

    auto cf = r.to_continued_fraction();
    auto convergents = compute_convergents(cf);

    // Find the last convergent with denominator <= max_denominator
    rational<T> best = rational<T>::zero();
    for (const auto& conv : convergents) {
        if (stepanov::norm(conv.q) <= stepanov::norm(max_denominator)) {
            best = rational<T>(conv.p, conv.q);
        } else {
            break;
        }
    }

    return best;
}

// Floor and ceiling operations
template <typename T>
    requires (integral_domain<T> || std::integral<T>) && totally_ordered<T>
constexpr T floor(const rational<T>& r) {
    if (r.is_infinity()) {
        throw std::domain_error("Floor of infinity");
    }
    return stepanov::quotient(r.numerator(), r.denominator());
}

template <typename T>
    requires (integral_domain<T> || std::integral<T>) && totally_ordered<T>
constexpr T ceil(const rational<T>& r) {
    if (r.is_infinity()) {
        throw std::domain_error("Ceiling of infinity");
    }
    T q = stepanov::quotient(r.numerator(), r.denominator());
    T rem = stepanov::remainder(r.numerator(), r.denominator());
    return (rem == T(0)) ? q : q + T(1);
}

// Fractional part
template <typename T>
    requires (integral_domain<T> || std::integral<T>) && totally_ordered<T>
constexpr rational<T> frac(const rational<T>& r) {
    if (r.is_infinity()) {
        throw std::domain_error("Fractional part of infinity");
    }
    T q = stepanov::quotient(r.numerator(), r.denominator());
    return r - rational<T>(q);
}

// Stream output operator
template <typename T>
    requires integral_domain<T> || std::integral<T>
std::ostream& operator<<(std::ostream& os, const rational<T>& r) {
    if (r.is_positive_infinity()) {
        os << "+inf";
    } else if (r.is_negative_infinity()) {
        os << "-inf";
    } else if (r.denominator() == T(1)) {
        os << r.numerator();
    } else {
        os << r.numerator() << "/" << r.denominator();
    }
    return os;
}

} // namespace stepanov