#pragma once

#include <concepts>
#include <limits>
#include <cmath>
#include <string>
#include <sstream>
#include <iomanip>
#include "concepts.hpp"
#include "math.hpp"

namespace stepanov {

/**
 * fixed_decimal<T, Scale> - Fixed-point decimal arithmetic
 *
 * Represents decimal numbers with a fixed number of decimal places.
 * Internally stores the value as an integer scaled by 10^Scale.
 *
 * Following Alex Stepanov's principles:
 * - Generic over the underlying integer type
 * - Efficient operations using integer arithmetic
 * - Composable with all generic algorithms
 * - Clear semantics for overflow and rounding
 *
 * Design decisions:
 * - Scale is a compile-time constant for efficiency
 * - Rounding modes follow IEEE 754 conventions
 * - Overflow behavior depends on underlying type T
 * - Supports all required operations for field concept
 *
 * Trade-offs vs floating point:
 * + Exact decimal arithmetic (no binary rounding errors)
 * + Predictable precision and range
 * + Faster on systems without FPU
 * + Deterministic across platforms
 * - Limited range compared to floating point
 * - Fixed precision (can't adapt to magnitude)
 * - More expensive division and transcendental functions
 */
template <std::integral T, size_t Scale>
    requires (Scale > 0)
class fixed_decimal {
private:
    T value_;  // Scaled integer representation

    static constexpr T scale_factor() {
        T result = 1;
        for (size_t i = 0; i < Scale; ++i) {
            result *= 10;
        }
        return result;
    }

    static constexpr T half_scale() {
        return scale_factor() / 2;
    }

public:
    using value_type = T;
    static constexpr size_t decimal_places = Scale;
    static constexpr T scaling_factor = scale_factor();

    // Constructors
    constexpr fixed_decimal() : value_(0) {}

    // Construct from integral type
    template <std::integral U>
    constexpr explicit fixed_decimal(U n) : value_(static_cast<T>(n) * scaling_factor) {
        // Check for overflow in multiplication
        if constexpr (std::numeric_limits<T>::is_bounded) {
            if (n != 0 && value_ / n != scaling_factor) {
                throw std::overflow_error("Overflow in fixed_decimal construction");
            }
        }
    }

    // Construct from floating point with rounding
    constexpr explicit fixed_decimal(double d) {
        if (std::isnan(d)) {
            throw std::invalid_argument("Cannot convert NaN to fixed_decimal");
        }
        if (std::isinf(d)) {
            throw std::overflow_error("Cannot convert infinity to fixed_decimal");
        }

        // Round to nearest
        double scaled = d * scaling_factor;
        value_ = static_cast<T>(scaled + (scaled >= 0 ? 0.5 : -0.5));
    }

    // Construct from raw scaled value
    static constexpr fixed_decimal from_raw(T raw_value) {
        fixed_decimal result;
        result.value_ = raw_value;
        return result;
    }

    // Copy and move constructors
    constexpr fixed_decimal(const fixed_decimal&) = default;
    constexpr fixed_decimal(fixed_decimal&&) = default;

    constexpr fixed_decimal& operator=(const fixed_decimal&) = default;
    constexpr fixed_decimal& operator=(fixed_decimal&&) = default;

    // Accessors
    constexpr T raw_value() const { return value_; }
    constexpr T integer_part() const { return value_ / scaling_factor; }
    constexpr T fractional_part() const { return value_ % scaling_factor; }

    // Conversion to floating point
    constexpr explicit operator double() const {
        return static_cast<double>(value_) / scaling_factor;
    }

    constexpr explicit operator float() const {
        return static_cast<float>(value_) / scaling_factor;
    }

    // Conversion to integral type (truncates)
    template <std::integral U>
    constexpr explicit operator U() const {
        return static_cast<U>(integer_part());
    }

    // String conversion
    std::string to_string() const {
        std::stringstream ss;
        T int_part = integer_part();
        T frac_part = std::abs(fractional_part());

        if (value_ < 0 && int_part == 0) {
            ss << '-';  // Handle -0.xxx case
        }
        ss << int_part << '.';
        ss << std::setfill('0') << std::setw(Scale) << frac_part;
        return ss.str();
    }

    // Special value checks
    constexpr bool is_zero() const { return value_ == 0; }
    constexpr bool is_positive() const { return value_ > 0; }
    constexpr bool is_negative() const { return value_ < 0; }

    // Sign operations
    constexpr int sign() const {
        if (value_ == 0) return 0;
        return (value_ > 0) ? 1 : -1;
    }

    // Static factory methods
    static constexpr fixed_decimal zero() { return fixed_decimal(); }
    static constexpr fixed_decimal one() { return fixed_decimal(1); }
    static constexpr fixed_decimal epsilon() { return from_raw(1); }

    static constexpr fixed_decimal min() {
        return from_raw(std::numeric_limits<T>::min());
    }

    static constexpr fixed_decimal max() {
        return from_raw(std::numeric_limits<T>::max());
    }
};

// Comparison operators

template <std::integral T, size_t Scale>
constexpr bool operator==(const fixed_decimal<T, Scale>& lhs, const fixed_decimal<T, Scale>& rhs) {
    return lhs.raw_value() == rhs.raw_value();
}

template <std::integral T, size_t Scale>
constexpr bool operator!=(const fixed_decimal<T, Scale>& lhs, const fixed_decimal<T, Scale>& rhs) {
    return !(lhs == rhs);
}

template <std::integral T, size_t Scale>
constexpr bool operator<(const fixed_decimal<T, Scale>& lhs, const fixed_decimal<T, Scale>& rhs) {
    return lhs.raw_value() < rhs.raw_value();
}

template <std::integral T, size_t Scale>
constexpr bool operator<=(const fixed_decimal<T, Scale>& lhs, const fixed_decimal<T, Scale>& rhs) {
    return !(rhs < lhs);
}

template <std::integral T, size_t Scale>
constexpr bool operator>(const fixed_decimal<T, Scale>& lhs, const fixed_decimal<T, Scale>& rhs) {
    return rhs < lhs;
}

template <std::integral T, size_t Scale>
constexpr bool operator>=(const fixed_decimal<T, Scale>& lhs, const fixed_decimal<T, Scale>& rhs) {
    return !(lhs < rhs);
}

// Arithmetic operations

template <std::integral T, size_t Scale>
constexpr fixed_decimal<T, Scale> operator+(const fixed_decimal<T, Scale>& lhs,
                                           const fixed_decimal<T, Scale>& rhs) {
    // Check for overflow
    if constexpr (std::numeric_limits<T>::is_bounded) {
        T result = lhs.raw_value() + rhs.raw_value();
        // Simple overflow check (not perfect but reasonable)
        if ((rhs.raw_value() > 0 && result < lhs.raw_value()) ||
            (rhs.raw_value() < 0 && result > lhs.raw_value())) {
            throw std::overflow_error("Overflow in fixed_decimal addition");
        }
        return fixed_decimal<T, Scale>::from_raw(result);
    } else {
        return fixed_decimal<T, Scale>::from_raw(lhs.raw_value() + rhs.raw_value());
    }
}

template <std::integral T, size_t Scale>
constexpr fixed_decimal<T, Scale> operator-(const fixed_decimal<T, Scale>& lhs,
                                           const fixed_decimal<T, Scale>& rhs) {
    // Check for overflow
    if constexpr (std::numeric_limits<T>::is_bounded) {
        T result = lhs.raw_value() - rhs.raw_value();
        // Simple overflow check
        if ((rhs.raw_value() < 0 && result < lhs.raw_value()) ||
            (rhs.raw_value() > 0 && result > lhs.raw_value())) {
            throw std::overflow_error("Overflow in fixed_decimal subtraction");
        }
        return fixed_decimal<T, Scale>::from_raw(result);
    } else {
        return fixed_decimal<T, Scale>::from_raw(lhs.raw_value() - rhs.raw_value());
    }
}

template <std::integral T, size_t Scale>
constexpr fixed_decimal<T, Scale> operator*(const fixed_decimal<T, Scale>& lhs,
                                           const fixed_decimal<T, Scale>& rhs) {
    // Multiplication: (a * 10^s) * (b * 10^s) = (a * b * 10^s) * 10^s
    // We need to divide by scale factor after multiplication

    // Use wider type for intermediate calculation to prevent overflow
    using wide_t = std::conditional_t<sizeof(T) <= 4, int64_t, __int128>;

    wide_t a = lhs.raw_value();
    wide_t b = rhs.raw_value();
    wide_t product = a * b;

    // Round to nearest
    wide_t scaled_result = product / fixed_decimal<T, Scale>::scaling_factor;
    wide_t remainder = product % fixed_decimal<T, Scale>::scaling_factor;

    wide_t abs_remainder = remainder < 0 ? -remainder : remainder;
    if (abs_remainder >= fixed_decimal<T, Scale>::scaling_factor / 2) {
        scaled_result += (product >= 0) ? 1 : -1;
    }

    // Check if result fits in T
    if constexpr (std::numeric_limits<T>::is_bounded) {
        if (scaled_result > std::numeric_limits<T>::max() ||
            scaled_result < std::numeric_limits<T>::min()) {
            throw std::overflow_error("Overflow in fixed_decimal multiplication");
        }
    }

    return fixed_decimal<T, Scale>::from_raw(static_cast<T>(scaled_result));
}

template <std::integral T, size_t Scale>
constexpr fixed_decimal<T, Scale> operator/(const fixed_decimal<T, Scale>& lhs,
                                           const fixed_decimal<T, Scale>& rhs) {
    if (rhs.is_zero()) {
        throw std::domain_error("Division by zero");
    }

    // Division: (a * 10^s) / (b * 10^s) = a/b
    // We need to multiply by scale factor before division

    // Use wider type for intermediate calculation
    using wide_t = std::conditional_t<sizeof(T) <= 4, int64_t, __int128>;

    wide_t a = static_cast<wide_t>(lhs.raw_value()) * fixed_decimal<T, Scale>::scaling_factor;
    wide_t b = rhs.raw_value();

    // Round to nearest
    wide_t quotient = a / b;
    wide_t remainder = a % b;

    wide_t abs_rem = remainder < 0 ? -remainder : remainder;
    wide_t abs_b = b < 0 ? -b : b;
    if (abs_rem * 2 >= abs_b) {
        quotient += (a >= 0) == (b >= 0) ? 1 : -1;
    }

    // Check if result fits in T
    if constexpr (std::numeric_limits<T>::is_bounded) {
        if (quotient > std::numeric_limits<T>::max() ||
            quotient < std::numeric_limits<T>::min()) {
            throw std::overflow_error("Overflow in fixed_decimal division");
        }
    }

    return fixed_decimal<T, Scale>::from_raw(static_cast<T>(quotient));
}

template <std::integral T, size_t Scale>
constexpr fixed_decimal<T, Scale> operator-(const fixed_decimal<T, Scale>& x) {
    return fixed_decimal<T, Scale>::from_raw(-x.raw_value());
}

// Modulo operation
template <std::integral T, size_t Scale>
constexpr fixed_decimal<T, Scale> operator%(const fixed_decimal<T, Scale>& lhs,
                                           const fixed_decimal<T, Scale>& rhs) {
    if (rhs.is_zero()) {
        throw std::domain_error("Modulo by zero");
    }
    return fixed_decimal<T, Scale>::from_raw(lhs.raw_value() % rhs.raw_value());
}

// Absolute value
template <std::integral T, size_t Scale>
constexpr fixed_decimal<T, Scale> abs(const fixed_decimal<T, Scale>& x) {
    return x.is_negative() ? -x : x;
}

// Min/max operations
template <std::integral T, size_t Scale>
constexpr fixed_decimal<T, Scale> min(const fixed_decimal<T, Scale>& a,
                                      const fixed_decimal<T, Scale>& b) {
    return (b < a) ? b : a;
}

template <std::integral T, size_t Scale>
constexpr fixed_decimal<T, Scale> max(const fixed_decimal<T, Scale>& a,
                                      const fixed_decimal<T, Scale>& b) {
    return (a < b) ? b : a;
}

// Fundamental operations for generic algorithms

template <std::integral T, size_t Scale>
constexpr bool even(const fixed_decimal<T, Scale>& x) {
    // Check if the integer part is even and fractional part is zero
    return x.fractional_part() == 0 && (x.integer_part() & 1) == 0;
}

template <std::integral T, size_t Scale>
constexpr fixed_decimal<T, Scale> twice(const fixed_decimal<T, Scale>& x) {
    return fixed_decimal<T, Scale>::from_raw(x.raw_value() * 2);
}

template <std::integral T, size_t Scale>
constexpr fixed_decimal<T, Scale> half(const fixed_decimal<T, Scale>& x) {
    // Round to nearest when halving
    T raw = x.raw_value();
    T half_raw = raw / 2;
    if (raw & 1) {  // Odd value needs rounding
        half_raw += (raw > 0) ? 0 : -1;  // Round toward zero
    }
    return fixed_decimal<T, Scale>::from_raw(half_raw);
}

template <std::integral T, size_t Scale>
constexpr fixed_decimal<T, Scale> increment(const fixed_decimal<T, Scale>& x) {
    return x + fixed_decimal<T, Scale>::one();
}

template <std::integral T, size_t Scale>
constexpr fixed_decimal<T, Scale> decrement(const fixed_decimal<T, Scale>& x) {
    return x - fixed_decimal<T, Scale>::one();
}

// Rounding operations

template <std::integral T, size_t Scale>
constexpr fixed_decimal<T, Scale> floor(const fixed_decimal<T, Scale>& x) {
    T int_part = x.integer_part();
    if (x.is_negative() && x.fractional_part() != 0) {
        int_part--;  // Floor of negative non-integer goes down
    }
    return fixed_decimal<T, Scale>(int_part);
}

template <std::integral T, size_t Scale>
constexpr fixed_decimal<T, Scale> ceil(const fixed_decimal<T, Scale>& x) {
    T int_part = x.integer_part();
    if (x.is_positive() && x.fractional_part() != 0) {
        int_part++;  // Ceiling of positive non-integer goes up
    }
    return fixed_decimal<T, Scale>(int_part);
}

template <std::integral T, size_t Scale>
constexpr fixed_decimal<T, Scale> round(const fixed_decimal<T, Scale>& x) {
    // Round to nearest integer (ties to even)
    T int_part = x.integer_part();
    T frac_part = x.fractional_part();
    T half = fixed_decimal<T, Scale>::scaling_factor / 2;

    if (frac_part > half || (frac_part == half && (int_part & 1))) {
        return fixed_decimal<T, Scale>(int_part + 1);
    } else if (frac_part < -half || (frac_part == -half && (int_part & 1))) {
        return fixed_decimal<T, Scale>(int_part - 1);
    } else {
        return fixed_decimal<T, Scale>(int_part);
    }
}

template <std::integral T, size_t Scale>
constexpr fixed_decimal<T, Scale> trunc(const fixed_decimal<T, Scale>& x) {
    return fixed_decimal<T, Scale>(x.integer_part());
}

// Fractional part
template <std::integral T, size_t Scale>
constexpr fixed_decimal<T, Scale> frac(const fixed_decimal<T, Scale>& x) {
    return fixed_decimal<T, Scale>::from_raw(x.fractional_part());
}

// Square root approximation using Newton's method
template <std::integral T, size_t Scale>
fixed_decimal<T, Scale> sqrt(const fixed_decimal<T, Scale>& x) {
    if (x.is_negative()) {
        throw std::domain_error("Square root of negative number");
    }
    if (x.is_zero()) {
        return fixed_decimal<T, Scale>::zero();
    }

    // Initial guess: use floating point sqrt
    fixed_decimal<T, Scale> guess(std::sqrt(static_cast<double>(x)));

    // Newton's method: x_{n+1} = (x_n + a/x_n) / 2
    fixed_decimal<T, Scale> prev;
    const int max_iterations = 10;

    for (int i = 0; i < max_iterations; ++i) {
        prev = guess;
        guess = (guess + x / guess) / fixed_decimal<T, Scale>(2);

        // Check for convergence
        if (abs(guess - prev) < fixed_decimal<T, Scale>::epsilon()) {
            break;
        }
    }

    return guess;
}

// Power function for integer exponents
template <std::integral T, size_t Scale>
constexpr fixed_decimal<T, Scale> power(const fixed_decimal<T, Scale>& base, int exp) {
    if (exp == 0) return fixed_decimal<T, Scale>::one();

    if (exp < 0) {
        return fixed_decimal<T, Scale>::one() / power(base, -exp);
    }

    // Use binary exponentiation
    fixed_decimal<T, Scale> result = fixed_decimal<T, Scale>::one();
    fixed_decimal<T, Scale> b = base;

    while (exp > 0) {
        if (exp & 1) {
            result = result * b;
        }
        b = b * b;
        exp >>= 1;
    }

    return result;
}

// Conversion between different scales
template <std::integral T, size_t FromScale, size_t ToScale>
constexpr fixed_decimal<T, ToScale> scale_convert(const fixed_decimal<T, FromScale>& from) {
    if constexpr (FromScale == ToScale) {
        return fixed_decimal<T, ToScale>::from_raw(from.raw_value());
    } else if constexpr (FromScale < ToScale) {
        // Scale up: multiply by 10^(ToScale - FromScale)
        constexpr T factor = []() {
            T f = 1;
            for (size_t i = 0; i < ToScale - FromScale; ++i) f *= 10;
            return f;
        }();
        return fixed_decimal<T, ToScale>::from_raw(from.raw_value() * factor);
    } else {
        // Scale down: divide by 10^(FromScale - ToScale) with rounding
        constexpr T factor = []() {
            T f = 1;
            for (size_t i = 0; i < FromScale - ToScale; ++i) f *= 10;
            return f;
        }();
        T raw = from.raw_value() / factor;
        T remainder = from.raw_value() % factor;
        if (std::abs(remainder) * 2 >= factor) {
            raw += (from.raw_value() >= 0) ? 1 : -1;
        }
        return fixed_decimal<T, ToScale>::from_raw(raw);
    }
}

// Enable use with generic algorithms by providing required operations
template <std::integral T, size_t Scale>
constexpr fixed_decimal<T, Scale> quotient(const fixed_decimal<T, Scale>& a,
                                          const fixed_decimal<T, Scale>& b) {
    return trunc(a / b);
}

template <std::integral T, size_t Scale>
constexpr fixed_decimal<T, Scale> remainder(const fixed_decimal<T, Scale>& a,
                                           const fixed_decimal<T, Scale>& b) {
    return a % b;
}

template <std::integral T, size_t Scale>
constexpr size_t norm(const fixed_decimal<T, Scale>& x) {
    // Norm as the number of significant digits
    if (x.is_zero()) return 0;

    T val = x.raw_value();
    if (val < 0) val = -val;
    size_t digits = 0;
    while (val > 0) {
        val /= 10;
        digits++;
    }
    return digits;
}

// Stream output operator
template <std::integral T, size_t Scale>
std::ostream& operator<<(std::ostream& os, const fixed_decimal<T, Scale>& fd) {
    os << fd.to_string();
    return os;
}

} // namespace stepanov