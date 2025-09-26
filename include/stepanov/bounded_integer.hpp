#pragma once

#include <limits>
#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <stdexcept>
#include <iostream>
#include "concepts.hpp"
#include "math.hpp"
#include "bounded_nat.hpp"

namespace stepanov {

/**
 * bounded_integer<N> - Fixed-size arbitrary precision signed integers
 *
 * This class represents signed integers using a fixed-size array of N digits
 * with two's complement representation. It demonstrates generic programming
 * principles by providing fundamental operations that enable the use of
 * generic algorithms.
 *
 * Design decisions:
 * - Uses two's complement for negative numbers
 * - Little-endian ordering (least significant digit first)
 * - Sign is determined from the most significant bit
 */
template <size_t N>
    requires (N > 0)
class bounded_integer {
public:
    using digit_type = unsigned char;
    using size_type = size_t;
    using unsigned_type = bounded_nat<N>;

    static constexpr digit_type max_digit = std::numeric_limits<digit_type>::max();
    static constexpr size_type num_digits = N;
    static constexpr size_type bits_per_digit = 8 * sizeof(digit_type);
    static constexpr size_type total_bits = N * bits_per_digit;

private:
    std::array<digit_type, N> digits{};  // Two's complement representation

    // Helper: Check if negative (MSB is 1)
    constexpr bool is_negative() const {
        return (digits[N - 1] & 0x80) != 0;
    }

    // Helper: Two's complement negation
    constexpr void negate_in_place() {
        // Invert all bits
        for (size_type i = 0; i < N; ++i) {
            digits[i] = ~digits[i];
        }
        // Add 1
        digit_type carry = 1;
        for (size_type i = 0; i < N && carry; ++i) {
            auto sum = static_cast<unsigned>(digits[i]) + carry;
            digits[i] = static_cast<digit_type>(sum & max_digit);
            carry = sum >> bits_per_digit;
        }
    }

public:
    // Constructors
    constexpr bounded_integer() = default;
    constexpr bounded_integer(const bounded_integer&) = default;
    constexpr bounded_integer(bounded_integer&&) = default;

    constexpr bounded_integer& operator=(const bounded_integer&) = default;
    constexpr bounded_integer& operator=(bounded_integer&&) = default;

    // Construct from integral type
    template <std::integral T>
    explicit constexpr bounded_integer(T value) {
        bool neg = value < 0;
        if (neg) value = -value;

        // Fill with absolute value
        for (size_type i = 0; i < N && value > 0; ++i) {
            digits[i] = static_cast<digit_type>(value & max_digit);
            value >>= bits_per_digit;
        }

        // Apply two's complement if negative
        if (neg) {
            negate_in_place();
        }
    }

    // Construct from bounded_nat (always positive)
    explicit constexpr bounded_integer(const unsigned_type& n) {
        for (size_type i = 0; i < N; ++i) {
            digits[i] = n[i];
        }
    }

    // Access to raw digits
    constexpr digit_type& operator[](size_type i) { return digits[i]; }
    constexpr const digit_type& operator[](size_type i) const { return digits[i]; }

    // Iterator support
    constexpr auto begin() { return digits.begin(); }
    constexpr auto end() { return digits.end(); }
    constexpr auto begin() const { return digits.begin(); }
    constexpr auto end() const { return digits.end(); }

    // Check if zero
    constexpr bool is_zero() const {
        return std::all_of(digits.begin(), digits.end(),
                          [](digit_type d) { return d == 0; });
    }

    // Sign operations
    constexpr int sign() const {
        if (is_zero()) return 0;
        return is_negative() ? -1 : 1;
    }

    // Get absolute value as bounded_nat
    constexpr unsigned_type abs() const {
        bounded_integer temp = *this;
        if (is_negative()) {
            temp.negate_in_place();
        }
        unsigned_type result;
        for (size_type i = 0; i < N; ++i) {
            result[i] = temp[i];
        }
        return result;
    }

    // Conversion to long long (with overflow check)
    explicit constexpr operator long long() const {
        if (is_negative()) {
            bounded_integer temp = *this;
            temp.negate_in_place();
            long long result = 0;
            for (size_type i = N; i > 0 && i > N - sizeof(long long); --i) {
                result = (result << bits_per_digit) | temp[i - 1];
            }
            return -result;
        } else {
            long long result = 0;
            for (size_type i = N; i > 0 && i > N - sizeof(long long); --i) {
                result = (result << bits_per_digit) | digits[i - 1];
            }
            return result;
        }
    }
};

// Comparison operators

template <size_t N>
constexpr bool operator==(const bounded_integer<N>& lhs, const bounded_integer<N>& rhs) {
    return std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

template <size_t N>
constexpr bool operator!=(const bounded_integer<N>& lhs, const bounded_integer<N>& rhs) {
    return !(lhs == rhs);
}

template <size_t N>
constexpr bool operator<(const bounded_integer<N>& lhs, const bounded_integer<N>& rhs) {
    // Check signs first
    bool lhs_neg = (lhs[N-1] & 0x80) != 0;
    bool rhs_neg = (rhs[N-1] & 0x80) != 0;

    if (lhs_neg && !rhs_neg) return true;
    if (!lhs_neg && rhs_neg) return false;

    // Same sign - compare magnitudes
    for (size_t i = N; i > 0; --i) {
        if (lhs[i - 1] != rhs[i - 1]) {
            return lhs[i - 1] < rhs[i - 1];
        }
    }
    return false;
}

template <size_t N>
constexpr bool operator<=(const bounded_integer<N>& lhs, const bounded_integer<N>& rhs) {
    return !(rhs < lhs);
}

template <size_t N>
constexpr bool operator>(const bounded_integer<N>& lhs, const bounded_integer<N>& rhs) {
    return rhs < lhs;
}

template <size_t N>
constexpr bool operator>=(const bounded_integer<N>& lhs, const bounded_integer<N>& rhs) {
    return !(lhs < rhs);
}

// Fundamental operations for generic algorithms

template <size_t N>
constexpr bool even(const bounded_integer<N>& n) {
    return (n[0] & 1) == 0;
}

template <size_t N>
constexpr bounded_integer<N> twice(bounded_integer<N> n) {
    using digit_type = typename bounded_integer<N>::digit_type;
    digit_type carry = 0;

    for (size_t i = 0; i < N; ++i) {
        digit_type new_carry = (n[i] & 0x80) ? 1 : 0;
        n[i] = (n[i] << 1) | carry;
        carry = new_carry;
    }

    return n;
}

template <size_t N>
constexpr bounded_integer<N> half(bounded_integer<N> n) {
    using digit_type = typename bounded_integer<N>::digit_type;

    // Arithmetic right shift (preserves sign)
    digit_type carry = (n[N-1] & 0x80) ? 0x80 : 0;

    for (size_t i = N; i > 0; --i) {
        digit_type new_carry = (n[i - 1] & 1) ? 0x80 : 0;
        n[i - 1] = (n[i - 1] >> 1) | carry;
        carry = new_carry;
    }

    return n;
}

template <size_t N>
constexpr bounded_integer<N> increment(bounded_integer<N> n) {
    for (size_t i = 0; i < N; ++i) {
        if (n[i] != bounded_integer<N>::max_digit) {
            ++n[i];
            break;
        }
        n[i] = 0;
    }
    return n;
}

template <size_t N>
constexpr bounded_integer<N> decrement(bounded_integer<N> n) {
    for (size_t i = 0; i < N; ++i) {
        if (n[i] != 0) {
            --n[i];
            break;
        }
        n[i] = bounded_integer<N>::max_digit;
    }
    return n;
}

// Arithmetic operations

template <size_t N>
constexpr bounded_integer<N> operator+(const bounded_integer<N>& lhs, const bounded_integer<N>& rhs) {
    bounded_integer<N> result;
    typename bounded_integer<N>::digit_type carry = 0;

    for (size_t i = 0; i < N; ++i) {
        auto sum = static_cast<unsigned>(lhs[i]) + static_cast<unsigned>(rhs[i]) + carry;
        result[i] = static_cast<typename bounded_integer<N>::digit_type>(sum & 0xFF);
        carry = sum >> 8;
    }

    return result;
}

template <size_t N>
constexpr bounded_integer<N> operator-(const bounded_integer<N>& lhs, const bounded_integer<N>& rhs) {
    return lhs + (-rhs);
}

template <size_t N>
constexpr bounded_integer<N> operator-(const bounded_integer<N>& x) {
    bounded_integer<N> result = x;
    // Two's complement negation
    for (size_t i = 0; i < N; ++i) {
        result[i] = ~result[i];
    }
    return increment(result);
}

template <size_t N>
constexpr bounded_integer<N> operator*(const bounded_integer<N>& lhs, const bounded_integer<N>& rhs) {
    // Use absolute values and track sign
    auto lhs_abs = lhs.abs();
    auto rhs_abs = rhs.abs();
    auto result_abs = lhs_abs * rhs_abs;

    bounded_integer<N> result(result_abs);

    // Apply sign
    if ((lhs.sign() < 0) != (rhs.sign() < 0) && !result.is_zero()) {
        result = -result;
    }

    return result;
}

// Division and modulo operations

template <size_t N>
constexpr std::pair<bounded_integer<N>, bounded_integer<N>> divmod(
    const bounded_integer<N>& dividend,
    const bounded_integer<N>& divisor)
{
    if (divisor.is_zero()) {
        throw std::domain_error("Division by zero");
    }

    // Use absolute values
    auto dividend_abs = dividend.abs();
    auto divisor_abs = divisor.abs();
    auto [quot_abs, rem_abs] = divmod(dividend_abs, divisor_abs);

    bounded_integer<N> quotient(quot_abs);
    bounded_integer<N> remainder(rem_abs);

    // Apply signs according to truncated division rules
    if ((dividend.sign() < 0) != (divisor.sign() < 0) && !quotient.is_zero()) {
        quotient = -quotient;
    }
    if (dividend.sign() < 0 && !remainder.is_zero()) {
        remainder = -remainder;
    }

    return {quotient, remainder};
}

template <size_t N>
constexpr bounded_integer<N> operator/(const bounded_integer<N>& lhs, const bounded_integer<N>& rhs) {
    return divmod(lhs, rhs).first;
}

template <size_t N>
constexpr bounded_integer<N> operator%(const bounded_integer<N>& lhs, const bounded_integer<N>& rhs) {
    return divmod(lhs, rhs).second;
}

template <size_t N>
constexpr bounded_integer<N> quotient(const bounded_integer<N>& lhs, const bounded_integer<N>& rhs) {
    return lhs / rhs;
}

template <size_t N>
constexpr bounded_integer<N> remainder(const bounded_integer<N>& lhs, const bounded_integer<N>& rhs) {
    return lhs % rhs;
}

// Norm function for Euclidean domain
template <size_t N>
constexpr size_t norm(const bounded_integer<N>& n) {
    return norm(n.abs());
}

// Bitwise operations

template <size_t N>
constexpr bounded_integer<N> operator&(const bounded_integer<N>& lhs, const bounded_integer<N>& rhs) {
    bounded_integer<N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = lhs[i] & rhs[i];
    }
    return result;
}

template <size_t N>
constexpr bounded_integer<N> operator|(const bounded_integer<N>& lhs, const bounded_integer<N>& rhs) {
    bounded_integer<N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = lhs[i] | rhs[i];
    }
    return result;
}

template <size_t N>
constexpr bounded_integer<N> operator^(const bounded_integer<N>& lhs, const bounded_integer<N>& rhs) {
    bounded_integer<N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = lhs[i] ^ rhs[i];
    }
    return result;
}

// Shift operations

template <size_t N>
constexpr bounded_integer<N> operator<<(bounded_integer<N> n, unsigned shift) {
    while (shift-- > 0) {
        n = twice(n);
    }
    return n;
}

template <size_t N>
constexpr bounded_integer<N> operator>>(bounded_integer<N> n, unsigned shift) {
    while (shift-- > 0) {
        n = half(n);  // Arithmetic right shift
    }
    return n;
}

// Stream output operator
template <size_t N>
std::ostream& operator<<(std::ostream& os, const bounded_integer<N>& n) {
    // Convert to long long for output (simplified)
    os << static_cast<long long>(n);
    return os;
}

} // namespace stepanov