#pragma once

#include <limits>
#include <cstring>
#include <utility>
#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <iostream>
#include <sstream>
#include "concepts.hpp"
#include "math.hpp"

namespace stepanov {

/**
 * bounded_nat<N> - Fixed-size arbitrary precision natural numbers
 *
 * This class represents unsigned integers using a fixed-size array of N digits.
 * It demonstrates generic programming principles by providing fundamental
 * operations (twice, half, even, increment, decrement) that enable the use
 * of generic algorithms for multiplication, addition, etc.
 *
 * The representation uses little-endian ordering (least significant digit first).
 */
template <size_t N>
    requires (N > 0)
class bounded_nat {
public:
    using digit_type = unsigned char;
    using size_type = size_t;

    static constexpr digit_type max_digit = std::numeric_limits<digit_type>::max();
    static constexpr size_type num_digits = N;
    static constexpr size_type bits_per_digit = 8 * sizeof(digit_type);

private:
    std::array<digit_type, N> digits{};  // Zero-initialized

public:
    // Constructors
    constexpr bounded_nat() = default;

    constexpr bounded_nat(const bounded_nat&) = default;
    constexpr bounded_nat(bounded_nat&&) = default;

    constexpr bounded_nat& operator=(const bounded_nat&) = default;
    constexpr bounded_nat& operator=(bounded_nat&&) = default;

    // Construct from integral type
    template <std::integral T>
    explicit constexpr bounded_nat(T value) {
        if (value < 0) value = 0;  // Clamp negative to zero

        for (size_type i = 0; i < N && value > 0; ++i) {
            digits[i] = static_cast<digit_type>(value & max_digit);
            value >>= bits_per_digit;
        }
    }

    // Access to raw digits (for advanced operations)
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

    // Find highest non-zero digit
    constexpr size_type highest_digit() const {
        for (size_type i = N; i > 0; --i) {
            if (digits[i - 1] != 0) return i - 1;
        }
        return 0;
    }
};

// Tag type to enable integral_domain concept
template <size_t N>
struct bounded_nat_traits {
    using integral_domain_tag = void;
};

// Comparison operators

template <size_t N>
constexpr bool operator==(const bounded_nat<N>& lhs, const bounded_nat<N>& rhs) {
    return std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

template <size_t N>
constexpr bool operator!=(const bounded_nat<N>& lhs, const bounded_nat<N>& rhs) {
    return !(lhs == rhs);
}

template <size_t N>
constexpr bool operator<(const bounded_nat<N>& lhs, const bounded_nat<N>& rhs) {
    for (size_t i = N; i > 0; --i) {
        if (lhs[i - 1] != rhs[i - 1]) {
            return lhs[i - 1] < rhs[i - 1];
        }
    }
    return false;
}

template <size_t N>
constexpr bool operator<=(const bounded_nat<N>& lhs, const bounded_nat<N>& rhs) {
    return !(rhs < lhs);
}

template <size_t N>
constexpr bool operator>(const bounded_nat<N>& lhs, const bounded_nat<N>& rhs) {
    return rhs < lhs;
}

template <size_t N>
constexpr bool operator>=(const bounded_nat<N>& lhs, const bounded_nat<N>& rhs) {
    return !(lhs < rhs);
}

// Fundamental operations for generic algorithms

template <size_t N>
constexpr bool even(const bounded_nat<N>& n) {
    return (n[0] & 1) == 0;
}

template <size_t N>
constexpr bounded_nat<N> twice(bounded_nat<N> n) {
    using digit_type = typename bounded_nat<N>::digit_type;
    digit_type carry = 0;

    for (size_t i = 0; i < N; ++i) {
        digit_type new_carry = (n[i] & 0x80) ? 1 : 0;  // High bit becomes carry
        n[i] = (n[i] << 1) | carry;
        carry = new_carry;
    }

    return n;
}

template <size_t N>
constexpr bounded_nat<N> half(bounded_nat<N> n) {
    using digit_type = typename bounded_nat<N>::digit_type;
    digit_type carry = 0;

    for (size_t i = N; i > 0; --i) {
        digit_type new_carry = (n[i - 1] & 1) ? 0x80 : 0;  // Low bit becomes carry
        n[i - 1] = (n[i - 1] >> 1) | carry;
        carry = new_carry;
    }

    return n;
}

template <size_t N>
constexpr bounded_nat<N> increment(bounded_nat<N> n) {
    for (size_t i = 0; i < N; ++i) {
        if (n[i] != bounded_nat<N>::max_digit) {
            ++n[i];
            break;
        }
        n[i] = 0;  // Wrap around and continue carry
    }
    return n;
}

template <size_t N>
constexpr bounded_nat<N> decrement(bounded_nat<N> n) {
    for (size_t i = 0; i < N; ++i) {
        if (n[i] != 0) {
            --n[i];
            break;
        }
        n[i] = bounded_nat<N>::max_digit;  // Borrow and continue
    }
    return n;
}

// Arithmetic operations

template <size_t N>
constexpr bounded_nat<N> operator+(const bounded_nat<N>& lhs, const bounded_nat<N>& rhs) {
    bounded_nat<N> result;
    typename bounded_nat<N>::digit_type carry = 0;

    for (size_t i = 0; i < N; ++i) {
        auto sum = static_cast<unsigned>(lhs[i]) + static_cast<unsigned>(rhs[i]) + carry;
        result[i] = static_cast<typename bounded_nat<N>::digit_type>(sum & 0xFF);
        carry = sum >> 8;
    }

    return result;
}

template <size_t N>
constexpr bounded_nat<N> operator*(const bounded_nat<N>& lhs, const bounded_nat<N>& rhs) {
    return product(lhs, rhs);
}

// Subtraction (assumes lhs >= rhs)
template <size_t N>
constexpr bounded_nat<N> operator-(bounded_nat<N> lhs, bounded_nat<N> rhs) {
    bounded_nat<N> result;
    typename bounded_nat<N>::digit_type borrow = 0;

    for (size_t i = 0; i < N; ++i) {
        auto temp = static_cast<int>(lhs[i]) - static_cast<int>(rhs[i]) - borrow;
        if (temp < 0) {
            temp += 256;
            borrow = 1;
        } else {
            borrow = 0;
        }
        result[i] = static_cast<typename bounded_nat<N>::digit_type>(temp);
    }

    return result;
}

// Division and modulo operations

template <size_t N>
constexpr std::pair<bounded_nat<N>, bounded_nat<N>> divmod(
    const bounded_nat<N>& dividend,
    const bounded_nat<N>& divisor)
{
    if (divisor.is_zero()) {
        throw std::domain_error("Division by zero");
    }

    bounded_nat<N> quotient;
    bounded_nat<N> remainder = dividend;

    // Simple division by repeated subtraction
    // (More efficient algorithms exist but this demonstrates the principle)
    while (remainder >= divisor) {
        remainder = remainder - divisor;
        quotient = increment(quotient);
    }

    return {quotient, remainder};
}

template <size_t N>
constexpr bounded_nat<N> operator/(const bounded_nat<N>& lhs, const bounded_nat<N>& rhs) {
    return divmod(lhs, rhs).first;
}

template <size_t N>
constexpr bounded_nat<N> operator%(const bounded_nat<N>& lhs, const bounded_nat<N>& rhs) {
    return divmod(lhs, rhs).second;
}

template <size_t N>
constexpr bounded_nat<N> quotient(const bounded_nat<N>& lhs, const bounded_nat<N>& rhs) {
    return lhs / rhs;
}

template <size_t N>
constexpr bounded_nat<N> remainder(const bounded_nat<N>& lhs, const bounded_nat<N>& rhs) {
    return lhs % rhs;
}

// Norm function for Euclidean domain
template <size_t N>
constexpr size_t norm(const bounded_nat<N>& n) {
    return n.highest_digit() + 1;
}

// Bitwise operations

template <size_t N>
constexpr bounded_nat<N> operator&(const bounded_nat<N>& lhs, const bounded_nat<N>& rhs) {
    bounded_nat<N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = lhs[i] & rhs[i];
    }
    return result;
}

template <size_t N>
constexpr bounded_nat<N> operator|(const bounded_nat<N>& lhs, const bounded_nat<N>& rhs) {
    bounded_nat<N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = lhs[i] | rhs[i];
    }
    return result;
}

template <size_t N>
constexpr bounded_nat<N> operator^(const bounded_nat<N>& lhs, const bounded_nat<N>& rhs) {
    bounded_nat<N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = lhs[i] ^ rhs[i];
    }
    return result;
}

// Shift operations

template <size_t N>
constexpr bounded_nat<N> operator<<(bounded_nat<N> n, unsigned shift) {
    while (shift-- > 0) {
        n = twice(n);
    }
    return n;
}

template <size_t N>
constexpr bounded_nat<N> operator>>(bounded_nat<N> n, unsigned shift) {
    while (shift-- > 0) {
        n = half(n);
    }
    return n;
}

// Specialization to enable integral_domain operations for bounded_nat
template <size_t N>
inline constexpr bounded_nat<N> operator-(const bounded_nat<N>& x) {
    // For unsigned types, negation wraps around
    return bounded_nat<N>(0) - x;
}

// Stream output operator - converts to decimal string
template <size_t N>
std::ostream& operator<<(std::ostream& os, const bounded_nat<N>& n) {
    if (n.is_zero()) {
        os << "0";
        return os;
    }

    // Convert to decimal by repeated division by 10
    bounded_nat<N> temp = n;
    std::string result;

    while (!temp.is_zero()) {
        // Get remainder when dividing by 10
        bounded_nat<N> ten(10);
        auto [quot, rem] = divmod(temp, ten);

        // Convert single digit to character
        result = char('0' + rem[0]) + result;
        temp = quot;
    }

    os << result;
    return os;
}

} // namespace stepanov