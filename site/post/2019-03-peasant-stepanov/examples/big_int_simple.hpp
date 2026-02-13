#pragma once

/**
 * @file big_int_simple.hpp
 * @brief Minimal unsigned big integer for teaching the peasant algorithm
 *
 * THIS MODULE TEACHES: How to adapt a custom type to work with axioms::product()
 * and axioms::power(). By providing the ADL functions (zero, one, twice, half,
 * even, increment, decrement), the generic algorithms "just work."
 *
 * This is NOT production-quality code. It's designed for clarity, not speed.
 */

#include <vector>
#include <cstdint>
#include <algorithm>
#include <compare>

namespace peasant::examples {

/**
 * Simple unsigned big integer using base 2^32
 * Digits stored little-endian (least significant first)
 */
struct big_nat {
    std::vector<uint32_t> digits;

    // Default: represents zero
    big_nat() : digits{0} {}

    // Construct from 64-bit unsigned
    explicit big_nat(uint64_t n) {
        if (n == 0) {
            digits = {0};
        } else {
            while (n > 0) {
                digits.push_back(static_cast<uint32_t>(n & 0xFFFFFFFF));
                n >>= 32;
            }
        }
    }

    // Remove leading zeros (but keep at least one digit)
    void normalize() {
        while (digits.size() > 1 && digits.back() == 0) {
            digits.pop_back();
        }
    }

    bool is_zero() const {
        return digits.size() == 1 && digits[0] == 0;
    }

    bool operator==(big_nat const& other) const = default;

    auto operator<=>(big_nat const& other) const {
        if (digits.size() != other.digits.size()) {
            return digits.size() <=> other.digits.size();
        }
        for (size_t i = digits.size(); i > 0; --i) {
            if (digits[i-1] != other.digits[i-1]) {
                return digits[i-1] <=> other.digits[i-1];
            }
        }
        return std::strong_ordering::equal;
    }

    // Unary negation (for unsigned type, returns zero - needed for generic algorithms to compile)
    big_nat operator-() const {
        return big_nat{0};  // Unsigned type has no negation
    }

    // Addition
    big_nat operator+(big_nat const& other) const {
        big_nat result;
        result.digits.resize(std::max(digits.size(), other.digits.size()) + 1, 0);

        uint64_t carry = 0;
        for (size_t i = 0; i < result.digits.size(); ++i) {
            uint64_t a = i < digits.size() ? digits[i] : 0;
            uint64_t b = i < other.digits.size() ? other.digits[i] : 0;
            uint64_t sum = a + b + carry;
            result.digits[i] = static_cast<uint32_t>(sum & 0xFFFFFFFF);
            carry = sum >> 32;
        }
        result.normalize();
        return result;
    }
};

// =============================================================================
// ADL functions for the peasant algorithm
// With these, axioms::product() and axioms::power() "just work" on big_nat!
// =============================================================================

// Additive identity
inline big_nat zero(big_nat const&) {
    return big_nat{0};
}

// Multiplicative identity
inline big_nat one(big_nat const&) {
    return big_nat{1};
}

// Double (left shift by 1 bit)
inline big_nat twice(big_nat const& x) {
    big_nat result;
    result.digits.resize(x.digits.size() + 1, 0);

    uint32_t carry = 0;
    for (size_t i = 0; i < x.digits.size(); ++i) {
        uint64_t shifted = (static_cast<uint64_t>(x.digits[i]) << 1) | carry;
        result.digits[i] = static_cast<uint32_t>(shifted & 0xFFFFFFFF);
        carry = static_cast<uint32_t>(shifted >> 32);
    }
    result.digits[x.digits.size()] = carry;
    result.normalize();
    return result;
}

// Halve (right shift by 1 bit)
inline big_nat half(big_nat const& x) {
    if (x.is_zero()) return x;

    big_nat result;
    result.digits.resize(x.digits.size(), 0);

    uint32_t carry = 0;
    for (size_t i = x.digits.size(); i > 0; --i) {
        uint64_t current = (static_cast<uint64_t>(carry) << 32) | x.digits[i-1];
        result.digits[i-1] = static_cast<uint32_t>(current >> 1);
        carry = x.digits[i-1] & 1;
    }
    result.normalize();
    return result;
}

// Check if even (test LSB)
inline bool even(big_nat const& x) {
    return (x.digits[0] & 1) == 0;
}

// Successor
inline big_nat increment(big_nat const& x) {
    return x + big_nat{1};
}

// Predecessor (assumes x > 0)
inline big_nat decrement(big_nat const& x) {
    if (x.is_zero()) return x;

    big_nat result = x;
    for (size_t i = 0; i < result.digits.size(); ++i) {
        if (result.digits[i] > 0) {
            result.digits[i]--;
            break;
        }
        result.digits[i] = 0xFFFFFFFF;
    }
    result.normalize();
    return result;
}

} // namespace peasant::examples
