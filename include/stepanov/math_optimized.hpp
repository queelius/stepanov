#pragma once

// Enable aggressive optimizations
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("inline")
#pragma GCC optimize("fast-math")
#pragma GCC target("avx2")
#pragma GCC target("fma")

#include "concepts.hpp"
#include "builtin_operations.hpp"
#include "simd_operations.hpp"
#include <bit>
#include <cmath>

namespace stepanov {

// =============================================================================
// Optimized Power Function with Hardware Intrinsics
// =============================================================================

/**
 * Optimized power using binary exponentiation with loop unrolling
 */
template <typename T>
    requires std::integral<T>
__attribute__((hot, flatten, const))
inline T power_optimized(T base, T exp) noexcept {
    if (__builtin_expect(exp == 0, 0)) return T(1);
    if (__builtin_expect(exp == 1, 0)) return base;
    if (__builtin_expect(exp == 2, 0)) return base * base;

    T result = 1;

    // Unroll by 4 for better pipeline utilization
    while (exp >= 4) {
        if (exp & 1) result *= base;
        base *= base;

        if ((exp >> 1) & 1) result *= base;
        base *= base;

        if ((exp >> 2) & 1) result *= base;
        base *= base;

        if ((exp >> 3) & 1) result *= base;
        base *= base;

        exp >>= 4;
    }

    // Handle remaining bits
    while (exp > 0) {
        if (exp & 1) result *= base;
        base *= base;
        exp >>= 1;
    }

    return result;
}

// Overloads for floating point types
__attribute__((always_inline))
inline double power_optimized(double base, double exp) noexcept {
    return std::pow(base, exp);
}

__attribute__((always_inline))
inline float power_optimized(float base, float exp) noexcept {
    return std::pow(base, exp);  // Use std::pow for float too
}

// =============================================================================
// Optimized Modular Arithmetic
// =============================================================================

/**
 * Optimized modular exponentiation using Montgomery reduction for large moduli
 */
template <typename T>
    requires std::unsigned_integral<T>
__attribute__((hot))
T power_mod_montgomery(T base, T exp, T mod) noexcept {
    if (mod == 1) return 0;

    // Montgomery parameters
    const int bits = std::bit_width(mod);
    const T R = T(1) << bits;  // R = 2^bits
    const T R_mask = R - 1;

    // Compute R^(-1) mod N and N' such that R * R^(-1) - N * N' = 1
    T R_inv = 1, N_prime = 1;
    for (int i = 0; i < bits; ++i) {
        if ((R_inv * mod) & (T(1) << i)) {
            R_inv |= T(1) << i;
            N_prime |= T(1) << i;
        }
    }

    // Convert to Montgomery form
    auto to_montgomery = [&](T x) -> T {
        return (x * R) % mod;
    };

    // Montgomery reduction
    auto mont_reduce = [&](T t) -> T {
        T m = (t * N_prime) & R_mask;
        T u = (t + m * mod) >> bits;
        return (u >= mod) ? u - mod : u;
    };

    // Montgomery multiplication
    auto mont_mul = [&](T a, T b) -> T {
        return mont_reduce(a * b);
    };

    T result = to_montgomery(1);
    base = to_montgomery(base % mod);

    while (exp > 0) {
        if (exp & 1) {
            result = mont_mul(result, base);
        }
        base = mont_mul(base, base);
        exp >>= 1;
    }

    // Convert back from Montgomery form
    return mont_reduce(result);
}

/**
 * Optimized modular power with automatic algorithm selection
 */
template <typename T, typename I>
    requires ring<T> && std::integral<I>
__attribute__((hot))
inline T power_mod_optimized(T base, I exp, T modulus) noexcept {
    // For small moduli, use simple algorithm
    if constexpr (sizeof(T) <= 4) {
        T result = T(1);
        base = base % modulus;

        while (exp > 0) {
            if (exp & 1) {
                result = (result * base) % modulus;
            }
            exp >>= 1;
            base = (base * base) % modulus;
        }
        return result;
    } else {
        // For large moduli, use Montgomery reduction if beneficial
        if (std::bit_width(static_cast<std::make_unsigned_t<T>>(modulus)) > sizeof(T) * 4) {
            return power_mod_montgomery(
                static_cast<std::make_unsigned_t<T>>(base),
                exp,
                static_cast<std::make_unsigned_t<T>>(modulus)
            );
        } else {
            // Standard algorithm with overflow prevention
            using U = std::conditional_t<sizeof(T) == 8, __uint128_t, uint64_t>;
            U result = 1;
            U b = static_cast<U>(base) % static_cast<U>(modulus);

            while (exp > 0) {
                if (exp & 1) {
                    result = (result * b) % static_cast<U>(modulus);
                }
                b = (b * b) % static_cast<U>(modulus);
                exp >>= 1;
            }
            return static_cast<T>(result);
        }
    }
}

// =============================================================================
// Optimized Product Using Russian Peasant with Unrolling
// =============================================================================

template <typename T>
    requires std::integral<T>
__attribute__((hot, flatten))
inline T product_optimized(T a, T b) noexcept {
    if (__builtin_expect(b == 0, 0)) return 0;
    if (__builtin_expect(b == 1, 0)) return a;
    if (__builtin_expect(b == 2, 0)) return a << 1;

    T result = 0;

    // Handle negative multiplier
    if (b < 0) {
        a = -a;
        b = -b;
    }

    // Unrolled Russian peasant multiplication
    while (b >= 8) {
        if (b & 1) result += a;
        a <<= 1;
        if ((b >> 1) & 1) result += a;
        a <<= 1;
        if ((b >> 2) & 1) result += a;
        a <<= 1;
        if ((b >> 3) & 1) result += a;
        a <<= 1;
        b >>= 4;
    }

    // Handle remaining bits
    while (b > 0) {
        if (b & 1) result += a;
        a <<= 1;
        b >>= 1;
    }

    return result;
}

// =============================================================================
// Optimized Square Function
// =============================================================================

template <typename T>
__attribute__((always_inline, const))
inline T square_optimized(T x) noexcept {
    return x * x;
}

// Specialized for types where we can use intrinsics
template <>
__attribute__((always_inline))
inline int32_t square_optimized<int32_t>(int32_t x) noexcept {
    return static_cast<int32_t>(static_cast<int64_t>(x) * x);
}

// =============================================================================
// Optimized Inner Product with SIMD
// =============================================================================

template <typename InputIt1, typename InputIt2, typename T>
    requires std::input_iterator<InputIt1> &&
             std::input_iterator<InputIt2> &&
             ring<T>
__attribute__((hot))
T inner_product_optimized(InputIt1 first1, InputIt1 last1,
                         InputIt2 first2, T init) {
    // For contiguous memory, use SIMD
    if constexpr (std::contiguous_iterator<InputIt1> &&
                  std::contiguous_iterator<InputIt2>) {
        size_t size = std::distance(first1, last1);
        const T* ptr1 = std::to_address(first1);
        const T* ptr2 = std::to_address(first2);

        if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
            return init + simd::dot_product_simd(ptr1, ptr2, size);
        }
    }

    // Fallback to standard algorithm with unrolling
    T result = init;

    // Unroll by 4
    auto dist = std::distance(first1, last1);
    auto end_unrolled = first1;
    if (dist >= 4) {
        std::advance(end_unrolled, dist - (dist % 4));

        while (first1 != end_unrolled) {
            result += (*first1++) * (*first2++);
            result += (*first1++) * (*first2++);
            result += (*first1++) * (*first2++);
            result += (*first1++) * (*first2++);
        }
    }

    // Handle remainder
    while (first1 != last1) {
        result += (*first1++) * (*first2++);
    }

    return result;
}

// =============================================================================
// Optimized Min/Max with Branch-Free Implementation
// =============================================================================

template <typename T>
    requires totally_ordered<T>
__attribute__((always_inline, const))
inline const T& min_optimized(const T& a, const T& b) noexcept {
    // Branch-free min for arithmetic types
    if constexpr (std::is_arithmetic_v<T>) {
        return b + ((a - b) & ((a - b) >> (sizeof(T) * 8 - 1)));
    } else {
        return (b < a) ? b : a;
    }
}

template <typename T>
    requires totally_ordered<T>
__attribute__((always_inline, const))
inline const T& max_optimized(const T& a, const T& b) noexcept {
    // Branch-free max for arithmetic types
    if constexpr (std::is_arithmetic_v<T>) {
        return a - ((a - b) & ((a - b) >> (sizeof(T) * 8 - 1)));
    } else {
        return (a < b) ? b : a;
    }
}

// =============================================================================
// Optimized Absolute Value
// =============================================================================

template <typename T>
    requires ordered_ring<T>
__attribute__((always_inline, const))
inline T abs_optimized(const T& x) noexcept {
    if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
        // Branch-free abs for signed integers
        const T mask = x >> (sizeof(T) * 8 - 1);
        return (x ^ mask) - mask;
    } else if constexpr (std::is_floating_point_v<T>) {
        return std::abs(x);
    } else {
        return (x < T(0)) ? -x : x;
    }
}

// =============================================================================
// Main Interface - Redirect to Optimized Versions
// =============================================================================

// Optimized power function (renamed to avoid conflicts)
template <typename T> requires algebraic<T>
__attribute__((flatten))
inline T power_opt(const T& base, const T& exp) {
    if constexpr (std::is_integral_v<T>) {
        return power_optimized(base, exp);
    } else {
        // Fallback to original implementation for non-integral types
        if (exp == T(0)) return T(1);
        if (exp == T(1)) return base;
        if (exp < T(0)) return T(0);

        return even(exp) ?
            square_optimized(power(base, half(exp))) :
            base * power(base, decrement(exp));
    }
}

template <typename T> requires algebraic<T>
__attribute__((always_inline))
inline T square_opt(const T& x) {
    return square_optimized(x);
}

template <typename T> requires algebraic<T>
__attribute__((flatten))
inline T product_opt(const T& lhs, const T& rhs) {
    if constexpr (std::is_integral_v<T>) {
        return product_optimized(lhs, rhs);
    } else {
        // Use builtin multiplication for non-integral types
        return lhs * rhs;
    }
}

template <typename T, typename I>
    requires ring<T> && std::integral<I>
__attribute__((flatten))
inline T power_mod_opt(T base, I exp, T modulus) {
    return power_mod_optimized(base, exp, modulus);
}

template <typename T>
    requires totally_ordered<T>
__attribute__((always_inline))
inline const T& min_opt(const T& a, const T& b) {
    return min_optimized(a, b);
}

template <typename T>
    requires totally_ordered<T>
__attribute__((always_inline))
inline const T& max_opt(const T& a, const T& b) {
    return max_optimized(a, b);
}

template <typename T>
    requires ordered_ring<T>
__attribute__((always_inline))
inline T abs_opt(const T& x) {
    return abs_optimized(x);
}

} // namespace stepanov