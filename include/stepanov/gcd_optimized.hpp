#pragma once

#include <utility>
#include <concepts>
#include <type_traits>
#include <optional>
#include <vector>
#include <bit>
#include "concepts.hpp"

namespace stepanov {

// =============================================================================
// Hardware-Optimized GCD Implementations
// =============================================================================

// Use builtin count trailing zeros for optimal performance
template<typename T>
    requires std::unsigned_integral<T>
__attribute__((always_inline))
inline int count_trailing_zeros(T x) noexcept {
    if (x == 0) return sizeof(T) * 8;

    if constexpr (sizeof(T) == 8) {
        return __builtin_ctzll(x);
    } else if constexpr (sizeof(T) == 4) {
        return __builtin_ctz(x);
    } else if constexpr (sizeof(T) == 2) {
        return __builtin_ctz(static_cast<unsigned>(x));
    } else {
        return __builtin_ctz(static_cast<unsigned>(x));
    }
}

// Binary GCD (Stein's algorithm) - optimized with hardware intrinsics
template <typename T>
    requires std::unsigned_integral<T>
__attribute__((hot, flatten))
constexpr T binary_gcd_optimized(T u, T v) noexcept {
    if (u == 0) return v;
    if (v == 0) return u;

    // Find common factor of 2 using hardware CTZ
    const int shift_u = count_trailing_zeros(u);
    const int shift_v = count_trailing_zeros(v);
    const int shift = std::min(shift_u, shift_v);

    // Remove all factors of 2
    u >>= shift_u;
    v >>= shift_v;

    // Main loop - optimized with branch prediction hints
    do {
        // v is odd at this point, remove factors of 2 from u
        u >>= count_trailing_zeros(u);

        // Ensure u >= v for subtraction
        if (__builtin_expect(u < v, 0)) {
            std::swap(u, v);
        }

        u -= v;
    } while (u != 0);

    return v << shift;
}

// Optimized GCD for signed integers
template <typename T>
    requires std::signed_integral<T>
__attribute__((hot))
constexpr T gcd_optimized(T a, T b) noexcept {
    using U = std::make_unsigned_t<T>;

    // Handle negative values
    U u = (a < 0) ? static_cast<U>(-static_cast<long long>(a)) : static_cast<U>(a);
    U v = (b < 0) ? static_cast<U>(-static_cast<long long>(b)) : static_cast<U>(b);

    return static_cast<T>(binary_gcd_optimized(u, v));
}

// Optimized GCD for unsigned integers
template <typename T>
    requires std::unsigned_integral<T>
__attribute__((hot))
constexpr T gcd_optimized(T a, T b) noexcept {
    return binary_gcd_optimized(a, b);
}

// Lehmer's GCD algorithm for very large integers
template <typename T>
    requires std::integral<T> && (sizeof(T) >= 8)
__attribute__((hot))
T lehmer_gcd(T a, T b) {
    using U = std::make_unsigned_t<T>;

    U u = std::abs(a);
    U v = std::abs(b);

    if (u == 0) return v;
    if (v == 0) return u;

    // Lehmer's algorithm with single precision arithmetic
    while (v > 0) {
        // Check if we can use single precision
        if (u < (U(1) << 32) && v < (U(1) << 32)) {
            // Use hardware GCD for small values
            return binary_gcd_optimized(static_cast<uint32_t>(u),
                                       static_cast<uint32_t>(v));
        }

        // Extract high bits
        int shift = std::countl_zero(u | v);
        U u_high = u >> (64 - 32 - shift);
        U v_high = v >> (64 - 32 - shift);

        // Single precision Euclidean steps
        U a_coef = 1, b_coef = 0;
        U c_coef = 0, d_coef = 1;

        while (v_high != 0) {
            U q = u_high / v_high;
            U r = u_high - q * v_high;

            if (r < b_coef || (q * d_coef) > (a_coef - r)) {
                break;
            }

            u_high = v_high;
            v_high = r;

            U temp = a_coef - q * c_coef;
            a_coef = c_coef;
            c_coef = temp;

            temp = b_coef - q * d_coef;
            b_coef = d_coef;
            d_coef = temp;
        }

        // Apply transformation to full precision
        U new_u = a_coef * u + b_coef * v;
        U new_v = c_coef * u + d_coef * v;

        u = new_u;
        v = new_v;

        // Ensure progress
        if (v >= u) {
            U temp = u % v;
            u = v;
            v = temp;
        }
    }

    return u;
}

// =============================================================================
// Extended GCD with Optimizations
// =============================================================================

// Optimized extended GCD using binary algorithm
template <typename T>
    requires std::integral<T>
__attribute__((hot))
extended_gcd_result<T> extended_gcd_optimized(T a, T b) {
    T old_r = a, r = b;
    T old_s = 1, s = 0;
    T old_t = 0, t = 1;

    // Unrolled loop with branch prediction
    while (r != 0) {
        T q = old_r / r;

        // Update r (remainder)
        T temp_r = old_r - q * r;
        old_r = r;
        r = temp_r;

        // Update s coefficient
        T temp_s = old_s - q * s;
        old_s = s;
        s = temp_s;

        // Update t coefficient
        T temp_t = old_t - q * t;
        old_t = t;
        t = temp_t;
    }

    return {old_r, old_s, old_t};
}

// =============================================================================
// SIMD GCD for Multiple Pairs
// =============================================================================

#ifdef __AVX2__
#include <immintrin.h>

// Vectorized GCD for multiple pairs of 32-bit integers
void gcd_batch_avx2(const uint32_t* a, const uint32_t* b, uint32_t* result, size_t count) {
    constexpr size_t SIMD_WIDTH = 8;

    size_t i = 0;
    for (; i + SIMD_WIDTH <= count; i += SIMD_WIDTH) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[i]));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&b[i]));

        // Process 8 GCDs in parallel
        alignas(32) uint32_t temp_a[SIMD_WIDTH];
        alignas(32) uint32_t temp_b[SIMD_WIDTH];
        alignas(32) uint32_t temp_result[SIMD_WIDTH];

        _mm256_store_si256(reinterpret_cast<__m256i*>(temp_a), va);
        _mm256_store_si256(reinterpret_cast<__m256i*>(temp_b), vb);

        // Compute GCDs
        #pragma omp simd aligned(temp_a, temp_b, temp_result:32)
        for (size_t j = 0; j < SIMD_WIDTH; ++j) {
            temp_result[j] = binary_gcd_optimized(temp_a[j], temp_b[j]);
        }

        __m256i vresult = _mm256_load_si256(reinterpret_cast<const __m256i*>(temp_result));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&result[i]), vresult);
    }

    // Handle remainder
    for (; i < count; ++i) {
        result[i] = binary_gcd_optimized(a[i], b[i]);
    }
}
#endif

// =============================================================================
// Optimized LCM
// =============================================================================

template <typename T>
    requires std::integral<T>
__attribute__((hot))
constexpr T lcm_optimized(T a, T b) noexcept {
    if (a == 0 || b == 0) return 0;

    // Prevent overflow by dividing first
    T g = gcd_optimized(a, b);
    return (a / g) * b;
}

// =============================================================================
// GCD with Compile-Time Optimizations
// =============================================================================

// Compile-time GCD using constexpr
template <typename T>
    requires std::integral<T>
constexpr T gcd_constexpr(T a, T b) noexcept {
    while (b != 0) {
        T temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// =============================================================================
// Main GCD Interface with Auto-Selection
// =============================================================================

template <typename T>
    requires std::integral<T>
__attribute__((flatten))
inline T gcd(T a, T b) noexcept {
    // For compile-time evaluation
    if (std::is_constant_evaluated()) {
        return gcd_constexpr(std::abs(a), std::abs(b));
    }

    // Runtime selection based on type size
    if constexpr (sizeof(T) >= 8 && std::is_unsigned_v<T>) {
        // Use Lehmer's algorithm for large unsigned integers
        return lehmer_gcd(a, b);
    } else if constexpr (std::is_unsigned_v<T>) {
        // Use optimized binary GCD for unsigned
        return binary_gcd_optimized(a, b);
    } else {
        // Use optimized GCD for signed integers
        return gcd_optimized(a, b);
    }
}

// No specializations needed - handled by main template

// =============================================================================
// Range GCD with Optimizations
// =============================================================================

template <typename InputIt>
    requires std::input_iterator<InputIt> &&
             std::integral<std::iter_value_t<InputIt>>
auto gcd_range_optimized(InputIt first, InputIt last) {
    using T = std::iter_value_t<InputIt>;

    if (first == last) return T(0);

    T result = *first++;

    // Early exit optimization
    while (first != last && result != 1) {
        result = gcd(result, *first++);
    }

    return result;
}

// Parallel GCD for large ranges
template <typename T>
    requires std::integral<T>
T gcd_range_parallel(const T* data, size_t count) {
    if (count == 0) return T(0);
    if (count == 1) return data[0];

    // Use parallel reduction for large arrays
    if (count > 1000) {
        T result = data[0];

        #pragma omp parallel
        {
            T local_gcd = T(0);

            #pragma omp for nowait
            for (size_t i = 0; i < count; ++i) {
                local_gcd = gcd(local_gcd, data[i]);
            }

            #pragma omp critical
            result = gcd(result, local_gcd);
        }

        return result;
    } else {
        // Serial computation for small arrays
        T result = data[0];
        for (size_t i = 1; i < count; ++i) {
            result = gcd(result, data[i]);
            if (result == 1) break; // Early exit
        }
        return result;
    }
}

} // namespace stepanov