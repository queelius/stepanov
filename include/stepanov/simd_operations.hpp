#pragma once

#include <immintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <nmmintrin.h>
#include <ammintrin.h>
#include <wmmintrin.h>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <type_traits>

namespace stepanov::simd {

// =============================================================================
// SIMD Feature Detection
// =============================================================================

// Compile-time detection of available SIMD instruction sets
#ifdef __AVX512F__
    constexpr bool has_avx512 = true;
    constexpr size_t vector_size = 64; // 512 bits
#else
    constexpr bool has_avx512 = false;
#endif

#ifdef __AVX2__
    constexpr bool has_avx2 = true;
    #ifndef __AVX512F__
        constexpr size_t vector_size = 32; // 256 bits
    #endif
#else
    constexpr bool has_avx2 = false;
#endif

#ifdef __AVX__
    constexpr bool has_avx = true;
    #if !defined(__AVX2__) && !defined(__AVX512F__)
        constexpr size_t vector_size = 32; // 256 bits
    #endif
#else
    constexpr bool has_avx = false;
#endif

#ifdef __SSE4_2__
    constexpr bool has_sse42 = true;
    #if !defined(__AVX__) && !defined(__AVX2__) && !defined(__AVX512F__)
        constexpr size_t vector_size = 16; // 128 bits
    #endif
#else
    constexpr bool has_sse42 = false;
#endif

#ifdef __SSE2__
    constexpr bool has_sse2 = true;
    #if !defined(__SSE4_2__) && !defined(__AVX__) && !defined(__AVX2__) && !defined(__AVX512F__)
        constexpr size_t vector_size = 16; // 128 bits
    #endif
#else
    constexpr bool has_sse2 = false;
    constexpr size_t vector_size = 8; // Fallback to 64-bit operations
#endif

// Helper to determine optimal alignment
constexpr size_t optimal_alignment = vector_size;

// =============================================================================
// SIMD Vector Operations for Different Types
// =============================================================================

// Double precision operations (AVX/AVX2/AVX512)
struct double_simd {
    static constexpr size_t simd_width = vector_size / sizeof(double);

    // Load aligned
    static inline auto load_aligned(const double* ptr) {
        #ifdef __AVX512F__
            return _mm512_load_pd(ptr);
        #elif defined(__AVX__)
            return _mm256_load_pd(ptr);
        #else
            return _mm_load_pd(ptr);
        #endif
    }

    // Load unaligned
    static inline auto load_unaligned(const double* ptr) {
        #ifdef __AVX512F__
            return _mm512_loadu_pd(ptr);
        #elif defined(__AVX__)
            return _mm256_loadu_pd(ptr);
        #else
            return _mm_loadu_pd(ptr);
        #endif
    }

    // Store aligned
    #ifdef __AVX512F__
    static inline void store_aligned(double* ptr, __m512d v) {
        _mm512_store_pd(ptr, v);
    }
    #elif defined(__AVX__)
    static inline void store_aligned(double* ptr, __m256d v) {
        _mm256_store_pd(ptr, v);
    }
    #else
    static inline void store_aligned(double* ptr, __m128d v) {
        _mm_store_pd(ptr, v);
    }
    #endif

    // Store unaligned
    #ifdef __AVX512F__
    static inline void store_unaligned(double* ptr, __m512d v) {
        _mm512_storeu_pd(ptr, v);
    }
    #elif defined(__AVX__)
    static inline void store_unaligned(double* ptr, __m256d v) {
        _mm256_storeu_pd(ptr, v);
    }
    #else
    static inline void store_unaligned(double* ptr, __m128d v) {
        _mm_storeu_pd(ptr, v);
    }
    #endif

    // Arithmetic operations
    #ifdef __AVX512F__
    static inline __m512d add(__m512d a, __m512d b) { return _mm512_add_pd(a, b); }
    static inline __m512d sub(__m512d a, __m512d b) { return _mm512_sub_pd(a, b); }
    static inline __m512d mul(__m512d a, __m512d b) { return _mm512_mul_pd(a, b); }
    static inline __m512d div(__m512d a, __m512d b) { return _mm512_div_pd(a, b); }
    static inline __m512d fmadd(__m512d a, __m512d b, __m512d c) { return _mm512_fmadd_pd(a, b, c); }
    static inline __m512d set1(double val) { return _mm512_set1_pd(val); }
    static inline __m512d setzero() { return _mm512_setzero_pd(); }
    #elif defined(__AVX__)
    static inline __m256d add(__m256d a, __m256d b) { return _mm256_add_pd(a, b); }
    static inline __m256d sub(__m256d a, __m256d b) { return _mm256_sub_pd(a, b); }
    static inline __m256d mul(__m256d a, __m256d b) { return _mm256_mul_pd(a, b); }
    static inline __m256d div(__m256d a, __m256d b) { return _mm256_div_pd(a, b); }
    #ifdef __FMA__
    static inline __m256d fmadd(__m256d a, __m256d b, __m256d c) { return _mm256_fmadd_pd(a, b, c); }
    #else
    static inline __m256d fmadd(__m256d a, __m256d b, __m256d c) { return _mm256_add_pd(_mm256_mul_pd(a, b), c); }
    #endif
    static inline __m256d set1(double val) { return _mm256_set1_pd(val); }
    static inline __m256d setzero() { return _mm256_setzero_pd(); }
    #else
    static inline __m128d add(__m128d a, __m128d b) { return _mm_add_pd(a, b); }
    static inline __m128d sub(__m128d a, __m128d b) { return _mm_sub_pd(a, b); }
    static inline __m128d mul(__m128d a, __m128d b) { return _mm_mul_pd(a, b); }
    static inline __m128d div(__m128d a, __m128d b) { return _mm_div_pd(a, b); }
    static inline __m128d fmadd(__m128d a, __m128d b, __m128d c) { return _mm_add_pd(_mm_mul_pd(a, b), c); }
    static inline __m128d set1(double val) { return _mm_set1_pd(val); }
    static inline __m128d setzero() { return _mm_setzero_pd(); }
    #endif
};

// Single precision operations
struct float_simd {
    static constexpr size_t simd_width = vector_size / sizeof(float);

    // Load aligned
    static inline auto load_aligned(const float* ptr) {
        #ifdef __AVX512F__
            return _mm512_load_ps(ptr);
        #elif defined(__AVX__)
            return _mm256_load_ps(ptr);
        #else
            return _mm_load_ps(ptr);
        #endif
    }

    // Load unaligned
    static inline auto load_unaligned(const float* ptr) {
        #ifdef __AVX512F__
            return _mm512_loadu_ps(ptr);
        #elif defined(__AVX__)
            return _mm256_loadu_ps(ptr);
        #else
            return _mm_loadu_ps(ptr);
        #endif
    }

    // Store aligned
    #ifdef __AVX512F__
    static inline void store_aligned(float* ptr, __m512 v) {
        _mm512_store_ps(ptr, v);
    }
    #elif defined(__AVX__)
    static inline void store_aligned(float* ptr, __m256 v) {
        _mm256_store_ps(ptr, v);
    }
    #else
    static inline void store_aligned(float* ptr, __m128 v) {
        _mm_store_ps(ptr, v);
    }
    #endif

    // Store unaligned
    #ifdef __AVX512F__
    static inline void store_unaligned(float* ptr, __m512 v) {
        _mm512_storeu_ps(ptr, v);
    }
    #elif defined(__AVX__)
    static inline void store_unaligned(float* ptr, __m256 v) {
        _mm256_storeu_ps(ptr, v);
    }
    #else
    static inline void store_unaligned(float* ptr, __m128 v) {
        _mm_storeu_ps(ptr, v);
    }
    #endif

    // Arithmetic operations
    #ifdef __AVX512F__
    static inline __m512 add(__m512 a, __m512 b) { return _mm512_add_ps(a, b); }
    static inline __m512 sub(__m512 a, __m512 b) { return _mm512_sub_ps(a, b); }
    static inline __m512 mul(__m512 a, __m512 b) { return _mm512_mul_ps(a, b); }
    static inline __m512 div(__m512 a, __m512 b) { return _mm512_div_ps(a, b); }
    static inline __m512 fmadd(__m512 a, __m512 b, __m512 c) { return _mm512_fmadd_ps(a, b, c); }
    static inline __m512 set1(float val) { return _mm512_set1_ps(val); }
    static inline __m512 setzero() { return _mm512_setzero_ps(); }
    #elif defined(__AVX__)
    static inline __m256 add(__m256 a, __m256 b) { return _mm256_add_ps(a, b); }
    static inline __m256 sub(__m256 a, __m256 b) { return _mm256_sub_ps(a, b); }
    static inline __m256 mul(__m256 a, __m256 b) { return _mm256_mul_ps(a, b); }
    static inline __m256 div(__m256 a, __m256 b) { return _mm256_div_ps(a, b); }
    #ifdef __FMA__
    static inline __m256 fmadd(__m256 a, __m256 b, __m256 c) { return _mm256_fmadd_ps(a, b, c); }
    #else
    static inline __m256 fmadd(__m256 a, __m256 b, __m256 c) { return _mm256_add_ps(_mm256_mul_ps(a, b), c); }
    #endif
    static inline __m256 set1(float val) { return _mm256_set1_ps(val); }
    static inline __m256 setzero() { return _mm256_setzero_ps(); }
    #else
    static inline __m128 add(__m128 a, __m128 b) { return _mm_add_ps(a, b); }
    static inline __m128 sub(__m128 a, __m128 b) { return _mm_sub_ps(a, b); }
    static inline __m128 mul(__m128 a, __m128 b) { return _mm_mul_ps(a, b); }
    static inline __m128 div(__m128 a, __m128 b) { return _mm_div_ps(a, b); }
    static inline __m128 fmadd(__m128 a, __m128 b, __m128 c) { return _mm_add_ps(_mm_mul_ps(a, b), c); }
    static inline __m128 set1(float val) { return _mm_set1_ps(val); }
    static inline __m128 setzero() { return _mm_setzero_ps(); }
    #endif
};

// =============================================================================
// Vectorized Matrix Operations
// =============================================================================

// Vectorized matrix addition
template<typename T>
void matrix_add_simd(const T* a, const T* b, T* result, size_t size) {
    if constexpr (std::is_same_v<T, double>) {
        const size_t simd_width = double_simd::simd_width;
        size_t simd_end = size - (size % simd_width);

        // Process SIMD width elements at a time
        for (size_t i = 0; i < simd_end; i += simd_width) {
            auto va = double_simd::load_unaligned(&a[i]);
            auto vb = double_simd::load_unaligned(&b[i]);
            auto vr = double_simd::add(va, vb);
            double_simd::store_unaligned(&result[i], vr);
        }

        // Handle remaining elements
        for (size_t i = simd_end; i < size; ++i) {
            result[i] = a[i] + b[i];
        }
    } else if constexpr (std::is_same_v<T, float>) {
        const size_t simd_width = float_simd::simd_width;
        size_t simd_end = size - (size % simd_width);

        for (size_t i = 0; i < simd_end; i += simd_width) {
            auto va = float_simd::load_unaligned(&a[i]);
            auto vb = float_simd::load_unaligned(&b[i]);
            auto vr = float_simd::add(va, vb);
            float_simd::store_unaligned(&result[i], vr);
        }

        for (size_t i = simd_end; i < size; ++i) {
            result[i] = a[i] + b[i];
        }
    } else {
        // Fallback to scalar
        for (size_t i = 0; i < size; ++i) {
            result[i] = a[i] + b[i];
        }
    }
}

// Vectorized matrix subtraction
template<typename T>
void matrix_sub_simd(const T* a, const T* b, T* result, size_t size) {
    if constexpr (std::is_same_v<T, double>) {
        const size_t simd_width = double_simd::simd_width;
        size_t simd_end = size - (size % simd_width);

        for (size_t i = 0; i < simd_end; i += simd_width) {
            auto va = double_simd::load_unaligned(&a[i]);
            auto vb = double_simd::load_unaligned(&b[i]);
            auto vr = double_simd::sub(va, vb);
            double_simd::store_unaligned(&result[i], vr);
        }

        for (size_t i = simd_end; i < size; ++i) {
            result[i] = a[i] - b[i];
        }
    } else if constexpr (std::is_same_v<T, float>) {
        const size_t simd_width = float_simd::simd_width;
        size_t simd_end = size - (size % simd_width);

        for (size_t i = 0; i < simd_end; i += simd_width) {
            auto va = float_simd::load_unaligned(&a[i]);
            auto vb = float_simd::load_unaligned(&b[i]);
            auto vr = float_simd::sub(va, vb);
            float_simd::store_unaligned(&result[i], vr);
        }

        for (size_t i = simd_end; i < size; ++i) {
            result[i] = a[i] - b[i];
        }
    } else {
        for (size_t i = 0; i < size; ++i) {
            result[i] = a[i] - b[i];
        }
    }
}

// Vectorized scalar multiplication
template<typename T>
void matrix_scalar_mul_simd(const T* a, T scalar, T* result, size_t size) {
    if constexpr (std::is_same_v<T, double>) {
        const size_t simd_width = double_simd::simd_width;
        size_t simd_end = size - (size % simd_width);
        auto vs = double_simd::set1(scalar);

        for (size_t i = 0; i < simd_end; i += simd_width) {
            auto va = double_simd::load_unaligned(&a[i]);
            auto vr = double_simd::mul(va, vs);
            double_simd::store_unaligned(&result[i], vr);
        }

        for (size_t i = simd_end; i < size; ++i) {
            result[i] = a[i] * scalar;
        }
    } else if constexpr (std::is_same_v<T, float>) {
        const size_t simd_width = float_simd::simd_width;
        size_t simd_end = size - (size % simd_width);
        auto vs = float_simd::set1(scalar);

        for (size_t i = 0; i < simd_end; i += simd_width) {
            auto va = float_simd::load_unaligned(&a[i]);
            auto vr = float_simd::mul(va, vs);
            float_simd::store_unaligned(&result[i], vr);
        }

        for (size_t i = simd_end; i < size; ++i) {
            result[i] = a[i] * scalar;
        }
    } else {
        for (size_t i = 0; i < size; ++i) {
            result[i] = a[i] * scalar;
        }
    }
}

// Vectorized dot product
template<typename T>
T dot_product_simd(const T* a, const T* b, size_t size) {
    if constexpr (std::is_same_v<T, double>) {
        const size_t simd_width = double_simd::simd_width;
        size_t simd_end = size - (size % simd_width);
        auto sum = double_simd::setzero();

        for (size_t i = 0; i < simd_end; i += simd_width) {
            auto va = double_simd::load_unaligned(&a[i]);
            auto vb = double_simd::load_unaligned(&b[i]);
            sum = double_simd::fmadd(va, vb, sum);
        }

        // Horizontal sum
        alignas(64) double temp[8];
        #ifdef __AVX512F__
            double hsum = _mm512_reduce_add_pd(sum);
        #elif defined(__AVX__)
            double_simd::store_aligned(temp, sum);
            double hsum = temp[0] + temp[1] + temp[2] + temp[3];
        #else
            double_simd::store_aligned(temp, sum);
            double hsum = temp[0] + temp[1];
        #endif

        // Handle remaining elements
        for (size_t i = simd_end; i < size; ++i) {
            hsum += a[i] * b[i];
        }

        return hsum;
    } else if constexpr (std::is_same_v<T, float>) {
        const size_t simd_width = float_simd::simd_width;
        size_t simd_end = size - (size % simd_width);
        auto sum = float_simd::setzero();

        for (size_t i = 0; i < simd_end; i += simd_width) {
            auto va = float_simd::load_unaligned(&a[i]);
            auto vb = float_simd::load_unaligned(&b[i]);
            sum = float_simd::fmadd(va, vb, sum);
        }

        alignas(64) float temp[16];
        #ifdef __AVX512F__
            float hsum = _mm512_reduce_add_ps(sum);
        #elif defined(__AVX__)
            float_simd::store_aligned(temp, sum);
            float hsum = temp[0] + temp[1] + temp[2] + temp[3] +
                        temp[4] + temp[5] + temp[6] + temp[7];
        #else
            float_simd::store_aligned(temp, sum);
            float hsum = temp[0] + temp[1] + temp[2] + temp[3];
        #endif

        for (size_t i = simd_end; i < size; ++i) {
            hsum += a[i] * b[i];
        }

        return hsum;
    } else {
        T sum = T(0);
        for (size_t i = 0; i < size; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }
}

// Vectorized matrix transpose (4x4 blocks for double, 8x8 for float)
template<typename T>
void transpose_block_simd(const T* src, T* dst, size_t rows, size_t cols,
                          size_t src_stride, size_t dst_stride) {
    if constexpr (std::is_same_v<T, double> && has_avx) {
        // 4x4 double transpose using AVX
        for (size_t i = 0; i < rows; i += 4) {
            for (size_t j = 0; j < cols; j += 4) {
                if (i + 4 <= rows && j + 4 <= cols) {
                    #ifdef __AVX__
                    __m256d row0 = _mm256_loadu_pd(&src[(i+0)*src_stride + j]);
                    __m256d row1 = _mm256_loadu_pd(&src[(i+1)*src_stride + j]);
                    __m256d row2 = _mm256_loadu_pd(&src[(i+2)*src_stride + j]);
                    __m256d row3 = _mm256_loadu_pd(&src[(i+3)*src_stride + j]);

                    __m256d tmp0 = _mm256_unpacklo_pd(row0, row1);
                    __m256d tmp1 = _mm256_unpackhi_pd(row0, row1);
                    __m256d tmp2 = _mm256_unpacklo_pd(row2, row3);
                    __m256d tmp3 = _mm256_unpackhi_pd(row2, row3);

                    row0 = _mm256_permute2f128_pd(tmp0, tmp2, 0x20);
                    row1 = _mm256_permute2f128_pd(tmp1, tmp3, 0x20);
                    row2 = _mm256_permute2f128_pd(tmp0, tmp2, 0x31);
                    row3 = _mm256_permute2f128_pd(tmp1, tmp3, 0x31);

                    _mm256_storeu_pd(&dst[(j+0)*dst_stride + i], row0);
                    _mm256_storeu_pd(&dst[(j+1)*dst_stride + i], row1);
                    _mm256_storeu_pd(&dst[(j+2)*dst_stride + i], row2);
                    _mm256_storeu_pd(&dst[(j+3)*dst_stride + i], row3);
                    #endif
                } else {
                    // Handle edge cases
                    for (size_t ii = i; ii < std::min(i + 4, rows); ++ii) {
                        for (size_t jj = j; jj < std::min(j + 4, cols); ++jj) {
                            dst[jj * dst_stride + ii] = src[ii * src_stride + jj];
                        }
                    }
                }
            }
        }
    } else {
        // Fallback scalar transpose
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                dst[j * dst_stride + i] = src[i * src_stride + j];
            }
        }
    }
}

// =============================================================================
// Optimized Matrix Multiplication Kernels
// =============================================================================

// AVX2/AVX512 optimized matrix multiplication kernel
template<typename T>
void matmul_kernel_simd(const T* __restrict__ a, const T* __restrict__ b,
                        T* __restrict__ c, size_t m, size_t n, size_t k,
                        size_t lda, size_t ldb, size_t ldc) {
    if constexpr (std::is_same_v<T, double>) {
        constexpr size_t BLOCK_M = 4;
        constexpr size_t BLOCK_N = double_simd::simd_width;

        for (size_t i = 0; i < m; i += BLOCK_M) {
            for (size_t j = 0; j < n; j += BLOCK_N) {
                // Load C block
                #ifdef __AVX512F__
                __m512d c0 = (j + 8 <= n) ? double_simd::load_unaligned(&c[i*ldc + j]) : double_simd::setzero();
                __m512d c1 = (i+1 < m && j + 8 <= n) ? double_simd::load_unaligned(&c[(i+1)*ldc + j]) : double_simd::setzero();
                __m512d c2 = (i+2 < m && j + 8 <= n) ? double_simd::load_unaligned(&c[(i+2)*ldc + j]) : double_simd::setzero();
                __m512d c3 = (i+3 < m && j + 8 <= n) ? double_simd::load_unaligned(&c[(i+3)*ldc + j]) : double_simd::setzero();
                #elif defined(__AVX__)
                __m256d c0 = (j + 4 <= n) ? double_simd::load_unaligned(&c[i*ldc + j]) : double_simd::setzero();
                __m256d c1 = (i+1 < m && j + 4 <= n) ? double_simd::load_unaligned(&c[(i+1)*ldc + j]) : double_simd::setzero();
                __m256d c2 = (i+2 < m && j + 4 <= n) ? double_simd::load_unaligned(&c[(i+2)*ldc + j]) : double_simd::setzero();
                __m256d c3 = (i+3 < m && j + 4 <= n) ? double_simd::load_unaligned(&c[(i+3)*ldc + j]) : double_simd::setzero();
                #else
                __m128d c0 = (j + 2 <= n) ? double_simd::load_unaligned(&c[i*ldc + j]) : double_simd::setzero();
                __m128d c1 = (i+1 < m && j + 2 <= n) ? double_simd::load_unaligned(&c[(i+1)*ldc + j]) : double_simd::setzero();
                __m128d c2 = (i+2 < m && j + 2 <= n) ? double_simd::load_unaligned(&c[(i+2)*ldc + j]) : double_simd::setzero();
                __m128d c3 = (i+3 < m && j + 2 <= n) ? double_simd::load_unaligned(&c[(i+3)*ldc + j]) : double_simd::setzero();
                #endif

                // Compute dot products
                for (size_t kk = 0; kk < k; ++kk) {
                    auto b_vec = (j + BLOCK_N <= n) ? double_simd::load_unaligned(&b[kk*ldb + j]) : double_simd::setzero();

                    if (i < m) {
                        auto a0 = double_simd::set1(a[i*lda + kk]);
                        c0 = double_simd::fmadd(a0, b_vec, c0);
                    }
                    if (i+1 < m) {
                        auto a1 = double_simd::set1(a[(i+1)*lda + kk]);
                        c1 = double_simd::fmadd(a1, b_vec, c1);
                    }
                    if (i+2 < m) {
                        auto a2 = double_simd::set1(a[(i+2)*lda + kk]);
                        c2 = double_simd::fmadd(a2, b_vec, c2);
                    }
                    if (i+3 < m) {
                        auto a3 = double_simd::set1(a[(i+3)*lda + kk]);
                        c3 = double_simd::fmadd(a3, b_vec, c3);
                    }
                }

                // Store results
                if (j + BLOCK_N <= n) {
                    if (i < m) double_simd::store_unaligned(&c[i*ldc + j], c0);
                    if (i+1 < m) double_simd::store_unaligned(&c[(i+1)*ldc + j], c1);
                    if (i+2 < m) double_simd::store_unaligned(&c[(i+2)*ldc + j], c2);
                    if (i+3 < m) double_simd::store_unaligned(&c[(i+3)*ldc + j], c3);
                } else {
                    // Handle remainder
                    alignas(64) double temp[8];
                    if (i < m) {
                        double_simd::store_aligned(temp, c0);
                        for (size_t jj = j; jj < n; ++jj) {
                            c[i*ldc + jj] = temp[jj - j];
                        }
                    }
                    if (i+1 < m) {
                        double_simd::store_aligned(temp, c1);
                        for (size_t jj = j; jj < n; ++jj) {
                            c[(i+1)*ldc + jj] = temp[jj - j];
                        }
                    }
                    if (i+2 < m) {
                        double_simd::store_aligned(temp, c2);
                        for (size_t jj = j; jj < n; ++jj) {
                            c[(i+2)*ldc + jj] = temp[jj - j];
                        }
                    }
                    if (i+3 < m) {
                        double_simd::store_aligned(temp, c3);
                        for (size_t jj = j; jj < n; ++jj) {
                            c[(i+3)*ldc + jj] = temp[jj - j];
                        }
                    }
                }
            }
        }
    } else {
        // Fallback to scalar multiplication
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                T sum = c[i*ldc + j];
                for (size_t kk = 0; kk < k; ++kk) {
                    sum += a[i*lda + kk] * b[kk*ldb + j];
                }
                c[i*ldc + j] = sum;
            }
        }
    }
}

// Prefetching helper
inline void prefetch(const void* ptr, int hint = 0) {
    #ifdef __builtin_prefetch
        __builtin_prefetch(ptr, 0, hint);
    #elif defined(_mm_prefetch)
        _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T0);
    #endif
}

} // namespace stepanov::simd