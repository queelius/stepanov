#pragma once

#include <immintrin.h>
#include <omp.h>
#include <vector>
#include <cstring>
#include "matrix.hpp"
#include "matrix_expressions.hpp"

namespace stepanov {

/**
 * High-performance matrix operations using OpenMP and SIMD
 *
 * These implementations use:
 * - AVX2/AVX512 SIMD instructions for vectorization
 * - OpenMP for multi-threading
 * - Cache-friendly blocking
 * - Loop unrolling
 */

// Cache line size (typically 64 bytes)
constexpr size_t CACHE_LINE = 64;
constexpr size_t BLOCK_SIZE = 64;  // Tile size for cache blocking

// =============================================================================
// SIMD Operations for different types
// =============================================================================

// Double precision AVX operations (4 doubles at once)
inline __m256d simd_load_pd(const double* ptr) {
    return _mm256_load_pd(ptr);
}

inline void simd_store_pd(double* ptr, __m256d val) {
    _mm256_store_pd(ptr, val);
}

inline __m256d simd_mul_pd(__m256d a, __m256d b) {
    return _mm256_mul_pd(a, b);
}

inline __m256d simd_add_pd(__m256d a, __m256d b) {
    return _mm256_add_pd(a, b);
}

inline __m256d simd_fmadd_pd(__m256d a, __m256d b, __m256d c) {
    return _mm256_fmadd_pd(a, b, c);  // a*b + c
}

// Single precision AVX operations (8 floats at once)
inline __m256 simd_load_ps(const float* ptr) {
    return _mm256_load_ps(ptr);
}

inline void simd_store_ps(float* ptr, __m256 val) {
    _mm256_store_ps(ptr, val);
}

inline __m256 simd_mul_ps(__m256 a, __m256 b) {
    return _mm256_mul_ps(a, b);
}

inline __m256 simd_add_ps(__m256 a, __m256 b) {
    return _mm256_add_ps(a, b);
}

inline __m256 simd_fmadd_ps(__m256 a, __m256 b, __m256 c) {
    return _mm256_fmadd_ps(a, b, c);  // a*b + c
}

// =============================================================================
// Optimized Matrix Multiplication
// =============================================================================

/**
 * Cache-blocked matrix multiplication with OpenMP and SIMD
 * Uses tiling to maximize cache reuse
 */
template<typename T>
    requires std::floating_point<T>
void matrix_multiply_blocked_simd(const T* A, const T* B, T* C,
                                  size_t M, size_t N, size_t K) {
    // Zero out result matrix
    #pragma omp parallel for
    for (size_t i = 0; i < M * N; ++i) {
        C[i] = 0;
    }

    // Cache-blocked multiplication
    #pragma omp parallel for collapse(2)
    for (size_t ii = 0; ii < M; ii += BLOCK_SIZE) {
        for (size_t jj = 0; jj < N; jj += BLOCK_SIZE) {
            for (size_t kk = 0; kk < K; kk += BLOCK_SIZE) {
                // Process one tile
                for (size_t i = ii; i < std::min(ii + BLOCK_SIZE, M); ++i) {
                    for (size_t j = jj; j < std::min(jj + BLOCK_SIZE, N); ++j) {
                        T sum = C[i * N + j];

                        // Inner loop - vectorize this
                        #pragma omp simd reduction(+:sum)
                        for (size_t k = kk; k < std::min(kk + BLOCK_SIZE, K); ++k) {
                            sum += A[i * K + k] * B[k * N + j];
                        }

                        C[i * N + j] = sum;
                    }
                }
            }
        }
    }
}

/**
 * AVX-optimized matrix multiplication for doubles
 * Processes 4 elements at once
 */
void matrix_multiply_avx_double(const double* A, const double* B, double* C,
                                size_t M, size_t N, size_t K) {
    #pragma omp parallel for
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            __m256d sum = _mm256_setzero_pd();

            // Process 4 elements at a time
            size_t k = 0;
            for (; k + 3 < K; k += 4) {
                __m256d a = _mm256_loadu_pd(&A[i * K + k]);
                __m256d b = _mm256_set_pd(
                    B[(k + 3) * N + j],
                    B[(k + 2) * N + j],
                    B[(k + 1) * N + j],
                    B[k * N + j]
                );
                sum = simd_fmadd_pd(a, b, sum);
            }

            // Sum the 4 elements
            double result[4];
            _mm256_storeu_pd(result, sum);
            double total = result[0] + result[1] + result[2] + result[3];

            // Handle remaining elements
            for (; k < K; ++k) {
                total += A[i * K + k] * B[k * N + j];
            }

            C[i * N + j] = total;
        }
    }
}

/**
 * Optimized diagonal matrix multiplication
 * Exploits sparsity pattern for O(n²) performance
 */
template<typename T>
void diagonal_multiply_simd(const T* diag, const T* M, T* result,
                            size_t n, size_t cols) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        T d = diag[i];

        // Vectorize the row scaling
        #pragma omp simd
        for (size_t j = 0; j < cols; ++j) {
            result[i * cols + j] = d * M[i * cols + j];
        }
    }
}

// Forward declare the aligned_allocator
template<typename T, size_t Alignment = 32>
class aligned_allocator;

/**
 * Optimized symmetric matrix storage and operations
 * Only stores upper triangle
 */
template<typename T>
class simd_symmetric_matrix {
private:
    std::vector<T> data_;  // Will use aligned allocation separately
    size_t n_;

    size_t index(size_t i, size_t j) const {
        if (i > j) std::swap(i, j);
        return i * n_ - i * (i - 1) / 2 + j - i;
    }

public:
    simd_symmetric_matrix(size_t n) : n_(n) {
        size_t size = n * (n + 1) / 2;
        data_.resize(size);
    }

    T& operator()(size_t i, size_t j) {
        return data_[index(i, j)];
    }

    const T& operator()(size_t i, size_t j) const {
        return data_[index(i, j)];
    }

    size_t size() const { return n_; }

    // Optimized symmetric matrix-vector multiplication
    std::vector<T> multiply_vector(const std::vector<T>& v) const {
        std::vector<T> result(n_, 0);

        #pragma omp parallel for
        for (size_t i = 0; i < n_; ++i) {
            T sum = 0;

            // Handle diagonal element
            sum += (*this)(i, i) * v[i];

            // Handle off-diagonal elements
            #pragma omp simd reduction(+:sum)
            for (size_t j = i + 1; j < n_; ++j) {
                T val = (*this)(i, j);
                sum += val * v[j];
                result[j] += val * v[i];  // Exploit symmetry
            }

            #pragma omp atomic
            result[i] += sum;
        }

        return result;
    }
};

/**
 * Optimized triangular system solver
 * Forward substitution with vectorization
 */
template<typename T>
void forward_substitution_simd(const T* L, const T* b, T* x, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        T sum = b[i];

        // Vectorize the dot product
        #pragma omp simd reduction(-:sum)
        for (size_t j = 0; j < i; ++j) {
            sum -= L[i * n + j] * x[j];
        }

        x[i] = sum / L[i * n + i];
    }
}

/**
 * Expression template evaluation with SIMD
 * Fuses operations to minimize memory traffic
 */
template<typename T>
void evaluate_expression_simd(const T* A, const T* B, const T* C,
                              T alpha, T beta, T gamma,
                              T* result, size_t size) {
    // Compute: result = alpha*A + beta*B - gamma*C
    // All in one pass with SIMD

    #pragma omp parallel for simd
    for (size_t i = 0; i < size; ++i) {
        result[i] = alpha * A[i] + beta * B[i] - gamma * C[i];
    }
}

// =============================================================================
// Specialized operators using SIMD
// =============================================================================

// Optimized matrix multiplication operator
template<typename T>
    requires std::floating_point<T>
matrix<T> multiply_simd(const matrix<T>& A, const matrix<T>& B) {
    size_t M = A.rows();
    size_t N = B.cols();
    size_t K = A.cols();

    matrix<T> C(M, N);

    if constexpr (std::is_same_v<T, double>) {
        matrix_multiply_avx_double(
            A.data(), B.data(), C.data(),
            M, N, K
        );
    } else {
        matrix_multiply_blocked_simd(
            A.data(), B.data(), C.data(),
            M, N, K
        );
    }

    return C;
}

// Optimized diagonal matrix multiplication
template<typename T>
matrix<T> multiply_diagonal_simd(const matrix_expr::diagonal_matrix<T>& D,
                                 const matrix<T>& M) {
    size_t n = D.size();
    size_t cols = M.cols();
    matrix<T> result(n, cols);

    diagonal_multiply_simd(
        D.diagonal_data(), M.data(), result.data(),
        n, cols
    );

    return result;
}

// =============================================================================
// Aligned allocator for SIMD
// =============================================================================

template<typename T, size_t Alignment>
class aligned_allocator {
public:
    using value_type = T;

    aligned_allocator() = default;

    template<typename U>
    aligned_allocator(const aligned_allocator<U, Alignment>&) {}

    T* allocate(size_t n) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, size_t) {
        free(ptr);
    }

    bool operator==(const aligned_allocator&) const { return true; }
    bool operator!=(const aligned_allocator&) const { return false; }
};

} // namespace stepanov