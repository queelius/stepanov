#pragma once

#include "matrix.hpp"
#include "simd_operations.hpp"
#include "parallel_algorithms.hpp"
#include "matrix_expressions_optimized.hpp"
#include <cstring>
#include <new>

namespace stepanov {

// =============================================================================
// Optimized Matrix Class with All Performance Enhancements
// =============================================================================

template<typename T>
class matrix_optimized : public matrix<T, row_major_storage<T>> {
private:
    using base = matrix<T, row_major_storage<T>>;

    // Alignment for SIMD operations
    static constexpr size_t ALIGNMENT = 64;

public:
    using base::rows;
    using base::cols;
    using base::data;

    // ==========================================================================
    // Constructors with Aligned Memory
    // ==========================================================================

    matrix_optimized() = default;

    matrix_optimized(size_t r, size_t c) : base(r, c) {
        ensure_alignment();
    }

    matrix_optimized(size_t r, size_t c, const T& val) : base(r, c, val) {
        ensure_alignment();
    }

    // ==========================================================================
    // Optimized Arithmetic Operations
    // ==========================================================================

    // Addition using SIMD and OpenMP
    matrix_optimized operator+(const matrix_optimized& rhs) const {
        matrix_optimized result(rows(), cols());

        parallel::matrix_add_parallel(
            this->data(), rhs.data(), result.data(),
            rows(), cols()
        );

        return result;
    }

    // Subtraction using SIMD and OpenMP
    matrix_optimized operator-(const matrix_optimized& rhs) const {
        matrix_optimized result(rows(), cols());

        parallel::matrix_sub_parallel(
            this->data(), rhs.data(), result.data(),
            rows(), cols()
        );

        return result;
    }

    // Scalar multiplication using SIMD
    matrix_optimized operator*(const T& scalar) const {
        matrix_optimized result(rows(), cols());

        const size_t total = rows() * cols();

        #pragma omp parallel for if(total > 10000)
        for (size_t i = 0; i < rows(); ++i) {
            simd::matrix_scalar_mul_simd(
                &this->data()[i * cols()],
                scalar,
                &result.data()[i * cols()],
                cols()
            );
        }

        return result;
    }

    // ==========================================================================
    // Optimized Matrix Multiplication
    // ==========================================================================

    matrix_optimized operator*(const matrix_optimized& b) const {
        if (cols() != b.rows()) {
            throw std::invalid_argument("Matrix dimensions incompatible");
        }

        matrix_optimized result(rows(), b.cols());

        // Use cache-blocked parallel multiplication
        parallel::blocked_matrix_multiply<T>::multiply(
            this->data(), b.data(), result.data(),
            rows(), b.cols(), cols()
        );

        return result;
    }

    // Optimized Strassen multiplication
    matrix_optimized multiply_strassen_optimized(const matrix_optimized& b) const {
        const size_t n = rows();

        // Use parallel blocked multiplication for smaller matrices
        if (n <= 256) {
            return *this * b;
        }

        // Strassen with parallel recursive calls
        return strassen_parallel(b);
    }

    // ==========================================================================
    // Optimized Transpose
    // ==========================================================================

    matrix_optimized transpose() const {
        matrix_optimized result(cols(), rows());

        parallel::transpose_parallel(
            this->data(), result.data(),
            rows(), cols()
        );

        return result;
    }

    // ==========================================================================
    // Expression Template Support
    // ==========================================================================

    // Assignment from expression
    template<typename E>
        requires matrix_expr::is_matrix_expression_v<std::decay_t<E>>
    matrix_optimized& operator=(E&& expr) {
        matrix_expr::assign_expression(*this, std::forward<E>(expr));
        return *this;
    }

    // Constructor from expression
    template<typename E>
        requires matrix_expr::is_matrix_expression_v<std::decay_t<E>>
    explicit matrix_optimized(E&& expr)
        : base(expr.rows(), expr.cols()) {
        matrix_expr::assign_expression(*this, std::forward<E>(expr));
    }

    // ==========================================================================
    // Optimized Linear Algebra Operations
    // ==========================================================================

    // Parallel LU decomposition
    std::tuple<matrix_optimized, matrix_optimized, std::vector<size_t>>
    lu_decomposition_optimized() const {
        size_t n = rows();
        matrix_optimized l = matrix_optimized::identity(n);
        matrix_optimized u = *this;
        std::vector<size_t> pivot(n);

        parallel::lu_decompose_parallel(u.data(), n, pivot.data());

        // Extract L from U
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < i; ++j) {
                l(i, j) = u(i, j);
                u(i, j) = T(0);
            }
        }

        return {l, u, pivot};
    }

    // Parallel QR decomposition
    std::pair<matrix_optimized, matrix_optimized>
    qr_decomposition_optimized() const {
        size_t m = rows(), n = cols();
        matrix_optimized q(m, m);
        matrix_optimized r(m, n);

        parallel::qr_decompose_parallel(
            this->data(), q.data(), r.data(), m, n
        );

        return {q, r};
    }

    // ==========================================================================
    // SIMD-Accelerated Vector Operations
    // ==========================================================================

    std::vector<T> operator*(const std::vector<T>& v) const {
        if (cols() != v.size()) {
            throw std::invalid_argument("Dimension mismatch");
        }

        std::vector<T> result(rows());

        #pragma omp parallel for if(rows() > 100)
        for (size_t i = 0; i < rows(); ++i) {
            result[i] = simd::dot_product_simd(
                &this->data()[i * cols()], v.data(), cols()
            );
        }

        return result;
    }

    // ==========================================================================
    // Memory Optimization Helpers
    // ==========================================================================

private:
    void ensure_alignment() {
        // Ensure data is aligned for SIMD operations
        // This would require modifying the storage class to use aligned allocation
    }

    // Parallel Strassen implementation
    matrix_optimized strassen_parallel(const matrix_optimized& b) const {
        const size_t n = rows();
        const size_t half = n / 2;

        // Base case - use optimized multiplication
        if (n <= 256) {
            return *this * b;
        }

        // Divide matrices into quadrants
        auto extract_quadrant = [](const matrix_optimized& m, size_t row_start,
                                   size_t col_start, size_t size) {
            matrix_optimized result(size, size);
            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < size; ++i) {
                for (size_t j = 0; j < size; ++j) {
                    result(i, j) = m(row_start + i, col_start + j);
                }
            }
            return result;
        };

        // Extract quadrants in parallel
        matrix_optimized a11, a12, a21, a22, b11, b12, b21, b22;

        #pragma omp parallel sections
        {
            #pragma omp section
            a11 = extract_quadrant(*this, 0, 0, half);
            #pragma omp section
            a12 = extract_quadrant(*this, 0, half, half);
            #pragma omp section
            a21 = extract_quadrant(*this, half, 0, half);
            #pragma omp section
            a22 = extract_quadrant(*this, half, half, half);
            #pragma omp section
            b11 = extract_quadrant(b, 0, 0, half);
            #pragma omp section
            b12 = extract_quadrant(b, 0, half, half);
            #pragma omp section
            b21 = extract_quadrant(b, half, 0, half);
            #pragma omp section
            b22 = extract_quadrant(b, half, half, half);
        }

        // Compute Strassen products in parallel
        matrix_optimized m1, m2, m3, m4, m5, m6, m7;

        #pragma omp parallel sections
        {
            #pragma omp section
            m1 = (a11 + a22).strassen_parallel(b11 + b22);
            #pragma omp section
            m2 = (a21 + a22).strassen_parallel(b11);
            #pragma omp section
            m3 = a11.strassen_parallel(b12 - b22);
            #pragma omp section
            m4 = a22.strassen_parallel(b21 - b11);
            #pragma omp section
            m5 = (a11 + a12).strassen_parallel(b22);
            #pragma omp section
            m6 = (a21 - a11).strassen_parallel(b11 + b12);
            #pragma omp section
            m7 = (a12 - a22).strassen_parallel(b21 + b22);
        }

        // Combine results in parallel
        matrix_optimized c11, c12, c21, c22;

        #pragma omp parallel sections
        {
            #pragma omp section
            c11 = m1 + m4 - m5 + m7;
            #pragma omp section
            c12 = m3 + m5;
            #pragma omp section
            c21 = m2 + m4;
            #pragma omp section
            c22 = m1 - m2 + m3 + m6;
        }

        // Assemble final result
        matrix_optimized result(n, n);

        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < half; ++i) {
            for (size_t j = 0; j < half; ++j) {
                result(i, j) = c11(i, j);
                result(i, j + half) = c12(i, j);
                result(i + half, j) = c21(i, j);
                result(i + half, j + half) = c22(i, j);
            }
        }

        return result;
    }

public:
    // Factory for identity matrix
    static matrix_optimized identity(size_t n) {
        matrix_optimized result(n, n, T(0));

        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            result(i, i) = T(1);
        }

        return result;
    }
};

// Type aliases for common optimized matrices
using matrix_d_opt = matrix_optimized<double>;
using matrix_f_opt = matrix_optimized<float>;
using matrix_i_opt = matrix_optimized<int>;

} // namespace stepanov