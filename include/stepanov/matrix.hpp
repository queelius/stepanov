#pragma once

#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <functional>
#include <concepts>
#include <utility>
#include <span>
#include <ranges>
#include <cmath>
#include <optional>
#include <variant>
#include <memory>
#include <execution>
#include <tuple>
#include "concepts.hpp"
#include "math.hpp"
#include "cache_oblivious.hpp"
#include "matrix_expressions.hpp"  // For property tags

namespace stepanov {

// =============================================================================
// Forward declarations and helper concepts
// =============================================================================

template<typename T> T abs(const T& x);
template<typename T> T sqrt(const T& x);

// Square root concept
template<typename T>
concept has_sqrt = requires(T a) {
    { sqrt(a) } -> std::convertible_to<T>;
};

// =============================================================================
// Matrix storage policies - following generic programming principles
// =============================================================================

/**
 * Storage policy concept - defines how matrix data is stored and accessed
 */
template<typename S>
concept storage_policy = requires(S s, size_t i, size_t j, size_t r, size_t c) {
    typename S::value_type;
    { s.allocate(r, c) };
    { s.rows() } -> std::convertible_to<size_t>;
    { s.cols() } -> std::convertible_to<size_t>;
    { s.get(i, j) } -> std::convertible_to<typename S::value_type>;
    { s.set(i, j, std::declval<typename S::value_type>()) };
};

/**
 * Dense row-major storage (default, cache-friendly for row traversal)
 */
template<typename T>
class row_major_storage {
public:
    using value_type = T;

private:
    std::vector<T> data_;
    size_t rows_;
    size_t cols_;

public:
    row_major_storage() : rows_(0), cols_(0) {}

    void allocate(size_t r, size_t c) {
        rows_ = r;
        cols_ = c;
        data_.resize(r * c);
    }

    void allocate(size_t r, size_t c, const T& val) {
        rows_ = r;
        cols_ = c;
        data_.resize(r * c, val);
    }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    T& get(size_t i, size_t j) {
        return data_[i * cols_ + j];
    }

    const T& get(size_t i, size_t j) const {
        return data_[i * cols_ + j];
    }

    void set(size_t i, size_t j, const T& val) {
        data_[i * cols_ + j] = val;
    }

    // Direct data access for optimized algorithms
    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }
    size_t stride() const { return cols_; }
};

/**
 * Dense column-major storage (Fortran-style, BLAS-friendly)
 */
template<typename T>
class column_major_storage {
public:
    using value_type = T;

private:
    std::vector<T> data_;
    size_t rows_;
    size_t cols_;

public:
    column_major_storage() : rows_(0), cols_(0) {}

    void allocate(size_t r, size_t c) {
        rows_ = r;
        cols_ = c;
        data_.resize(r * c);
    }

    void allocate(size_t r, size_t c, const T& val) {
        rows_ = r;
        cols_ = c;
        data_.resize(r * c, val);
    }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    T& get(size_t i, size_t j) {
        return data_[j * rows_ + i];
    }

    const T& get(size_t i, size_t j) const {
        return data_[j * rows_ + i];
    }

    void set(size_t i, size_t j, const T& val) {
        data_[j * rows_ + i] = val;
    }

    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }
    size_t stride() const { return rows_; }
};

/**
 * Compressed sparse row (CSR) storage
 */
template<typename T>
class csr_storage {
public:
    using value_type = T;

private:
    std::vector<T> values_;
    std::vector<size_t> col_indices_;
    std::vector<size_t> row_ptrs_;
    size_t rows_;
    size_t cols_;
    T zero_;

public:
    csr_storage() : rows_(0), cols_(0), zero_(T(0)) {}

    void allocate(size_t r, size_t c) {
        rows_ = r;
        cols_ = c;
        row_ptrs_.resize(r + 1, 0);
        zero_ = T(0);
    }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    T get(size_t i, size_t j) const {
        for (size_t idx = row_ptrs_[i]; idx < row_ptrs_[i + 1]; ++idx) {
            if (col_indices_[idx] == j) {
                return values_[idx];
            }
            if (col_indices_[idx] > j) {
                break;
            }
        }
        return zero_;
    }

    void set(size_t i, size_t j, const T& val);

    // Build from COO format
    void build_from_coo(std::vector<std::tuple<size_t, size_t, T>>& entries);

    // Efficient sparse operations
    size_t nnz() const { return values_.size(); }
    const std::vector<T>& values() const { return values_; }
    const std::vector<size_t>& col_indices() const { return col_indices_; }
    const std::vector<size_t>& row_pointers() const { return row_ptrs_; }
};

// Note: Expression templates can be added later as an optimization

// =============================================================================
// Main matrix class template
// =============================================================================

/**
 * Generic matrix class following Stepanov's principles
 *
 * - Works with any numeric type modeling required concepts
 * - Supports multiple storage formats transparently
 * - Provides mathematical operations via expression templates
 * - Integrates with optimization, autodiff, and other library components
 */
template<typename T, typename Storage = row_major_storage<T>>
class matrix {
public:
    using value_type = T;
    using storage_type = Storage;
    using size_type = size_t;
    using property_tag = matrix_expr::general_tag;  // General matrix by default

private:
    Storage storage_;

    // Threshold for algorithm selection
    static constexpr size_t STRASSEN_THRESHOLD = 128;
    static constexpr size_t PARALLEL_THRESHOLD = 512;

public:
    // ==========================================================================
    // Constructors and factory methods
    // ==========================================================================

    matrix() = default;

    matrix(size_t r, size_t c) {
        storage_.allocate(r, c, T(0));
    }

    matrix(size_t r, size_t c, const T& val) {
        storage_.allocate(r, c, val);
    }

    // Construct from initializer list
    matrix(std::initializer_list<std::initializer_list<T>> init) {
        size_t r = init.size();
        size_t c = r > 0 ? init.begin()->size() : 0;
        storage_.allocate(r, c);

        size_t i = 0;
        for (const auto& row : init) {
            size_t j = 0;
            for (const auto& val : row) {
                storage_.set(i, j, val);
                ++j;
            }
            ++i;
        }
    }


    // Factory methods for special matrices
    static matrix identity(size_t n) {
        matrix result(n, n, T(0));
        for (size_t i = 0; i < n; ++i) {
            result(i, i) = T(1);
        }
        return result;
    }

    static matrix diagonal(const std::vector<T>& diag) {
        size_t n = diag.size();
        matrix result(n, n, T(0));
        for (size_t i = 0; i < n; ++i) {
            result(i, i) = diag[i];
        }
        return result;
    }

    static matrix zeros(size_t r, size_t c) {
        return matrix(r, c, T(0));
    }

    static matrix ones(size_t r, size_t c) {
        return matrix(r, c, T(1));
    }

    static matrix random(size_t r, size_t c, std::function<T()> generator) {
        matrix result(r, c);
        for (size_t i = 0; i < r; ++i) {
            for (size_t j = 0; j < c; ++j) {
                result(i, j) = generator();
            }
        }
        return result;
    }

    // ==========================================================================
    // Element access and properties
    // ==========================================================================

    size_t rows() const { return storage_.rows(); }
    size_t cols() const { return storage_.cols(); }
    size_t size() const { return rows() * cols(); }
    bool empty() const { return size() == 0; }
    bool is_square() const { return rows() == cols(); }

    T& operator()(size_t i, size_t j) {
        return storage_.get(i, j);
    }

    const T& operator()(size_t i, size_t j) const {
        return storage_.get(i, j);
    }

    T& at(size_t i, size_t j) {
        if (i >= rows() || j >= cols()) {
            throw std::out_of_range("Matrix index out of range");
        }
        return storage_.get(i, j);
    }

    const T& at(size_t i, size_t j) const {
        if (i >= rows() || j >= cols()) {
            throw std::out_of_range("Matrix index out of range");
        }
        return storage_.get(i, j);
    }

    // Direct data access for SIMD operations
    T* data() {
        return storage_.data();
    }

    const T* data() const {
        return storage_.data();
    }

    // ==========================================================================
    // Views and slices (without copying)
    // ==========================================================================

    class matrix_view {
        matrix& parent_;
        size_t row_start_, col_start_;
        size_t rows_, cols_;

    public:
        matrix_view(matrix& parent, size_t rs, size_t cs, size_t r, size_t c)
            : parent_(parent), row_start_(rs), col_start_(cs), rows_(r), cols_(c) {}

        T& operator()(size_t i, size_t j) {
            return parent_(row_start_ + i, col_start_ + j);
        }

        const T& operator()(size_t i, size_t j) const {
            return parent_(row_start_ + i, col_start_ + j);
        }

        size_t rows() const { return rows_; }
        size_t cols() const { return cols_; }
    };

    matrix_view submatrix(size_t row_start, size_t col_start, size_t rows, size_t cols) {
        return matrix_view(*this, row_start, col_start, rows, cols);
    }

    // Row and column views
    auto row(size_t i) {
        return std::views::iota(size_t(0), cols()) |
               std::views::transform([this, i](size_t j) -> T& { return (*this)(i, j); });
    }

    auto col(size_t j) {
        return std::views::iota(size_t(0), rows()) |
               std::views::transform([this, j](size_t i) -> T& { return (*this)(i, j); });
    }

    // ==========================================================================
    // Basic arithmetic operations (using expression templates)
    // ==========================================================================

    // Matrix addition
    matrix operator+(const matrix& rhs) const {
        matrix result(rows(), rhs.cols());
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                result(i, j) = (*this)(i, j) + rhs(i, j);
            }
        }
        return result;
    }

    // Matrix subtraction
    matrix operator-(const matrix& rhs) const {
        matrix result(rows(), rhs.cols());
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                result(i, j) = (*this)(i, j) - rhs(i, j);
            }
        }
        return result;
    }

    // Matrix negation
    matrix operator-() const {
        matrix result(rows(), cols());
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                result(i, j) = -(*this)(i, j);
            }
        }
        return result;
    }

    // Scalar multiplication
    matrix operator*(const T& scalar) const {
        matrix result(rows(), cols());
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                result(i, j) = (*this)(i, j) * scalar;
            }
        }
        return result;
    }

    friend matrix operator*(const T& scalar, const matrix& m) {
        return m * scalar;
    }

    // Matrix-vector multiplication
    std::vector<T> operator*(const std::vector<T>& v) const {
        if (cols() != v.size()) {
            throw std::invalid_argument("Matrix-vector dimension mismatch");
        }

        std::vector<T> result(rows(), T(0));
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                result[i] += (*this)(i, j) * v[j];
            }
        }
        return result;
    }

    // Scalar division (for fields)
    matrix operator/(const T& scalar) const
        requires field<T>
    {
        matrix result(rows(), cols());
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                result(i, j) = (*this)(i, j) / scalar;
            }
        }
        return result;
    }

    // ==========================================================================
    // Matrix multiplication algorithms
    // ==========================================================================

    // Standard matrix multiplication (O(n³))
    matrix multiply_standard(const matrix& b) const {
        matrix result(rows(), b.cols(), T(0));

        for (size_t i = 0; i < rows(); ++i) {
            for (size_t k = 0; k < cols(); ++k) {
                T a_ik = (*this)(i, k);
                for (size_t j = 0; j < b.cols(); ++j) {
                    result(i, j) = result(i, j) + a_ik * b(k, j);
                }
            }
        }
        return result;
    }

    // Strassen's algorithm (O(n^2.807))
    matrix multiply_strassen(const matrix& b) const {
        size_t n = rows();

        // Base case
        if (n <= STRASSEN_THRESHOLD) {
            return multiply_standard(b);
        }

        // Pad to next power of 2 if necessary
        size_t new_size = 1;
        while (new_size < n) new_size *= 2;

        if (new_size != n) {
            matrix a_padded(new_size, new_size, T(0));
            matrix b_padded(new_size, new_size, T(0));

            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    a_padded(i, j) = (*this)(i, j);
                    b_padded(i, j) = b(i, j);
                }
            }

            matrix c_padded = a_padded.strassen_recursive(b_padded);

            matrix result(n, n);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    result(i, j) = c_padded(i, j);
                }
            }
            return result;
        }

        return strassen_recursive(b);
    }

private:
    matrix strassen_recursive(const matrix& b) const {
        size_t n = rows();
        size_t half = n / 2;

        if (n <= STRASSEN_THRESHOLD) {
            return multiply_standard(b);
        }

        // Divide matrices into quadrants
        matrix a11(half, half), a12(half, half), a21(half, half), a22(half, half);
        matrix b11(half, half), b12(half, half), b21(half, half), b22(half, half);

        for (size_t i = 0; i < half; ++i) {
            for (size_t j = 0; j < half; ++j) {
                a11(i, j) = (*this)(i, j);
                a12(i, j) = (*this)(i, j + half);
                a21(i, j) = (*this)(i + half, j);
                a22(i, j) = (*this)(i + half, j + half);

                b11(i, j) = b(i, j);
                b12(i, j) = b(i, j + half);
                b21(i, j) = b(i + half, j);
                b22(i, j) = b(i + half, j + half);
            }
        }

        // Compute Strassen products
        matrix m1 = (a11 + a22).strassen_recursive(b11 + b22);
        matrix m2 = (a21 + a22).strassen_recursive(b11);
        matrix m3 = a11.strassen_recursive(b12 - b22);
        matrix m4 = a22.strassen_recursive(b21 - b11);
        matrix m5 = (a11 + a12).strassen_recursive(b22);
        matrix m6 = (a21 - a11).strassen_recursive(b11 + b12);
        matrix m7 = (a12 - a22).strassen_recursive(b21 + b22);

        // Combine results
        matrix c11 = m1 + m4 - m5 + m7;
        matrix c12 = m3 + m5;
        matrix c21 = m2 + m4;
        matrix c22 = m1 - m2 + m3 + m6;

        // Assemble result
        matrix result(n, n);
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
    // Cache-oblivious multiplication
    matrix multiply_cache_oblivious(const matrix& b) const {
        // For now, just use standard multiplication
        // A full cache-oblivious implementation would require
        // recursive subdivision with the right memory layout
        return multiply_standard(b);
    }

    // Operator* selects best algorithm based on matrix size
    matrix operator*(const matrix& b) const {
        if (cols() != b.rows()) {
            throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
        }

        if (rows() >= STRASSEN_THRESHOLD && cols() >= STRASSEN_THRESHOLD &&
            b.cols() >= STRASSEN_THRESHOLD && is_square() && b.is_square()) {
            return multiply_strassen(b);
        } else {
            return multiply_cache_oblivious(b);
        }
    }

    // ==========================================================================
    // Special products
    // ==========================================================================

    // Hadamard (element-wise) product
    matrix hadamard(const matrix& b) const {
        if (rows() != b.rows() || cols() != b.cols()) {
            throw std::invalid_argument("Matrices must have same dimensions for Hadamard product");
        }

        matrix result(rows(), cols());
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                result(i, j) = (*this)(i, j) * b(i, j);
            }
        }
        return result;
    }

    // Kronecker product
    matrix kronecker(const matrix& b) const {
        size_t m = rows(), n = cols();
        size_t p = b.rows(), q = b.cols();

        matrix result(m * p, n * q);

        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                T a_ij = (*this)(i, j);
                for (size_t k = 0; k < p; ++k) {
                    for (size_t l = 0; l < q; ++l) {
                        result(i * p + k, j * q + l) = a_ij * b(k, l);
                    }
                }
            }
        }

        return result;
    }

    // ==========================================================================
    // Basic matrix operations
    // ==========================================================================

    // Transpose
    matrix transpose() const {
        matrix result(cols(), rows());
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }

    // Trace (sum of diagonal elements)
    T trace() const {
        if (!is_square()) {
            throw std::invalid_argument("Trace requires square matrix");
        }

        T result = T(0);
        for (size_t i = 0; i < rows(); ++i) {
            result = result + (*this)(i, i);
        }
        return result;
    }

    // Frobenius norm
    T frobenius_norm() const
        requires has_sqrt<T>
    {
        T sum = T(0);
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                T val = (*this)(i, j);
                sum = sum + val * val;
            }
        }
        return ::stepanov::sqrt(sum);
    }

    // ==========================================================================
    // Decompositions
    // ==========================================================================

    // LU decomposition with partial pivoting
    std::tuple<matrix, matrix, std::vector<size_t>> lu_decomposition() const
        requires field<T>
    {
        if (!is_square()) {
            throw std::invalid_argument("LU decomposition requires square matrix");
        }

        size_t n = rows();
        matrix l = matrix::identity(n);
        matrix u = *this;
        std::vector<size_t> pivot(n);
        std::iota(pivot.begin(), pivot.end(), 0);

        for (size_t k = 0; k < n - 1; ++k) {
            // Find pivot
            size_t pivot_row = k;
            T max_val = abs(u(k, k));
            for (size_t i = k + 1; i < n; ++i) {
                T val = abs(u(i, k));
                if (val > max_val) {
                    max_val = val;
                    pivot_row = i;
                }
            }

            // Swap rows if needed
            if (pivot_row != k) {
                std::swap(pivot[k], pivot[pivot_row]);
                for (size_t j = 0; j < n; ++j) {
                    std::swap(u(k, j), u(pivot_row, j));
                }
                for (size_t j = 0; j < k; ++j) {
                    std::swap(l(k, j), l(pivot_row, j));
                }
            }

            // Eliminate column
            for (size_t i = k + 1; i < n; ++i) {
                T factor = u(i, k) / u(k, k);
                l(i, k) = factor;
                for (size_t j = k; j < n; ++j) {
                    u(i, j) = u(i, j) - factor * u(k, j);
                }
            }
        }

        return {l, u, pivot};
    }

    // QR decomposition using Gram-Schmidt
    std::pair<matrix, matrix> qr_decomposition() const
        requires field<T> && has_sqrt<T>
    {
        size_t m = rows(), n = cols();
        matrix q(m, n);
        matrix r(n, n, T(0));

        for (size_t j = 0; j < n; ++j) {
            // Copy column j
            for (size_t i = 0; i < m; ++i) {
                q(i, j) = (*this)(i, j);
            }

            // Subtract projections onto previous columns
            for (size_t k = 0; k < j; ++k) {
                T dot_product = T(0);
                for (size_t i = 0; i < m; ++i) {
                    dot_product = dot_product + q(i, k) * (*this)(i, j);
                }
                r(k, j) = dot_product;

                for (size_t i = 0; i < m; ++i) {
                    q(i, j) = q(i, j) - dot_product * q(i, k);
                }
            }

            // Normalize
            T norm = T(0);
            for (size_t i = 0; i < m; ++i) {
                norm = norm + q(i, j) * q(i, j);
            }
            norm = ::stepanov::sqrt(norm);
            r(j, j) = norm;

            if (norm != T(0)) {
                for (size_t i = 0; i < m; ++i) {
                    q(i, j) = q(i, j) / norm;
                }
            }
        }

        return {q, r};
    }

    // Cholesky decomposition for positive definite matrices
    matrix cholesky() const
        requires field<T> && has_sqrt<T>
    {
        if (!is_square()) {
            throw std::invalid_argument("Cholesky decomposition requires square matrix");
        }

        size_t n = rows();
        matrix l(n, n, T(0));

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                T sum = (*this)(i, j);

                for (size_t k = 0; k < j; ++k) {
                    sum = sum - l(i, k) * l(j, k);
                }

                if (i == j) {
                    l(i, j) = ::stepanov::sqrt(sum);
                } else {
                    l(i, j) = sum / l(j, j);
                }
            }
        }

        return l;
    }

    // ==========================================================================
    // Linear system solvers
    // ==========================================================================

    // Gaussian elimination with partial pivoting
    matrix solve_gaussian(const matrix& b) const
        requires field<T>
    {
        if (!is_square()) {
            throw std::invalid_argument("System must be square");
        }
        if (rows() != b.rows()) {
            throw std::invalid_argument("Dimension mismatch");
        }

        // Create augmented matrix
        size_t n = rows();
        size_t m = b.cols();
        matrix aug(n, n + m);

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                aug(i, j) = (*this)(i, j);
            }
            for (size_t j = 0; j < m; ++j) {
                aug(i, n + j) = b(i, j);
            }
        }

        // Forward elimination
        for (size_t k = 0; k < n; ++k) {
            // Find pivot
            size_t pivot_row = k;
            T max_val = abs(aug(k, k));
            for (size_t i = k + 1; i < n; ++i) {
                T val = abs(aug(i, k));
                if (val > max_val) {
                    max_val = val;
                    pivot_row = i;
                }
            }

            // Swap rows
            if (pivot_row != k) {
                for (size_t j = 0; j < n + m; ++j) {
                    std::swap(aug(k, j), aug(pivot_row, j));
                }
            }

            // Check for singularity
            if (aug(k, k) == T(0)) {
                throw std::runtime_error("Matrix is singular");
            }

            // Eliminate column
            for (size_t i = k + 1; i < n; ++i) {
                T factor = aug(i, k) / aug(k, k);
                for (size_t j = k; j < n + m; ++j) {
                    aug(i, j) = aug(i, j) - factor * aug(k, j);
                }
            }
        }

        // Back substitution
        matrix x(n, m);
        for (int i = n - 1; i >= 0; --i) {
            for (size_t j = 0; j < m; ++j) {
                T sum = aug(i, n + j);
                for (size_t k = i + 1; k < n; ++k) {
                    sum = sum - aug(i, k) * x(k, j);
                }
                x(i, j) = sum / aug(i, i);
            }
        }

        return x;
    }

    // Conjugate gradient method for symmetric positive definite matrices
    template<typename Vec>
    Vec conjugate_gradient(const Vec& b, T tolerance = T(1e-10), size_t max_iter = 1000) const
        requires field<T>
    {
        size_t n = rows();
        Vec x(n, T(0));
        Vec r = b - (*this) * x;
        Vec p = r;
        T rsold = dot_product(r, r);

        for (size_t iter = 0; iter < max_iter; ++iter) {
            Vec ap = (*this) * p;
            T alpha = rsold / dot_product(p, ap);
            x = x + alpha * p;
            r = r - alpha * ap;
            T rsnew = dot_product(r, r);

            if (::stepanov::sqrt(rsnew) < tolerance) {
                break;
            }

            p = r + (rsnew / rsold) * p;
            rsold = rsnew;
        }

        return x;
    }

    // ==========================================================================
    // Matrix properties and special computations
    // ==========================================================================

    // Determinant using LU decomposition
    T determinant() const
        requires field<T>
    {
        if (!is_square()) {
            throw std::invalid_argument("Determinant requires square matrix");
        }

        auto [l, u, pivot] = lu_decomposition();

        T det = T(1);
        for (size_t i = 0; i < rows(); ++i) {
            det = det * u(i, i);
        }

        // Account for row swaps
        size_t swaps = 0;
        for (size_t i = 0; i < pivot.size(); ++i) {
            if (pivot[i] != i) {
                ++swaps;
            }
        }

        return (swaps % 2 == 0) ? det : -det;
    }

    // Matrix inverse using Gaussian elimination
    matrix inverse() const
        requires field<T>
    {
        if (!is_square()) {
            throw std::invalid_argument("Inverse requires square matrix");
        }

        return solve_gaussian(matrix::identity(rows()));
    }

    // Rank computation
    size_t rank(T tolerance = T(1e-10)) const
        requires field<T>
    {
        auto [q, r] = qr_decomposition();

        size_t rank = 0;
        size_t n = std::min(rows(), cols());
        for (size_t i = 0; i < n; ++i) {
            if (abs(r(i, i)) > tolerance) {
                ++rank;
            }
        }

        return rank;
    }

    // ==========================================================================
    // Eigenvalue algorithms
    // ==========================================================================

    // Power iteration for dominant eigenvalue
    std::pair<T, std::vector<T>> power_iteration(size_t max_iter = 1000, T tolerance = T(1e-10)) const
        requires field<T> && has_sqrt<T>
    {
        if (!is_square()) {
            throw std::invalid_argument("Eigenvalues require square matrix");
        }

        size_t n = rows();
        std::vector<T> v(n, T(1));
        T eigenvalue = T(0);

        // Normalize initial vector
        T norm = T(0);
        for (size_t i = 0; i < n; ++i) {
            norm = norm + v[i] * v[i];
        }
        norm = ::stepanov::sqrt(norm);
        for (size_t i = 0; i < n; ++i) {
            v[i] = v[i] / norm;
        }

        for (size_t iter = 0; iter < max_iter; ++iter) {
            // Multiply by matrix
            std::vector<T> v_new(n, T(0));
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    v_new[i] = v_new[i] + (*this)(i, j) * v[j];
                }
            }

            // Compute eigenvalue (Rayleigh quotient)
            T new_eigenvalue = T(0);
            T denominator = T(0);
            for (size_t i = 0; i < n; ++i) {
                new_eigenvalue = new_eigenvalue + v[i] * v_new[i];
                denominator = denominator + v[i] * v[i];
            }
            new_eigenvalue = new_eigenvalue / denominator;

            // Check convergence
            if (abs(new_eigenvalue - eigenvalue) < tolerance) {
                return {new_eigenvalue, v};
            }
            eigenvalue = new_eigenvalue;

            // Normalize
            norm = T(0);
            for (size_t i = 0; i < n; ++i) {
                norm = norm + v_new[i] * v_new[i];
            }
            norm = ::stepanov::sqrt(norm);
            for (size_t i = 0; i < n; ++i) {
                v[i] = v_new[i] / norm;
            }
        }

        return {eigenvalue, v};
    }

    // QR algorithm for all eigenvalues
    std::vector<T> qr_eigenvalues(size_t max_iter = 1000, T tolerance = T(1e-10)) const
        requires field<T> && has_sqrt<T>
    {
        if (!is_square()) {
            throw std::invalid_argument("Eigenvalues require square matrix");
        }

        matrix a = *this;
        size_t n = rows();

        for (size_t iter = 0; iter < max_iter; ++iter) {
            auto [q, r] = a.qr_decomposition();
            matrix a_new = r * q;

            // Check convergence (off-diagonal elements approach zero)
            T off_diag = T(0);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    if (i != j) {
                        off_diag = off_diag + abs(a_new(i, j));
                    }
                }
            }

            if (off_diag < tolerance) {
                std::vector<T> eigenvalues(n);
                for (size_t i = 0; i < n; ++i) {
                    eigenvalues[i] = a_new(i, i);
                }
                return eigenvalues;
            }

            a = a_new;
        }

        // Return diagonal even if not fully converged
        std::vector<T> eigenvalues(n);
        for (size_t i = 0; i < n; ++i) {
            eigenvalues[i] = a(i, i);
        }
        return eigenvalues;
    }

    // ==========================================================================
    // Matrix functions
    // ==========================================================================

    // Matrix exponential using Taylor series
    matrix exp(size_t terms = 20) const
        requires field<T>
    {
        if (!is_square()) {
            throw std::invalid_argument("Matrix exponential requires square matrix");
        }

        matrix result = matrix::identity(rows());
        matrix term = matrix::identity(rows());

        for (size_t k = 1; k <= terms; ++k) {
            term = term * (*this) / T(k);
            result = result + term;
        }

        return result;
    }

    // Matrix logarithm (principal logarithm)
    matrix log(size_t terms = 50) const
        requires field<T>
    {
        if (!is_square()) {
            throw std::invalid_argument("Matrix logarithm requires square matrix");
        }

        // log(A) = log(I + (A - I)) = sum_{k=1}^∞ (-1)^{k+1}/k * (A - I)^k
        matrix x = *this - matrix::identity(rows());
        matrix result = x;
        matrix term = x;

        for (size_t k = 2; k <= terms; ++k) {
            term = term * x;
            T coeff = (k % 2 == 0) ? -T(1) / T(k) : T(1) / T(k);
            result = result + coeff * term;
        }

        return result;
    }

    // Matrix square root using Newton's method
    matrix sqrt(size_t max_iter = 100, T tolerance = T(1e-10)) const
        requires field<T>
    {
        if (!is_square()) {
            throw std::invalid_argument("Matrix square root requires square matrix");
        }

        matrix x = *this;
        matrix x_inv = x.inverse();

        for (size_t iter = 0; iter < max_iter; ++iter) {
            matrix x_new = T(0.5) * (x + x_inv);

            // Check convergence
            T diff = (x_new - x).frobenius_norm();
            if (diff < tolerance) {
                return x_new;
            }

            x = x_new;
            x_inv = x.inverse();
        }

        return x;
    }

    // ==========================================================================
    // Block matrix operations
    // ==========================================================================

    // Block multiplication for 2x2 block matrices
    static matrix block_multiply_2x2(
        const matrix& a11, const matrix& a12,
        const matrix& a21, const matrix& a22,
        const matrix& b11, const matrix& b12,
        const matrix& b21, const matrix& b22)
    {
        matrix c11 = a11 * b11 + a12 * b21;
        matrix c12 = a11 * b12 + a12 * b22;
        matrix c21 = a21 * b11 + a22 * b21;
        matrix c22 = a21 * b12 + a22 * b22;

        size_t m1 = c11.rows(), n1 = c11.cols();
        size_t m2 = c21.rows(), n2 = c22.cols();

        matrix result(m1 + m2, n1 + n2);

        // Copy blocks into result
        for (size_t i = 0; i < m1; ++i) {
            for (size_t j = 0; j < n1; ++j) {
                result(i, j) = c11(i, j);
            }
            for (size_t j = 0; j < n2; ++j) {
                result(i, n1 + j) = c12(i, j);
            }
        }
        for (size_t i = 0; i < m2; ++i) {
            for (size_t j = 0; j < n1; ++j) {
                result(m1 + i, j) = c21(i, j);
            }
            for (size_t j = 0; j < n2; ++j) {
                result(m1 + i, n1 + j) = c22(i, j);
            }
        }

        return result;
    }

    // ==========================================================================
    // Iterative methods
    // ==========================================================================

    // Jacobi iteration
    matrix jacobi_solve(const matrix& b, size_t max_iter = 1000, T tolerance = T(1e-10)) const
        requires field<T>
    {
        if (!is_square() || rows() != b.rows()) {
            throw std::invalid_argument("Invalid dimensions for Jacobi method");
        }

        size_t n = rows();
        size_t m = b.cols();
        matrix x(n, m, T(0));
        matrix x_new(n, m);

        for (size_t iter = 0; iter < max_iter; ++iter) {
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    T sum = b(i, j);
                    for (size_t k = 0; k < n; ++k) {
                        if (k != i) {
                            sum = sum - (*this)(i, k) * x(k, j);
                        }
                    }
                    x_new(i, j) = sum / (*this)(i, i);
                }
            }

            // Check convergence
            T diff = (x_new - x).frobenius_norm();
            if (diff < tolerance) {
                return x_new;
            }

            x = x_new;
        }

        return x;
    }

    // Gauss-Seidel iteration
    matrix gauss_seidel_solve(const matrix& b, size_t max_iter = 1000, T tolerance = T(1e-10)) const
        requires field<T>
    {
        if (!is_square() || rows() != b.rows()) {
            throw std::invalid_argument("Invalid dimensions for Gauss-Seidel method");
        }

        size_t n = rows();
        size_t m = b.cols();
        matrix x(n, m, T(0));

        for (size_t iter = 0; iter < max_iter; ++iter) {
            matrix x_old = x;

            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    T sum = b(i, j);
                    for (size_t k = 0; k < i; ++k) {
                        sum = sum - (*this)(i, k) * x(k, j);
                    }
                    for (size_t k = i + 1; k < n; ++k) {
                        sum = sum - (*this)(i, k) * x(k, j);
                    }
                    x(i, j) = sum / (*this)(i, i);
                }
            }

            // Check convergence
            T diff = (x - x_old).frobenius_norm();
            if (diff < tolerance) {
                return x;
            }
        }

        return x;
    }

    // ==========================================================================
    // Utility functions
    // ==========================================================================

    void swap_rows(size_t i, size_t j) {
        if (i >= rows() || j >= rows()) {
            throw std::out_of_range("Row index out of range");
        }

        for (size_t k = 0; k < cols(); ++k) {
            std::swap((*this)(i, k), (*this)(j, k));
        }
    }

    void swap_cols(size_t i, size_t j) {
        if (i >= cols() || j >= cols()) {
            throw std::out_of_range("Column index out of range");
        }

        for (size_t k = 0; k < rows(); ++k) {
            std::swap((*this)(k, i), (*this)(k, j));
        }
    }

    // Apply function to each element
    template<typename F>
    matrix apply(F&& f) const {
        matrix result(rows(), cols());
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                result(i, j) = f((*this)(i, j));
            }
        }
        return result;
    }

    // Reduce matrix to single value
    template<typename F>
    T reduce(F&& f, T init = T(0)) const {
        T result = init;
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                result = f(result, (*this)(i, j));
            }
        }
        return result;
    }
};

// =============================================================================
// Helper functions
// =============================================================================

// Absolute value helper for generic types
template<typename T>
T abs(const T& x) {
    if constexpr (std::is_arithmetic_v<T>) {
        return std::abs(x);
    } else {
        return x < T(0) ? -x : x;
    }
}

// Square root helper
template<typename T>
T sqrt(const T& x) {
    if constexpr (std::is_arithmetic_v<T>) {
        return std::sqrt(x);
    } else {
        // Newton's method for generic types
        T guess = x / T(2);
        T tolerance = T(1e-10);
        for (size_t i = 0; i < 100; ++i) {
            T next = (guess + x / guess) / T(2);
            if (abs(next - guess) < tolerance) {
                return next;
            }
            guess = next;
        }
        return guess;
    }
}

// Dot product for vectors
template<typename T>
T dot_product(const std::vector<T>& a, const std::vector<T>& b) {
    T result = T(0);
    for (size_t i = 0; i < a.size(); ++i) {
        result = result + a[i] * b[i];
    }
    return result;
}

// =============================================================================
// Type aliases for common matrix types
// =============================================================================

template<typename T>
using dense_matrix = matrix<T, row_major_storage<T>>;

template<typename T>
using column_matrix = matrix<T, column_major_storage<T>>;

template<typename T>
using sparse_matrix_csr = matrix<T, csr_storage<T>>;

// Specific numeric type matrices
using matrix_i = dense_matrix<int>;
using matrix_d = dense_matrix<double>;
using matrix_f = dense_matrix<float>;

// =============================================================================
// CSR storage implementation details
// =============================================================================

template<typename T>
void csr_storage<T>::set(size_t i, size_t j, const T& val) {
    if (val == zero_) {
        // Remove entry if it exists
        for (size_t idx = row_ptrs_[i]; idx < row_ptrs_[i + 1]; ++idx) {
            if (col_indices_[idx] == j) {
                values_.erase(values_.begin() + idx);
                col_indices_.erase(col_indices_.begin() + idx);
                for (size_t r = i + 1; r <= rows_; ++r) {
                    row_ptrs_[r]--;
                }
                break;
            }
        }
    } else {
        // Find position to insert/update
        size_t idx = row_ptrs_[i];
        while (idx < row_ptrs_[i + 1] && col_indices_[idx] < j) {
            ++idx;
        }

        if (idx < row_ptrs_[i + 1] && col_indices_[idx] == j) {
            // Update existing entry
            values_[idx] = val;
        } else {
            // Insert new entry
            values_.insert(values_.begin() + idx, val);
            col_indices_.insert(col_indices_.begin() + idx, j);
            for (size_t r = i + 1; r <= rows_; ++r) {
                row_ptrs_[r]++;
            }
        }
    }
}

template<typename T>
void csr_storage<T>::build_from_coo(std::vector<std::tuple<size_t, size_t, T>>& entries) {
    // Sort entries by (row, col)
    std::sort(entries.begin(), entries.end());

    // Clear existing data
    values_.clear();
    col_indices_.clear();
    row_ptrs_.assign(rows_ + 1, 0);

    // Build CSR format
    size_t current_row = 0;
    for (const auto& [row, col, val] : entries) {
        if (val != zero_) {
            while (current_row < row) {
                row_ptrs_[++current_row] = values_.size();
            }
            values_.push_back(val);
            col_indices_.push_back(col);
        }
    }

    // Fill remaining row pointers
    while (current_row < rows_) {
        row_ptrs_[++current_row] = values_.size();
    }
}

} // namespace stepanov