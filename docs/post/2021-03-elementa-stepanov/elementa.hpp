// elementa.hpp - Pedagogical Linear Algebra Library
//
// A teaching-oriented implementation of dense linear algebra operations.
// Designed to be readable and mathematically transparent, not optimized.
//
// Key design principles:
// - Generic over scalar type T (any type satisfying arithmetic operations)
// - Row-major storage for cache-friendly row access
// - Value semantics throughout
// - Educational comments explaining the mathematics
//
// Author: [Your Name]
// License: MIT

#ifndef ELEMENTA_HPP
#define ELEMENTA_HPP

#include <algorithm>
#include <cmath>
#include <concepts>
#include <initializer_list>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace elementa {

// =============================================================================
// Concepts
// =============================================================================

// Scalar types must support basic arithmetic
template <typename T>
concept Arithmetic = requires(T a, T b) {
    { a + b } -> std::convertible_to<T>;
    { a - b } -> std::convertible_to<T>;
    { a * b } -> std::convertible_to<T>;
    { a / b } -> std::convertible_to<T>;
    { -a } -> std::convertible_to<T>;
};

// Forward declaration
template <Arithmetic T>
class matrix;

// The Matrix concept defines the interface any matrix type must satisfy.
// This enables gradator (and other libraries) to be generic over matrix types.
//
// Note: Scalar multiplication is not in the concept to avoid circular
// constraint issues with the matmul operator*. Use scale() for generic code.
template <typename M>
concept Matrix = requires(M m, const M cm, std::size_t i, std::size_t j) {
    // Type requirements
    typename M::scalar_type;

    // Dimension queries
    { cm.rows() } -> std::same_as<std::size_t>;
    { cm.cols() } -> std::same_as<std::size_t>;

    // Element access (mutable and const)
    { m(i, j) } -> std::same_as<typename M::scalar_type&>;
    { cm(i, j) } -> std::same_as<const typename M::scalar_type&>;

    // Arithmetic operations returning same type
    { cm + cm } -> std::same_as<M>;
    { cm - cm } -> std::same_as<M>;
    { -cm } -> std::same_as<M>;
};

// =============================================================================
// matrix<T> - Dense Matrix Class
// =============================================================================

// Dense matrix with row-major storage.
//
// Row-major means elements are stored row by row in memory:
//   [ a b c ]
//   [ d e f ]  stored as: [a, b, c, d, e, f]
//
// This gives good cache locality when iterating over rows.
template <Arithmetic T>
class matrix {
public:
    using scalar_type = T;
    using size_type = std::size_t;

private:
    std::vector<T> data_;
    size_type rows_;
    size_type cols_;

    // Convert 2D index to 1D (row-major)
    [[nodiscard]] constexpr auto index(size_type i, size_type j) const noexcept -> size_type {
        return i * cols_ + j;
    }

public:
    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------

    // Default: empty 0x0 matrix
    matrix() : data_{}, rows_{0}, cols_{0} {}

    // Create rows x cols matrix filled with value
    matrix(size_type rows, size_type cols, T value = T{})
        : data_(rows * cols, value), rows_{rows}, cols_{cols} {}

    // Create from initializer list (row-major order)
    // Usage: matrix<double> m(2, 3, {1, 2, 3, 4, 5, 6});
    matrix(size_type rows, size_type cols, std::initializer_list<T> init)
        : data_(init), rows_{rows}, cols_{cols} {
        if (data_.size() != rows * cols) {
            throw std::invalid_argument(
                "Initializer list size does not match matrix dimensions");
        }
    }

    // Create from nested initializer list
    // Usage: matrix<double> m{{1, 2, 3}, {4, 5, 6}};
    matrix(std::initializer_list<std::initializer_list<T>> init) {
        rows_ = init.size();
        if (rows_ == 0) {
            cols_ = 0;
            return;
        }
        cols_ = init.begin()->size();
        data_.reserve(rows_ * cols_);

        for (const auto& row : init) {
            if (row.size() != cols_) {
                throw std::invalid_argument("All rows must have the same length");
            }
            data_.insert(data_.end(), row.begin(), row.end());
        }
    }

    // -------------------------------------------------------------------------
    // Dimension Queries
    // -------------------------------------------------------------------------

    [[nodiscard]] constexpr auto rows() const noexcept -> size_type { return rows_; }
    [[nodiscard]] constexpr auto cols() const noexcept -> size_type { return cols_; }
    [[nodiscard]] constexpr auto size() const noexcept -> size_type { return data_.size(); }
    [[nodiscard]] constexpr auto empty() const noexcept -> bool { return data_.empty(); }

    // -------------------------------------------------------------------------
    // Element Access
    // -------------------------------------------------------------------------

    // Access element at (i, j) with bounds checking
    [[nodiscard]] auto operator()(size_type i, size_type j) -> T& {
        if (i >= rows_ || j >= cols_) {
            throw std::out_of_range("Matrix index out of bounds");
        }
        return data_[index(i, j)];
    }

    [[nodiscard]] auto operator()(size_type i, size_type j) const -> const T& {
        if (i >= rows_ || j >= cols_) {
            throw std::out_of_range("Matrix index out of bounds");
        }
        return data_[index(i, j)];
    }

    // Raw data access for advanced use
    [[nodiscard]] auto data() noexcept -> T* { return data_.data(); }
    [[nodiscard]] auto data() const noexcept -> const T* { return data_.data(); }

    // -------------------------------------------------------------------------
    // Arithmetic Operators
    // -------------------------------------------------------------------------

    // Matrix addition: C = A + B
    // Requires same dimensions
    [[nodiscard]] auto operator+(const matrix& other) const -> matrix {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix dimensions must match for addition");
        }
        matrix result(rows_, cols_);
        for (size_type i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] + other.data_[i];
        }
        return result;
    }

    // Matrix subtraction: C = A - B
    [[nodiscard]] auto operator-(const matrix& other) const -> matrix {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix dimensions must match for subtraction");
        }
        matrix result(rows_, cols_);
        for (size_type i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] - other.data_[i];
        }
        return result;
    }

    // Unary negation: B = -A
    [[nodiscard]] auto operator-() const -> matrix {
        matrix result(rows_, cols_);
        for (size_type i = 0; i < data_.size(); ++i) {
            result.data_[i] = -data_[i];
        }
        return result;
    }

    // Scalar multiplication: B = A * s
    [[nodiscard]] auto operator*(T scalar) const -> matrix {
        matrix result(rows_, cols_);
        for (size_type i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] * scalar;
        }
        return result;
    }

    // Scalar division: B = A / s
    [[nodiscard]] auto operator/(T scalar) const -> matrix {
        matrix result(rows_, cols_);
        for (size_type i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] / scalar;
        }
        return result;
    }

    // Compound assignment operators
    auto operator+=(const matrix& other) -> matrix& {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix dimensions must match");
        }
        for (size_type i = 0; i < data_.size(); ++i) {
            data_[i] += other.data_[i];
        }
        return *this;
    }

    auto operator-=(const matrix& other) -> matrix& {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix dimensions must match");
        }
        for (size_type i = 0; i < data_.size(); ++i) {
            data_[i] -= other.data_[i];
        }
        return *this;
    }

    auto operator*=(T scalar) -> matrix& {
        for (auto& elem : data_) {
            elem *= scalar;
        }
        return *this;
    }

    auto operator/=(T scalar) -> matrix& {
        for (auto& elem : data_) {
            elem /= scalar;
        }
        return *this;
    }

    // -------------------------------------------------------------------------
    // Comparison
    // -------------------------------------------------------------------------

    [[nodiscard]] auto operator==(const matrix& other) const -> bool {
        return rows_ == other.rows_ && cols_ == other.cols_ && data_ == other.data_;
    }

    [[nodiscard]] auto operator!=(const matrix& other) const -> bool {
        return !(*this == other);
    }
};

// Left scalar multiplication: B = s * A
template <Arithmetic T>
[[nodiscard]] auto operator*(T scalar, const matrix<T>& m) -> matrix<T> {
    return m * scalar;
}

// =============================================================================
// Factory Functions
// =============================================================================

// Create identity matrix of size n x n
// The identity matrix I satisfies: A * I = I * A = A for any conformable A
template <Arithmetic T = double>
[[nodiscard]] auto eye(std::size_t n) -> matrix<T> {
    matrix<T> result(n, n, T{0});
    for (std::size_t i = 0; i < n; ++i) {
        result(i, i) = T{1};
    }
    return result;
}

// Create matrix of zeros
template <Arithmetic T = double>
[[nodiscard]] auto zeros(std::size_t rows, std::size_t cols) -> matrix<T> {
    return matrix<T>(rows, cols, T{0});
}

// Create matrix of ones
template <Arithmetic T = double>
[[nodiscard]] auto ones(std::size_t rows, std::size_t cols) -> matrix<T> {
    return matrix<T>(rows, cols, T{1});
}

// Create diagonal matrix from vector
template <Arithmetic T>
[[nodiscard]] auto diag(const std::vector<T>& d) -> matrix<T> {
    auto n = d.size();
    matrix<T> result(n, n, T{0});
    for (std::size_t i = 0; i < n; ++i) {
        result(i, i) = d[i];
    }
    return result;
}

// Extract diagonal from matrix as vector
template <Matrix M>
[[nodiscard]] auto diag(const M& A) -> std::vector<typename M::scalar_type> {
    auto n = std::min(A.rows(), A.cols());
    std::vector<typename M::scalar_type> d(n);
    for (std::size_t i = 0; i < n; ++i) {
        d[i] = A(i, i);
    }
    return d;
}

// Generic scalar multiplication for any Matrix type
// Use this in generic code instead of operator* to avoid concept issues
template <Matrix M>
[[nodiscard]] auto scale(typename M::scalar_type s, const M& A) -> matrix<typename M::scalar_type> {
    using T = typename M::scalar_type;
    matrix<T> result(A.rows(), A.cols());
    for (std::size_t i = 0; i < A.rows(); ++i) {
        for (std::size_t j = 0; j < A.cols(); ++j) {
            result(i, j) = s * A(i, j);
        }
    }
    return result;
}

template <Matrix M>
[[nodiscard]] auto scale(const M& A, typename M::scalar_type s) -> matrix<typename M::scalar_type> {
    return scale(s, A);
}

// =============================================================================
// Matrix Operations
// =============================================================================

// Matrix transpose: B = A^T
// Swaps rows and columns: B(i,j) = A(j,i)
template <Matrix M>
[[nodiscard]] auto transpose(const M& A) -> matrix<typename M::scalar_type> {
    using T = typename M::scalar_type;
    matrix<T> result(A.cols(), A.rows());
    for (std::size_t i = 0; i < A.rows(); ++i) {
        for (std::size_t j = 0; j < A.cols(); ++j) {
            result(j, i) = A(i, j);
        }
    }
    return result;
}

// Matrix multiplication: C = A * B
// Uses the standard O(n³) algorithm:
//   C(i,j) = Σ_k A(i,k) * B(k,j)
//
// Requires: A.cols() == B.rows()
// Result dimensions: (A.rows() x B.cols())
template <Matrix M1, Matrix M2>
    requires std::same_as<typename M1::scalar_type, typename M2::scalar_type>
[[nodiscard]] auto matmul(const M1& A, const M2& B) -> matrix<typename M1::scalar_type> {
    if (A.cols() != B.rows()) {
        throw std::invalid_argument(
            "Matrix dimensions incompatible for multiplication: "
            "A is " + std::to_string(A.rows()) + "x" + std::to_string(A.cols()) +
            ", B is " + std::to_string(B.rows()) + "x" + std::to_string(B.cols()));
    }

    using T = typename M1::scalar_type;
    matrix<T> C(A.rows(), B.cols(), T{0});

    for (std::size_t i = 0; i < A.rows(); ++i) {
        for (std::size_t k = 0; k < A.cols(); ++k) {
            // Loop interchange for better cache performance:
            // A(i,k) is reused across all j
            auto aik = A(i, k);
            for (std::size_t j = 0; j < B.cols(); ++j) {
                C(i, j) += aik * B(k, j);
            }
        }
    }
    return C;
}

// Operator* for matrix multiplication
template <Matrix M1, Matrix M2>
    requires std::same_as<typename M1::scalar_type, typename M2::scalar_type>
[[nodiscard]] auto operator*(const M1& A, const M2& B) -> matrix<typename M1::scalar_type> {
    return matmul(A, B);
}

// Element-wise multiplication (Hadamard product): C = A ⊙ B
template <Matrix M>
[[nodiscard]] auto hadamard(const M& A, const M& B) -> matrix<typename M::scalar_type> {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::invalid_argument("Matrix dimensions must match for Hadamard product");
    }

    using T = typename M::scalar_type;
    matrix<T> C(A.rows(), A.cols());
    for (std::size_t i = 0; i < A.rows(); ++i) {
        for (std::size_t j = 0; j < A.cols(); ++j) {
            C(i, j) = A(i, j) * B(i, j);
        }
    }
    return C;
}

// Element-wise division
template <Matrix M>
[[nodiscard]] auto elem_div(const M& A, const M& B) -> matrix<typename M::scalar_type> {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::invalid_argument("Matrix dimensions must match for element-wise division");
    }

    using T = typename M::scalar_type;
    matrix<T> C(A.rows(), A.cols());
    for (std::size_t i = 0; i < A.rows(); ++i) {
        for (std::size_t j = 0; j < A.cols(); ++j) {
            C(i, j) = A(i, j) / B(i, j);
        }
    }
    return C;
}

// =============================================================================
// Reductions
// =============================================================================

// Trace: sum of diagonal elements
// tr(A) = Σ_i A(i,i)
// Only defined for square matrices
template <Matrix M>
[[nodiscard]] auto trace(const M& A) -> typename M::scalar_type {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("Trace is only defined for square matrices");
    }

    using T = typename M::scalar_type;
    T sum{0};
    for (std::size_t i = 0; i < A.rows(); ++i) {
        sum += A(i, i);
    }
    return sum;
}

// Sum of all elements
template <Matrix M>
[[nodiscard]] auto sum(const M& A) -> typename M::scalar_type {
    using T = typename M::scalar_type;
    T total{0};
    for (std::size_t i = 0; i < A.rows(); ++i) {
        for (std::size_t j = 0; j < A.cols(); ++j) {
            total += A(i, j);
        }
    }
    return total;
}

// Mean of all elements
template <Matrix M>
[[nodiscard]] auto mean(const M& A) -> typename M::scalar_type {
    return sum(A) / static_cast<typename M::scalar_type>(A.rows() * A.cols());
}

// =============================================================================
// Norms
// =============================================================================

// Frobenius norm: ||A||_F = sqrt(Σ_ij |A(i,j)|²)
// This is the matrix analogue of the Euclidean vector norm.
template <Matrix M>
[[nodiscard]] auto frobenius_norm(const M& A) -> typename M::scalar_type {
    using T = typename M::scalar_type;
    T sum_sq{0};
    for (std::size_t i = 0; i < A.rows(); ++i) {
        for (std::size_t j = 0; j < A.cols(); ++j) {
            sum_sq += A(i, j) * A(i, j);
        }
    }
    return std::sqrt(sum_sq);
}

// L1 norm (max column sum): ||A||_1 = max_j Σ_i |A(i,j)|
template <Matrix M>
[[nodiscard]] auto l1_norm(const M& A) -> typename M::scalar_type {
    using T = typename M::scalar_type;
    T max_col_sum{0};
    for (std::size_t j = 0; j < A.cols(); ++j) {
        T col_sum{0};
        for (std::size_t i = 0; i < A.rows(); ++i) {
            col_sum += std::abs(A(i, j));
        }
        max_col_sum = std::max(max_col_sum, col_sum);
    }
    return max_col_sum;
}

// Infinity norm (max row sum): ||A||_∞ = max_i Σ_j |A(i,j)|
template <Matrix M>
[[nodiscard]] auto linf_norm(const M& A) -> typename M::scalar_type {
    using T = typename M::scalar_type;
    T max_row_sum{0};
    for (std::size_t i = 0; i < A.rows(); ++i) {
        T row_sum{0};
        for (std::size_t j = 0; j < A.cols(); ++j) {
            row_sum += std::abs(A(i, j));
        }
        max_row_sum = std::max(max_row_sum, row_sum);
    }
    return max_row_sum;
}

// =============================================================================
// LU Decomposition
// =============================================================================

// Result of LU decomposition with partial pivoting
// For matrix A, we compute P, L, U such that:
//   P * A = L * U
// where:
//   P = permutation matrix (row swaps)
//   L = lower triangular with 1s on diagonal
//   U = upper triangular
template <Arithmetic T>
struct lu_result {
    matrix<T> L;               // Lower triangular (unit diagonal)
    matrix<T> U;               // Upper triangular
    std::vector<std::size_t> perm;  // Permutation vector
    int sign;                  // Sign of permutation (+1 or -1)
    bool singular;             // True if matrix is singular
};

// LU decomposition with partial pivoting (Gaussian elimination)
//
// The algorithm:
// 1. For each column k:
//    a. Find the pivot (largest absolute value in column k, rows k to n)
//    b. Swap rows to bring pivot to diagonal
//    c. Eliminate entries below pivot using row operations
//
// Partial pivoting improves numerical stability by avoiding division
// by small numbers.
template <Matrix M>
[[nodiscard]] auto lu(const M& A) -> lu_result<typename M::scalar_type> {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("LU decomposition requires a square matrix");
    }

    using T = typename M::scalar_type;
    auto n = A.rows();

    // Work on a copy - this will become U
    matrix<T> U(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            U(i, j) = A(i, j);
        }
    }

    // L starts as identity, will accumulate multipliers
    matrix<T> L = eye<T>(n);

    // Permutation vector: perm[i] = row that ended up in position i
    std::vector<std::size_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0);

    int sign = 1;
    bool singular = false;

    // Tolerance for singularity detection
    constexpr T eps = std::numeric_limits<T>::epsilon() * T{100};

    for (std::size_t k = 0; k < n; ++k) {
        // Find pivot: largest absolute value in column k, rows k to n-1
        std::size_t pivot_row = k;
        T pivot_val = std::abs(U(k, k));

        for (std::size_t i = k + 1; i < n; ++i) {
            if (std::abs(U(i, k)) > pivot_val) {
                pivot_val = std::abs(U(i, k));
                pivot_row = i;
            }
        }

        // Check for singularity
        if (pivot_val < eps) {
            singular = true;
            continue;  // Skip this column
        }

        // Swap rows if needed
        if (pivot_row != k) {
            // Swap in U
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(U(k, j), U(pivot_row, j));
            }
            // Swap in L (only the part we've computed so far)
            for (std::size_t j = 0; j < k; ++j) {
                std::swap(L(k, j), L(pivot_row, j));
            }
            // Update permutation
            std::swap(perm[k], perm[pivot_row]);
            sign = -sign;
        }

        // Eliminate entries below pivot
        for (std::size_t i = k + 1; i < n; ++i) {
            // Multiplier: how much of row k to subtract from row i
            T mult = U(i, k) / U(k, k);
            L(i, k) = mult;  // Store multiplier in L

            // Update row i of U
            U(i, k) = T{0};  // Explicitly zero (would be zero anyway)
            for (std::size_t j = k + 1; j < n; ++j) {
                U(i, j) -= mult * U(k, j);
            }
        }
    }

    return {std::move(L), std::move(U), std::move(perm), sign, singular};
}

// =============================================================================
// Derived Operations (using LU)
// =============================================================================

// Determinant via LU decomposition
// det(A) = det(P^-1) * det(L) * det(U) = sign * Π_i U(i,i)
// since det(L) = 1 (unit diagonal) and det(P^-1) = sign of permutation
template <Matrix M>
[[nodiscard]] auto det(const M& A) -> typename M::scalar_type {
    auto [L, U, perm, sign, singular] = lu(A);

    if (singular) {
        return typename M::scalar_type{0};
    }

    using T = typename M::scalar_type;
    T result = static_cast<T>(sign);
    for (std::size_t i = 0; i < U.rows(); ++i) {
        result *= U(i, i);
    }
    return result;
}

// Log determinant (more numerically stable for large matrices)
// Returns {sign, log|det|} where det = sign * exp(log|det|)
template <Matrix M>
[[nodiscard]] auto logdet(const M& A) -> std::pair<int, typename M::scalar_type> {
    auto [L, U, perm, sign, singular] = lu(A);

    if (singular) {
        return {0, -std::numeric_limits<typename M::scalar_type>::infinity()};
    }

    using T = typename M::scalar_type;
    T log_abs_det{0};
    int result_sign = sign;

    for (std::size_t i = 0; i < U.rows(); ++i) {
        auto diag = U(i, i);
        if (diag < T{0}) {
            result_sign = -result_sign;
            diag = -diag;
        }
        log_abs_det += std::log(diag);
    }

    return {result_sign, log_abs_det};
}

// Forward substitution: solve L * x = b where L is lower triangular
// Used as part of solving A * x = b after LU decomposition
template <Matrix M, Matrix V>
    requires std::same_as<typename M::scalar_type, typename V::scalar_type>
[[nodiscard]] auto forward_substitute(const M& L, const V& b) -> matrix<typename M::scalar_type> {
    auto n = L.rows();
    if (b.rows() != n) {
        throw std::invalid_argument("Dimension mismatch in forward substitution");
    }

    using T = typename M::scalar_type;
    matrix<T> x(n, b.cols());

    for (std::size_t col = 0; col < b.cols(); ++col) {
        for (std::size_t i = 0; i < n; ++i) {
            T sum = b(i, col);
            for (std::size_t j = 0; j < i; ++j) {
                sum -= L(i, j) * x(j, col);
            }
            x(i, col) = sum / L(i, i);
        }
    }
    return x;
}

// Back substitution: solve U * x = b where U is upper triangular
template <Matrix M, Matrix V>
    requires std::same_as<typename M::scalar_type, typename V::scalar_type>
[[nodiscard]] auto back_substitute(const M& U, const V& b) -> matrix<typename M::scalar_type> {
    auto n = U.rows();
    if (b.rows() != n) {
        throw std::invalid_argument("Dimension mismatch in back substitution");
    }

    using T = typename M::scalar_type;
    matrix<T> x(n, b.cols());

    for (std::size_t col = 0; col < b.cols(); ++col) {
        for (std::size_t i = n; i-- > 0;) {
            T sum = b(i, col);
            for (std::size_t j = i + 1; j < n; ++j) {
                sum -= U(i, j) * x(j, col);
            }
            x(i, col) = sum / U(i, i);
        }
    }
    return x;
}

// Solve A * X = B for X using LU decomposition
// This is the standard way to solve linear systems:
// 1. Compute P * A = L * U
// 2. Permute B according to P
// 3. Solve L * Y = P * B (forward substitution)
// 4. Solve U * X = Y (back substitution)
template <Matrix M, Matrix V>
    requires std::same_as<typename M::scalar_type, typename V::scalar_type>
[[nodiscard]] auto solve(const M& A, const V& B) -> matrix<typename M::scalar_type> {
    auto [L, U, perm, sign, singular] = lu(A);

    if (singular) {
        throw std::runtime_error("Cannot solve: matrix is singular");
    }

    using T = typename M::scalar_type;
    auto n = A.rows();

    // Apply permutation to B
    matrix<T> Pb(n, B.cols());
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < B.cols(); ++j) {
            Pb(i, j) = B(perm[i], j);
        }
    }

    // Forward then back substitution
    auto Y = forward_substitute(L, Pb);
    return back_substitute(U, Y);
}

// Matrix inverse via solving A * X = I
// A^(-1) satisfies: A * A^(-1) = A^(-1) * A = I
template <Matrix M>
[[nodiscard]] auto inverse(const M& A) -> matrix<typename M::scalar_type> {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("Inverse is only defined for square matrices");
    }

    auto n = A.rows();
    auto I = eye<typename M::scalar_type>(n);
    return solve(A, I);
}

// =============================================================================
// Element-wise Functions
// =============================================================================

// Apply function to each element
template <Matrix M, typename F>
[[nodiscard]] auto apply(const M& A, F&& f) -> matrix<typename M::scalar_type> {
    using T = typename M::scalar_type;
    matrix<T> result(A.rows(), A.cols());
    for (std::size_t i = 0; i < A.rows(); ++i) {
        for (std::size_t j = 0; j < A.cols(); ++j) {
            result(i, j) = f(A(i, j));
        }
    }
    return result;
}

// Element-wise exp
template <Matrix M>
[[nodiscard]] auto exp(const M& A) -> matrix<typename M::scalar_type> {
    return apply(A, [](auto x) { return std::exp(x); });
}

// Element-wise log
template <Matrix M>
[[nodiscard]] auto log(const M& A) -> matrix<typename M::scalar_type> {
    return apply(A, [](auto x) { return std::log(x); });
}

// Element-wise sqrt
template <Matrix M>
[[nodiscard]] auto sqrt(const M& A) -> matrix<typename M::scalar_type> {
    return apply(A, [](auto x) { return std::sqrt(x); });
}

// Element-wise pow
template <Matrix M>
[[nodiscard]] auto pow(const M& A, typename M::scalar_type p) -> matrix<typename M::scalar_type> {
    return apply(A, [p](auto x) { return std::pow(x, p); });
}

// Element-wise abs
template <Matrix M>
[[nodiscard]] auto abs(const M& A) -> matrix<typename M::scalar_type> {
    return apply(A, [](auto x) { return std::abs(x); });
}

// =============================================================================
// Comparison Utilities
// =============================================================================

// Check if two matrices are approximately equal
template <Matrix M>
[[nodiscard]] auto approx_equal(const M& A, const M& B,
                                typename M::scalar_type rtol = 1e-5,
                                typename M::scalar_type atol = 1e-8) -> bool {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        return false;
    }

    for (std::size_t i = 0; i < A.rows(); ++i) {
        for (std::size_t j = 0; j < A.cols(); ++j) {
            auto diff = std::abs(A(i, j) - B(i, j));
            auto tol = atol + rtol * std::abs(B(i, j));
            if (diff > tol) {
                return false;
            }
        }
    }
    return true;
}

// Maximum absolute difference between matrices
template <Matrix M>
[[nodiscard]] auto max_diff(const M& A, const M& B) -> typename M::scalar_type {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::invalid_argument("Matrix dimensions must match");
    }

    using T = typename M::scalar_type;
    T max_d{0};
    for (std::size_t i = 0; i < A.rows(); ++i) {
        for (std::size_t j = 0; j < A.cols(); ++j) {
            max_d = std::max(max_d, std::abs(A(i, j) - B(i, j)));
        }
    }
    return max_d;
}

// =============================================================================
// I/O Utilities
// =============================================================================

// Convert matrix to string for debugging
template <Matrix M>
[[nodiscard]] auto to_string(const M& A, int precision = 4) -> std::string {
    std::ostringstream oss;
    oss << std::fixed;
    oss.precision(precision);

    for (std::size_t i = 0; i < A.rows(); ++i) {
        oss << (i == 0 ? "[" : " ");
        oss << "[";
        for (std::size_t j = 0; j < A.cols(); ++j) {
            if (j > 0) oss << ", ";
            oss << A(i, j);
        }
        oss << "]";
        oss << (i == A.rows() - 1 ? "]" : "\n");
    }
    return oss.str();
}

}  // namespace elementa

#endif  // ELEMENTA_HPP
