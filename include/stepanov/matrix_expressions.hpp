#pragma once

#include <type_traits>
#include <concepts>
#include <memory>
#include <vector>
#include "concepts.hpp"

namespace stepanov {

// Forward declaration
template<typename T, typename Storage>
class matrix;

template<typename T>
class row_major_storage;

} // namespace stepanov

namespace stepanov::matrix_expr {

// Import matrix class into this namespace for operator definitions
using stepanov::matrix;
using stepanov::row_major_storage;

/**
 * Matrix expression templates with compile-time property tracking
 *
 * This system tracks matrix properties through the type system:
 * - Symmetry, diagonal, triangular, orthogonal, etc.
 * - Enables optimal algorithms based on compile-time knowledge
 * - Zero runtime overhead through template metaprogramming
 */

// =============================================================================
// Matrix property tags (compile-time properties)
// =============================================================================

struct general_tag {};
struct symmetric_tag : general_tag {};
struct diagonal_tag : symmetric_tag {};
struct identity_tag : diagonal_tag {};
struct triangular_lower_tag : general_tag {};
struct triangular_upper_tag : general_tag {};
struct orthogonal_tag : general_tag {};
struct hermitian_tag : symmetric_tag {};  // For complex matrices
struct sparse_tag : general_tag {};
struct banded_tag : general_tag {};
struct tridiagonal_tag : banded_tag {};

// Property concepts
template<typename T>
concept matrix_property = std::is_base_of_v<general_tag, T>;

template<typename T>
concept is_symmetric = std::is_base_of_v<symmetric_tag, T>;

template<typename T>
concept is_diagonal = std::is_base_of_v<diagonal_tag, T>;

template<typename T>
concept is_triangular = std::is_base_of_v<triangular_lower_tag, T> ||
                        std::is_base_of_v<triangular_upper_tag, T>;

// =============================================================================
// Expression templates
// =============================================================================

/**
 * Base expression template for lazy evaluation
 */
template<typename Derived, typename ValueType, typename PropertyTag = general_tag>
class matrix_expr_base {
public:
    using value_type = ValueType;
    using property_tag = PropertyTag;

    // CRTP downcasting
    const Derived& derived() const {
        return static_cast<const Derived&>(*this);
    }

    // Size queries (delegated to derived)
    size_t rows() const { return derived().rows(); }
    size_t cols() const { return derived().cols(); }

    // Element access (delegated to derived)
    value_type operator()(size_t i, size_t j) const {
        return derived()(i, j);
    }

    // Check if element is structurally zero
    bool is_zero(size_t i, size_t j) const {
        return derived().is_structural_zero(i, j);
    }
};

/**
 * Expression for matrix addition
 * Property preservation rules:
 * - symmetric + symmetric = symmetric
 * - diagonal + diagonal = diagonal
 * - triangular + triangular (same) = triangular
 */
template<typename E1, typename E2, typename ResultTag = general_tag>
class matrix_add_expr : public matrix_expr_base<
    matrix_add_expr<E1, E2, ResultTag>,
    typename E1::value_type,
    ResultTag> {
private:
    const E1& lhs_;
    const E2& rhs_;

public:
    using value_type = typename E1::value_type;
    using property_tag = ResultTag;

    matrix_add_expr(const E1& lhs, const E2& rhs)
        : lhs_(lhs), rhs_(rhs) {}

    size_t rows() const { return lhs_.rows(); }
    size_t cols() const { return lhs_.cols(); }

    value_type operator()(size_t i, size_t j) const {
        return lhs_(i, j) + rhs_(i, j);
    }

    bool is_structural_zero(size_t i, size_t j) const {
        return lhs_.is_zero(i, j) && rhs_.is_zero(i, j);
    }
};

/**
 * Expression for matrix multiplication
 * Property preservation rules:
 * - diagonal * diagonal = diagonal
 * - triangular * triangular (same) = triangular
 * - orthogonal * orthogonal = orthogonal
 * - symmetric * symmetric ≠ symmetric (in general)
 */
template<typename E1, typename E2, typename ResultTag = general_tag>
class matrix_mul_expr : public matrix_expr_base<
    matrix_mul_expr<E1, E2, ResultTag>,
    typename E1::value_type,
    ResultTag> {
private:
    const E1& lhs_;
    const E2& rhs_;

public:
    using value_type = typename E1::value_type;
    using property_tag = ResultTag;

    matrix_mul_expr(const E1& lhs, const E2& rhs)
        : lhs_(lhs), rhs_(rhs) {}

    size_t rows() const { return lhs_.rows(); }
    size_t cols() const { return rhs_.cols(); }

    value_type operator()(size_t i, size_t j) const {
        value_type sum = value_type(0);

        // Optimize based on known structure
        if constexpr (is_diagonal<typename E1::property_tag> &&
                      is_diagonal<typename E2::property_tag>) {
            // Diagonal * Diagonal: O(1) per element
            if (i == j) {
                sum = lhs_(i, i) * rhs_(i, i);
            }
        } else if constexpr (is_diagonal<typename E1::property_tag>) {
            // Diagonal * General: Skip off-diagonal of left
            sum = lhs_(i, i) * rhs_(i, j);
        } else if constexpr (is_diagonal<typename E2::property_tag>) {
            // General * Diagonal: Skip off-diagonal of right
            sum = lhs_(i, j) * rhs_(j, j);
        } else {
            // General case
            for (size_t k = 0; k < lhs_.cols(); ++k) {
                if (!lhs_.is_zero(i, k) && !rhs_.is_zero(k, j)) {
                    sum += lhs_(i, k) * rhs_(k, j);
                }
            }
        }

        return sum;
    }

    bool is_structural_zero(size_t i, size_t j) const {
        // Check if this position must be zero based on structure
        if constexpr (is_diagonal<ResultTag>) {
            return i != j;
        }
        // Could add more structural zero checks
        return false;
    }
};

/**
 * Expression for matrix transpose
 * Property transformations:
 * - symmetric^T = symmetric
 * - lower_triangular^T = upper_triangular
 * - orthogonal^T = orthogonal^(-1)
 */
template<typename E, typename ResultTag = general_tag>
class matrix_transpose_expr : public matrix_expr_base<
    matrix_transpose_expr<E, ResultTag>,
    typename E::value_type,
    ResultTag> {
private:
    const E& expr_;

public:
    using value_type = typename E::value_type;
    using property_tag = ResultTag;

    explicit matrix_transpose_expr(const E& expr) : expr_(expr) {}

    size_t rows() const { return expr_.cols(); }
    size_t cols() const { return expr_.rows(); }

    value_type operator()(size_t i, size_t j) const {
        return expr_(j, i);
    }

    bool is_structural_zero(size_t i, size_t j) const {
        return expr_.is_zero(j, i);
    }
};

/**
 * Scalar multiplication expression
 * Properties are preserved under scalar multiplication
 */
template<typename E>
class matrix_scalar_expr : public matrix_expr_base<
    matrix_scalar_expr<E>,
    typename E::value_type,
    typename E::property_tag> {
private:
    const E& expr_;
    typename E::value_type scalar_;

public:
    using value_type = typename E::value_type;
    using property_tag = typename E::property_tag;

    matrix_scalar_expr(const E& expr, value_type scalar)
        : expr_(expr), scalar_(scalar) {}

    size_t rows() const { return expr_.rows(); }
    size_t cols() const { return expr_.cols(); }

    value_type operator()(size_t i, size_t j) const {
        return scalar_ * expr_(i, j);
    }

    bool is_structural_zero(size_t i, size_t j) const {
        return expr_.is_zero(i, j) || scalar_ == value_type(0);
    }
};

// =============================================================================
// Specialized matrix types with properties
// =============================================================================

/**
 * Diagonal matrix - O(n) storage
 */
template<typename T>
class diagonal_matrix : public matrix_expr_base<
    diagonal_matrix<T>, T, diagonal_tag> {
private:
    std::vector<T> diag_;
    size_t size_;

public:
    using value_type = T;
    using property_tag = diagonal_tag;

    explicit diagonal_matrix(size_t n) : diag_(n), size_(n) {}

    diagonal_matrix(std::initializer_list<T> diag)
        : diag_(diag), size_(diag.size()) {}

    size_t rows() const { return size_; }
    size_t cols() const { return size_; }

    T operator()(size_t i, size_t j) const {
        return (i == j) ? diag_[i] : T(0);
    }

    T& diagonal(size_t i) { return diag_[i]; }
    const T& diagonal(size_t i) const { return diag_[i]; }

    size_t size() const { return size_; }

    bool is_structural_zero(size_t i, size_t j) const {
        return i != j;
    }

    // Efficient inverse for diagonal matrices
    diagonal_matrix inverse() const
        requires field<T>
    {
        diagonal_matrix result(size_);
        for (size_t i = 0; i < size_; ++i) {
            result.diagonal(i) = T(1) / diag_[i];
        }
        return result;
    }

    // Determinant is product of diagonal
    T determinant() const {
        T det = T(1);
        for (const auto& d : diag_) {
            det *= d;
        }
        return det;
    }
};

/**
 * Triangular matrix (upper or lower)
 */
template<typename T, typename TriangularTag>
    requires std::is_same_v<TriangularTag, triangular_lower_tag> ||
             std::is_same_v<TriangularTag, triangular_upper_tag>
class triangular_matrix : public matrix_expr_base<
    triangular_matrix<T, TriangularTag>, T, TriangularTag> {
private:
    std::vector<T> data_;  // Stores only the triangular part
    size_t size_;
    static constexpr bool is_lower = std::is_same_v<TriangularTag, triangular_lower_tag>;

    size_t index(size_t i, size_t j) const {
        if constexpr (is_lower) {
            // Lower triangular: i >= j
            return (i * (i + 1)) / 2 + j;
        } else {
            // Upper triangular: i <= j
            return (i * (2 * size_ - i - 1)) / 2 + (j - i);
        }
    }

public:
    using value_type = T;
    using property_tag = TriangularTag;

    explicit triangular_matrix(size_t n)
        : data_((n * (n + 1)) / 2, T(0)), size_(n) {}

    size_t rows() const { return size_; }
    size_t cols() const { return size_; }

    T operator()(size_t i, size_t j) const {
        if constexpr (is_lower) {
            return (i >= j) ? data_[index(i, j)] : T(0);
        } else {
            return (i <= j) ? data_[index(i, j)] : T(0);
        }
    }

    T& at(size_t i, size_t j) {
        if constexpr (is_lower) {
            if (i < j) throw std::out_of_range("Upper triangle access in lower triangular matrix");
        } else {
            if (i > j) throw std::out_of_range("Lower triangle access in upper triangular matrix");
        }
        return data_[index(i, j)];
    }

    bool is_structural_zero(size_t i, size_t j) const {
        if constexpr (is_lower) {
            return i < j;
        } else {
            return i > j;
        }
    }

    // Efficient forward/back substitution for solving triangular systems
    std::vector<T> solve(const std::vector<T>& b) const
        requires field<T>
    {
        std::vector<T> x(size_);

        if constexpr (is_lower) {
            // Forward substitution: Lx = b
            for (size_t i = 0; i < size_; ++i) {
                T sum = b[i];
                for (size_t j = 0; j < i; ++j) {
                    sum -= (*this)(i, j) * x[j];
                }
                x[i] = sum / (*this)(i, i);
            }
        } else {
            // Back substitution: Ux = b
            for (int i = size_ - 1; i >= 0; --i) {
                T sum = b[i];
                for (size_t j = i + 1; j < size_; ++j) {
                    sum -= (*this)(i, j) * x[j];
                }
                x[i] = sum / (*this)(i, i);
            }
        }

        return x;
    }
};

// Type aliases
template<typename T>
using lower_triangular = triangular_matrix<T, triangular_lower_tag>;

template<typename T>
using upper_triangular = triangular_matrix<T, triangular_upper_tag>;

// =============================================================================
// Operator overloads with property deduction
// =============================================================================

// Helper to deduce result property from operation
template<typename Tag1, typename Tag2>
struct add_result_tag {
    using type = std::conditional_t<
        std::is_same_v<Tag1, Tag2> && is_symmetric<Tag1>,
        Tag1,
        general_tag
    >;
};

template<typename Tag1, typename Tag2>
struct mul_result_tag {
    using type = std::conditional_t<
        is_diagonal<Tag1> && is_diagonal<Tag2>,
        diagonal_tag,
        general_tag
    >;
};

// Addition
template<typename E1, typename E2>
    requires std::is_same_v<typename E1::value_type, typename E2::value_type>
auto operator+(const matrix_expr_base<E1, typename E1::value_type, typename E1::property_tag>& lhs,
               const matrix_expr_base<E2, typename E2::value_type, typename E2::property_tag>& rhs) {
    using result_tag = typename add_result_tag<
        typename E1::property_tag,
        typename E2::property_tag>::type;

    return matrix_add_expr<E1, E2, result_tag>(
        static_cast<const E1&>(lhs),
        static_cast<const E2&>(rhs)
    );
}

// Multiplication
template<typename E1, typename E2>
    requires std::is_same_v<typename E1::value_type, typename E2::value_type>
auto operator*(const matrix_expr_base<E1, typename E1::value_type, typename E1::property_tag>& lhs,
               const matrix_expr_base<E2, typename E2::value_type, typename E2::property_tag>& rhs) {
    using result_tag = typename mul_result_tag<
        typename E1::property_tag,
        typename E2::property_tag>::type;

    return matrix_mul_expr<E1, E2, result_tag>(
        static_cast<const E1&>(lhs),
        static_cast<const E2&>(rhs)
    );
}

// Scalar multiplication
template<typename E, typename T>
    requires std::is_same_v<T, typename E::value_type>
auto operator*(const T& scalar,
               const matrix_expr_base<E, typename E::value_type, typename E::property_tag>& expr) {
    return matrix_scalar_expr<E>(static_cast<const E&>(expr), scalar);
}

// Matrix multiplication operators

// Diagonal matrix * regular matrix
template<typename T, typename Storage = row_major_storage<T>>
matrix<T, Storage> operator*(const diagonal_matrix<T>& D, const matrix<T, Storage>& M) {
    size_t n = D.size();
    matrix<T, Storage> result(n, M.cols());

    // O(n²) instead of O(n³)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < M.cols(); ++j) {
            result(i, j) = D.diagonal(i) * M(i, j);
        }
    }

    return result;
}

// Regular matrix * diagonal matrix
template<typename T, typename Storage = row_major_storage<T>>
matrix<T, Storage> operator*(const matrix<T, Storage>& M, const diagonal_matrix<T>& D) {
    size_t n = D.size();
    matrix<T, Storage> result(M.rows(), n);

    // O(n²) instead of O(n³)
    for (size_t i = 0; i < M.rows(); ++i) {
        for (size_t j = 0; j < n; ++j) {
            result(i, j) = M(i, j) * D.diagonal(j);
        }
    }

    return result;
}

// Diagonal * diagonal = diagonal
template<typename T>
diagonal_matrix<T> operator*(const diagonal_matrix<T>& D1, const diagonal_matrix<T>& D2) {
    size_t n = D1.size();
    diagonal_matrix<T> result(n);

    // O(n) instead of O(n³)
    for (size_t i = 0; i < n; ++i) {
        result.diagonal(i) = D1.diagonal(i) * D2.diagonal(i);
    }

    return result;
}

// Matrix addition and subtraction
template<typename T, typename Storage = row_major_storage<T>>
matrix<T, Storage> operator+(const matrix<T, Storage>& A, const matrix<T, Storage>& B) {
    size_t m = A.rows();
    size_t n = A.cols();
    matrix<T, Storage> result(m, n);

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result(i, j) = A(i, j) + B(i, j);
        }
    }

    return result;
}

template<typename T, typename Storage = row_major_storage<T>>
matrix<T, Storage> operator-(const matrix<T, Storage>& A, const matrix<T, Storage>& B) {
    size_t m = A.rows();
    size_t n = A.cols();
    matrix<T, Storage> result(m, n);

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result(i, j) = A(i, j) - B(i, j);
        }
    }

    return result;
}

// Scalar multiplication for matrix
template<typename T, typename Storage = row_major_storage<T>>
matrix<T, Storage> operator*(const T& scalar, const matrix<T, Storage>& M) {
    size_t m = M.rows();
    size_t n = M.cols();
    matrix<T, Storage> result(m, n);

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result(i, j) = scalar * M(i, j);
        }
    }

    return result;
}

template<typename T, typename Storage = row_major_storage<T>>
matrix<T, Storage> operator*(const matrix<T, Storage>& M, const T& scalar) {
    return scalar * M;
}

// Matrix-vector multiplication
template<typename T, typename Storage = row_major_storage<T>>
std::vector<T> operator*(const matrix<T, Storage>& M, const std::vector<T>& v) {
    size_t m = M.rows();
    size_t n = M.cols();
    std::vector<T> result(m, T(0));

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result[i] += M(i, j) * v[j];
        }
    }

    return result;
}

// Diagonal matrix-vector multiplication
template<typename T>
std::vector<T> operator*(const diagonal_matrix<T>& D, const std::vector<T>& v) {
    size_t n = D.size();
    std::vector<T> result(n);

    // O(n) instead of O(n²)
    for (size_t i = 0; i < n; ++i) {
        result[i] = D.diagonal(i) * v[i];
    }

    return result;
}

// =============================================================================
// Expression evaluation and materialization
// =============================================================================

/**
 * Force evaluation of expression into concrete matrix
 * This is where the actual computation happens
 */
template<typename E, template<typename> class MatrixType = diagonal_matrix>
    requires is_diagonal<typename E::property_tag>
auto evaluate(const E& expr) {
    using T = typename E::value_type;
    diagonal_matrix<T> result(expr.rows());

    for (size_t i = 0; i < expr.rows(); ++i) {
        result.diagonal(i) = expr(i, i);
    }

    return result;
}

// General evaluation (would need a general matrix class)
template<typename E>
std::vector<std::vector<typename E::value_type>> evaluate_dense(const E& expr) {
    using T = typename E::value_type;
    size_t m = expr.rows();
    size_t n = expr.cols();

    std::vector<std::vector<T>> result(m, std::vector<T>(n));

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (!expr.is_structural_zero(i, j)) {
                result[i][j] = expr(i, j);
            }
        }
    }

    return result;
}

} // namespace stepanov::matrix_expr