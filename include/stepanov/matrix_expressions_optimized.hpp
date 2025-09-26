#pragma once

#include <type_traits>
#include <concepts>
#include <memory>
#include <vector>
#include <utility>
#include <cstring>
#include "concepts.hpp"
#include "simd_operations.hpp"

namespace stepanov {

// Forward declaration
template<typename T, typename Storage>
class matrix;

template<typename T>
class row_major_storage;

} // namespace stepanov

namespace stepanov::matrix_expr {

// Import for operator definitions
using stepanov::matrix;
using stepanov::row_major_storage;

// =============================================================================
// Optimized Expression Templates with Perfect Forwarding and Move Semantics
// =============================================================================

// Expression traits to detect expression types
template<typename T>
struct is_matrix_expression : std::false_type {};

template<typename T>
inline constexpr bool is_matrix_expression_v = is_matrix_expression<T>::value;

// Base expression with CRTP and optimizations
template<typename Derived>
class expression_base {
public:
    using derived_type = Derived;

    // Perfect forwarding to derived class
    __attribute__((always_inline))
    const Derived& derived() const noexcept {
        return static_cast<const Derived&>(*this);
    }

    __attribute__((always_inline))
    Derived& derived() noexcept {
        return static_cast<Derived&>(*this);
    }

    // Size accessors
    size_t rows() const noexcept { return derived().rows_impl(); }
    size_t cols() const noexcept { return derived().cols_impl(); }

    // Element access - marked for aggressive inlining
    __attribute__((always_inline, hot))
    auto operator()(size_t i, size_t j) const {
        return derived().at_impl(i, j);
    }

    // Enable move semantics
    expression_base() = default;
    expression_base(const expression_base&) = default;
    expression_base(expression_base&&) noexcept = default;
    expression_base& operator=(const expression_base&) = default;
    expression_base& operator=(expression_base&&) noexcept = default;
};

// Mark all expressions as matrix expressions
template<typename Derived>
struct is_matrix_expression<expression_base<Derived>> : std::true_type {};

// =============================================================================
// Optimized Binary Expression with Reference Semantics
// =============================================================================

template<typename E1, typename E2, typename Op>
class binary_expression : public expression_base<binary_expression<E1, E2, Op>> {
private:
    using lhs_stored = std::conditional_t<is_matrix_expression_v<std::decay_t<E1>>,
                                          E1, const E1&>;
    using rhs_stored = std::conditional_t<is_matrix_expression_v<std::decay_t<E2>>,
                                          E2, const E2&>;

    lhs_stored lhs_;
    rhs_stored rhs_;
    Op op_;

public:
    using value_type = decltype(std::declval<Op>()(
        std::declval<E1>()(0, 0),
        std::declval<E2>()(0, 0)));

    // Perfect forwarding constructor
    template<typename L, typename R>
    binary_expression(L&& l, R&& r, Op op = Op{})
        : lhs_(std::forward<L>(l)),
          rhs_(std::forward<R>(r)),
          op_(std::move(op)) {}

    size_t rows_impl() const noexcept { return lhs_.rows(); }
    size_t cols_impl() const noexcept { return lhs_.cols(); }

    __attribute__((always_inline, hot))
    auto at_impl(size_t i, size_t j) const {
        return op_(lhs_(i, j), rhs_(i, j));
    }

    // Enable vectorized evaluation
    template<typename T>
    void evaluate_to(T* result, size_t stride) const {
        const size_t m = this->rows();
        const size_t n = this->cols();

        if constexpr (std::is_same_v<Op, std::plus<>> ||
                     std::is_same_v<Op, std::minus<>>) {
            // Use SIMD for element-wise operations
            #pragma omp parallel for if(m * n > 10000)
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; j += simd::double_simd::simd_width) {
                    if (j + simd::double_simd::simd_width <= n) {
                        alignas(64) value_type temp_lhs[simd::double_simd::simd_width];
                        alignas(64) value_type temp_rhs[simd::double_simd::simd_width];

                        // Gather elements
                        #pragma omp simd
                        for (size_t k = 0; k < simd::double_simd::simd_width; ++k) {
                            temp_lhs[k] = lhs_(i, j + k);
                            temp_rhs[k] = rhs_(i, j + k);
                        }

                        // Apply operation
                        #pragma omp simd
                        for (size_t k = 0; k < simd::double_simd::simd_width; ++k) {
                            result[i * stride + j + k] = op_(temp_lhs[k], temp_rhs[k]);
                        }
                    } else {
                        // Handle remainder
                        for (size_t jj = j; jj < n; ++jj) {
                            result[i * stride + jj] = op_(lhs_(i, jj), rhs_(i, jj));
                        }
                    }
                }
            }
        } else {
            // General case
            #pragma omp parallel for collapse(2) if(m * n > 10000)
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    result[i * stride + j] = op_(lhs_(i, j), rhs_(i, j));
                }
            }
        }
    }
};

// =============================================================================
// Optimized Unary Expression
// =============================================================================

template<typename E, typename Op>
class unary_expression : public expression_base<unary_expression<E, Op>> {
private:
    using expr_stored = std::conditional_t<is_matrix_expression_v<std::decay_t<E>>,
                                           E, const E&>;
    expr_stored expr_;
    Op op_;

public:
    using value_type = decltype(std::declval<Op>()(std::declval<E>()(0, 0)));

    template<typename Expr>
    unary_expression(Expr&& e, Op op = Op{})
        : expr_(std::forward<Expr>(e)), op_(std::move(op)) {}

    size_t rows_impl() const noexcept { return expr_.rows(); }
    size_t cols_impl() const noexcept { return expr_.cols(); }

    __attribute__((always_inline, hot))
    auto at_impl(size_t i, size_t j) const {
        return op_(expr_(i, j));
    }
};

// =============================================================================
// Scalar Expression Optimizations
// =============================================================================

template<typename E, typename Scalar>
class scalar_multiply_expression : public expression_base<scalar_multiply_expression<E, Scalar>> {
private:
    using expr_stored = std::conditional_t<is_matrix_expression_v<std::decay_t<E>>,
                                           E, const E&>;
    expr_stored expr_;
    Scalar scalar_;

public:
    using value_type = decltype(std::declval<Scalar>() * std::declval<E>()(0, 0));

    template<typename Expr>
    scalar_multiply_expression(Expr&& e, Scalar s)
        : expr_(std::forward<Expr>(e)), scalar_(s) {}

    size_t rows_impl() const noexcept { return expr_.rows(); }
    size_t cols_impl() const noexcept { return expr_.cols(); }

    __attribute__((always_inline, hot))
    auto at_impl(size_t i, size_t j) const {
        return scalar_ * expr_(i, j);
    }

    // Optimized evaluation with SIMD
    template<typename T>
    void evaluate_to(T* result, size_t stride) const {
        const size_t m = this->rows();
        const size_t n = this->cols();

        #pragma omp parallel for if(m * n > 10000)
        for (size_t i = 0; i < m; ++i) {
            if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
                // Vectorized scalar multiplication
                alignas(64) T temp[simd::double_simd::simd_width];

                size_t j = 0;
                for (; j + simd::double_simd::simd_width <= n; j += simd::double_simd::simd_width) {
                    #pragma omp simd aligned(temp:64)
                    for (size_t k = 0; k < simd::double_simd::simd_width; ++k) {
                        temp[k] = expr_(i, j + k);
                    }

                    simd::matrix_scalar_mul_simd(temp, scalar_, &result[i * stride + j],
                                                 simd::double_simd::simd_width);
                }

                // Handle remainder
                for (; j < n; ++j) {
                    result[i * stride + j] = scalar_ * expr_(i, j);
                }
            } else {
                for (size_t j = 0; j < n; ++j) {
                    result[i * stride + j] = scalar_ * expr_(i, j);
                }
            }
        }
    }
};

// =============================================================================
// Matrix Wrapper for Expressions
// =============================================================================

template<typename T, typename Storage = row_major_storage<T>>
class matrix_ref : public expression_base<matrix_ref<T, Storage>> {
private:
    const matrix<T, Storage>* mat_;

public:
    using value_type = T;

    explicit matrix_ref(const matrix<T, Storage>& m) : mat_(&m) {}

    size_t rows_impl() const noexcept { return mat_->rows(); }
    size_t cols_impl() const noexcept { return mat_->cols(); }

    __attribute__((always_inline, hot))
    const T& at_impl(size_t i, size_t j) const {
        return (*mat_)(i, j);
    }

    // Direct data access for optimizations
    const T* data() const { return mat_->data(); }
};

template<typename T, typename Storage>
struct is_matrix_expression<matrix_ref<T, Storage>> : std::true_type {};

// =============================================================================
// Optimized Operators with Expression Building
// =============================================================================

// Addition
template<typename E1, typename E2>
    requires is_matrix_expression_v<std::decay_t<E1>> ||
             is_matrix_expression_v<std::decay_t<E2>>
__attribute__((always_inline))
auto operator+(E1&& lhs, E2&& rhs) {
    return binary_expression<E1, E2, std::plus<>>(
        std::forward<E1>(lhs),
        std::forward<E2>(rhs),
        std::plus<>{}
    );
}

// Subtraction
template<typename E1, typename E2>
    requires is_matrix_expression_v<std::decay_t<E1>> ||
             is_matrix_expression_v<std::decay_t<E2>>
__attribute__((always_inline))
auto operator-(E1&& lhs, E2&& rhs) {
    return binary_expression<E1, E2, std::minus<>>(
        std::forward<E1>(lhs),
        std::forward<E2>(rhs),
        std::minus<>{}
    );
}

// Scalar multiplication
template<typename E, typename S>
    requires is_matrix_expression_v<std::decay_t<E>> && std::is_arithmetic_v<S>
__attribute__((always_inline))
auto operator*(S scalar, E&& expr) {
    return scalar_multiply_expression<E, S>(std::forward<E>(expr), scalar);
}

template<typename E, typename S>
    requires is_matrix_expression_v<std::decay_t<E>> && std::is_arithmetic_v<S>
__attribute__((always_inline))
auto operator*(E&& expr, S scalar) {
    return scalar_multiply_expression<E, S>(std::forward<E>(expr), scalar);
}

// Negation
template<typename E>
    requires is_matrix_expression_v<std::decay_t<E>>
__attribute__((always_inline))
auto operator-(E&& expr) {
    return unary_expression<E, std::negate<>>(
        std::forward<E>(expr),
        std::negate<>{}
    );
}

// =============================================================================
// Expression Evaluation with Loop Fusion
// =============================================================================

// Force evaluation of expression into a matrix
template<typename E, typename T = typename std::decay_t<E>::value_type,
         typename Storage = row_major_storage<T>>
    requires is_matrix_expression_v<std::decay_t<E>>
matrix<T, Storage> evaluate(E&& expr) {
    const size_t m = expr.rows();
    const size_t n = expr.cols();
    matrix<T, Storage> result(m, n);

    // Use optimized evaluation if available
    if constexpr (requires { expr.evaluate_to(result.data(), n); }) {
        expr.evaluate_to(result.data(), n);
    } else {
        // Fallback to element-wise evaluation with OpenMP
        T* data = result.data();

        #pragma omp parallel for collapse(2) if(m * n > 10000)
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                data[i * n + j] = expr(i, j);
            }
        }
    }

    return result;
}

// =============================================================================
// Lazy Evaluation Assignment Operator
// =============================================================================

// Enable expression assignment to matrix
template<typename T, typename Storage, typename E>
    requires is_matrix_expression_v<std::decay_t<E>>
matrix<T, Storage>& assign_expression(matrix<T, Storage>& mat, E&& expr) {
    const size_t m = expr.rows();
    const size_t n = expr.cols();

    if (mat.rows() != m || mat.cols() != n) {
        mat = matrix<T, Storage>(m, n);
    }

    // Use optimized evaluation
    if constexpr (requires { expr.evaluate_to(mat.data(), n); }) {
        expr.evaluate_to(mat.data(), n);
    } else {
        T* data = mat.data();

        #pragma omp parallel for collapse(2) if(m * n > 10000)
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                data[i * n + j] = expr(i, j);
            }
        }
    }

    return mat;
}

// =============================================================================
// Expression Chain Optimization
// =============================================================================

// Detect and optimize common expression patterns
template<typename E>
struct expression_optimizer {
    using type = E;

    static E optimize(E&& expr) {
        return std::forward<E>(expr);
    }
};

// Optimize (A + B) + C to A + (B + C) for better cache usage
template<typename E1, typename E2, typename E3>
struct expression_optimizer<
    binary_expression<binary_expression<E1, E2, std::plus<>>, E3, std::plus<>>> {
    using inner = binary_expression<E1, E2, std::plus<>>;
    using outer = binary_expression<inner, E3, std::plus<>>;
    using optimized = binary_expression<E1, binary_expression<E2, E3, std::plus<>>, std::plus<>>;

    static optimized optimize(outer&& expr) {
        return optimized(
            std::move(expr.lhs_.lhs_),
            binary_expression<E2, E3, std::plus<>>(
                std::move(expr.lhs_.rhs_),
                std::move(expr.rhs_)
            )
        );
    }
};

// =============================================================================
// Matrix Expression Wrappers
// =============================================================================

// Convert matrix to expression
template<typename T, typename Storage>
__attribute__((always_inline))
auto as_expression(const matrix<T, Storage>& m) {
    return matrix_ref<T, Storage>(m);
}

// Enable expression templates for matrix operations
template<typename T, typename Storage>
__attribute__((always_inline))
auto operator+(const matrix<T, Storage>& a, const matrix<T, Storage>& b) {
    return as_expression(a) + as_expression(b);
}

template<typename T, typename Storage>
__attribute__((always_inline))
auto operator-(const matrix<T, Storage>& a, const matrix<T, Storage>& b) {
    return as_expression(a) - as_expression(b);
}

template<typename T, typename Storage, typename S>
    requires std::is_arithmetic_v<S>
__attribute__((always_inline))
auto operator*(S scalar, const matrix<T, Storage>& m) {
    return scalar * as_expression(m);
}

template<typename T, typename Storage, typename S>
    requires std::is_arithmetic_v<S>
__attribute__((always_inline))
auto operator*(const matrix<T, Storage>& m, S scalar) {
    return as_expression(m) * scalar;
}

} // namespace stepanov::matrix_expr