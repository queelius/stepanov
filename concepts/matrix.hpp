// matrix.hpp - Matrix and vector concepts
//
// Pure concept definitions for linear algebra types. Zero implementation.
// Extracted and generalized from elementa's Matrix concept so that
// gradator, alea, and any future library can depend on the concept
// without depending on elementa's implementation.

#pragma once

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace stepanov {

// =============================================================================
// Matrix concept
// =============================================================================

/// A Matrix is a 2D array of scalars with arithmetic operations.
///
/// Any type satisfying this concept can be used with gradator (AD),
/// alea (multivariate distributions), and any generic matrix algorithm.
///
/// Note: Scalar multiplication is deliberately excluded from the concept
/// to avoid circular constraint issues with matmul operator*. Use scale()
/// for generic scalar-matrix multiplication.
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
// SquareMatrix concept
// =============================================================================

/// A SquareMatrix is a Matrix whose dimensions are known to be equal.
/// This is a runtime-checked refinement — we cannot verify squareness
/// at compile time with dynamic dimensions, so this concept is for
/// documentation and overload resolution.
template <typename M>
concept SquareMatrix = Matrix<M>;

// =============================================================================
// Vector concept
// =============================================================================

/// A column vector is a Matrix with cols() == 1.
/// Like SquareMatrix, this is a semantic refinement.
template <typename M>
concept Vector = Matrix<M>;

// =============================================================================
// PositiveDefinite tag
// =============================================================================

// PositiveDefinite is not checkable at compile time.
// It is a precondition, documented in function signatures.
// Algorithms like Cholesky decomposition require positive definiteness
// but cannot verify it statically — they detect failure at runtime.

} // namespace stepanov
