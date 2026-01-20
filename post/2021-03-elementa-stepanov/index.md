---
author:
  name: "Alex Towell"
  email: "queelius@gmail.com"
  url: "https://metafunctor.com"
title: "Teaching Linear Algebra with C++20 Concepts"
date: 2021-03-08
draft: false
tags:
  - C++
  - C++20
  - linear-algebra
  - concepts
  - matrices
  - generic-programming
categories:
  - Computer Science
  - Mathematics
series: ["stepanov"]
series_weight: 6
math: true
description: "elementa is a pedagogical linear algebra library where every design decision prioritizes clarity over cleverness---code that reads like a textbook that happens to compile."
---

Why build yet another linear algebra library? The world already has Eigen, Armadillo, Blaze, and countless others. The answer lies not in performance or features, but in *pedagogy*.

elementa exists to teach three things simultaneously: linear algebra, modern C++, and numerical computing. Every design decision prioritizes clarity over cleverness. The code reads like a textbook that happens to compile.

## The Matrix Concept

C++20 concepts let us express "what a matrix is" as a compile-time contract. Here's the core definition:

```cpp
template <typename M>
concept Matrix = requires(M m, const M cm, std::size_t i, std::size_t j) {
    typename M::scalar_type;

    { cm.rows() } -> std::same_as<std::size_t>;
    { cm.cols() } -> std::same_as<std::size_t>;

    { m(i, j) } -> std::same_as<typename M::scalar_type&>;
    { cm(i, j) } -> std::same_as<const typename M::scalar_type&>;

    { cm + cm } -> std::same_as<M>;
    { cm - cm } -> std::same_as<M>;
    { -cm } -> std::same_as<M>;
};
```

This concept says: a type `M` is a Matrix if it has a `scalar_type`, dimension queries, element access (both mutable and const), and basic arithmetic. Notice what's *absent*: scalar multiplication. This omission avoids circular constraint issues with the `operator*` overload for matrix multiplication. Instead, we provide a `scale()` function for generic code.

The beauty of concepts is that any type satisfying these constraints works with our algorithms---no inheritance required. We can write:

```cpp
template <Matrix M>
auto det(const M& A) -> typename M::scalar_type;
```

and this function works for `matrix<double>`, `matrix<float>`, or any future type satisfying `Matrix`.

## Clean API Design

A pedagogical library needs an ergonomic API. elementa provides multiple ways to construct matrices:

```cpp
// Default: empty 0x0
matrix<double> empty;

// Filled with value
matrix<double> zeros(3, 4, 0.0);  // 3x4 of zeros

// Flat initializer list (row-major)
matrix<double> flat(2, 3, {1, 2, 3, 4, 5, 6});

// Nested initializer list (most natural)
matrix<double> natural{{1, 2, 3},
                       {4, 5, 6}};
```

Value semantics are enforced throughout. Operators like `+` and `-` return new matrices, marked `[[nodiscard]]` to prevent accidental discards.

## LU Decomposition: The Heart of the Library

LU decomposition factors a matrix A into a lower triangular L and upper triangular U such that PA = LU, where P is a permutation matrix capturing row swaps. This single decomposition powers determinants, inverses, and linear system solving.

The implementation uses *partial pivoting*: at each step, we find the largest absolute value in the current column to use as the pivot. This prevents division by small numbers that would amplify rounding errors.

```cpp
template <Arithmetic T>
struct lu_result {
    matrix<T> L;                     // Lower triangular (unit diagonal)
    matrix<T> U;                     // Upper triangular
    std::vector<std::size_t> perm;   // Permutation vector
    int sign;                        // Sign of permutation (+1 or -1)
    bool singular;                   // True if matrix is singular
};
```

## Derived Operations

Once you have LU, everything else follows.

**Determinant**: Since det(PA) = det(L)det(U) and L has unit diagonal:

```cpp
template <Matrix M>
auto det(const M& A) -> typename M::scalar_type {
    auto [L, U, perm, sign, singular] = lu(A);
    if (singular) return 0;

    T result = static_cast<T>(sign);  // +/- 1
    for (std::size_t i = 0; i < U.rows(); ++i) {
        result *= U(i, i);  // Product of diagonal
    }
    return result;
}
```

**Solve Ax = b**: Apply permutation to b, then forward-substitute through L, then back-substitute through U.

**Inverse**: Solve A * X = I, where I is the identity. This approach---computing the inverse by solving n linear systems---is more numerically stable than explicit formulas.

## Numerical Considerations

For large matrices, computing the determinant directly risks overflow or underflow. The solution is `logdet`:

```cpp
template <Matrix M>
auto logdet(const M& A) -> std::pair<int, typename M::scalar_type> {
    auto [L, U, perm, sign, singular] = lu(A);
    if (singular) return {0, -std::numeric_limits<T>::infinity()};

    T log_abs_det{0};
    int result_sign = sign;

    for (std::size_t i = 0; i < U.rows(); ++i) {
        auto diag = U(i, i);
        if (diag < 0) {
            result_sign = -result_sign;
            diag = -diag;
        }
        log_abs_det += std::log(diag);  // Sum of logs = log of product
    }

    return {result_sign, log_abs_det};
}
```

## Cache-Conscious Matrix Multiplication

The standard matrix multiplication formula is:

    C(i,j) = sum over k of A(i,k) * B(k,j)

A naive implementation uses i-j-k loop order. But with row-major storage, accessing B(k,j) for varying k causes cache misses---we're jumping through memory by column.

The fix is i-k-j ordering:

```cpp
template <Matrix M1, Matrix M2>
auto matmul(const M1& A, const M2& B) -> matrix<typename M1::scalar_type> {
    matrix<T> C(A.rows(), B.cols(), T{0});

    for (std::size_t i = 0; i < A.rows(); ++i) {
        for (std::size_t k = 0; k < A.cols(); ++k) {
            auto aik = A(i, k);  // Load once, reuse across j loop
            for (std::size_t j = 0; j < B.cols(); ++j) {
                C(i, j) += aik * B(k, j);
            }
        }
    }
    return C;
}
```

Now `A(i,k)` is loaded once and reused for all j values. This simple reordering can yield 2-10x speedups on real hardware.

## Conclusion

elementa demonstrates that linear algebra code can be both efficient and readable. The C++20 concept defines what matrices *are*, not how they're stored. LU decomposition serves as the algorithmic foundation for determinants, inverses, and system solving. Numerical considerations like logdet and approximate equality acknowledge the realities of floating-point.

Most importantly, this library enables generic algorithms. The automatic differentiation library builds directly on elementa's Matrix concept, computing gradients for matrix operations without knowing the concrete type. That's the power of concept-based generic programming: algorithms that work for any conforming type, today and in the future.
