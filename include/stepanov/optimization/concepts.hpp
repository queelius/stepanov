#pragma once

#include "../concepts.hpp"
#include <functional>
#include <limits>

namespace stepanov::optimization {

/**
 * Optimization-specific concepts following Stepanov's generic programming principles
 */

// Function that can be evaluated at a point (returns a scalar)
template<typename F, typename T>
concept evaluable = requires(F f, T x) {
    { f(x) };  // Just requires that f(x) is callable
};

// Function with computable gradient
template<typename F, typename T, typename V>
concept differentiable = requires(F f, T x, V v) {
    { f(x) } -> std::convertible_to<T>;
    { gradient(f, x) } -> std::convertible_to<V>;
};

// Function with computable Hessian
template<typename F, typename T, typename V, typename M>
concept twice_differentiable = differentiable<F, T, V> && requires(F f, T x, M m) {
    { hessian(f, x) } -> std::convertible_to<M>;
};

// Stopping criterion for iterative algorithms
template<typename S, typename T>
concept stopping_criterion = requires(S s, T current, T previous, size_t iterations) {
    { s(current, previous, iterations) } -> std::convertible_to<bool>;
};

// Line search strategy
template<typename L, typename F, typename T>
concept line_search = requires(L ls, F f, T x, T direction, T alpha) {
    { ls(f, x, direction, alpha) } -> std::convertible_to<T>;
};

// Constraint function
template<typename C, typename T>
concept constraint = requires(C c, T x) {
    { c(x) } -> std::convertible_to<bool>;
};

// Optimization result
template<typename T, typename V = T>
struct optimization_result {
    V solution;           // The optimal point
    T value;             // Function value at optimal point
    size_t iterations;   // Number of iterations performed
    bool converged;      // Whether convergence was achieved

    optimization_result() = default;
    optimization_result(V sol, T val, size_t iter, bool conv)
        : solution(sol), value(val), iterations(iter), converged(conv) {}
};

// Default stopping criteria
template<typename T>
    requires field<T>
struct tolerance_criterion {
    T absolute_tolerance;
    T relative_tolerance;
    size_t max_iterations;

    tolerance_criterion(T abs_tol = std::numeric_limits<T>::epsilon() * T(100),
                       T rel_tol = std::numeric_limits<T>::epsilon() * T(10),
                       size_t max_iter = 1000)
        : absolute_tolerance(abs_tol), relative_tolerance(rel_tol), max_iterations(max_iter) {}

    bool operator()(const T& current, const T& previous, size_t iterations) const {
        if (iterations >= max_iterations) return true;
        T diff = abs(current - previous);
        return diff < absolute_tolerance || diff < relative_tolerance * abs(previous);
    }
};

// Vector space operations for optimization
template<typename V>
concept vector_space = requires(V v1, V v2, typename V::value_type scalar) {
    typename V::value_type;
    { v1 + v2 } -> std::convertible_to<V>;
    { v1 - v2 } -> std::convertible_to<V>;
    { scalar * v1 } -> std::convertible_to<V>;
    { v1 * scalar } -> std::convertible_to<V>;
    // Note: norm is defined separately to avoid circular dependency
};

// Matrix operations for second-order methods
template<typename M>
concept matrix = requires(M m1, M m2, typename M::value_type scalar) {
    typename M::value_type;
    { m1 + m2 } -> std::convertible_to<M>;
    { m1 * m2 } -> std::convertible_to<M>;
    { scalar * m1 } -> std::convertible_to<M>;
    { inverse(m1) } -> std::convertible_to<M>;
};

} // namespace stepanov::optimization