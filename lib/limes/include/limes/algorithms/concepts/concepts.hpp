#pragma once

#include <concepts>
#include <type_traits>

namespace limes::algorithms::concepts {

// Fundamental algebraic concept: a type supporting field operations
template<typename T>
concept Field = requires(T a, T b) {
    { a + b } -> std::convertible_to<T>;
    { a - b } -> std::convertible_to<T>;
    { a * b } -> std::convertible_to<T>;
    { a / b } -> std::convertible_to<T>;
    { -a } -> std::convertible_to<T>;
    { T(0) };
    { T(1) };
};

// A callable that accepts a single argument of type T
template<typename F, typename T>
concept UnivariateFunction = std::invocable<F, T>;

// Accumulator: incremental summation with operator+=, operator(), and default/value construction
template<typename A, typename T>
concept Accumulator = requires(A acc, T value) {
    { acc += value } -> std::same_as<A&>;
    { acc() } -> std::convertible_to<T>;
    { A{} };
    { A{T{}} };
};

// Integration result: value, error, iteration count, and conversion to T
template<typename R, typename T>
concept IntegrationResult = requires(R result) {
    { result.value() } -> std::convertible_to<T>;
    { result.error() } -> std::convertible_to<T>;
    { result.iterations() } -> std::convertible_to<std::size_t>;
    { static_cast<T>(result) } -> std::convertible_to<T>;
};

// Quadrature rule: fixed set of nodes and weights on [-1, 1]
template<typename Q, typename T>
concept QuadratureRule = requires(Q rule) {
    typename Q::value_type;
    typename Q::size_type;
    { rule.size() } -> std::convertible_to<std::size_t>;
    { rule.weight(std::size_t{}) } -> std::convertible_to<T>;
    { rule.abscissa(std::size_t{}) } -> std::convertible_to<T>;
};

// Integrator: a type with value_type and a result_type satisfying IntegrationResult
template<typename I, typename T>
concept Integrator = requires {
    typename I::value_type;
    typename I::result_type;
    requires IntegrationResult<typename I::result_type, T>;
};

} // namespace limes::algorithms::concepts
