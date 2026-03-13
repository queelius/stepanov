#pragma once

#include "nodes/const.hpp"
#include "nodes/var.hpp"
#include "nodes/binary.hpp"
#include "nodes/unary.hpp"
#include "nodes/primitives.hpp"
#include <functional>

namespace limes::expr {

// Compile-time derivative: preferred method
// Returns an expression representing the derivative with respect to dimension Dim
template<std::size_t Dim, typename E>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto derivative(E expr) {
    return expr.template derivative<Dim>();
}

// =============================================================================
// AnyExpr: Type-erased expression wrapper for runtime derivative dispatch
// =============================================================================
// With type-level simplification (Zero<T>, One<T>), derivatives for different
// dimensions can return different types. AnyExpr provides type erasure to
// support runtime dimension selection.

template<typename T>
struct AnyExpr {
    using value_type = T;
    static constexpr std::size_t arity_v = 8;  // Maximum supported arity

    std::function<T(std::span<T const>)> eval_fn;
    std::function<std::string()> to_string_fn;

    AnyExpr() = default;

    template<typename E>
        requires is_expr_node_v<E>
    explicit AnyExpr(E expr)
        : eval_fn([expr](std::span<T const> args) { return expr.eval(args); })
        , to_string_fn([expr]() { return expr.to_string(); })
    {}

    [[nodiscard]] T eval(std::span<T const> args) const {
        return eval_fn(args);
    }

    // Deprecated: use eval() instead
    [[nodiscard]] [[deprecated("use eval() instead")]]
    T evaluate(std::span<T const> args) const {
        return eval(args);
    }

    [[nodiscard]] std::string to_string() const {
        return to_string_fn();
    }
};

// Runtime derivative dispatch using type erasure
// Supports dimensions 0-7

namespace detail {

template<typename E, std::size_t... Is>
auto derivative_dispatch_impl(E expr, std::size_t dim, std::index_sequence<Is...>) {
    using T = typename E::value_type;

    // Table of type-erased derivative constructors
    AnyExpr<T> (*funcs[])(E) = {
        [](E e) { return AnyExpr<T>{e.template derivative<Is>()}; }...
    };

    if (dim >= sizeof...(Is)) {
        return AnyExpr<T>{Zero<T>{}};  // Beyond supported dimensions, return zero
    }

    return funcs[dim](expr);
}

} // namespace detail

// Runtime derivative with dimension as parameter
// Returns type-erased AnyExpr<T> to handle varying derivative types
// Supports dimensions 0-7
template<typename E>
    requires is_expr_node_v<E>
[[nodiscard]] auto derivative(E expr, std::size_t dim) {
    return detail::derivative_dispatch_impl(expr, dim, std::make_index_sequence<8>{});
}

// Gradient: compute all partial derivatives as a tuple
// Returns tuple<d/dx0, d/dx1, ..., d/dx(N-1)>
namespace detail {

template<typename E, std::size_t... Is>
constexpr auto gradient_impl(E expr, std::index_sequence<Is...>) {
    return std::make_tuple(expr.template derivative<Is>()...);
}

} // namespace detail

template<typename E>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto gradient(E expr) {
    return detail::gradient_impl(expr, std::make_index_sequence<E::arity_v>{});
}

// Higher-order derivatives
// derivative<Dim1, Dim2, ...>(expr) computes d/dx_Dim1 d/dx_Dim2 ... expr
template<std::size_t Dim, std::size_t... Dims, typename E>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto derivative_n(E expr) {
    if constexpr (sizeof...(Dims) == 0) {
        return expr.template derivative<Dim>();
    } else {
        return derivative_n<Dims...>(expr.template derivative<Dim>());
    }
}

} // namespace limes::expr
