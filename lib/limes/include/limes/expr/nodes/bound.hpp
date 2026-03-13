#pragma once

#include <span>
#include <string>
#include <vector>
#include <cstddef>
#include <sstream>
#include "const.hpp"
#include "binary.hpp"

namespace limes::expr {

// BoundExpr<E, Dim, BoundValue>: An expression E with dimension Dim bound to a value.
//
// When evaluated, injects the bound value at position Dim and uses
// the remaining arguments for other dimensions.
//
// Example:
//   auto f = x * y;                    // f(x, y), arity = 2
//   auto g = bind<0>(f, 3.0);          // g(y) = 3 * y, arity = 1
//   g.eval({4.0}) == 12.0
//
template<typename E, std::size_t Dim, typename BoundValue>
struct BoundExpr {
    using value_type = typename E::value_type;
    using expr_type = E;
    using bound_type = BoundValue;

    static constexpr std::size_t dim_v = Dim;
    static constexpr std::size_t original_arity_v = E::arity_v;
    static constexpr std::size_t arity_v = (E::arity_v > 0) ? (E::arity_v - 1) : 0;

    E expr;
    BoundValue bound_value;

    constexpr BoundExpr(E e, BoundValue v) noexcept
        : expr{e}, bound_value{v} {}

    [[nodiscard]] constexpr value_type eval(std::span<value_type const> args) const {
        std::vector<value_type> full_args;
        full_args.reserve(original_arity_v);

        std::size_t args_idx = 0;
        for (std::size_t i = 0; i < original_arity_v; ++i) {
            if (i == Dim) {
                full_args.push_back(get_bound_value());
            } else if (args_idx < args.size()) {
                full_args.push_back(args[args_idx++]);
            } else {
                full_args.push_back(value_type{0});
            }
        }

        return expr.eval(std::span<value_type const>{full_args});
    }

    [[nodiscard]] constexpr value_type eval() const {
        return eval(std::span<value_type const>{});
    }

    [[nodiscard]] [[deprecated("use eval() instead")]]
    constexpr value_type evaluate(std::span<value_type const> args) const {
        return eval(args);
    }

    [[nodiscard]] [[deprecated("use eval() instead")]]
    constexpr value_type evaluate() const {
        return eval();
    }

    // Derivative: map DerivDim to the original dimension, accounting for the bound slot
    template<std::size_t DerivDim>
    [[nodiscard]] constexpr auto derivative() const {
        constexpr std::size_t original_dim = (DerivDim >= Dim) ? (DerivDim + 1) : DerivDim;
        auto d_expr = expr.template derivative<original_dim>();
        return BoundExpr<decltype(d_expr), Dim, BoundValue>{d_expr, bound_value};
    }

    [[nodiscard]] std::string to_string() const {
        std::ostringstream oss;
        oss << "(bind " << Dim << " " << get_bound_value() << " " << expr.to_string() << ")";
        return oss.str();
    }

private:
    [[nodiscard]] constexpr value_type get_bound_value() const {
        if constexpr (std::is_arithmetic_v<BoundValue>) {
            return static_cast<value_type>(bound_value);
        } else {
            return bound_value.value;
        }
    }
};

// bind<Dim>(expr, value): Bind dimension Dim to a constant value
template<std::size_t Dim, typename E, typename T>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto bind(E expr, T value) {
    static_assert(Dim < E::arity_v || E::arity_v == 0,
        "Cannot bind a dimension that doesn't exist in the expression");

    using VT = typename E::value_type;

    if constexpr (std::is_arithmetic_v<T>) {
        return BoundExpr<E, Dim, Const<VT>>{expr, Const<VT>{static_cast<VT>(value)}};
    } else {
        return BoundExpr<E, Dim, T>{expr, value};
    }
}

// bind_all<D1, D2, ...>(expr, v1, v2, ...): Bind multiple dimensions in sequence
template<std::size_t Dim, std::size_t... Dims, typename E, typename T, typename... Ts>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto bind_all(E expr, T value, Ts... values) {
    auto bound_once = bind<Dim>(expr, value);
    if constexpr (sizeof...(Dims) == 0) {
        return bound_once;
    } else {
        return bind_all<Dims...>(bound_once, values...);
    }
}

// partial(expr, value): Bind dimension 0 (curry from left)
template<typename E, typename T>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto partial(E expr, T value) {
    return bind<0>(expr, value);
}

// partial_right(expr, value): Bind the last dimension (curry from right)
template<typename E, typename T>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto partial_right(E expr, T value) {
    constexpr std::size_t last_dim = (E::arity_v > 0) ? (E::arity_v - 1) : 0;
    return bind<last_dim>(expr, value);
}

} // namespace limes::expr
