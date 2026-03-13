#pragma once

#include <span>
#include <string>
#include <cstddef>
#include <algorithm>
#include "binary.hpp"

namespace limes::expr {

// Conditional<Cond, Then, Else>: Piecewise/conditional expressions.
// Evaluates to then_branch if condition > 0, else_branch otherwise.
template<typename Cond, typename Then, typename Else>
struct Conditional {
    using value_type = typename Then::value_type;
    using condition_type = Cond;
    using then_type = Then;
    using else_type = Else;

    static constexpr std::size_t arity_v =
        std::max({Cond::arity_v, Then::arity_v, Else::arity_v});

    Cond condition;
    Then then_branch;
    Else else_branch;

    constexpr Conditional(Cond c, Then t, Else e) noexcept
        : condition{c}, then_branch{t}, else_branch{e} {}

    [[nodiscard]] constexpr value_type eval(std::span<value_type const> args) const {
        if (condition.eval(args) > value_type(0)) {
            return then_branch.eval(args);
        } else {
            return else_branch.eval(args);
        }
    }

    [[nodiscard]] [[deprecated("use eval() instead")]]
    constexpr value_type evaluate(std::span<value_type const> args) const {
        return eval(args);
    }

    // Subgradient: d/dx[if c>0 then f else g] = if c>0 then df else dg
    template<std::size_t Dim>
    [[nodiscard]] constexpr auto derivative() const {
        auto df = then_branch.template derivative<Dim>();
        auto dg = else_branch.template derivative<Dim>();
        return Conditional<Cond, decltype(df), decltype(dg)>{condition, df, dg};
    }

    [[nodiscard]] std::string to_string() const {
        return "(if " + condition.to_string() + " "
             + then_branch.to_string() + " "
             + else_branch.to_string() + ")";
    }
};

template<typename T>
struct is_conditional : std::false_type {};

template<typename C, typename T, typename E>
struct is_conditional<Conditional<C, T, E>> : std::true_type {};

template<typename T>
inline constexpr bool is_conditional_v = is_conditional<T>::value;

// Factory functions

template<typename C, typename T, typename E>
    requires (is_expr_node_v<C> && is_expr_node_v<T> && is_expr_node_v<E>)
[[nodiscard]] constexpr auto if_then_else(C cond, T then_expr, E else_expr) {
    return Conditional<C, T, E>{cond, then_expr, else_expr};
}

template<typename C, typename T>
    requires (is_expr_node_v<C> && std::is_arithmetic_v<T>)
[[nodiscard]] constexpr auto if_then_else(C cond, T then_val, T else_val) {
    using VT = typename C::value_type;
    return Conditional<C, Const<VT>, Const<VT>>{
        cond,
        Const<VT>{static_cast<VT>(then_val)},
        Const<VT>{static_cast<VT>(else_val)}
    };
}

// Common piecewise functions

// heaviside(e): H(x) = 1 if x > 0, 0 otherwise
template<typename E>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto heaviside(E e) {
    using T = typename E::value_type;
    return if_then_else(e, One<T>{}, Zero<T>{});
}

// ramp(e): max(e, 0) -- the positive part / ReLU function
template<typename E>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto ramp(E e) {
    using T = typename E::value_type;
    return if_then_else(e, e, Zero<T>{});
}

// sign(e): 1 if x > 0, -1 if x < 0, 0 if x = 0
template<typename E>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto sign(E e) {
    using T = typename E::value_type;
    auto inner = if_then_else(-e, Const<T>{T(-1)}, Zero<T>{});
    return if_then_else(e, One<T>{}, inner);
}

// clamp(e, lo, hi): Clamp e to [lo, hi]
template<typename E, typename T>
    requires (is_expr_node_v<E> && std::is_arithmetic_v<T>)
[[nodiscard]] constexpr auto clamp(E e, T lo, T hi) {
    using VT = typename E::value_type;
    auto lo_const = Const<VT>{static_cast<VT>(lo)};
    auto hi_const = Const<VT>{static_cast<VT>(hi)};
    auto upper_clamped = if_then_else(hi_const - e, e, hi_const);
    return if_then_else(e - lo_const, upper_clamped, lo_const);
}

// indicator(e): Alias for heaviside, emphasizing indicator function semantics
template<typename E>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto indicator(E e) {
    return heaviside(e);
}

} // namespace limes::expr
