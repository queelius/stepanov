#pragma once

#include <span>
#include <string>
#include <string_view>
#include <cstddef>
#include <cmath>
#include "binary.hpp"
#include "unary.hpp"

namespace limes::expr {

// Primitive function tags
struct ExpTag {};
struct LogTag {};
struct SinTag {};
struct CosTag {};
struct SqrtTag {};
struct AbsTag {};
struct TanTag {};
struct SinhTag {};
struct CoshTag {};
struct TanhTag {};
struct AsinTag {};
struct AcosTag {};
struct AtanTag {};
struct AsinhTag {};
struct AcoshTag {};
struct AtanhTag {};

namespace detail {

// Map function tag to its display name
template<typename Tag>
constexpr std::string_view tag_name() {
    if constexpr (std::is_same_v<Tag, ExpTag>)   return "exp";
    else if constexpr (std::is_same_v<Tag, LogTag>)   return "log";
    else if constexpr (std::is_same_v<Tag, SinTag>)   return "sin";
    else if constexpr (std::is_same_v<Tag, CosTag>)   return "cos";
    else if constexpr (std::is_same_v<Tag, SqrtTag>)  return "sqrt";
    else if constexpr (std::is_same_v<Tag, AbsTag>)   return "abs";
    else if constexpr (std::is_same_v<Tag, TanTag>)   return "tan";
    else if constexpr (std::is_same_v<Tag, SinhTag>)  return "sinh";
    else if constexpr (std::is_same_v<Tag, CoshTag>)  return "cosh";
    else if constexpr (std::is_same_v<Tag, TanhTag>)  return "tanh";
    else if constexpr (std::is_same_v<Tag, AsinTag>)  return "asin";
    else if constexpr (std::is_same_v<Tag, AcosTag>)  return "acos";
    else if constexpr (std::is_same_v<Tag, AtanTag>)  return "atan";
    else if constexpr (std::is_same_v<Tag, AsinhTag>)  return "asinh";
    else if constexpr (std::is_same_v<Tag, AcoshTag>)  return "acosh";
    else if constexpr (std::is_same_v<Tag, AtanhTag>)  return "atanh";
}

} // namespace detail

// UnaryFunc<Tag, E>: A unary function applied to a child expression.
// This is the expression node for primitives like exp, sin, cos, sqrt, etc.
template<typename Tag, typename E>
struct UnaryFunc {
    using value_type = typename E::value_type;
    using tag_type = Tag;
    using child_type = E;

    static constexpr std::size_t arity_v = E::arity_v;

    E child;

    constexpr explicit UnaryFunc(E c) noexcept : child{c} {}

    [[nodiscard]] constexpr value_type eval(std::span<value_type const> args) const {
        value_type c_val = child.eval(args);

        if constexpr (std::is_same_v<Tag, ExpTag>)        return std::exp(c_val);
        else if constexpr (std::is_same_v<Tag, LogTag>)   return std::log(c_val);
        else if constexpr (std::is_same_v<Tag, SinTag>)   return std::sin(c_val);
        else if constexpr (std::is_same_v<Tag, CosTag>)   return std::cos(c_val);
        else if constexpr (std::is_same_v<Tag, SqrtTag>)  return std::sqrt(c_val);
        else if constexpr (std::is_same_v<Tag, AbsTag>)   return std::abs(c_val);
        else if constexpr (std::is_same_v<Tag, TanTag>)   return std::tan(c_val);
        else if constexpr (std::is_same_v<Tag, SinhTag>)  return std::sinh(c_val);
        else if constexpr (std::is_same_v<Tag, CoshTag>)  return std::cosh(c_val);
        else if constexpr (std::is_same_v<Tag, TanhTag>)  return std::tanh(c_val);
        else if constexpr (std::is_same_v<Tag, AsinTag>)  return std::asin(c_val);
        else if constexpr (std::is_same_v<Tag, AcosTag>)  return std::acos(c_val);
        else if constexpr (std::is_same_v<Tag, AtanTag>)  return std::atan(c_val);
        else if constexpr (std::is_same_v<Tag, AsinhTag>) return std::asinh(c_val);
        else if constexpr (std::is_same_v<Tag, AcoshTag>) return std::acosh(c_val);
        else if constexpr (std::is_same_v<Tag, AtanhTag>) return std::atanh(c_val);
    }

    [[nodiscard]] [[deprecated("use eval() instead")]]
    constexpr value_type evaluate(std::span<value_type const> args) const {
        return eval(args);
    }

    // Chain rule: d/dx[f(u)] = f'(u) * du/dx
    template<std::size_t Dim>
    [[nodiscard]] constexpr auto derivative() const {
        auto dc = child.template derivative<Dim>();

        if constexpr (std::is_same_v<Tag, ExpTag>) {
            // d[exp(u)] = exp(u) * du
            return (*this) * dc;
        } else if constexpr (std::is_same_v<Tag, LogTag>) {
            // d[log(u)] = du / u
            return (One<value_type>{} / child) * dc;
        } else if constexpr (std::is_same_v<Tag, SinTag>) {
            // d[sin(u)] = cos(u) * du
            return UnaryFunc<CosTag, E>{child} * dc;
        } else if constexpr (std::is_same_v<Tag, CosTag>) {
            // d[cos(u)] = -sin(u) * du
            return -UnaryFunc<SinTag, E>{child} * dc;
        } else if constexpr (std::is_same_v<Tag, SqrtTag>) {
            // d[sqrt(u)] = du / (2 * sqrt(u))
            auto two_sqrt = Const<value_type>{value_type(2)} * UnaryFunc<SqrtTag, E>{child};
            return (One<value_type>{} / two_sqrt) * dc;
        } else if constexpr (std::is_same_v<Tag, AbsTag>) {
            // d[|u|] = sign(u) * du, where sign(u) = u / |u|
            return (child / UnaryFunc<AbsTag, E>{child}) * dc;
        } else if constexpr (std::is_same_v<Tag, TanTag>) {
            // d[tan(u)] = sec^2(u) * du = du / cos^2(u)
            auto cos_child = UnaryFunc<CosTag, E>{child};
            return (One<value_type>{} / (cos_child * cos_child)) * dc;
        } else if constexpr (std::is_same_v<Tag, SinhTag>) {
            // d[sinh(u)] = cosh(u) * du
            return UnaryFunc<CoshTag, E>{child} * dc;
        } else if constexpr (std::is_same_v<Tag, CoshTag>) {
            // d[cosh(u)] = sinh(u) * du
            return UnaryFunc<SinhTag, E>{child} * dc;
        } else if constexpr (std::is_same_v<Tag, TanhTag>) {
            // d[tanh(u)] = (1 - tanh^2(u)) * du
            auto tanh_child = *this;
            return (One<value_type>{} - tanh_child * tanh_child) * dc;
        } else if constexpr (std::is_same_v<Tag, AsinTag>) {
            // d[asin(u)] = du / sqrt(1 - u^2)
            auto one = One<value_type>{};
            auto denom = UnaryFunc<SqrtTag, decltype(one - child * child)>{one - child * child};
            return (one / denom) * dc;
        } else if constexpr (std::is_same_v<Tag, AcosTag>) {
            // d[acos(u)] = -du / sqrt(1 - u^2)
            auto one = One<value_type>{};
            auto denom = UnaryFunc<SqrtTag, decltype(one - child * child)>{one - child * child};
            return -(one / denom) * dc;
        } else if constexpr (std::is_same_v<Tag, AtanTag>) {
            // d[atan(u)] = du / (1 + u^2)
            auto one = One<value_type>{};
            return (one / (one + child * child)) * dc;
        } else if constexpr (std::is_same_v<Tag, AsinhTag>) {
            // d[asinh(u)] = du / sqrt(1 + u^2)
            auto one = One<value_type>{};
            auto denom = UnaryFunc<SqrtTag, decltype(one + child * child)>{one + child * child};
            return (one / denom) * dc;
        } else if constexpr (std::is_same_v<Tag, AcoshTag>) {
            // d[acosh(u)] = du / sqrt(u^2 - 1)
            auto one = One<value_type>{};
            auto denom = UnaryFunc<SqrtTag, decltype(child * child - one)>{child * child - one};
            return (one / denom) * dc;
        } else if constexpr (std::is_same_v<Tag, AtanhTag>) {
            // d[atanh(u)] = du / (1 - u^2)
            auto one = One<value_type>{};
            return (one / (one - child * child)) * dc;
        }
    }

    [[nodiscard]] std::string to_string() const {
        return "(" + std::string(detail::tag_name<Tag>()) + " " + child.to_string() + ")";
    }
};

// Factory functions for creating primitive function expressions

template<typename E>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto exp(E e) { return UnaryFunc<ExpTag, E>{e}; }

template<typename E>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto log(E e) { return UnaryFunc<LogTag, E>{e}; }

template<typename E>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto sin(E e) { return UnaryFunc<SinTag, E>{e}; }

template<typename E>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto cos(E e) { return UnaryFunc<CosTag, E>{e}; }

template<typename E>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto sqrt(E e) { return UnaryFunc<SqrtTag, E>{e}; }

template<typename E>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto abs(E e) { return UnaryFunc<AbsTag, E>{e}; }

template<typename E>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto tan(E e) { return UnaryFunc<TanTag, E>{e}; }

template<typename E>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto sinh(E e) { return UnaryFunc<SinhTag, E>{e}; }

template<typename E>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto cosh(E e) { return UnaryFunc<CoshTag, E>{e}; }

template<typename E>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto tanh(E e) { return UnaryFunc<TanhTag, E>{e}; }

template<typename E>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto asin(E e) { return UnaryFunc<AsinTag, E>{e}; }

template<typename E>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto acos(E e) { return UnaryFunc<AcosTag, E>{e}; }

template<typename E>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto atan(E e) { return UnaryFunc<AtanTag, E>{e}; }

template<typename E>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto asinh(E e) { return UnaryFunc<AsinhTag, E>{e}; }

template<typename E>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto acosh(E e) { return UnaryFunc<AcoshTag, E>{e}; }

template<typename E>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto atanh(E e) { return UnaryFunc<AtanhTag, E>{e}; }

// Type aliases
template<typename E> using Exp   = UnaryFunc<ExpTag, E>;
template<typename E> using Log   = UnaryFunc<LogTag, E>;
template<typename E> using Sin   = UnaryFunc<SinTag, E>;
template<typename E> using Cos   = UnaryFunc<CosTag, E>;
template<typename E> using Sqrt  = UnaryFunc<SqrtTag, E>;
template<typename E> using Abs   = UnaryFunc<AbsTag, E>;
template<typename E> using Tan   = UnaryFunc<TanTag, E>;
template<typename E> using Sinh  = UnaryFunc<SinhTag, E>;
template<typename E> using Cosh  = UnaryFunc<CoshTag, E>;
template<typename E> using Tanh  = UnaryFunc<TanhTag, E>;
template<typename E> using Asin  = UnaryFunc<AsinTag, E>;
template<typename E> using Acos  = UnaryFunc<AcosTag, E>;
template<typename E> using Atan  = UnaryFunc<AtanTag, E>;
template<typename E> using Asinh = UnaryFunc<AsinhTag, E>;
template<typename E> using Acosh = UnaryFunc<AcoshTag, E>;
template<typename E> using Atanh = UnaryFunc<AtanhTag, E>;

} // namespace limes::expr
