#pragma once

#include <span>
#include <string>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include "binary.hpp"
#include "primitives.hpp"

namespace limes::expr {

// Binary function tags
struct PowTag {};
struct MaxTag {};
struct MinTag {};

template<typename Tag, typename L, typename R> struct BinaryFunc;

// Type traits for BinaryFunc and specific tags
template<typename E> inline constexpr bool is_binary_func_v = false;
template<typename Tag, typename L, typename R>
inline constexpr bool is_binary_func_v<BinaryFunc<Tag, L, R>> = true;

template<typename E> inline constexpr bool is_runtime_pow_v = false;
template<typename L, typename R>
inline constexpr bool is_runtime_pow_v<BinaryFunc<PowTag, L, R>> = true;

template<typename E> inline constexpr bool is_max_v = false;
template<typename L, typename R>
inline constexpr bool is_max_v<BinaryFunc<MaxTag, L, R>> = true;

template<typename E> inline constexpr bool is_min_v = false;
template<typename L, typename R>
inline constexpr bool is_min_v<BinaryFunc<MinTag, L, R>> = true;

// BinaryFunc<Tag, L, R>: A binary function applied to two child expressions
template<typename Tag, typename L, typename R>
struct BinaryFunc {
    using value_type = typename L::value_type;
    using tag_type = Tag;
    using left_type = L;
    using right_type = R;

    static constexpr std::size_t arity_v = std::max(L::arity_v, R::arity_v);

    L left;
    R right;

    constexpr BinaryFunc(L l, R r) noexcept : left{l}, right{r} {}

    [[nodiscard]] constexpr value_type eval(std::span<value_type const> args) const {
        value_type l_val = left.eval(args);
        value_type r_val = right.eval(args);

        if constexpr (std::is_same_v<Tag, PowTag>) {
            return std::pow(l_val, r_val);
        } else if constexpr (std::is_same_v<Tag, MaxTag>) {
            return std::max(l_val, r_val);
        } else if constexpr (std::is_same_v<Tag, MinTag>) {
            return std::min(l_val, r_val);
        }
    }

    [[nodiscard]] [[deprecated("use eval() instead")]]
    constexpr value_type evaluate(std::span<value_type const> args) const {
        return eval(args);
    }

    // Derivatives:
    //   pow(f, g): d/dx[f^g] = f^(g-1) * (g*f' + f*g'*ln(f))
    //   max/min:   subgradient via 0.5*(f'+g') +/- 0.5*sign(f-g)*(f'-g')
    template<std::size_t Dim>
    [[nodiscard]] constexpr auto derivative() const {
        auto df = left.template derivative<Dim>();
        auto dg = right.template derivative<Dim>();

        if constexpr (std::is_same_v<Tag, PowTag>) {
            auto g_minus_1 = right - One<value_type>{};
            auto f_to_g_minus_1 = BinaryFunc<PowTag, L, decltype(g_minus_1)>{left, g_minus_1};
            auto log_f = UnaryFunc<LogTag, L>{left};
            return f_to_g_minus_1 * (right * df + left * dg * log_f);
        } else if constexpr (std::is_same_v<Tag, MaxTag>) {
            auto half = Const<value_type>{value_type(0.5)};
            auto diff = left - right;
            auto sign_diff = diff / UnaryFunc<AbsTag, decltype(diff)>{diff};
            return half * (df + dg) + half * sign_diff * (df - dg);
        } else if constexpr (std::is_same_v<Tag, MinTag>) {
            auto half = Const<value_type>{value_type(0.5)};
            auto diff = left - right;
            auto sign_diff = diff / UnaryFunc<AbsTag, decltype(diff)>{diff};
            return half * (df + dg) - half * sign_diff * (df - dg);
        }
    }

    [[nodiscard]] std::string to_string() const {
        std::string func_name;
        if constexpr (std::is_same_v<Tag, PowTag>) {
            func_name = "pow";
        } else if constexpr (std::is_same_v<Tag, MaxTag>) {
            func_name = "max";
        } else if constexpr (std::is_same_v<Tag, MinTag>) {
            func_name = "min";
        }
        return "(" + func_name + " " + left.to_string() + " " + right.to_string() + ")";
    }
};

// Factory functions with compile-time simplification

// pow(expr, expr)
template<typename L, typename R>
    requires (is_expr_node_v<L> && is_expr_node_v<R>)
[[nodiscard]] constexpr auto pow(L base, R exponent) {
    if constexpr (is_zero_v<R>) {
        return One<typename L::value_type>{};
    } else if constexpr (is_one_v<R>) {
        return base;
    } else if constexpr (is_one_v<L>) {
        return One<typename L::value_type>{};
    } else if constexpr (is_const_expr_v<L> && is_const_expr_v<R>) {
        return Const<typename L::value_type>{std::pow(base.value, exponent.value)};
    } else {
        return BinaryFunc<PowTag, L, R>{base, exponent};
    }
}

// pow(expr, scalar)
template<typename L, typename T>
    requires (is_expr_node_v<L> && std::is_arithmetic_v<T>)
[[nodiscard]] constexpr auto pow(L base, T exponent) {
    using VT = typename L::value_type;
    return pow(base, Const<VT>{static_cast<VT>(exponent)});
}

// pow(scalar, expr)
template<typename T, typename R>
    requires (std::is_arithmetic_v<T> && is_expr_node_v<R>)
[[nodiscard]] constexpr auto pow(T base, R exponent) {
    using VT = typename R::value_type;
    return pow(Const<VT>{static_cast<VT>(base)}, exponent);
}

// max(expr, expr)
template<typename L, typename R>
    requires (is_expr_node_v<L> && is_expr_node_v<R>)
[[nodiscard]] constexpr auto max(L a, R b) {
    if constexpr (is_const_expr_v<L> && is_const_expr_v<R>) {
        return Const<typename L::value_type>{std::max(a.value, b.value)};
    } else if constexpr (std::is_same_v<L, R>) {
        return a;
    } else {
        return BinaryFunc<MaxTag, L, R>{a, b};
    }
}

// max(expr, scalar)
template<typename L, typename T>
    requires (is_expr_node_v<L> && std::is_arithmetic_v<T>)
[[nodiscard]] constexpr auto max(L a, T b) {
    using VT = typename L::value_type;
    return max(a, Const<VT>{static_cast<VT>(b)});
}

// max(scalar, expr)
template<typename T, typename R>
    requires (std::is_arithmetic_v<T> && is_expr_node_v<R>)
[[nodiscard]] constexpr auto max(T a, R b) {
    using VT = typename R::value_type;
    return max(Const<VT>{static_cast<VT>(a)}, b);
}

// min(expr, expr)
template<typename L, typename R>
    requires (is_expr_node_v<L> && is_expr_node_v<R>)
[[nodiscard]] constexpr auto min(L a, R b) {
    if constexpr (is_const_expr_v<L> && is_const_expr_v<R>) {
        return Const<typename L::value_type>{std::min(a.value, b.value)};
    } else if constexpr (std::is_same_v<L, R>) {
        return a;
    } else {
        return BinaryFunc<MinTag, L, R>{a, b};
    }
}

// min(expr, scalar)
template<typename L, typename T>
    requires (is_expr_node_v<L> && std::is_arithmetic_v<T>)
[[nodiscard]] constexpr auto min(L a, T b) {
    using VT = typename L::value_type;
    return min(a, Const<VT>{static_cast<VT>(b)});
}

// min(scalar, expr)
template<typename T, typename R>
    requires (std::is_arithmetic_v<T> && is_expr_node_v<R>)
[[nodiscard]] constexpr auto min(T a, R b) {
    using VT = typename R::value_type;
    return min(Const<VT>{static_cast<VT>(a)}, b);
}

// Type aliases
template<typename L, typename R>
using RuntimePow = BinaryFunc<PowTag, L, R>;

template<typename L, typename R>
using Max = BinaryFunc<MaxTag, L, R>;

template<typename L, typename R>
using Min = BinaryFunc<MinTag, L, R>;

} // namespace limes::expr
