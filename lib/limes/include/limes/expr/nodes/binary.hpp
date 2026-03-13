#pragma once

#include <span>
#include <string>
#include <cstddef>
#include <algorithm>
#include "../concepts.hpp"

namespace limes::expr {

// Forward declaration of Integral type trait (defined in integral.hpp).
// Used to exclude Integral types from generic operator* to avoid ambiguity.
template<typename T>
struct is_integral : std::false_type {};

// Operation tags for binary operations
struct Add {};
struct Sub {};
struct Mul {};
struct Div {};

// Binary<Op, L, R>: A binary operation node combining two child expressions.
// Arity is max of children's arities.
template<typename Op, typename L, typename R>
struct Binary {
    using value_type = typename L::value_type;
    using op_type = Op;
    using left_type = L;
    using right_type = R;

    static constexpr std::size_t arity_v = std::max(L::arity_v, R::arity_v);

    L left;
    R right;

    constexpr Binary(L l, R r) noexcept : left{l}, right{r} {}

    [[nodiscard]] constexpr value_type eval(std::span<value_type const> args) const {
        value_type l_val = left.eval(args);
        value_type r_val = right.eval(args);

        if constexpr (std::is_same_v<Op, Add>) {
            return l_val + r_val;
        } else if constexpr (std::is_same_v<Op, Sub>) {
            return l_val - r_val;
        } else if constexpr (std::is_same_v<Op, Mul>) {
            return l_val * r_val;
        } else if constexpr (std::is_same_v<Op, Div>) {
            return l_val / r_val;
        }
    }

    [[nodiscard]] [[deprecated("use eval() instead")]]
    constexpr value_type evaluate(std::span<value_type const> args) const {
        return eval(args);
    }

    // Derivative via sum/difference, product, and quotient rules.
    // Uses operator overloads for automatic simplification with Zero/One types.
    template<std::size_t Dim>
    [[nodiscard]] constexpr auto derivative() const {
        auto dl = left.template derivative<Dim>();
        auto dr = right.template derivative<Dim>();

        if constexpr (std::is_same_v<Op, Add>) {
            return dl + dr;
        } else if constexpr (std::is_same_v<Op, Sub>) {
            return dl - dr;
        } else if constexpr (std::is_same_v<Op, Mul>) {
            return (dl * right) + (left * dr);
        } else if constexpr (std::is_same_v<Op, Div>) {
            auto numer = (dl * right) - (left * dr);
            auto denom = right * right;
            return numer / denom;
        }
    }

    [[nodiscard]] std::string to_string() const {
        std::string op_str;
        if constexpr (std::is_same_v<Op, Add>) {
            op_str = "+";
        } else if constexpr (std::is_same_v<Op, Sub>) {
            op_str = "-";
        } else if constexpr (std::is_same_v<Op, Mul>) {
            op_str = "*";
        } else if constexpr (std::is_same_v<Op, Div>) {
            op_str = "/";
        }
        return "(" + op_str + " " + left.to_string() + " " + right.to_string() + ")";
    }
};

// Type traits

template<typename T>
struct is_const_expr : std::false_type {};

template<typename T>
struct is_const_expr<Const<T>> : std::true_type {};

template<typename T>
inline constexpr bool is_const_expr_v = is_const_expr<T>::value;

template<typename T, typename = void>
struct is_expr_node : std::false_type {};

template<typename T>
struct is_expr_node<T, std::void_t<decltype(T::arity_v)>> : std::true_type {};

template<typename T>
inline constexpr bool is_expr_node_v = is_expr_node<T>::value;

// Operator overloads for expression composition.
// These use compile-time simplification via Zero<T> and One<T> marker types.

// expr + expr
template<typename L, typename R>
    requires (is_expr_node_v<L> && is_expr_node_v<R>)
[[nodiscard]] constexpr auto operator+(L l, R r) {
    if constexpr (is_zero_v<L>) {
        return r;
    } else if constexpr (is_zero_v<R>) {
        return l;
    } else if constexpr (is_const_expr_v<L> && is_const_expr_v<R>) {
        return Const<typename L::value_type>{l.value + r.value};
    } else if constexpr (std::is_same_v<L, R>) {
        using T = typename L::value_type;
        return Const<T>{T(2)} * l;
    } else {
        return Binary<Add, L, R>{l, r};
    }
}

// expr + scalar
template<typename L, typename T>
    requires (is_expr_node_v<L> && std::is_arithmetic_v<T>)
[[nodiscard]] constexpr auto operator+(L l, T r) {
    using VT = typename L::value_type;
    return Binary<Add, L, Const<VT>>{l, Const<VT>{static_cast<VT>(r)}};
}

// scalar + expr
template<typename T, typename R>
    requires (std::is_arithmetic_v<T> && is_expr_node_v<R>)
[[nodiscard]] constexpr auto operator+(T l, R r) {
    using VT = typename R::value_type;
    return Binary<Add, Const<VT>, R>{Const<VT>{static_cast<VT>(l)}, r};
}

// expr - expr
template<typename L, typename R>
    requires (is_expr_node_v<L> && is_expr_node_v<R>)
[[nodiscard]] constexpr auto operator-(L l, R r) {
    if constexpr (is_zero_v<R>) {
        return l;
    } else if constexpr (is_zero_v<L>) {
        return -r;
    } else if constexpr (is_const_expr_v<L> && is_const_expr_v<R>) {
        return Const<typename L::value_type>{l.value - r.value};
    } else if constexpr (std::is_same_v<L, R>) {
        return Zero<typename L::value_type>{};
    } else {
        return Binary<Sub, L, R>{l, r};
    }
}

// expr - scalar
template<typename L, typename T>
    requires (is_expr_node_v<L> && std::is_arithmetic_v<T>)
[[nodiscard]] constexpr auto operator-(L l, T r) {
    using VT = typename L::value_type;
    return Binary<Sub, L, Const<VT>>{l, Const<VT>{static_cast<VT>(r)}};
}

// scalar - expr
template<typename T, typename R>
    requires (std::is_arithmetic_v<T> && is_expr_node_v<R>)
[[nodiscard]] constexpr auto operator-(T l, R r) {
    using VT = typename R::value_type;
    return Binary<Sub, Const<VT>, R>{Const<VT>{static_cast<VT>(l)}, r};
}

// expr * expr (excludes Integral types -- they use ProductIntegral)
template<typename L, typename R>
    requires (is_expr_node_v<L> && is_expr_node_v<R> && !is_integral<L>::value && !is_integral<R>::value)
[[nodiscard]] constexpr auto operator*(L l, R r) {
    if constexpr (is_zero_v<L>) {
        return Zero<typename R::value_type>{};
    } else if constexpr (is_zero_v<R>) {
        return Zero<typename L::value_type>{};
    } else if constexpr (is_one_v<L>) {
        return r;
    } else if constexpr (is_one_v<R>) {
        return l;
    } else if constexpr (is_const_expr_v<L> && is_const_expr_v<R>) {
        return Const<typename L::value_type>{l.value * r.value};
    } else {
        return Binary<Mul, L, R>{l, r};
    }
}

// expr * scalar
template<typename L, typename T>
    requires (is_expr_node_v<L> && std::is_arithmetic_v<T>)
[[nodiscard]] constexpr auto operator*(L l, T r) {
    using VT = typename L::value_type;
    return Binary<Mul, L, Const<VT>>{l, Const<VT>{static_cast<VT>(r)}};
}

// scalar * expr
template<typename T, typename R>
    requires (std::is_arithmetic_v<T> && is_expr_node_v<R>)
[[nodiscard]] constexpr auto operator*(T l, R r) {
    using VT = typename R::value_type;
    return Binary<Mul, Const<VT>, R>{Const<VT>{static_cast<VT>(l)}, r};
}

// expr / expr
template<typename L, typename R>
    requires (is_expr_node_v<L> && is_expr_node_v<R>)
[[nodiscard]] constexpr auto operator/(L l, R r) {
    if constexpr (is_zero_v<L>) {
        return Zero<typename L::value_type>{};
    } else if constexpr (is_one_v<R>) {
        return l;
    } else if constexpr (is_const_expr_v<L> && is_const_expr_v<R>) {
        return Const<typename L::value_type>{l.value / r.value};
    } else if constexpr (std::is_same_v<L, R>) {
        return One<typename L::value_type>{};
    } else {
        return Binary<Div, L, R>{l, r};
    }
}

// expr / scalar
template<typename L, typename T>
    requires (is_expr_node_v<L> && std::is_arithmetic_v<T>)
[[nodiscard]] constexpr auto operator/(L l, T r) {
    using VT = typename L::value_type;
    return Binary<Div, L, Const<VT>>{l, Const<VT>{static_cast<VT>(r)}};
}

// scalar / expr
template<typename T, typename R>
    requires (std::is_arithmetic_v<T> && is_expr_node_v<R>)
[[nodiscard]] constexpr auto operator/(T l, R r) {
    using VT = typename R::value_type;
    return Binary<Div, Const<VT>, R>{Const<VT>{static_cast<VT>(l)}, r};
}

} // namespace limes::expr
