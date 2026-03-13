#pragma once

#include <span>
#include <string>
#include <cstddef>
#include "binary.hpp"

namespace limes::expr {

struct Neg {};

template<typename Op, typename E> struct Unary;

template<typename E> inline constexpr bool is_negation_v = false;
template<typename E> inline constexpr bool is_negation_v<Unary<Neg, E>> = true;

template<typename E> struct negation_inner { using type = void; };
template<typename E> struct negation_inner<Unary<Neg, E>> { using type = E; };
template<typename E> using negation_inner_t = typename negation_inner<E>::type;

// Unary<Op, E>: A unary operation node wrapping a child expression.
// Currently only supports Neg (negation).
template<typename Op, typename E>
struct Unary {
    using value_type = typename E::value_type;
    using op_type = Op;
    using child_type = E;

    static constexpr std::size_t arity_v = E::arity_v;

    E child;

    constexpr explicit Unary(E c) noexcept : child{c} {}

    [[nodiscard]] constexpr value_type eval(std::span<value_type const> args) const {
        if constexpr (std::is_same_v<Op, Neg>) {
            return -child.eval(args);
        }
    }

    [[nodiscard]] [[deprecated("use eval() instead")]]
    constexpr value_type evaluate(std::span<value_type const> args) const {
        return eval(args);
    }

    // d/dx[-e] = -(de/dx)
    template<std::size_t Dim>
    [[nodiscard]] constexpr auto derivative() const {
        if constexpr (std::is_same_v<Op, Neg>) {
            return -child.template derivative<Dim>();
        }
    }

    [[nodiscard]] std::string to_string() const {
        if constexpr (std::is_same_v<Op, Neg>) {
            return "(- " + child.to_string() + ")";
        }
    }
};

// Unary negation with simplification
template<typename E>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto operator-(E e) {
    if constexpr (is_zero_v<E>) {
        return e;           // -0 = 0
    } else if constexpr (is_negation_v<E>) {
        return e.child;     // --x = x
    } else {
        return Unary<Neg, E>{e};
    }
}

// Unary plus (identity)
template<typename E>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto operator+(E e) {
    return e;
}

} // namespace limes::expr
