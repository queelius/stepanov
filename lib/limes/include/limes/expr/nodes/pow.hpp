#pragma once

#include <span>
#include <string>
#include <cstddef>
#include <cmath>
#include "binary.hpp"

namespace limes::expr {

template<typename E, int N> struct Pow;

template<typename E> inline constexpr bool is_pow_v = false;
template<typename E, int N> inline constexpr bool is_pow_v<Pow<E, N>> = true;

template<typename E> struct pow_base { using type = void; };
template<typename E, int N> struct pow_base<Pow<E, N>> { using type = E; };
template<typename E> using pow_base_t = typename pow_base<E>::type;

template<typename E> inline constexpr int pow_exponent_v = 0;
template<typename E, int N> inline constexpr int pow_exponent_v<Pow<E, N>> = N;

// Pow<E, N>: Expression raised to a compile-time integer power N
template<typename E, int N>
struct Pow {
    using value_type = typename E::value_type;
    using base_type = E;
    static constexpr std::size_t arity_v = E::arity_v;
    static constexpr int exponent = N;

    E base;

    constexpr explicit Pow(E b) noexcept : base{b} {}

    [[nodiscard]] constexpr value_type eval(std::span<value_type const> args) const {
        if constexpr (N == 0) {
            return value_type(1);
        } else if constexpr (N == 1) {
            return base.eval(args);
        } else if constexpr (N == 2) {
            auto v = base.eval(args);
            return v * v;
        } else if constexpr (N == 3) {
            auto v = base.eval(args);
            return v * v * v;
        } else {
            return std::pow(base.eval(args), N);
        }
    }

    [[nodiscard]] [[deprecated("use eval() instead")]]
    constexpr value_type evaluate(std::span<value_type const> args) const {
        return eval(args);
    }

    // Power rule: d/dx[u^n] = n * u^(n-1) * du/dx
    template<std::size_t Dim>
    [[nodiscard]] constexpr auto derivative() const {
        auto du = base.template derivative<Dim>();

        if constexpr (N == 0) {
            return Zero<value_type>{};
        } else if constexpr (N == 1) {
            return du;
        } else if constexpr (N == 2) {
            return Const<value_type>{value_type(2)} * base * du;
        } else {
            return Const<value_type>{value_type(N)} * Pow<E, N-1>{base} * du;
        }
    }

    [[nodiscard]] std::string to_string() const {
        return "(^ " + base.to_string() + " " + std::to_string(N) + ")";
    }
};

// Factory: pow<N>(expr) with compile-time simplification
template<int N, typename E>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto pow(E e) {
    if constexpr (N == 0) {
        return One<typename E::value_type>{};
    } else if constexpr (N == 1) {
        return e;
    } else {
        return Pow<E, N>{e};
    }
}

template<typename E>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto square(E e) {
    return Pow<E, 2>{e};
}

template<typename E>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto cube(E e) {
    return Pow<E, 3>{e};
}

} // namespace limes::expr
