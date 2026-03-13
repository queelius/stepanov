#pragma once

#include <span>
#include <string>
#include <cstddef>
#include <cassert>
#include "const.hpp"

namespace limes::expr {

// Var<N, T>: A variable reference at position N (arity = N + 1)
template<std::size_t N, typename T = double>
struct Var {
    using value_type = T;
    static constexpr std::size_t arity_v = N + 1;
    static constexpr std::size_t index_v = N;

    constexpr Var() noexcept = default;

    [[nodiscard]] constexpr T eval(std::span<T const> args) const noexcept {
        assert(args.size() > N && "Not enough arguments for variable");
        return args[N];
    }

    [[nodiscard]] [[deprecated("use eval() instead")]]
    constexpr T evaluate(std::span<T const> args) const noexcept {
        return eval(args);
    }

    // Returns One<T>/Zero<T> marker types for compile-time simplification
    template<std::size_t Dim>
    [[nodiscard]] constexpr auto derivative() const noexcept {
        if constexpr (Dim == N) {
            return One<T>{};
        } else {
            return Zero<T>{};
        }
    }

    [[nodiscard]] std::string to_string() const {
        return "x" + std::to_string(N);
    }
};

template<typename T = double>
inline constexpr Var<0, T> x{};

template<typename T = double>
inline constexpr Var<1, T> y{};

template<typename T = double>
inline constexpr Var<2, T> z{};

template<std::size_t N, typename T = double>
inline constexpr Var<N, T> arg{};

} // namespace limes::expr
