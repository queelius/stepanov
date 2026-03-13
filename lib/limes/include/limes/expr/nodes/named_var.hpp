#pragma once

#include <span>
#include <string>
#include <string_view>
#include <cstddef>
#include <cassert>
#include "const.hpp"

namespace limes::expr {

// NamedVar<T>: Like Var<N, T> but carries a runtime name for debugging/to_string()
//
// Example:
//   auto x = var(0, "x");
//   auto y = var(1, "y");
//   auto f = sin(x) * cos(y);
//   f.to_string();  // "(* (sin x) (cos y))"
//
template<typename T = double>
struct NamedVar {
    using value_type = T;
    static constexpr std::size_t arity_v = 1;

    std::size_t dim;
    std::string_view name;

    constexpr NamedVar(std::size_t dimension, std::string_view n) noexcept
        : dim{dimension}, name{n} {}

    [[nodiscard]] constexpr std::size_t arity() const noexcept {
        return dim + 1;
    }

    [[nodiscard]] constexpr T eval(std::span<T const> args) const noexcept {
        assert(args.size() > dim && "Not enough arguments for named variable");
        return args[dim];
    }

    [[nodiscard]] [[deprecated("use eval() instead")]]
    constexpr T evaluate(std::span<T const> args) const noexcept {
        return eval(args);
    }

    // Compile-time derivative: limited because dim is a runtime value.
    // Only correct when Dim matches the actual runtime dim.
    template<std::size_t Dim>
    [[nodiscard]] constexpr auto derivative() const noexcept {
        if constexpr (Dim == 0) {
            return (dim == Dim) ? One<T>{} : Zero<T>{};
        } else {
            return Zero<T>{};
        }
    }

    [[nodiscard]] constexpr auto derivative_rt(std::size_t d) const noexcept {
        return (d == dim) ? One<T>{} : Zero<T>{};
    }

    [[nodiscard]] std::string to_string() const {
        return std::string(name);
    }

    [[nodiscard]] constexpr std::size_t dimension() const noexcept {
        return dim;
    }
};

template<typename T>
struct is_named_var : std::false_type {};

template<typename T>
struct is_named_var<NamedVar<T>> : std::true_type {};

template<typename T>
inline constexpr bool is_named_var_v = is_named_var<T>::value;

template<typename T = double>
[[nodiscard]] constexpr NamedVar<T> var(std::size_t dim, std::string_view name) noexcept {
    return NamedVar<T>{dim, name};
}

// StaticNamedVar<N, T>: Combines compile-time dimension (like Var<N>) with a runtime name.
//
// Example:
//   auto x = named<0>("x");
//   auto y = named<1>("y");
//
template<std::size_t N, typename T = double>
struct StaticNamedVar {
    using value_type = T;
    static constexpr std::size_t arity_v = N + 1;
    static constexpr std::size_t index_v = N;

    std::string_view name;

    constexpr explicit StaticNamedVar(std::string_view n) noexcept : name{n} {}

    [[nodiscard]] constexpr T eval(std::span<T const> args) const noexcept {
        assert(args.size() > N && "Not enough arguments for named variable");
        return args[N];
    }

    [[nodiscard]] [[deprecated("use eval() instead")]]
    constexpr T evaluate(std::span<T const> args) const noexcept {
        return eval(args);
    }

    template<std::size_t Dim>
    [[nodiscard]] constexpr auto derivative() const noexcept {
        if constexpr (Dim == N) {
            return One<T>{};
        } else {
            return Zero<T>{};
        }
    }

    [[nodiscard]] std::string to_string() const {
        return std::string(name);
    }

    [[nodiscard]] static constexpr std::size_t dimension() noexcept {
        return N;
    }
};

template<typename T>
struct is_static_named_var : std::false_type {};

template<std::size_t N, typename T>
struct is_static_named_var<StaticNamedVar<N, T>> : std::true_type {};

template<typename T>
inline constexpr bool is_static_named_var_v = is_static_named_var<T>::value;

template<std::size_t N, typename T = double>
[[nodiscard]] constexpr StaticNamedVar<N, T> named(std::string_view name) noexcept {
    return StaticNamedVar<N, T>{name};
}

// Multi-variable declaration helpers

template<std::size_t N, typename T, std::size_t... Is>
constexpr auto declare_vars_impl(std::array<std::string_view, N> const& names,
                                   std::index_sequence<Is...>) {
    return std::make_tuple(StaticNamedVar<Is, T>{names[Is]}...);
}

/// Usage: auto [x, y, z] = declare_vars<3>("x", "y", "z");
template<std::size_t N, typename T = double, typename... Names>
    requires (sizeof...(Names) == N)
[[nodiscard]] constexpr auto declare_vars(Names... names) {
    std::array<std::string_view, N> name_array{names...};
    return declare_vars_impl<N, T>(name_array, std::make_index_sequence<N>{});
}

template<typename T = double>
[[nodiscard]] constexpr auto vars_xy(std::string_view x_name = "x",
                                      std::string_view y_name = "y") {
    return std::make_pair(StaticNamedVar<0, T>{x_name},
                          StaticNamedVar<1, T>{y_name});
}

template<typename T = double>
[[nodiscard]] constexpr auto vars_xyz(std::string_view x_name = "x",
                                       std::string_view y_name = "y",
                                       std::string_view z_name = "z") {
    return std::make_tuple(StaticNamedVar<0, T>{x_name},
                           StaticNamedVar<1, T>{y_name},
                           StaticNamedVar<2, T>{z_name});
}

} // namespace limes::expr
