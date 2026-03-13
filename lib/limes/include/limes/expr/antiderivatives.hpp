#pragma once

#include <type_traits>
#include <cstddef>
#include "nodes/const.hpp"
#include "nodes/var.hpp"
#include "nodes/binary.hpp"
#include "nodes/unary.hpp"
#include "nodes/pow.hpp"
#include "nodes/primitives.hpp"

namespace limes::expr {

// =============================================================================
// Antiderivative Trait: Check if expression has known symbolic antiderivative
// =============================================================================

// Primary template: assume no antiderivative is known
template<typename E, std::size_t Dim, typename = void>
struct has_antiderivative : std::false_type {};

// Helper variable template
template<typename E, std::size_t Dim>
inline constexpr bool has_antiderivative_v = has_antiderivative<E, Dim>::value;

// =============================================================================
// Specializations for known antiderivatives
// =============================================================================

// ∫ c dx = c*x (constant with respect to integration variable)
template<typename T, std::size_t Dim>
struct has_antiderivative<Const<T>, Dim> : std::true_type {};

// ∫ 0 dx = 0
template<typename T, std::size_t Dim>
struct has_antiderivative<Zero<T>, Dim> : std::true_type {};

// ∫ 1 dx = x
template<typename T, std::size_t Dim>
struct has_antiderivative<One<T>, Dim> : std::true_type {};

// ∫ x dx = x²/2 (when integrating Var<Dim> over Dim)
template<std::size_t N, typename T, std::size_t Dim>
struct has_antiderivative<Var<N, T>, Dim>
    : std::bool_constant<N == Dim> {};

// ∫ x^n dx = x^(n+1)/(n+1) for integer n ≠ -1
template<typename E, int N, std::size_t Dim>
struct has_antiderivative<Pow<E, N>, Dim,
    std::enable_if_t<N != -1 && has_antiderivative_v<E, Dim> &&
                     std::is_same_v<E, Var<Dim, typename E::value_type>>>>
    : std::true_type {};

// ∫ sin(x) dx = -cos(x)
template<typename E, std::size_t Dim>
struct has_antiderivative<UnaryFunc<SinTag, E>, Dim,
    std::enable_if_t<std::is_same_v<E, Var<Dim, typename E::value_type>>>>
    : std::true_type {};

// ∫ cos(x) dx = sin(x)
template<typename E, std::size_t Dim>
struct has_antiderivative<UnaryFunc<CosTag, E>, Dim,
    std::enable_if_t<std::is_same_v<E, Var<Dim, typename E::value_type>>>>
    : std::true_type {};

// ∫ exp(x) dx = exp(x)
template<typename E, std::size_t Dim>
struct has_antiderivative<UnaryFunc<ExpTag, E>, Dim,
    std::enable_if_t<std::is_same_v<E, Var<Dim, typename E::value_type>>>>
    : std::true_type {};

// ∫ 1/x dx = ln|x| (as Binary<Div, One, Var>)
// This is tricky - we'd need pattern matching for 1/x

// ∫ (f + g) dx = ∫f dx + ∫g dx
template<typename L, typename R, std::size_t Dim>
struct has_antiderivative<Binary<Add, L, R>, Dim>
    : std::bool_constant<has_antiderivative_v<L, Dim> && has_antiderivative_v<R, Dim>> {};

// ∫ (f - g) dx = ∫f dx - ∫g dx
template<typename L, typename R, std::size_t Dim>
struct has_antiderivative<Binary<Sub, L, R>, Dim>
    : std::bool_constant<has_antiderivative_v<L, Dim> && has_antiderivative_v<R, Dim>> {};

// ∫ c*f dx = c * ∫f dx (when c is constant w.r.t. Dim)
template<typename L, typename R, std::size_t Dim>
struct has_antiderivative<Binary<Mul, L, R>, Dim,
    std::enable_if_t<
        // L is constant and R has antiderivative
        (is_const_expr_v<L> || is_zero_v<L> || is_one_v<L> ||
         (is_expr_node_v<L> && !std::is_same_v<L, Var<Dim, typename L::value_type>> &&
          L::arity_v <= Dim)) &&
        has_antiderivative_v<R, Dim>
    >> : std::true_type {};

// ∫ f*c dx = ∫f dx * c (when c is constant w.r.t. Dim)
template<typename L, typename R, std::size_t Dim>
struct has_antiderivative<Binary<Mul, L, R>, Dim,
    std::enable_if_t<
        has_antiderivative_v<L, Dim> &&
        (is_const_expr_v<R> || is_zero_v<R> || is_one_v<R>) &&
        !is_const_expr_v<L> && !is_zero_v<L> && !is_one_v<L>
    >> : std::true_type {};

// ∫ -f dx = -∫f dx
template<typename E, std::size_t Dim>
struct has_antiderivative<Unary<Neg, E>, Dim>
    : std::bool_constant<has_antiderivative_v<E, Dim>> {};

// =============================================================================
// Antiderivative Computation
// =============================================================================

// Primary template declaration (defined via specialization)
template<std::size_t Dim, typename E>
struct antiderivative_impl;

// Helper function
template<std::size_t Dim, typename E>
[[nodiscard]] constexpr auto antiderivative(E expr) {
    static_assert(has_antiderivative_v<E, Dim>,
        "No symbolic antiderivative known for this expression");
    return antiderivative_impl<Dim, E>::compute(expr);
}

// =============================================================================
// Antiderivative Implementations
// =============================================================================

// ∫ c dx = c*x
template<std::size_t Dim, typename T>
struct antiderivative_impl<Dim, Const<T>> {
    static constexpr auto compute(Const<T> c) {
        auto x = Var<Dim, T>{};
        return c * x;
    }
};

// ∫ 0 dx = 0
template<std::size_t Dim, typename T>
struct antiderivative_impl<Dim, Zero<T>> {
    static constexpr auto compute(Zero<T>) {
        return Zero<T>{};
    }
};

// ∫ 1 dx = x
template<std::size_t Dim, typename T>
struct antiderivative_impl<Dim, One<T>> {
    static constexpr auto compute(One<T>) {
        return Var<Dim, T>{};
    }
};

// ∫ x dx = x²/2
template<std::size_t Dim, typename T>
struct antiderivative_impl<Dim, Var<Dim, T>> {
    static constexpr auto compute(Var<Dim, T>) {
        auto x = Var<Dim, T>{};
        auto half = Const<T>{T(0.5)};
        return half * x * x;
    }
};

// ∫ x^n dx = x^(n+1)/(n+1)
template<std::size_t Dim, typename T, int N>
struct antiderivative_impl<Dim, Pow<Var<Dim, T>, N>> {
    static constexpr auto compute(Pow<Var<Dim, T>, N>) {
        auto x = Var<Dim, T>{};
        constexpr int new_exp = N + 1;
        auto coeff = Const<T>{T(1) / T(new_exp)};
        return coeff * pow<new_exp>(x);
    }
};

// ∫ sin(x) dx = -cos(x)
template<std::size_t Dim, typename T>
struct antiderivative_impl<Dim, UnaryFunc<SinTag, Var<Dim, T>>> {
    static constexpr auto compute(UnaryFunc<SinTag, Var<Dim, T>>) {
        auto x = Var<Dim, T>{};
        return -cos(x);
    }
};

// ∫ cos(x) dx = sin(x)
template<std::size_t Dim, typename T>
struct antiderivative_impl<Dim, UnaryFunc<CosTag, Var<Dim, T>>> {
    static constexpr auto compute(UnaryFunc<CosTag, Var<Dim, T>>) {
        auto x = Var<Dim, T>{};
        return sin(x);
    }
};

// ∫ exp(x) dx = exp(x)
template<std::size_t Dim, typename T>
struct antiderivative_impl<Dim, UnaryFunc<ExpTag, Var<Dim, T>>> {
    static constexpr auto compute(UnaryFunc<ExpTag, Var<Dim, T>>) {
        auto x = Var<Dim, T>{};
        return exp(x);
    }
};

// ∫ (f + g) dx = ∫f dx + ∫g dx
template<std::size_t Dim, typename L, typename R>
struct antiderivative_impl<Dim, Binary<Add, L, R>> {
    static constexpr auto compute(Binary<Add, L, R> expr) {
        auto F = antiderivative<Dim>(expr.left);
        auto G = antiderivative<Dim>(expr.right);
        return F + G;
    }
};

// ∫ (f - g) dx = ∫f dx - ∫g dx
template<std::size_t Dim, typename L, typename R>
struct antiderivative_impl<Dim, Binary<Sub, L, R>> {
    static constexpr auto compute(Binary<Sub, L, R> expr) {
        auto F = antiderivative<Dim>(expr.left);
        auto G = antiderivative<Dim>(expr.right);
        return F - G;
    }
};

// ∫ c*f dx = c * ∫f dx (Const * Expr)
template<std::size_t Dim, typename T, typename R>
struct antiderivative_impl<Dim, Binary<Mul, Const<T>, R>> {
    static constexpr auto compute(Binary<Mul, Const<T>, R> expr) {
        auto F = antiderivative<Dim>(expr.right);
        return expr.left * F;
    }
};

// ∫ f*c dx = ∫f dx * c (Expr * Const)
template<std::size_t Dim, typename L, typename T>
struct antiderivative_impl<Dim, Binary<Mul, L, Const<T>>> {
    static constexpr auto compute(Binary<Mul, L, Const<T>> expr) {
        auto F = antiderivative<Dim>(expr.left);
        return F * expr.right;
    }
};

// ∫ -f dx = -∫f dx
template<std::size_t Dim, typename E>
struct antiderivative_impl<Dim, Unary<Neg, E>> {
    static constexpr auto compute(Unary<Neg, E> expr) {
        auto F = antiderivative<Dim>(expr.child);
        return -F;
    }
};

// =============================================================================
// Definite Integral using Fundamental Theorem of Calculus
// =============================================================================

// Evaluate definite integral symbolically: ∫[a,b] f dx = F(b) - F(a)
template<std::size_t Dim, typename E, typename T>
[[nodiscard]] constexpr T definite_integral_symbolic(E expr, T a, T b) {
    static_assert(has_antiderivative_v<E, Dim>,
        "No symbolic antiderivative known for this expression");

    auto F = antiderivative<Dim>(expr);

    // Evaluate F at bounds
    std::array<T, Dim + 1> args_b{};
    args_b[Dim] = b;
    T F_b = F.eval(std::span<T const>{args_b});

    std::array<T, Dim + 1> args_a{};
    args_a[Dim] = a;
    T F_a = F.eval(std::span<T const>{args_a});

    return F_b - F_a;
}

} // namespace limes::expr
