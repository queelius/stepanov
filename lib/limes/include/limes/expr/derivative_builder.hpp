#pragma once

/**
 * @file derivative_builder.hpp
 * @brief Fluent builder pattern for symbolic differentiation.
 *
 * This module provides the `DerivativeBuilder` class and `derivative()` function
 * for computing symbolic derivatives using a fluent API.
 *
 * @section derivative_usage Usage
 *
 * @code{.cpp}
 * using namespace limes::expr;
 *
 * auto x = arg<0>;
 * auto y = arg<1>;
 * auto f = sin(x*x) + cos(y);
 *
 * // Single partial derivative
 * auto df_dx = derivative(f).wrt<0>();      // ∂f/∂x = 2x·cos(x²)
 *
 * // Mixed partial derivative
 * auto d2f = derivative(f).wrt<0>().wrt<1>(); // ∂²f/∂x∂y
 *
 * // Convenience for second-order
 * auto d2f_dx2 = derivative(f).wrt<0, 0>();   // ∂²f/∂x²
 *
 * // Gradient (all first partials)
 * auto [fx, fy] = derivative(f).gradient();
 * @endcode
 *
 * @see derivative() Entry point function
 * @see DerivativeBuilder Builder class
 * @ingroup expr_ops
 */

#include <tuple>
#include <utility>
#include "nodes/binary.hpp"

/**
 * @addtogroup expr_ops
 * @{
 */

namespace limes::expr {

// =============================================================================
// DerivativeBuilder: Fluent builder for derivatives
// =============================================================================

/**
 * @brief Fluent builder for computing symbolic derivatives.
 *
 * DerivativeBuilder wraps an expression and provides chainable methods for
 * computing partial derivatives. Derivatives are computed symbolically at
 * compile time using the chain rule.
 *
 * @tparam E The expression type being differentiated
 *
 * @par Fluent API
 * @code{.cpp}
 * derivative(f).wrt<0>()              // ∂f/∂x₀
 * derivative(f).wrt<0>().wrt<1>()     // ∂²f/∂x₀∂x₁
 * derivative(f).wrt<0, 0>()           // ∂²f/∂x₀² (convenience)
 * derivative(f).gradient()            // all partials as tuple
 * @endcode
 *
 * @par Evaluation
 * The result of differentiation is itself an expression that can be evaluated:
 * @code{.cpp}
 * auto df = derivative(sin(x*x)).wrt<0>();
 * double result = df.eval({2.0});  // Evaluate at x=2
 * @endcode
 *
 * @see derivative() Entry point function
 */
template<typename E>
struct DerivativeBuilder {
    using value_type = typename E::value_type;
    using expr_type = E;

    E expr;

    constexpr DerivativeBuilder(E e) noexcept : expr{e} {}

    // =========================================================================
    // wrt<Dim>(): Compute partial derivative with respect to dimension Dim
    // =========================================================================

    /// Compute ∂/∂x_Dim and return a new builder for chaining
    template<std::size_t Dim>
    [[nodiscard]] constexpr auto wrt() const {
        auto result = expr.template derivative<Dim>();
        return DerivativeBuilder<decltype(result)>{result};
    }

    /// Compute ∂²/∂x_D1 ∂x_D2 (convenience for second-order mixed partials)
    template<std::size_t D1, std::size_t D2>
    [[nodiscard]] constexpr auto wrt() const {
        return wrt<D1>().template wrt<D2>();
    }

    /// Compute ∂³/∂x_D1 ∂x_D2 ∂x_D3 (convenience for third-order mixed partials)
    template<std::size_t D1, std::size_t D2, std::size_t D3>
    [[nodiscard]] constexpr auto wrt() const {
        return wrt<D1>().template wrt<D2>().template wrt<D3>();
    }

    // =========================================================================
    // gradient(): Compute all partial derivatives as a tuple
    // =========================================================================

    /// Returns tuple<∂f/∂x₀, ∂f/∂x₁, ..., ∂f/∂x_{n-1}>
    [[nodiscard]] constexpr auto gradient() const {
        return gradient_impl(std::make_index_sequence<E::arity_v>{});
    }

    // =========================================================================
    // Implicit conversion to underlying expression
    // =========================================================================

    /// Allow implicit conversion to the underlying derivative expression
    [[nodiscard]] constexpr E get() const noexcept { return expr; }

    /// Evaluate the derivative expression
    [[nodiscard]] constexpr value_type eval(std::span<value_type const> args) const {
        return expr.eval(args);
    }

    /// Deprecated: use eval() instead
    [[nodiscard]] [[deprecated("use eval() instead")]]
    constexpr value_type evaluate(std::span<value_type const> args) const {
        return eval(args);
    }

    /// String representation
    [[nodiscard]] std::string to_string() const {
        return expr.to_string();
    }

    /// Arity of the derivative expression
    static constexpr std::size_t arity_v = E::arity_v;

    /// Further differentiation through the member function
    template<std::size_t Dim>
    [[nodiscard]] constexpr auto derivative() const {
        return expr.template derivative<Dim>();
    }

private:
    template<std::size_t... Is>
    [[nodiscard]] constexpr auto gradient_impl(std::index_sequence<Is...>) const {
        return std::make_tuple(expr.template derivative<Is>()...);
    }
};

// Type trait for DerivativeBuilder detection
template<typename T>
struct is_derivative_builder : std::false_type {};

template<typename E>
struct is_derivative_builder<DerivativeBuilder<E>> : std::true_type {};

template<typename T>
inline constexpr bool is_derivative_builder_v = is_derivative_builder<T>::value;

// =============================================================================
// Builder entry point (new API, overloads existing derivative)
// =============================================================================

/**
 * @brief Create a DerivativeBuilder for fluent derivative computation.
 *
 * This is the main entry point for the derivative builder pattern.
 * Returns a DerivativeBuilder that can be chained with `.wrt<Dim>()` calls.
 *
 * @tparam E Expression type (automatically deduced)
 * @param expr The expression to differentiate
 * @return DerivativeBuilder<E> A builder for fluent differentiation
 *
 * @par Example
 * @code{.cpp}
 * auto x = arg<0>;
 * auto y = arg<1>;
 * auto f = x*x + y*y;
 *
 * auto df_dx = derivative(f).wrt<0>();      // 2x
 * auto d2f   = derivative(f).wrt<0, 1>();   // 0 (no mixed term)
 * auto grad  = derivative(f).gradient();    // (2x, 2y)
 * @endcode
 */
template<typename E>
    requires (is_expr_node_v<E> && !is_derivative_builder_v<E>)
[[nodiscard]] constexpr auto derivative(E expr) {
    return DerivativeBuilder<E>{expr};
}

// Note: The existing derivative<Dim>(expr) free function in derivative.hpp
// is still available for direct use without the builder pattern.

/** @} */ // end of expr_ops group

} // namespace limes::expr
