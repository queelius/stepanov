#pragma once

/**
 * @file integral.hpp
 * @brief Fluent builder and node types for numerical integration.
 *
 * This module provides the `Integral` expression node, `IntegralBuilder` for
 * fluent construction, and related types like `TransformedIntegral` for
 * change-of-variables.
 *
 * @section integral_usage Usage
 *
 * @code{.cpp}
 * using namespace limes::expr;
 *
 * auto x = arg<0>;
 *
 * // Basic integral: ∫₀¹ x² dx
 * auto I = integral(x*x).over<0>(0.0, 1.0);
 * auto result = I.eval();  // ≈ 0.333...
 *
 * // With specific method
 * auto r2 = I.eval(gauss<7>());
 *
 * // Nested integral: ∫₀¹ ∫₀¹ xy dx dy
 * auto y = arg<1>;
 * auto J = integral(integral(x*y).over<0>(0.0, 1.0)).over<1>(0.0, 1.0);
 *
 * // Domain operations
 * auto [left, right] = I.split(0.5);  // Split at x=0.5
 *
 * // Change of variables (remove singularity)
 * auto K = integral(1.0/sqrt(x)).over<0>(0.0, 1.0);
 * auto T = K.transform(
 *     [](double t) { return t*t; },    // x = t²
 *     [](double t) { return 2*t; },    // dx/dt = 2t
 *     0.0, 1.0                          // new bounds
 * );
 * @endcode
 *
 * @see IntegralBuilder Entry point via integral()
 * @see Integral Integration expression node
 * @see TransformedIntegral Change of variables
 * @ingroup expr_ops
 */

#include <span>
#include <string>
#include <array>
#include <vector>
#include <cstddef>
#include <sstream>
#include <utility>
#include "../algorithms/integrators/integrators.hpp"
#include "../methods/methods.hpp"
#include "nodes/const.hpp"
#include "nodes/binary.hpp"
#include "box_integral.hpp"

/**
 * @addtogroup expr_ops
 * @{
 */

namespace limes::expr {

// Forward declarations
template<typename E, std::size_t Dim, typename Lo, typename Hi>
struct Integral;

template<typename E>
struct IntegralBuilder;

template<typename OriginalIntegral, typename Phi, typename Jacobian>
struct TransformedIntegral;

namespace detail {

/// Build a 1D function from an N-D integrand by fixing outer variables
/// and varying only dimension Dim. Shared by Integral and TransformedIntegral.
template<std::size_t Dim, typename E, typename T>
auto make_integrand_fn(E const& integrand, std::span<T const> outer_args) {
    return [&integrand, outer_args](T x) -> T {
        std::vector<T> full_args;
        full_args.reserve(E::arity_v);

        std::size_t args_idx = 0;
        for (std::size_t i = 0; i < E::arity_v; ++i) {
            if (i == Dim) {
                full_args.push_back(x);
            } else if (args_idx < outer_args.size()) {
                full_args.push_back(outer_args[args_idx++]);
            } else {
                full_args.push_back(T{0});
            }
        }

        return integrand.eval(std::span<T const>{full_args});
    };
}

} // namespace detail

// =============================================================================
// Type traits for Integral
// =============================================================================

/// Specialization: Integral<E, Dim, Lo, Hi> is recognized as an integral type
/// (Primary template is_integral is defined in nodes/binary.hpp)
template<typename E, std::size_t Dim, typename Lo, typename Hi>
struct is_integral<Integral<E, Dim, Lo, Hi>> : std::true_type {};

template<typename T>
inline constexpr bool is_integral_v = is_integral<T>::value;

/// Constant integration bound (e.g., 0.0, 1.0)
template<typename T>
struct ConstBound {
    T value;

    constexpr ConstBound() noexcept : value{} {}
    constexpr explicit ConstBound(T v) noexcept : value{v} {}

    template<typename Args>
    [[nodiscard]] constexpr T eval(Args const& /*args*/) const noexcept {
        return value;
    }

    template<typename Args>
    [[nodiscard]] [[deprecated("use eval() instead")]]
    constexpr T evaluate(Args const& args) const noexcept {
        return eval(args);
    }

    [[nodiscard]] std::string to_string() const {
        std::ostringstream oss;
        oss << value;
        return oss.str();
    }
};

/// Expression-valued integration bound (depends on outer variables)
template<typename E>
struct ExprBound {
    E expr;

    constexpr explicit ExprBound(E e) noexcept : expr{e} {}

    template<typename Args>
    [[nodiscard]] constexpr auto eval(Args const& args) const {
        return expr.eval(args);
    }

    template<typename Args>
    [[nodiscard]] [[deprecated("use eval() instead")]]
    constexpr auto evaluate(Args const& args) const {
        return eval(args);
    }

    [[nodiscard]] std::string to_string() const {
        return "(bound " + expr.to_string() + ")";
    }
};

// Helper to create bounds
template<typename T>
    requires std::is_arithmetic_v<T>
[[nodiscard]] constexpr auto make_bound(T value) {
    return ConstBound<T>{value};
}

template<typename E>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto make_bound(E expr) {
    return ExprBound<E>{expr};
}

/**
 * @brief Definite integral expression node.
 *
 * Represents the integral ∫ₐᵇ f(x₀,...,xₙ) dxₐᵢₘ where:
 * - `E` is the integrand expression
 * - `Dim` is the dimension being integrated over
 * - `Lo` and `Hi` are the bounds (constant or expression-valued)
 *
 * The arity of the result is one less than the integrand's arity,
 * since dimension `Dim` is consumed by the integration.
 *
 * @tparam E Integrand expression type
 * @tparam Dim The variable dimension being integrated (0-indexed)
 * @tparam Lo Lower bound type (ConstBound or ExprBound)
 * @tparam Hi Upper bound type (ConstBound or ExprBound)
 *
 * @par Domain Operations
 * - `split(point)` — Split integral at a point: [a,b] → [a,c] + [c,b]
 * - `swap<D1,D2>()` — Swap integration order (Fubini's theorem)
 * - `transform(φ, φ', c, d)` — Change of variables
 *
 * @par Method-based Evaluation
 * @code{.cpp}
 * I.eval();                          // Default (adaptive)
 * I.eval(gauss<7>());                // Gauss-Legendre
 * I.eval(monte_carlo_method(10000)); // Monte Carlo
 * @endcode
 *
 * @see IntegralBuilder Fluent builder
 * @see integral() Entry point function
 */
template<typename E, std::size_t Dim, typename Lo, typename Hi>
struct Integral {
    using value_type = typename E::value_type;
    using integrand_type = E;
    using lower_bound_type = Lo;
    using upper_bound_type = Hi;

    static constexpr std::size_t dim_v = Dim;
    static constexpr std::size_t arity_v = (E::arity_v > 0) ? (E::arity_v - 1) : 0;

    E integrand;
    Lo lower;
    Hi upper;
    value_type tolerance;

    constexpr Integral(E e, Lo lo, Hi hi, value_type tol = value_type(1e-10)) noexcept
        : integrand{e}, lower{lo}, upper{hi}, tolerance{tol} {}

    // Evaluate the integral numerically
    [[nodiscard]] algorithms::integration_result<value_type>
    eval(std::span<value_type const> args) const {
        value_type lo = lower.eval(args);
        value_type hi = upper.eval(args);
        auto f = detail::make_integrand_fn<Dim>(integrand, args);

        algorithms::adaptive_integrator<value_type> integrator;
        return integrator(f, lo, hi, tolerance);
    }

    // Convenience: eval with no outer arguments (for fully-integrated expressions)
    [[nodiscard]] algorithms::integration_result<value_type> eval() const {
        return eval(std::span<value_type const>{});
    }

    /// Evaluate the integral using a specific integration method
    template<typename Method>
        requires methods::is_integration_method_v<Method>
    [[nodiscard]] algorithms::integration_result<value_type>
    eval(Method const& method, std::span<value_type const> args) const {
        value_type lo = lower.eval(args);
        value_type hi = upper.eval(args);
        auto f = detail::make_integrand_fn<Dim>(integrand, args);

        return method(f, lo, hi);
    }

    /// Evaluate the integral using a specific method (no outer arguments)
    template<typename Method>
        requires methods::is_integration_method_v<Method>
    [[nodiscard]] algorithms::integration_result<value_type>
    eval(Method const& method) const {
        return eval(method, std::span<value_type const>{});
    }

    // Deprecated: use eval() instead
    [[nodiscard]] [[deprecated("use eval() instead")]]
    algorithms::integration_result<value_type> evaluate(std::span<value_type const> args) const {
        return eval(args);
    }

    // Deprecated: use eval() instead
    [[nodiscard]] [[deprecated("use eval() instead")]]
    algorithms::integration_result<value_type> evaluate() const {
        return eval();
    }

    // Chain integration: add another integral layer
    template<std::size_t NewDim, typename NewLo, typename NewHi>
    [[nodiscard]] constexpr auto over(NewLo lo, NewHi hi) const {
        return Integral<Integral, NewDim, decltype(make_bound(lo)), decltype(make_bound(hi))>{
            *this, make_bound(lo), make_bound(hi), tolerance
        };
    }

    // Convenience: over with explicit dimension template parameter
    // integral(f).over<0>(0.0, 1.0)
    template<std::size_t NewDim>
    [[nodiscard]] constexpr auto over(value_type lo, value_type hi) const {
        return Integral<Integral, NewDim, ConstBound<value_type>, ConstBound<value_type>>{
            *this, ConstBound<value_type>{lo}, ConstBound<value_type>{hi}, tolerance
        };
    }

    // String representation
    [[nodiscard]] std::string to_string() const {
        return "(integral " + integrand.to_string() +
               " [" + std::to_string(Dim) + " " +
               lower.to_string() + " " + upper.to_string() + "])";
    }

    // =========================================================================
    // Domain Operations
    // =========================================================================

    /// Split the integration domain at a given point
    ///
    /// Returns a pair of integrals: [lower, point] and [point, upper]
    /// Invariant: left.evaluate() + right.evaluate() ≈ original.evaluate()
    ///
    /// Example:
    ///   auto I = integral(f).over<0>(0.0, 1.0);
    ///   auto [left, right] = I.split(0.5);
    ///   // left integrates from 0 to 0.5
    ///   // right integrates from 0.5 to 1
    ///
    template<std::size_t SplitDim = Dim>
    [[nodiscard]] constexpr auto split(value_type point) const {
        static_assert(SplitDim == Dim,
            "split() can only split on the integration dimension");

        using LeftIntegral = Integral<E, Dim, Lo, ConstBound<value_type>>;
        using RightIntegral = Integral<E, Dim, ConstBound<value_type>, Hi>;

        auto left = LeftIntegral{integrand, lower, ConstBound<value_type>{point}, tolerance};
        auto right = RightIntegral{integrand, ConstBound<value_type>{point}, upper, tolerance};

        return std::make_pair(left, right);
    }

    /// Swap integration order for nested integrals (Fubini's theorem)
    ///
    /// For a double integral ∫∫ f(x,y) dx dy, swapping gives ∫∫ f(x,y) dy dx
    /// Only available when the integrand is itself an Integral
    ///
    /// Example:
    ///   auto I = integral(integral(x*y).over<0>(0,1)).over<1>(0,1);
    ///   auto J = I.swap<0, 1>();  // Swap x and y integration order
    ///   // I.evaluate() ≈ J.evaluate()
    ///
    template<std::size_t D1, std::size_t D2>
        requires is_integral_v<E>
    [[nodiscard]] constexpr auto swap() const {
        // Get the inner integral
        auto& inner = integrand;

        // For swap to work, we need:
        // - Outer integral over dimension Dim (this)
        // - Inner integral over dimension E::dim_v (inner)
        // We swap them by creating:
        // - New outer integral over E::dim_v
        // - New inner integral over Dim

        // Create the swapped structure
        using InnerIntegrand = typename E::integrand_type;
        using InnerLo = typename E::lower_bound_type;
        using InnerHi = typename E::upper_bound_type;
        constexpr std::size_t InnerDim = E::dim_v;

        // New inner integral: same integrand, but integrated over what was the outer dimension
        using NewInnerType = Integral<InnerIntegrand, Dim, Lo, Hi>;
        auto new_inner = NewInnerType{inner.integrand, lower, upper, tolerance};

        // New outer integral: over what was the inner dimension
        using NewOuterType = Integral<NewInnerType, InnerDim, InnerLo, InnerHi>;
        return NewOuterType{new_inner, inner.lower, inner.upper, tolerance};
    }

    /// Apply a change of variables (substitution) to the integral
    ///
    /// For ∫[a,b] f(x) dx, using x = φ(t) gives:
    ///   ∫[c,d] f(φ(t)) |φ'(t)| dt
    ///
    /// Example (removing sqrt singularity using x = t²):
    ///   auto I = integral(1.0 / sqrt(x)).over<0>(0.0, 1.0);
    ///   auto T = I.transform(
    ///       [](double t) { return t*t; },           // φ: t → t²
    ///       [](double t) { return 2*t; },           // φ': t → 2t
    ///       0.0, 1.0                                 // new bounds
    ///   );
    ///
    template<typename Phi, typename Jacobian>
    [[nodiscard]] constexpr auto transform(
        Phi phi,
        Jacobian jacobian,
        value_type new_lower,
        value_type new_upper
    ) const {
        return TransformedIntegral<Integral, Phi, Jacobian>{
            *this, phi, jacobian, new_lower, new_upper, tolerance
        };
    }

    /// Set new tolerance for this integral
    [[nodiscard]] constexpr auto with_tolerance(value_type tol) const {
        return Integral{integrand, lower, upper, tol};
    }
};

/**
 * @brief Fluent builder for constructing integrals.
 *
 * IntegralBuilder wraps an expression and provides methods for specifying
 * integration bounds. Use `integral(expr)` to create a builder.
 *
 * @tparam E The integrand expression type
 *
 * @par Usage
 * @code{.cpp}
 * auto x = arg<0>;
 *
 * // 1D integral
 * auto I = integral(x*x).over<0>(0.0, 1.0);
 *
 * // Box integral (N-dimensional)
 * auto J = integral(x*y).over_box({{0,1}, {0,1}});
 *
 * // With custom tolerance
 * auto K = integral(f).with_tolerance(1e-12).over<0>(0.0, 1.0);
 * @endcode
 *
 * @see integral() Entry point function
 * @see Integral The resulting integral node
 * @see BoxIntegral For N-dimensional box integration
 */
template<typename E>
struct IntegralBuilder {
    using value_type = typename E::value_type;

    E expr;
    value_type tolerance;

    constexpr IntegralBuilder(E e, value_type tol = value_type(1e-10)) noexcept
        : expr{e}, tolerance{tol} {}

    // over<Dim>(lo, hi): Create an integral over dimension Dim
    template<std::size_t Dim, typename Lo, typename Hi>
    [[nodiscard]] constexpr auto over(Lo lo, Hi hi) const {
        return Integral<E, Dim, decltype(make_bound(lo)), decltype(make_bound(hi))>{
            expr, make_bound(lo), make_bound(hi), tolerance
        };
    }

    // over(dim, lo, hi): Runtime dimension (limited support)
    // For now, only supports dimension 0 at runtime
    template<typename Lo, typename Hi>
    [[nodiscard]] constexpr auto over(std::size_t dim, Lo lo, Hi hi) const {
        // Runtime dispatch - support dimensions 0-7
        if (dim == 0) return over<0>(lo, hi);
        if (dim == 1) return over<1>(lo, hi);
        if (dim == 2) return over<2>(lo, hi);
        if (dim == 3) return over<3>(lo, hi);
        // Default to dimension 0 for higher dimensions
        return over<0>(lo, hi);
    }

    // Set tolerance
    [[nodiscard]] constexpr IntegralBuilder with_tolerance(value_type tol) const {
        return IntegralBuilder{expr, tol};
    }

    // over_box(bounds): Create an N-dimensional box integral for Monte Carlo
    template<std::size_t Dims>
    [[nodiscard]] constexpr auto over_box(
        std::array<std::pair<value_type, value_type>, Dims> bounds
    ) const {
        return BoxIntegral<E, Dims>{expr, bounds};
    }

    // Convenience: over_box with initializer list for 2D
    [[nodiscard]] constexpr auto over_box(
        std::pair<value_type, value_type> b0,
        std::pair<value_type, value_type> b1
    ) const {
        return BoxIntegral<E, 2>{expr, {{{b0.first, b0.second}, {b1.first, b1.second}}}};
    }

    // Convenience: over_box with initializer list for 3D
    [[nodiscard]] constexpr auto over_box(
        std::pair<value_type, value_type> b0,
        std::pair<value_type, value_type> b1,
        std::pair<value_type, value_type> b2
    ) const {
        return BoxIntegral<E, 3>{expr, {{{b0.first, b0.second}, {b1.first, b1.second}, {b2.first, b2.second}}}};
    }
};

/**
 * @brief Create an IntegralBuilder for fluent integral construction.
 *
 * This is the main entry point for building integrals. Returns an
 * IntegralBuilder that can be chained with `.over<Dim>(a, b)` or
 * `.over_box(bounds)` to specify integration bounds.
 *
 * @tparam E Expression type (automatically deduced)
 * @param expr The integrand expression
 * @return IntegralBuilder<E> A builder for specifying bounds
 *
 * @par Example
 * @code{.cpp}
 * auto x = arg<0>;
 * auto I = integral(x*x).over<0>(0.0, 1.0);  // ∫₀¹ x² dx
 * auto result = I.eval();                     // ≈ 0.333
 * @endcode
 */
template<typename E>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto integral(E expr) {
    return IntegralBuilder<E>{expr};
}

// =============================================================================
// TransformedIntegral: Change of variables for integration
// =============================================================================

/**
 * @brief Integral with change of variables (substitution).
 *
 * Applies the transformation rule:
 * @f[
 *   \int_a^b f(x) \, dx = \int_c^d f(\phi(t)) \cdot |\phi'(t)| \, dt
 * @f]
 *
 * This is useful for removing singularities or improving convergence.
 *
 * @tparam OriginalIntegral The original integral type
 * @tparam Phi The substitution function type (t → x)
 * @tparam Jacobian The Jacobian |dφ/dt| function type
 *
 * @par Example: Removing √x singularity
 * @code{.cpp}
 * // ∫₀¹ 1/√x dx has singularity at x=0
 * auto I = integral(1.0/sqrt(x)).over<0>(0.0, 1.0);
 *
 * // Substitute x = t² to get ∫₀¹ 1/t · 2t dt = ∫₀¹ 2 dt
 * auto T = I.transform(
 *     [](double t) { return t*t; },    // φ: t → t²
 *     [](double t) { return 2*t; },    // |φ'|: 2t
 *     0.0, 1.0                          // new bounds
 * );
 * @endcode
 *
 * @see Integral::transform() Factory method
 * @see transforms::quadratic Predefined quadratic transform
 */
template<typename OriginalIntegral, typename Phi, typename Jacobian>
struct TransformedIntegral {
    using value_type = typename OriginalIntegral::value_type;

    static constexpr std::size_t arity_v = OriginalIntegral::arity_v;

    OriginalIntegral original;
    Phi phi;                    // t → x substitution
    Jacobian jacobian;          // |dφ/dt|
    value_type new_lower;
    value_type new_upper;
    value_type tolerance;

    constexpr TransformedIntegral(
        OriginalIntegral orig,
        Phi p,
        Jacobian j,
        value_type new_lo,
        value_type new_hi,
        value_type tol = value_type(1e-10)
    ) noexcept
        : original{orig}
        , phi{p}
        , jacobian{j}
        , new_lower{new_lo}
        , new_upper{new_hi}
        , tolerance{tol}
    {}

    /// Evaluate the transformed integral numerically
    [[nodiscard]] algorithms::integration_result<value_type>
    eval(std::span<value_type const> args) const {
        constexpr std::size_t dim = OriginalIntegral::dim_v;
        auto base_fn = detail::make_integrand_fn<dim>(original.integrand, args);

        auto transformed_f = [this, &base_fn](value_type t) -> value_type {
            value_type x = phi(t);
            value_type jac = std::abs(jacobian(t));
            return base_fn(x) * jac;
        };

        algorithms::adaptive_integrator<value_type> integrator;
        return integrator(transformed_f, new_lower, new_upper, tolerance);
    }

    [[nodiscard]] algorithms::integration_result<value_type> eval() const {
        return eval(std::span<value_type const>{});
    }

    // Deprecated: use eval() instead
    [[nodiscard]] [[deprecated("use eval() instead")]]
    algorithms::integration_result<value_type> evaluate(std::span<value_type const> args) const {
        return eval(args);
    }

    // Deprecated: use eval() instead
    [[nodiscard]] [[deprecated("use eval() instead")]]
    algorithms::integration_result<value_type> evaluate() const {
        return eval();
    }

    [[nodiscard]] std::string to_string() const {
        std::ostringstream oss;
        oss << "(transformed-integral " << original.to_string()
            << " [" << new_lower << " " << new_upper << "])";
        return oss.str();
    }
};

// =============================================================================
// Common Transforms
// =============================================================================

namespace transforms {

/// Linear transform: x = a*t + b
template<typename T>
struct linear {
    T a;  // scale
    T b;  // shift

    constexpr linear(T scale, T shift) noexcept : a{scale}, b{shift} {}

    [[nodiscard]] constexpr T operator()(T t) const noexcept {
        return a * t + b;
    }
};

/// Linear transform Jacobian: |dx/dt| = |a|
template<typename T>
struct linear_jacobian {
    T a;

    constexpr explicit linear_jacobian(T scale) noexcept : a{scale} {}

    [[nodiscard]] constexpr T operator()(T /*t*/) const noexcept {
        return std::abs(a);
    }
};

/// Quadratic transform: x = t^2 (removes sqrt singularities at x=0)
template<typename T>
struct quadratic {
    [[nodiscard]] constexpr T operator()(T t) const noexcept {
        return t * t;
    }
};

/// Quadratic transform Jacobian: |dx/dt| = 2|t|
template<typename T>
struct quadratic_jacobian {
    [[nodiscard]] constexpr T operator()(T t) const noexcept {
        return T(2) * std::abs(t);
    }
};

} // namespace transforms

/** @} */ // end of expr_ops group

} // namespace limes::expr
