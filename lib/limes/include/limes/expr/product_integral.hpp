#pragma once

/**
 * @file product_integral.hpp
 * @brief Separable integral composition for independent integrals.
 *
 * This module provides `ProductIntegral` for composing independent integrals.
 * When two integrals depend on disjoint sets of variables, their product can
 * be computed as the product of the individual results—a significant optimization.
 *
 * @section product_usage Usage
 *
 * @code{.cpp}
 * using namespace limes::expr;
 *
 * auto x = arg<0>;
 * auto y = arg<1>;
 *
 * // Two independent integrals
 * auto I = integral(sin(x)).over<0>(0.0, 3.14159);  // depends on x only
 * auto J = integral(exp(y)).over<1>(0.0, 1.0);      // depends on y only
 *
 * // Product integral: evaluates each separately, then multiplies
 * auto IJ = I * J;
 * auto result = IJ.eval();  // Much faster than nested integration!
 *
 * // Chaining
 * auto z = arg<2>;
 * auto K = integral(cos(z)).over<2>(0.0, 1.0);
 * auto IJK = I * J * K;
 * @endcode
 *
 * @par Compile-time Independence Check
 * The library uses `variable_set` analysis to verify at compile time that
 * the integrals are over disjoint variable sets. If they share variables,
 * compilation fails with a clear error message.
 *
 * @see ProductIntegral The composition type
 * @see product() Variadic factory function
 * @see are_independent_integrals_v Compile-time independence check
 * @ingroup expr_ops
 */

#include <type_traits>
#include <cmath>
#include <utility>
#include "../algorithms/core/result.hpp"
#include "analysis.hpp"

/**
 * @addtogroup expr_ops
 * @{
 */

namespace limes::expr {

// Forward declarations
template<typename E, std::size_t Dim, typename Lo, typename Hi>
struct Integral;

template<typename E, std::size_t Dims>
struct BoxIntegral;

template<typename I1, typename I2>
struct ProductIntegral;

// Helper to get variable set from any evaluable integral type
namespace detail {

template<typename T, typename = void>
struct integral_variable_set {
    static constexpr std::uint64_t value = 0;
};

// For types with integrand_type (Integral, BoxIntegral)
template<typename T>
struct integral_variable_set<T, std::void_t<typename T::integrand_type>> {
    static constexpr std::uint64_t value = variable_set_v<typename T::integrand_type>;
};

// For ProductIntegral: combine the variable sets of both children
template<typename I1, typename I2>
struct integral_variable_set<ProductIntegral<I1, I2>> {
    static constexpr std::uint64_t value =
        integral_variable_set<I1>::value | integral_variable_set<I2>::value;
};

template<typename T>
inline constexpr std::uint64_t integral_variable_set_v = integral_variable_set<T>::value;

} // namespace detail

// =============================================================================
// ProductIntegral: Composition of independent integrals
// =============================================================================

/**
 * @brief Product of two independent integrals.
 *
 * For integrals over independent variables:
 * @f[
 *   \left(\int f(x) \, dx\right) \cdot \left(\int g(y) \, dy\right)
 * @f]
 *
 * The product can be computed by evaluating each integral separately and
 * multiplying the results, which is much faster than nested integration.
 *
 * @tparam I1 First integral type
 * @tparam I2 Second integral type
 *
 * @par Independence Requirement
 * The two integrals must depend on disjoint variable sets. This is verified
 * at compile time using `variable_set` analysis. If the integrals share
 * variables, a `static_assert` fires with a descriptive error message.
 *
 * @par Error Propagation
 * Errors are propagated using the product rule:
 * @f[
 *   \delta(ab) = |b|\delta a + |a|\delta b
 * @f]
 *
 * @par Example
 * @code{.cpp}
 * auto x = arg<0>;
 * auto y = arg<1>;
 *
 * auto I = integral(sin(x)).over<0>(0.0, pi);
 * auto J = integral(exp(y)).over<1>(0.0, 1.0);
 *
 * auto IJ = I * J;
 * auto result = IJ.eval();  // Evaluates I and J separately
 * @endcode
 *
 * @see operator*(I1, I2) Factory operator
 * @see product() Variadic factory function
 */
template<typename I1, typename I2>
struct ProductIntegral {
    using value_type = typename I1::value_type;

    I1 left;
    I2 right;

    // Verify independence at compile time
    // The integrals must depend on disjoint sets of variables
    static_assert(
        (detail::integral_variable_set_v<I1> & detail::integral_variable_set_v<I2>) == 0,
        "ProductIntegral requires integrals over independent variables. "
        "The integrand variable sets must be disjoint."
    );

    constexpr ProductIntegral(I1 i1, I2 i2) noexcept
        : left{i1}, right{i2} {}

    /// Evaluate the product integral (no outer arguments)
    [[nodiscard]] algorithms::integration_result<value_type>
    eval() const {
        return combine_results(left.eval(), right.eval());
    }

    /// Evaluate with arguments (for integrals with free variables)
    template<typename... Args>
    [[nodiscard]] algorithms::integration_result<value_type>
    eval(Args&&... args) const {
        return combine_results(
            left.eval(std::forward<Args>(args)...),
            right.eval(std::forward<Args>(args)...));
    }

private:
    /// Combine two integration results using product error propagation:
    /// delta(ab) = |b| * delta_a + |a| * delta_b
    static algorithms::integration_result<value_type>
    combine_results(algorithms::integration_result<value_type> const& r1,
                    algorithms::integration_result<value_type> const& r2) {
        value_type value = r1.value() * r2.value();
        value_type error = std::abs(r2.value()) * r1.error()
                         + std::abs(r1.value()) * r2.error();
        return {value, error,
                r1.iterations() + r2.iterations(),
                r1.evaluations() + r2.evaluations()};
    }

public:

    /// String representation
    [[nodiscard]] std::string to_string() const {
        return "(product-integral " + left.to_string() + " " + right.to_string() + ")";
    }

    /// Multiply with another integral (chaining)
    template<typename I3>
    [[nodiscard]] constexpr auto operator*(I3 const& other) const {
        return ProductIntegral<ProductIntegral, I3>{*this, other};
    }
};

// =============================================================================
// Type traits for ProductIntegral
// =============================================================================

template<typename T>
struct is_product_integral : std::false_type {};

template<typename I1, typename I2>
struct is_product_integral<ProductIntegral<I1, I2>> : std::true_type {};

template<typename T>
inline constexpr bool is_product_integral_v = is_product_integral<T>::value;

// =============================================================================
// Concepts for integral types
// =============================================================================

/// Concept for types that can be evaluated as integrals
template<typename T>
concept EvaluableIntegral = requires(T t) {
    { t.eval() } -> std::same_as<algorithms::integration_result<typename T::value_type>>;
};

// =============================================================================
// operator* for integrals (creates ProductIntegral)
// =============================================================================

/// Multiply two independent integrals
/// Returns a ProductIntegral that evaluates both separately
template<typename I1, typename I2>
    requires (is_integral_v<I1> && is_integral_v<I2>)
[[nodiscard]] constexpr auto operator*(I1 const& i1, I2 const& i2) {
    return ProductIntegral<I1, I2>{i1, i2};
}

/// Multiply ProductIntegral with another integral
template<typename I1, typename I2, typename I3>
    requires is_integral_v<I3>
[[nodiscard]] constexpr auto operator*(ProductIntegral<I1, I2> const& pi, I3 const& i3) {
    return ProductIntegral<ProductIntegral<I1, I2>, I3>{pi, i3};
}

/// Multiply integral with ProductIntegral
template<typename I1, typename I2, typename I3>
    requires is_integral_v<I3>
[[nodiscard]] constexpr auto operator*(I3 const& i3, ProductIntegral<I1, I2> const& pi) {
    return ProductIntegral<I3, ProductIntegral<I1, I2>>{i3, pi};
}

// =============================================================================
// Separable integral detection and optimization
// =============================================================================

/// Check if multiplying integrals is valid (they must be over independent dimensions)
template<typename I1, typename I2>
inline constexpr bool are_independent_integrals_v =
    (detail::integral_variable_set_v<I1> & detail::integral_variable_set_v<I2>) == 0;

// =============================================================================
// Convenience: product() function
// =============================================================================

/**
 * @brief Create a product of two independent integrals.
 *
 * @tparam I1 First integral type
 * @tparam I2 Second integral type
 * @param i1 First integral
 * @param i2 Second integral
 * @return ProductIntegral<I1, I2>
 *
 * @par Example
 * @code{.cpp}
 * auto IJ = product(I, J);  // Equivalent to I * J
 * @endcode
 */
template<typename I1, typename I2>
    requires (EvaluableIntegral<I1> && EvaluableIntegral<I2>)
[[nodiscard]] constexpr auto product(I1 const& i1, I2 const& i2) {
    return ProductIntegral<I1, I2>{i1, i2};
}

/**
 * @brief Create a product of multiple independent integrals.
 *
 * @tparam I1 First integral type
 * @tparam I2 Second integral type
 * @tparam Is Additional integral types
 * @return Nested ProductIntegral
 *
 * @par Example
 * @code{.cpp}
 * auto IJK = product(I, J, K);  // Equivalent to I * J * K
 * @endcode
 */
template<typename I1, typename I2, typename... Is>
    requires (EvaluableIntegral<I1> && EvaluableIntegral<I2> && (EvaluableIntegral<Is> && ...))
[[nodiscard]] constexpr auto product(I1 const& i1, I2 const& i2, Is const&... is) {
    return product(ProductIntegral<I1, I2>{i1, i2}, is...);
}

/** @} */ // end of expr_ops group

} // namespace limes::expr
