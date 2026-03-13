#pragma once

/**
 * @file box_integral.hpp
 * @brief N-dimensional box integration with Monte Carlo methods.
 *
 * This module provides `BoxIntegral` for integrating expressions over
 * rectangular N-dimensional regions, and `ConstrainedBoxIntegral` for
 * irregular regions via rejection sampling.
 *
 * @section box_usage Usage
 *
 * @code{.cpp}
 * using namespace limes::expr;
 * using namespace limes::methods;
 *
 * auto x = arg<0>;
 * auto y = arg<1>;
 * auto z = arg<2>;
 *
 * // 3D integral over unit cube
 * auto I = integral(x*y*z).over_box({{0,1}, {0,1}, {0,1}});
 * auto result = I.eval(monte_carlo_method(100000));
 *
 * // Integral with constraint (triangle region y < x)
 * auto J = integral(x*y)
 *     .over_box({{0,1}, {0,1}})
 *     .where([](auto x, auto y) { return y < x; });
 * auto r2 = J.eval(100000);  // sample count
 * @endcode
 *
 * @see BoxIntegral For rectangular regions
 * @see ConstrainedBoxIntegral For irregular regions
 * @see monte_carlo Monte Carlo method object
 * @ingroup expr_ops
 */

#include <span>
#include <string>
#include <array>
#include <cstddef>
#include <sstream>
#include <utility>
#include <random>
#include "../algorithms/core/result.hpp"
#include "../methods/methods.hpp"
#include "nodes/binary.hpp"

/**
 * @addtogroup expr_ops
 * @{
 */

namespace limes::expr {

// Forward declarations
template<typename E, std::size_t Dims, typename Constraint>
struct ConstrainedBoxIntegral;

namespace detail {

/// Initialize RNG and per-dimension uniform distributions for Monte Carlo sampling.
/// Returns the RNG engine, distribution array, and box volume.
template<typename T, std::size_t Dims>
struct mc_box_sampler {
    std::mt19937_64 rng;
    std::array<std::uniform_real_distribution<T>, Dims> dists;
    T volume;

    explicit mc_box_sampler(
        std::array<std::pair<T, T>, Dims> const& bounds,
        std::optional<std::size_t> seed)
        : volume{T(1)}
    {
        if (seed) {
            rng.seed(*seed);
        } else {
            std::random_device rd;
            rng.seed(rd());
        }
        for (std::size_t d = 0; d < Dims; ++d) {
            dists[d] = std::uniform_real_distribution<T>(bounds[d].first, bounds[d].second);
            volume *= (bounds[d].second - bounds[d].first);
        }
    }

    void sample(std::array<T, Dims>& point) {
        for (std::size_t d = 0; d < Dims; ++d) {
            point[d] = dists[d](rng);
        }
    }
};

} // namespace detail

// =============================================================================
// BoxIntegral: Integration over rectangular N-dimensional regions
// =============================================================================

/**
 * @brief N-dimensional integral over a rectangular box.
 *
 * Represents:
 * @f[
 *   \int_{a_0}^{b_0} \cdots \int_{a_{n-1}}^{b_{n-1}} f(x_0, \ldots, x_{n-1}) \, dx_0 \cdots dx_{n-1}
 * @f]
 *
 * Box integrals are evaluated using Monte Carlo integration, which scales
 * well to high dimensions (error decreases as O(1/√n) regardless of dimension).
 *
 * @tparam E Integrand expression type
 * @tparam Dims Number of dimensions
 *
 * @par Example
 * @code{.cpp}
 * auto x = arg<0>;
 * auto y = arg<1>;
 * auto z = arg<2>;
 *
 * // Volume of unit cube (should be 1.0)
 * auto I = integral(1.0).over_box({{0,1}, {0,1}, {0,1}});
 *
 * // Triple integral
 * auto J = integral(x*y*z).over_box({{0,1}, {0,1}, {0,1}});
 * auto result = J.eval(monte_carlo_method(1000000));  // ≈ 0.125
 * @endcode
 *
 * @see integral().over_box() Builder method
 * @see ConstrainedBoxIntegral For irregular regions
 */
template<typename E, std::size_t Dims>
struct BoxIntegral {
    using value_type = typename E::value_type;
    using integrand_type = E;
    using bounds_type = std::array<std::pair<value_type, value_type>, Dims>;

    static constexpr std::size_t dims_v = Dims;
    static constexpr std::size_t arity_v = (E::arity_v > Dims) ? (E::arity_v - Dims) : 0;

    E integrand;
    bounds_type bounds;

    constexpr BoxIntegral(E e, bounds_type b) noexcept
        : integrand{e}, bounds{b} {}

    // =========================================================================
    // Evaluation using Monte Carlo (default for N-D integration)
    // =========================================================================

    /// Evaluate using Monte Carlo integration
    [[nodiscard]] algorithms::integration_result<value_type>
    eval(methods::monte_carlo<value_type> const& method) const {
        detail::mc_box_sampler<value_type, Dims> sampler(bounds, method.seed);

        value_type sum = value_type(0);
        value_type sum_sq = value_type(0);
        std::array<value_type, Dims> point;

        for (std::size_t i = 0; i < method.samples; ++i) {
            sampler.sample(point);
            value_type y = integrand.eval(std::span<value_type const>{point.data(), Dims});
            sum += y;
            sum_sq += y * y;
        }

        value_type n = static_cast<value_type>(method.samples);
        value_type mean = sum / n;
        value_type variance = (sum_sq / n - mean * mean) / n;
        value_type value = mean * sampler.volume;
        value_type error = std::sqrt(variance) * sampler.volume;

        algorithms::integration_result<value_type> result{value, error, method.samples, method.samples};
        result.variance_ = variance;
        return result;
    }

    /// Evaluate with default Monte Carlo (10000 samples)
    [[nodiscard]] algorithms::integration_result<value_type>
    eval() const {
        return eval(methods::monte_carlo<value_type>{10000});
    }

    /// Evaluate with sample count
    [[nodiscard]] algorithms::integration_result<value_type>
    eval(std::size_t samples) const {
        return eval(methods::monte_carlo<value_type>{samples});
    }

    // Deprecated: use eval() instead
    [[nodiscard]] [[deprecated("use eval() instead")]]
    algorithms::integration_result<value_type> evaluate() const {
        return eval();
    }

    /// String representation
    [[nodiscard]] std::string to_string() const {
        std::ostringstream oss;
        oss << "(box-integral " << integrand.to_string() << " [";
        for (std::size_t d = 0; d < Dims; ++d) {
            if (d > 0) oss << ", ";
            oss << "[" << bounds[d].first << ", " << bounds[d].second << "]";
        }
        oss << "])";
        return oss.str();
    }

    /// Add a constraint to the box integral (for rejection sampling)
    template<typename Constraint>
    [[nodiscard]] constexpr auto where(Constraint c) const {
        return ConstrainedBoxIntegral<E, Dims, Constraint>{integrand, bounds, c};
    }
};

// =============================================================================
// ConstrainedBoxIntegral: Box integral with a constraint function
// =============================================================================

/**
 * @brief Box integral with constraint for irregular regions.
 *
 * Uses rejection sampling to integrate over regions defined by a constraint
 * function. Points are sampled from the bounding box and rejected if they
 * don't satisfy the constraint.
 *
 * @tparam E Integrand expression type
 * @tparam Dims Number of dimensions
 * @tparam Constraint Constraint predicate type
 *
 * @par Example: Triangle region
 * @code{.cpp}
 * // Integrate over triangle where y < x
 * auto I = integral(x*y)
 *     .over_box({{0,1}, {0,1}})
 *     .where([](auto x, auto y) { return y < x; });
 *
 * auto result = I.eval(100000);  // 100k samples
 * @endcode
 *
 * @par Example: Circle region
 * @code{.cpp}
 * // Integrate over unit disk
 * auto J = integral(f)
 *     .over_box({{-1,1}, {-1,1}})
 *     .where([](auto x, auto y) { return x*x + y*y < 1; });
 * @endcode
 *
 * @note The acceptance rate affects variance. For regions that are a small
 * fraction of the bounding box, more samples are needed for accuracy.
 *
 * @see BoxIntegral::where() Factory method
 */
template<typename E, std::size_t Dims, typename Constraint>
struct ConstrainedBoxIntegral {
    using value_type = typename E::value_type;
    using integrand_type = E;
    using bounds_type = std::array<std::pair<value_type, value_type>, Dims>;

    static constexpr std::size_t dims_v = Dims;
    static constexpr std::size_t arity_v = (E::arity_v > Dims) ? (E::arity_v - Dims) : 0;

    E integrand;
    bounds_type bounds;
    Constraint constraint;

    constexpr ConstrainedBoxIntegral(E e, bounds_type b, Constraint c) noexcept
        : integrand{e}, bounds{b}, constraint{c} {}

    /// Evaluate using Monte Carlo with rejection sampling
    [[nodiscard]] algorithms::integration_result<value_type>
    eval(methods::monte_carlo<value_type> const& method) const {
        detail::mc_box_sampler<value_type, Dims> sampler(bounds, method.seed);

        value_type sum = value_type(0);
        value_type sum_sq = value_type(0);
        std::size_t accepted = 0;
        std::array<value_type, Dims> point;

        for (std::size_t i = 0; i < method.samples; ++i) {
            sampler.sample(point);

            if (evaluate_constraint(point)) {
                value_type y = integrand.eval(std::span<value_type const>{point.data(), Dims});
                sum += y;
                sum_sq += y * y;
                ++accepted;
            }
        }

        if (accepted == 0) {
            return algorithms::integration_result<value_type>{
                value_type(0), value_type(0), method.samples, method.samples};
        }

        value_type acceptance_rate = static_cast<value_type>(accepted)
                                   / static_cast<value_type>(method.samples);
        value_type region_volume = sampler.volume * acceptance_rate;

        value_type n = static_cast<value_type>(accepted);
        value_type mean = sum / n;
        value_type variance = (sum_sq / n - mean * mean) / n;
        value_type value = mean * region_volume;
        value_type error = std::sqrt(variance) * region_volume;

        algorithms::integration_result<value_type> result{value, error, method.samples, method.samples};
        result.variance_ = variance;
        return result;
    }

    /// Evaluate with default Monte Carlo
    [[nodiscard]] algorithms::integration_result<value_type>
    eval() const {
        return eval(methods::monte_carlo<value_type>{10000});
    }

    /// Evaluate with sample count
    [[nodiscard]] algorithms::integration_result<value_type>
    eval(std::size_t samples) const {
        return eval(methods::monte_carlo<value_type>{samples});
    }

    /// String representation
    [[nodiscard]] std::string to_string() const {
        std::ostringstream oss;
        oss << "(constrained-box-integral " << integrand.to_string() << " [";
        for (std::size_t d = 0; d < Dims; ++d) {
            if (d > 0) oss << ", ";
            oss << "[" << bounds[d].first << ", " << bounds[d].second << "]";
        }
        oss << "] <constraint>)";
        return oss.str();
    }

private:
    // Helper to call constraint with array unpacked to arguments
    template<std::size_t... Is>
    [[nodiscard]] bool evaluate_constraint_impl(
        std::array<value_type, Dims> const& point,
        std::index_sequence<Is...>) const {
        return constraint(point[Is]...);
    }

    [[nodiscard]] bool evaluate_constraint(std::array<value_type, Dims> const& point) const {
        return evaluate_constraint_impl(point, std::make_index_sequence<Dims>{});
    }
};

// =============================================================================
// IntegralBuilder extension for over_box
// =============================================================================

// Note: This is added to the IntegralBuilder via a free function pattern
// to avoid modifying the existing IntegralBuilder class

/// Create a box integral from an expression
template<typename E, std::size_t Dims>
[[nodiscard]] constexpr auto over_box(
    E expr,
    std::array<std::pair<typename E::value_type, typename E::value_type>, Dims> bounds
) {
    return BoxIntegral<E, Dims>{expr, bounds};
}


// =============================================================================
// Convenience functions
// =============================================================================

/// Create a 2D box integral (specialization for common case)
template<typename E>
[[nodiscard]] auto box2d(E expr, typename E::value_type x0, typename E::value_type x1,
                          typename E::value_type y0, typename E::value_type y1) {
    return BoxIntegral<E, 2>{expr, {{{x0, x1}, {y0, y1}}}};
}

/// Create a 3D box integral (specialization for common case)
template<typename E>
[[nodiscard]] auto box3d(E expr,
                          typename E::value_type x0, typename E::value_type x1,
                          typename E::value_type y0, typename E::value_type y1,
                          typename E::value_type z0, typename E::value_type z1) {
    return BoxIntegral<E, 3>{expr, {{{x0, x1}, {y0, y1}, {z0, z1}}}};
}

// =============================================================================
// Type traits
// =============================================================================

template<typename T>
struct is_box_integral : std::false_type {};

template<typename E, std::size_t D>
struct is_box_integral<BoxIntegral<E, D>> : std::true_type {};

template<typename T>
inline constexpr bool is_box_integral_v = is_box_integral<T>::value;

template<typename T>
struct is_constrained_box_integral : std::false_type {};

template<typename E, std::size_t D, typename C>
struct is_constrained_box_integral<ConstrainedBoxIntegral<E, D, C>> : std::true_type {};

template<typename T>
inline constexpr bool is_constrained_box_integral_v = is_constrained_box_integral<T>::value;

/** @} */ // end of expr_ops group

} // namespace limes::expr
