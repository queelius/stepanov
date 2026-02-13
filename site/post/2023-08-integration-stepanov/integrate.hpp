#pragma once

/**
 * @file integrate.hpp
 * @brief Numerical Integration (Quadrature)
 *
 * The definite integral ∫[a,b] f(x) dx represents the signed area under f.
 * Since most functions lack closed-form antiderivatives, we approximate
 * the integral numerically using quadrature rules.
 *
 * Key insight: All quadrature rules are weighted sums:
 *     ∫[a,b] f(x) dx ≈ Σᵢ wᵢ · f(xᵢ)
 *
 * Rules differ in their choice of nodes xᵢ and weights wᵢ.
 *
 * Error analysis:
 *   - Midpoint rule:    O(h²)   - surprisingly good!
 *   - Trapezoidal rule: O(h²)
 *   - Simpson's rule:   O(h⁴)
 *   - Gauss-Legendre:   O(h^{2n}) for n points - optimal for polynomials
 *
 * The Euler-Maclaurin formula reveals why: error depends on derivatives
 * at the boundary. Midpoint doesn't evaluate at endpoints!
 */

#include <cmath>
#include <concepts>
#include <functional>
#include <array>
#include <utility>
#include <stdexcept>
#include <limits>

namespace integration {

// =============================================================================
// Concepts: Stepanov-style minimal operations
// =============================================================================

/**
 * The Stepanov insight: Algorithms arise from MINIMAL OPERATIONS.
 *
 * Instead of constraining to std::floating_point, we ask:
 * "What operations does this algorithm ACTUALLY need?"
 *
 * For numerical integration, we need:
 *   - Field operations (+, -, *, /)
 *   - Ordering (<) for adaptive methods
 *   - Construction from small integers (for weights like 2, 4, 6)
 *
 * This allows rational<T>, dual<T>, and user-defined types to work
 * automatically—no special-casing required.
 */

template<typename T>
concept has_add = requires(T a, T b) {
    { a + b } -> std::convertible_to<T>;
};

template<typename T>
concept has_sub = requires(T a, T b) {
    { a - b } -> std::convertible_to<T>;
};

template<typename T>
concept has_mul = requires(T a, T b) {
    { a * b } -> std::convertible_to<T>;
};

template<typename T>
concept has_div = requires(T a, T b) {
    { a / b } -> std::convertible_to<T>;
};

template<typename T>
concept has_neg = requires(T a) {
    { -a } -> std::convertible_to<T>;
};

template<typename T>
concept has_order = requires(T a, T b) {
    { a < b } -> std::convertible_to<bool>;
};

/**
 * ordered_field: The algebraic structure needed for numerical integration.
 *
 * A field with ordering and small integer construction. This is the MINIMAL
 * requirement—any type satisfying this works with midpoint_rule, trapezoidal_rule,
 * simpsons_rule, and the composite versions.
 *
 * Examples of types satisfying ordered_field:
 *   - double, float (built-in)
 *   - rational<int> (exact fractions)
 *   - dual<double> (automatic differentiation)
 *   - interval<double> (interval arithmetic)
 *   - User-defined real number types
 */
template<typename T>
concept ordered_field =
    has_add<T> && has_sub<T> && has_mul<T> && has_div<T> && has_neg<T> && has_order<T> &&
    requires {
        T(0);  // Additive identity
        T(1);  // Multiplicative identity
        T(2);  // Small integers for weights
    };

// =============================================================================
// ADL-discoverable abs()
// =============================================================================

/**
 * Absolute value adaptor for ordered fields.
 *
 * For adaptive methods, we need abs() to measure error. This default
 * implementation works for any ordered type. Custom types can provide
 * their own abs() via ADL (like rational<T> already does).
 */
template<typename T>
    requires ordered_field<T> && (!std::floating_point<T>)
constexpr T abs(T x) {
    return x < T(0) ? -x : x;
}

// =============================================================================
// Integrand concept
// =============================================================================

template<typename F, typename T>
concept Integrand = requires(F f, T x) {
    { f(x) } -> std::convertible_to<T>;
};

// =============================================================================
// Basic Rules (Single Interval)
// =============================================================================

/**
 * Midpoint rule: ∫[a,b] f(x) dx ≈ (b-a) · f((a+b)/2)
 *
 * Uses the value at the center. Error: O(h³) per interval.
 * Better than trapezoidal for smooth functions!
 *
 * Works with any ordered_field: double, rational<int>, dual<double>, etc.
 */
template<typename T, Integrand<T> F>
    requires ordered_field<T>
T midpoint_rule(F&& f, T a, T b) {
    return (b - a) * f((a + b) / T(2));
}

/**
 * Trapezoidal rule: ∫[a,b] f(x) dx ≈ (b-a)/2 · (f(a) + f(b))
 *
 * Linear interpolation between endpoints. Error: O(h³) per interval.
 *
 * Works with any ordered_field: double, rational<int>, dual<double>, etc.
 */
template<typename T, Integrand<T> F>
    requires ordered_field<T>
T trapezoidal_rule(F&& f, T a, T b) {
    return (b - a) / T(2) * (f(a) + f(b));
}

/**
 * Simpson's rule: ∫[a,b] f(x) dx ≈ (b-a)/6 · (f(a) + 4f(m) + f(b))
 *
 * Quadratic interpolation through three points. Error: O(h⁵) per interval.
 * Exact for polynomials up to degree 3 (bonus degree!).
 *
 * Works with any ordered_field: double, rational<int>, dual<double>, etc.
 */
template<typename T, Integrand<T> F>
    requires ordered_field<T>
T simpsons_rule(F&& f, T a, T b) {
    T m = (a + b) / T(2);
    return (b - a) / T(6) * (f(a) + T(4)*f(m) + f(b));
}

// =============================================================================
// Composite Rules (Multiple Intervals)
// =============================================================================

/**
 * Composite midpoint: Divide [a,b] into n subintervals.
 * Error: O(h²) where h = (b-a)/n
 *
 * Works with any ordered_field: double, rational<int>, dual<double>, etc.
 */
template<typename T, Integrand<T> F>
    requires ordered_field<T>
T composite_midpoint(F&& f, T a, T b, int n) {
    T h = (b - a) / T(n);
    T sum = T(0);
    for (int i = 0; i < n; ++i) {
        T x_mid = a + (T(i) + T(1) / T(2)) * h;
        sum += f(x_mid);
    }
    return h * sum;
}

/**
 * Composite trapezoidal: The classic numerical integration method.
 * Error: O(h²) where h = (b-a)/n
 *
 * Special property: For periodic functions on a full period,
 * convergence is exponentially fast (spectral accuracy)!
 *
 * Works with any ordered_field: double, rational<int>, dual<double>, etc.
 */
template<typename T, Integrand<T> F>
    requires ordered_field<T>
T composite_trapezoidal(F&& f, T a, T b, int n) {
    T h = (b - a) / T(n);
    T sum = (f(a) + f(b)) / T(2);
    for (int i = 1; i < n; ++i) {
        sum += f(a + T(i) * h);
    }
    return h * sum;
}

/**
 * Composite Simpson's: Requires even number of intervals.
 * Error: O(h⁴) where h = (b-a)/n
 *
 * Works with any ordered_field: double, rational<int>, dual<double>, etc.
 */
template<typename T, Integrand<T> F>
    requires ordered_field<T>
T composite_simpsons(F&& f, T a, T b, int n) {
    if (n % 2 != 0) {
        throw std::invalid_argument("Simpson's rule requires even n");
    }
    T h = (b - a) / T(n);
    T sum = f(a) + f(b);

    for (int i = 1; i < n; i += 2) {
        sum += T(4) * f(a + T(i) * h);
    }
    for (int i = 2; i < n; i += 2) {
        sum += T(2) * f(a + T(i) * h);
    }
    return h / T(3) * sum;
}

// =============================================================================
// Gauss-Legendre Quadrature
// =============================================================================

/**
 * Gauss-Legendre nodes and weights for n-point quadrature on [-1, 1].
 *
 * These are the optimal choice: n points exactly integrate polynomials
 * of degree up to 2n-1. No other quadrature rule achieves this.
 *
 * The nodes are roots of Legendre polynomials.
 *
 * NOTE: This is constrained to std::floating_point because the pre-computed
 * nodes and weights are irrational numbers (roots of Legendre polynomials).
 * For rational arithmetic, you would need to provide your own nodes/weights
 * as rational approximations or use a different quadrature rule.
 */
template<std::floating_point T, std::size_t N>
struct gauss_legendre_table;

template<std::floating_point T>
struct gauss_legendre_table<T, 2> {
    static constexpr std::array<T, 2> nodes = {
        T(-0.5773502691896257),  // -1/√3
        T(0.5773502691896257)    //  1/√3
    };
    static constexpr std::array<T, 2> weights = {T(1), T(1)};
};

template<std::floating_point T>
struct gauss_legendre_table<T, 3> {
    static constexpr std::array<T, 3> nodes = {
        T(-0.7745966692414834),  // -√(3/5)
        T(0),
        T(0.7745966692414834)    //  √(3/5)
    };
    static constexpr std::array<T, 3> weights = {
        T(0.5555555555555556),   // 5/9
        T(0.8888888888888888),   // 8/9
        T(0.5555555555555556)    // 5/9
    };
};

template<std::floating_point T>
struct gauss_legendre_table<T, 4> {
    static constexpr std::array<T, 4> nodes = {
        T(-0.8611363115940526),
        T(-0.3399810435848563),
        T(0.3399810435848563),
        T(0.8611363115940526)
    };
    static constexpr std::array<T, 4> weights = {
        T(0.3478548451374538),
        T(0.6521451548625461),
        T(0.6521451548625461),
        T(0.3478548451374538)
    };
};

template<std::floating_point T>
struct gauss_legendre_table<T, 5> {
    static constexpr std::array<T, 5> nodes = {
        T(-0.9061798459386640),
        T(-0.5384693101056831),
        T(0),
        T(0.5384693101056831),
        T(0.9061798459386640)
    };
    static constexpr std::array<T, 5> weights = {
        T(0.2369268850561891),
        T(0.4786286704993665),
        T(0.5688888888888889),
        T(0.4786286704993665),
        T(0.2369268850561891)
    };
};

/**
 * Gauss-Legendre quadrature for arbitrary interval [a, b].
 *
 * Transform from [-1, 1] to [a, b]: x = (b-a)/2 · t + (a+b)/2
 * Jacobian: dx = (b-a)/2 · dt
 */
template<std::size_t N, std::floating_point T, Integrand<T> F>
T gauss_legendre(F&& f, T a, T b) {
    const auto& table = gauss_legendre_table<T, N>{};
    T scale = (b - a) / T(2);
    T shift = (a + b) / T(2);

    T sum = T(0);
    for (std::size_t i = 0; i < N; ++i) {
        T x = scale * table.nodes[i] + shift;
        sum += table.weights[i] * f(x);
    }
    return scale * sum;
}

// =============================================================================
// Adaptive Integration
// =============================================================================

/**
 * Adaptive Simpson's integration.
 *
 * Recursively subdivides intervals where the error is large.
 * Uses Simpson's rule and estimates error by comparing with the
 * two-panel composite Simpson result.
 *
 * Works with any ordered_field that provides abs() via ADL.
 *
 * @param f         Function to integrate
 * @param a         Left endpoint
 * @param b         Right endpoint
 * @param tolerance Desired absolute error
 * @param max_depth Maximum recursion depth
 */
template<typename T, Integrand<T> F>
    requires ordered_field<T>
T adaptive_simpsons(F&& f, T a, T b, T tolerance, int max_depth = 50) {
    // ADL-discoverable abs: uses integration::abs for non-floating-point ordered fields,
    // std::abs for floating-point, or custom abs() via ADL for user types
    using std::abs;

    auto helper = [&](auto& self, T a, T b, T fa, T fb, T fm, T whole, T tol, int depth) -> T {
        T m = (a + b) / T(2);
        T lm = (a + m) / T(2);  // Left midpoint
        T rm = (m + b) / T(2);  // Right midpoint

        T flm = f(lm);
        T frm = f(rm);

        T left = (m - a) / T(6) * (fa + T(4)*flm + fm);
        T right = (b - m) / T(6) * (fm + T(4)*frm + fb);
        T refined = left + right;

        // Richardson extrapolation estimate of error
        T error = (refined - whole) / T(15);

        if (depth <= 0 || abs(error) <= tol) {
            return refined + error;  // Include error estimate for correction
        }

        // Recurse on both halves
        return self(self, a, m, fa, fm, flm, left, tol/T(2), depth - 1)
             + self(self, m, b, fm, fb, frm, right, tol/T(2), depth - 1);
    };

    T fa = f(a), fb = f(b), fm = f((a + b) / T(2));
    T whole = (b - a) / T(6) * (fa + T(4)*fm + fb);

    return helper(helper, a, b, fa, fb, fm, whole, tolerance, max_depth);
}

// Overload with default tolerance for floating-point types
template<std::floating_point T, Integrand<T> F>
T adaptive_simpsons(F&& f, T a, T b) {
    return adaptive_simpsons(std::forward<F>(f), a, b, T(1e-10), 50);
}

// =============================================================================
// Convenience Interface
// =============================================================================

/**
 * Integrate f from a to b using adaptive quadrature.
 * This is the recommended general-purpose function.
 *
 * Works with any ordered_field: double, dual<double>, etc.
 */
template<typename T, Integrand<T> F>
    requires ordered_field<T>
T integrate(F&& f, T a, T b, T tolerance) {
    return adaptive_simpsons(std::forward<F>(f), a, b, tolerance);
}

// Overload with default tolerance for floating-point types
template<std::floating_point T, Integrand<T> F>
T integrate(F&& f, T a, T b) {
    return integrate(std::forward<F>(f), a, b, T(1e-10));
}

/**
 * Result type for integration with error estimate.
 */
template<typename T>
    requires ordered_field<T>
struct integration_result {
    T value;
    T error_estimate;
    int evaluations;
};

/**
 * Integrate with error estimate using Gauss-Kronrod.
 * Compares n-point and (2n+1)-point rules.
 *
 * NOTE: Constrained to std::floating_point because it uses gauss_legendre
 * which requires pre-computed irrational nodes.
 */
template<std::floating_point T, Integrand<T> F>
integration_result<T> integrate_with_error(F&& f, T a, T b) {
    using std::abs;
    // Use 5-point Gauss and compare with adaptive Simpson
    T g5 = gauss_legendre<5>(f, a, b);
    T adapt = adaptive_simpsons(f, a, b, T(1e-12));

    return {adapt, abs(adapt - g5), 0};  // evaluation count not tracked
}

// =============================================================================
// Special Integrands
// =============================================================================

// NOTE: These special integrand handlers are constrained to std::floating_point
// because they use std::sqrt and std::numeric_limits. To support custom numeric
// types, you would need to provide sqrt() and epsilon via ADL.

/**
 * Integrate a function with a singularity at an endpoint.
 *
 * Uses substitution x = a + t² for left singularity,
 * or x = b - t² for right singularity.
 *
 * The transformed integrand g(t) = 2t·f(a+t²) vanishes at t=0,
 * removing the singularity when f(x) ~ 1/√(x-a).
 */
template<std::floating_point T, Integrand<T> F>
T integrate_left_singularity(F&& f, T a, T b, T tolerance = T(1e-10)) {
    // Substitution: x = a + t², dx = 2t dt
    // ∫[a,b] f(x) dx = ∫[0,√(b-a)] f(a + t²) · 2t dt
    T sqrt_len = std::sqrt(b - a);
    // Start from small positive t to avoid 0 * ∞ = nan at t=0
    // The contribution from [0, eps] is O(eps²) for 1/√x singularities
    T eps = std::sqrt(std::numeric_limits<T>::epsilon());
    auto g = [&](T t) { return T(2) * t * f(a + t*t); };
    return integrate(g, eps, sqrt_len, tolerance);
}

template<std::floating_point T, Integrand<T> F>
T integrate_right_singularity(F&& f, T a, T b, T tolerance = T(1e-10)) {
    // Substitution: x = b - t², dx = -2t dt
    T sqrt_len = std::sqrt(b - a);
    T eps = std::sqrt(std::numeric_limits<T>::epsilon());
    auto g = [&](T t) { return T(2) * t * f(b - t*t); };
    return integrate(g, eps, sqrt_len, tolerance);
}

/**
 * Integrate on [0, ∞) using substitution x = t/(1-t).
 */
template<std::floating_point T, Integrand<T> F>
T integrate_semi_infinite(F&& f, T tolerance = T(1e-8)) {
    // x = t/(1-t), dx = 1/(1-t)² dt
    // ∫[0,∞) f(x) dx = ∫[0,1) f(t/(1-t)) / (1-t)² dt
    auto g = [&](T t) {
        if (t >= T(1) - T(1e-14)) return T(0);  // Avoid division by zero
        T x = t / (T(1) - t);
        T jacobian = T(1) / ((T(1) - t) * (T(1) - t));
        return f(x) * jacobian;
    };
    return integrate(g, T(0), T(1) - T(1e-10), tolerance);
}

} // namespace integration
