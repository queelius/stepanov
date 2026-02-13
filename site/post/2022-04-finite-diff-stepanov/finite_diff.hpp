#pragma once

/**
 * @file finite_diff.hpp
 * @brief Numerical Differentiation via Finite Differences
 *
 * The derivative f'(x) is defined as:
 *     f'(x) = lim_{h→0} (f(x+h) - f(x)) / h
 *
 * We cannot take h = 0 on a computer, but we can choose small h.
 * Different stencils (patterns of evaluation points) yield different
 * accuracy vs. computational cost tradeoffs.
 *
 * Key insight: The error has two components:
 *   1. Truncation error: From Taylor series approximation (decreases with h)
 *   2. Round-off error: From floating-point arithmetic (increases as h→0)
 *
 * Optimal h balances these. For double precision:
 *   - Forward/backward: h ≈ √ε ≈ 1e-8
 *   - Central:          h ≈ ε^(1/3) ≈ 1e-5
 *   - Five-point:       h ≈ ε^(1/5) ≈ 1e-3
 *
 * where ε ≈ 2.2e-16 is machine epsilon.
 */

#include <cmath>
#include <concepts>
#include <functional>
#include <vector>
#include <array>
#include <limits>
#include <stdexcept>

namespace finite_diff {

// =============================================================================
// Concepts
// =============================================================================

template<typename F, typename T>
concept ScalarFunction = requires(F f, T x) {
    { f(x) } -> std::convertible_to<T>;
};

template<typename F, typename T>
concept VectorFunction = requires(F f, std::vector<T> x) {
    { f(x) } -> std::convertible_to<T>;
};

// =============================================================================
// Optimal Step Sizes
// =============================================================================

/**
 * Optimal step size for different finite difference methods.
 * Based on minimizing total error = truncation + round-off.
 */
template<std::floating_point T>
struct optimal_h {
    // Machine epsilon for type T
    static constexpr T eps = std::numeric_limits<T>::epsilon();

    // Forward/backward difference: error = O(h) + O(ε/h)
    // Optimal: h = √(ε) ≈ 1.5e-8 for double
    static constexpr T forward = std::sqrt(eps);

    // Central difference: error = O(h²) + O(ε/h)
    // Optimal: h = ε^(1/3) ≈ 6e-6 for double
    static constexpr T central = std::cbrt(eps);

    // Five-point stencil: error = O(h⁴) + O(ε/h)
    // Optimal: h = ε^(1/5) ≈ 7e-4 for double
    static constexpr T five_point = std::pow(eps, T(0.2));
};

// =============================================================================
// First Derivatives (Univariate)
// =============================================================================

/**
 * Forward difference: f'(x) ≈ (f(x+h) - f(x)) / h
 * Error: O(h) - first order accurate
 * Uses: One-sided derivatives, boundary conditions
 */
template<std::floating_point T, ScalarFunction<T> F>
T forward_difference(F&& f, T x, T h = optimal_h<T>::forward) {
    return (f(x + h) - f(x)) / h;
}

/**
 * Backward difference: f'(x) ≈ (f(x) - f(x-h)) / h
 * Error: O(h) - first order accurate
 */
template<std::floating_point T, ScalarFunction<T> F>
T backward_difference(F&& f, T x, T h = optimal_h<T>::forward) {
    return (f(x) - f(x - h)) / h;
}

/**
 * Central difference: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
 * Error: O(h²) - second order accurate
 * The workhorse of numerical differentiation.
 */
template<std::floating_point T, ScalarFunction<T> F>
T central_difference(F&& f, T x, T h = optimal_h<T>::central) {
    return (f(x + h) - f(x - h)) / (T(2) * h);
}

/**
 * Five-point stencil: f'(x) ≈ (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
 * Error: O(h⁴) - fourth order accurate
 * Higher accuracy when function evaluations are cheap.
 */
template<std::floating_point T, ScalarFunction<T> F>
T five_point_stencil(F&& f, T x, T h = optimal_h<T>::five_point) {
    return (-f(x + T(2)*h) + T(8)*f(x + h) - T(8)*f(x - h) + f(x - T(2)*h)) / (T(12) * h);
}

// =============================================================================
// Second Derivatives
// =============================================================================

/**
 * Second derivative: f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
 * Error: O(h²)
 */
template<std::floating_point T, ScalarFunction<T> F>
T second_derivative(F&& f, T x, T h = optimal_h<T>::central) {
    return (f(x + h) - T(2)*f(x) + f(x - h)) / (h * h);
}

/**
 * Five-point second derivative:
 * f''(x) ≈ (-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)) / (12h²)
 * Error: O(h⁴)
 */
template<std::floating_point T, ScalarFunction<T> F>
T second_derivative_five_point(F&& f, T x, T h = optimal_h<T>::five_point) {
    T h2 = h * h;
    return (-f(x + T(2)*h) + T(16)*f(x + h) - T(30)*f(x) + T(16)*f(x - h) - f(x - T(2)*h)) / (T(12) * h2);
}

// =============================================================================
// Richardson Extrapolation
// =============================================================================

/**
 * Richardson extrapolation improves accuracy by combining estimates at
 * different step sizes. If D(h) = f'(x) + C*h^p + O(h^{p+1}), then:
 *
 *   D*(h) = (2^p * D(h/2) - D(h)) / (2^p - 1)
 *
 * eliminates the leading error term, giving O(h^{p+1}) accuracy.
 *
 * @param f    Function to differentiate
 * @param x    Point of differentiation
 * @param h    Initial step size
 * @param p    Order of the base method (1 for forward, 2 for central)
 */
template<std::floating_point T, ScalarFunction<T> F>
T richardson_extrapolation(F&& f, T x, T h, int p = 2) {
    auto deriv = [&](T step) { return central_difference(f, x, step); };
    T d_h = deriv(h);
    T d_h2 = deriv(h / T(2));
    T factor = std::pow(T(2), T(p));
    return (factor * d_h2 - d_h) / (factor - T(1));
}

// =============================================================================
// Multivariate: Gradient
// =============================================================================

/**
 * Gradient: ∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
 * Each partial derivative computed via central difference.
 */
template<std::floating_point T, VectorFunction<T> F>
std::vector<T> gradient(F&& f, const std::vector<T>& x, T h = optimal_h<T>::central) {
    std::vector<T> grad(x.size());
    std::vector<T> x_plus = x, x_minus = x;

    for (std::size_t i = 0; i < x.size(); ++i) {
        x_plus[i] = x[i] + h;
        x_minus[i] = x[i] - h;
        grad[i] = (f(x_plus) - f(x_minus)) / (T(2) * h);
        x_plus[i] = x[i];  // Restore
        x_minus[i] = x[i];
    }
    return grad;
}

/**
 * Directional derivative: D_v f(x) = ∇f(x) · v
 * Rate of change of f at x in direction v.
 */
template<std::floating_point T, VectorFunction<T> F>
T directional_derivative(F&& f, const std::vector<T>& x, const std::vector<T>& v,
                         T h = optimal_h<T>::central) {
    if (x.size() != v.size()) throw std::invalid_argument("dimension mismatch");

    std::vector<T> x_plus(x.size()), x_minus(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        x_plus[i] = x[i] + h * v[i];
        x_minus[i] = x[i] - h * v[i];
    }
    return (f(x_plus) - f(x_minus)) / (T(2) * h);
}

// =============================================================================
// Multivariate: Hessian
// =============================================================================

/**
 * Hessian matrix: H[i][j] = ∂²f/∂xᵢ∂xⱼ
 * Symmetric matrix of second partial derivatives.
 */
template<std::floating_point T, VectorFunction<T> F>
std::vector<std::vector<T>> hessian(F&& f, const std::vector<T>& x,
                                     T h = optimal_h<T>::central) {
    std::size_t n = x.size();
    std::vector<std::vector<T>> H(n, std::vector<T>(n));
    std::vector<T> x_pp = x, x_pm = x, x_mp = x, x_mm = x;

    for (std::size_t i = 0; i < n; ++i) {
        // Diagonal: ∂²f/∂xᵢ²
        x_pp[i] = x[i] + h;
        x_mm[i] = x[i] - h;
        H[i][i] = (f(x_pp) - T(2)*f(x) + f(x_mm)) / (h * h);
        x_pp[i] = x[i];
        x_mm[i] = x[i];

        // Off-diagonal: ∂²f/∂xᵢ∂xⱼ (use mixed partial formula)
        for (std::size_t j = i + 1; j < n; ++j) {
            x_pp[i] = x[i] + h; x_pp[j] = x[j] + h;
            x_pm[i] = x[i] + h; x_pm[j] = x[j] - h;
            x_mp[i] = x[i] - h; x_mp[j] = x[j] + h;
            x_mm[i] = x[i] - h; x_mm[j] = x[j] - h;

            H[i][j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (T(4) * h * h);
            H[j][i] = H[i][j];  // Symmetry

            x_pp[i] = x[i]; x_pp[j] = x[j];
            x_pm[i] = x[i]; x_pm[j] = x[j];
            x_mp[i] = x[i]; x_mp[j] = x[j];
            x_mm[i] = x[i]; x_mm[j] = x[j];
        }
    }
    return H;
}

// =============================================================================
// Jacobian
// =============================================================================

/**
 * Jacobian matrix for f: Rⁿ → Rᵐ
 * J[i][j] = ∂fᵢ/∂xⱼ
 */
template<std::floating_point T>
std::vector<std::vector<T>> jacobian(
    std::function<std::vector<T>(const std::vector<T>&)> f,
    const std::vector<T>& x,
    T h = optimal_h<T>::central)
{
    std::size_t n = x.size();
    std::vector<T> f_plus, f_minus;
    std::vector<T> x_plus = x, x_minus = x;

    // First, determine output dimension
    std::vector<T> f0 = f(x);
    std::size_t m = f0.size();

    std::vector<std::vector<T>> J(m, std::vector<T>(n));

    for (std::size_t j = 0; j < n; ++j) {
        x_plus[j] = x[j] + h;
        x_minus[j] = x[j] - h;
        f_plus = f(x_plus);
        f_minus = f(x_minus);

        for (std::size_t i = 0; i < m; ++i) {
            J[i][j] = (f_plus[i] - f_minus[i]) / (T(2) * h);
        }

        x_plus[j] = x[j];
        x_minus[j] = x[j];
    }
    return J;
}

// =============================================================================
// Error Estimation
// =============================================================================

/**
 * Estimate the error in a derivative approximation by comparing
 * results at different step sizes.
 */
template<std::floating_point T, ScalarFunction<T> F>
struct derivative_result {
    T value;        // Estimated derivative
    T error;        // Estimated error
    int order;      // Order of accuracy
};

template<std::floating_point T, ScalarFunction<T> F>
derivative_result<T, F> central_with_error(F&& f, T x, T h = optimal_h<T>::central) {
    T d1 = central_difference(f, x, h);
    T d2 = central_difference(f, x, h / T(2));
    // For O(h²) method, error ≈ |d2 - d1| / 3
    return {d2, std::abs(d2 - d1) / T(3), 2};
}

// =============================================================================
// Comparison with Exact Derivatives
// =============================================================================

/**
 * Compare finite difference with an exact derivative (e.g., from autodiff).
 * Useful for validation and choosing optimal step sizes.
 */
template<std::floating_point T>
struct comparison_result {
    T finite_diff;
    T exact;
    T absolute_error;
    T relative_error;

    bool agrees(T tolerance = T(1e-6)) const {
        return absolute_error < tolerance ||
               relative_error < tolerance;
    }
};

template<std::floating_point T, ScalarFunction<T> F>
comparison_result<T> compare(F&& f, T x, T exact_derivative, T h = optimal_h<T>::central) {
    T fd = central_difference(f, x, h);
    T abs_err = std::abs(fd - exact_derivative);
    T rel_err = exact_derivative != T(0) ? abs_err / std::abs(exact_derivative) : abs_err;
    return {fd, exact_derivative, abs_err, rel_err};
}

} // namespace finite_diff
