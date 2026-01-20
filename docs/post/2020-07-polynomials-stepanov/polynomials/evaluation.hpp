#pragma once

/**
 * @file evaluation.hpp
 * @brief Polynomial evaluation and root finding
 *
 * THIS MODULE TEACHES: Efficient polynomial evaluation.
 *
 * The naive approach to evaluate a_n*x^n + ... + a_1*x + a_0 uses O(n^2)
 * multiplications. Horner's method reduces this to O(n) by rewriting as:
 *   ((a_n*x + a_{n-1})*x + a_{n-2})*x + ... + a_0
 *
 * This is not just faster - it's also more numerically stable.
 */

#include "sparse.hpp"
#include <optional>
#include <vector>
#include <cmath>

namespace poly {

// =============================================================================
// Evaluation
// =============================================================================

/**
 * Evaluate polynomial at a point using Horner's method.
 *
 * For dense polynomials, this is optimal: O(n) multiplications and additions.
 * For sparse polynomials, we use a modified approach that skips zero terms.
 *
 * @param p The polynomial to evaluate
 * @param x The point at which to evaluate
 * @return p(x)
 */
template<typename T>
T evaluate(const polynomial<T>& p, const T& x) {
    if (p.is_zero()) return T(0);

    // For sparse polynomials, iterate through terms
    // and accumulate: sum of coef * x^deg
    T result = T(0);
    T x_power = T(1);
    int current_deg = 0;

    for (const auto& [deg, coef] : p) {
        // Compute x^deg from current power
        while (current_deg < deg) {
            x_power = x_power * x;
            ++current_deg;
        }
        result = result + coef * x_power;
    }

    return result;
}

/**
 * Evaluate polynomial using Horner's method (dense evaluation).
 *
 * More efficient when the polynomial is dense or nearly dense.
 * Evaluates p(x) = a_n*x^n + ... + a_1*x + a_0 as:
 *   ((a_n*x + a_{n-1})*x + ... + a_1)*x + a_0
 */
template<typename T>
T evaluate_horner(const polynomial<T>& p, const T& x) {
    if (p.is_zero()) return T(0);

    int deg = p.degree();
    T result = p[deg];

    for (int i = deg - 1; i >= 0; --i) {
        result = result * x + p[i];
    }

    return result;
}

// Note: polynomial class should define operator() inline if desired.
// For now, use the evaluate() function directly.

/**
 * Evaluate polynomial at multiple points.
 *
 * @param p The polynomial
 * @param points Vector of points to evaluate at
 * @return Vector of p(point) for each point
 */
template<typename T>
std::vector<T> evaluate_many(const polynomial<T>& p, const std::vector<T>& points) {
    std::vector<T> results;
    results.reserve(points.size());
    for (const auto& x : points) {
        results.push_back(evaluate(p, x));
    }
    return results;
}

// =============================================================================
// Calculus operations
// =============================================================================

/**
 * Compute derivative of polynomial.
 *
 * d/dx (a_n*x^n) = n*a_n*x^(n-1)
 */
template<typename T>
polynomial<T> derivative(const polynomial<T>& p) {
    polynomial<T> result;

    for (const auto& [deg, coef] : p) {
        if (deg > 0) {
            // Use repeated addition to multiply by degree
            // This works for any ring, not just fields
            T new_coef = coef;
            for (int i = 1; i < deg; ++i) {
                new_coef = new_coef + coef;
            }
            result = result + polynomial<T>::monomial(new_coef, deg - 1);
        }
    }

    return result;
}

/**
 * Compute antiderivative (indefinite integral) of polynomial.
 *
 * integral (a_n*x^n) dx = a_n*x^(n+1)/(n+1) + C
 *
 * Requires field coefficients for division by (n+1).
 *
 * @param p The polynomial to integrate
 * @param constant The constant of integration (default 0)
 */
template<typename T>
    requires field<T>
polynomial<T> antiderivative(const polynomial<T>& p, T constant = T(0)) {
    polynomial<T> result{constant};

    for (const auto& [deg, coef] : p) {
        T divisor = T(deg + 1);
        result = result + polynomial<T>::monomial(coef / divisor, deg + 1);
    }

    return result;
}

/**
 * Definite integral from a to b.
 *
 * integral_a^b p(x) dx = F(b) - F(a) where F is antiderivative
 */
template<typename T>
    requires field<T>
T definite_integral(const polynomial<T>& p, T a, T b) {
    auto F = antiderivative(p);
    return evaluate(F, b) - evaluate(F, a);
}

// =============================================================================
// Root finding
// =============================================================================

/**
 * Find a root of polynomial using Newton-Raphson method.
 *
 * Newton's method: x_{n+1} = x_n - p(x_n) / p'(x_n)
 *
 * @param p The polynomial
 * @param initial_guess Starting point for iteration
 * @param tolerance Convergence threshold
 * @param max_iter Maximum iterations
 * @return Root if found, nullopt if not converged
 */
template<typename T>
    requires ordered_field<T>
std::optional<T> find_root_newton(
    const polynomial<T>& p,
    T initial_guess,
    T tolerance = T(1e-10),
    int max_iter = 100
) {
    auto dp = derivative(p);
    T x = initial_guess;

    for (int i = 0; i < max_iter; ++i) {
        T fx = evaluate(p, x);

        // Check convergence
        if (fx < tolerance && fx > -tolerance) {
            return x;
        }

        T fpx = evaluate(dp, x);

        // Check for zero derivative
        if (fpx < tolerance && fpx > -tolerance) {
            return std::nullopt;
        }

        T x_new = x - fx / fpx;

        // Check for convergence in x
        T diff = x_new - x;
        if (diff < tolerance && diff > -tolerance) {
            return x_new;
        }

        x = x_new;
    }

    return std::nullopt;
}

/**
 * Find all real roots in a given interval.
 *
 * Uses Newton's method with multiple starting points to find distinct roots.
 *
 * @param p The polynomial
 * @param lower Lower bound of search interval
 * @param upper Upper bound of search interval
 * @param num_samples Number of starting points
 * @return Vector of distinct roots found
 */
template<typename T>
    requires ordered_field<T>
std::vector<T> find_roots(
    const polynomial<T>& p,
    T lower,
    T upper,
    int num_samples = 20,
    T tolerance = T(1e-10)
) {
    std::vector<T> roots;
    T step = (upper - lower) / T(num_samples);

    for (int i = 0; i <= num_samples; ++i) {
        T initial = lower + T(i) * step;

        if (auto root = find_root_newton(p, initial, tolerance)) {
            // Check if root is in bounds and not a duplicate
            if (*root >= lower && *root <= upper) {
                bool is_new = true;
                for (const auto& r : roots) {
                    T diff = *root - r;
                    if (diff < tolerance * T(10) && diff > -tolerance * T(10)) {
                        is_new = false;
                        break;
                    }
                }
                if (is_new) {
                    roots.push_back(*root);
                }
            }
        }
    }

    std::sort(roots.begin(), roots.end());
    return roots;
}

/**
 * Find stationary points (where derivative = 0).
 */
template<typename T>
    requires ordered_field<T>
std::vector<T> stationary_points(const polynomial<T>& p, T lower, T upper) {
    return find_roots(derivative(p), lower, upper);
}

/**
 * Find inflection points (where second derivative = 0).
 */
template<typename T>
    requires ordered_field<T>
std::vector<T> inflection_points(const polynomial<T>& p, T lower, T upper) {
    return find_roots(derivative(derivative(p)), lower, upper);
}

// =============================================================================
// Synthetic division
// =============================================================================

/**
 * Divide polynomial by (x - c) using synthetic division.
 *
 * If p(c) = 0, then (x - c) is a factor and the remainder is 0.
 *
 * @param p The polynomial to divide
 * @param c The root to divide out
 * @return Pair (quotient, remainder) where p = (x-c)*quotient + remainder
 */
template<typename T>
    requires field<T>
std::pair<polynomial<T>, T> synthetic_division(const polynomial<T>& p, T c) {
    if (p.is_zero()) {
        return {polynomial<T>{}, T(0)};
    }

    int deg = p.degree();
    std::vector<T> coeffs(deg + 1, T(0));
    for (const auto& [d, coef] : p) {
        coeffs[d] = coef;
    }

    // Synthetic division
    std::vector<T> result(deg, T(0));
    T carry = coeffs[deg];

    for (int i = deg - 1; i >= 0; --i) {
        result[i] = carry;
        carry = coeffs[i] + carry * c;
    }

    // Build quotient polynomial
    polynomial<T> quotient;
    for (int i = 0; i < deg; ++i) {
        if (result[i] != T(0)) {
            quotient = quotient + polynomial<T>::monomial(result[i], i);
        }
    }

    return {quotient, carry};
}

/**
 * Factor out a known root, returning the deflated polynomial.
 */
template<typename T>
    requires field<T>
polynomial<T> deflate(const polynomial<T>& p, T root) {
    return synthetic_division(p, root).first;
}

} // namespace poly
