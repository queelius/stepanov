#pragma once

#include "concepts.hpp"
#include "../concepts.hpp"
#include <cmath>
#include <tuple>

namespace stepanov::optimization {

/**
 * Golden section search for finding minimum of unimodal functions
 *
 * Uses the golden ratio to efficiently narrow down the search interval
 * Works with any ordered field type following generic programming principles
 */

template<typename T>
    requires ordered_field<T>
constexpr T golden_ratio() {
    // φ = (1 + √5) / 2
    return (T(1) + sqrt(T(5))) / T(2);
}

template<typename T>
    requires ordered_field<T>
constexpr T golden_ratio_conjugate() {
    // φ̂ = (√5 - 1) / 2 = 1 / φ
    return (sqrt(T(5)) - T(1)) / T(2);
}

// Golden section search for minimization
template<typename T, typename F>
    requires ordered_field<T> && evaluable<F, T>
optimization_result<T> golden_section_minimize(
    F f,
    T a,
    T b,
    tolerance_criterion<T> tolerance = {})
{
    // Golden ratio conjugate for optimal division
    T gr = golden_ratio_conjugate<T>();

    // Ensure a < b
    if (a > b) std::swap(a, b);

    // Initial probe points
    T x1 = a + (T(1) - gr) * (b - a);  // x1 is at ~0.382 of interval
    T x2 = a + gr * (b - a);            // x2 is at ~0.618 of interval

    T f1 = f(x1);
    T f2 = f(x2);

    size_t iter = 0;
    for (; iter < tolerance.max_iterations; ++iter) {
        // Check convergence
        if (abs(b - a) < tolerance.absolute_tolerance) {
            T x_min = half(a) + half(b);
            return optimization_result<T>{x_min, f(x_min), iter, true};
        }

        if (f1 < f2) {
            // Minimum is in [a, x2]
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = a + (T(1) - gr) * (b - a);
            f1 = f(x1);
        } else {
            // Minimum is in [x1, b]
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + gr * (b - a);
            f2 = f(x2);
        }
    }

    T x_min = half(a) + half(b);
    return optimization_result<T>{x_min, f(x_min), iter, false};
}

// Golden section search for maximization
template<typename T, typename F>
    requires ordered_field<T> && evaluable<F, T>
optimization_result<T> golden_section_maximize(
    F f,
    T a,
    T b,
    tolerance_criterion<T> tolerance = {})
{
    // Maximize f by minimizing -f
    auto neg_f = [&f](T x) { return -f(x); };
    auto result = golden_section_minimize(neg_f, a, b, tolerance);
    result.value = -result.value; // Correct the function value
    return result;
}

// Fibonacci search - more optimal for fixed number of evaluations
template<typename T, typename F>
    requires ordered_field<T> && evaluable<F, T>
optimization_result<T> fibonacci_search(
    F f,
    T a,
    T b,
    size_t n_evaluations)
{
    // Generate Fibonacci numbers up to n
    std::vector<T> fib(n_evaluations + 2);
    fib[0] = T(1);
    fib[1] = T(1);
    for (size_t i = 2; i < fib.size(); ++i) {
        fib[i] = fib[i-1] + fib[i-2];
    }

    // Ensure a < b
    if (a > b) std::swap(a, b);

    // Initial probe points
    T x1 = a + (fib[n_evaluations-2] / fib[n_evaluations]) * (b - a);
    T x2 = a + (fib[n_evaluations-1] / fib[n_evaluations]) * (b - a);

    T f1 = f(x1);
    T f2 = f(x2);

    for (size_t k = n_evaluations; k > 2; --k) {
        if (f1 < f2) {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = a + (fib[k-3] / fib[k-1]) * (b - a);
            f1 = f(x1);
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + (fib[k-2] / fib[k-1]) * (b - a);
            f2 = f(x2);
        }
    }

    T x_min = f1 < f2 ? x1 : x2;
    T f_min = f1 < f2 ? f1 : f2;

    return optimization_result<T>{x_min, f_min, n_evaluations, true};
}

// Ternary search - alternative to golden section
template<typename T, typename F>
    requires ordered_field<T> && evaluable<F, T>
optimization_result<T> ternary_search(
    F f,
    T a,
    T b,
    tolerance_criterion<T> tolerance = {})
{
    // Ensure a < b
    if (a > b) std::swap(a, b);

    size_t iter = 0;
    for (; iter < tolerance.max_iterations; ++iter) {
        // Check convergence
        if (abs(b - a) < tolerance.absolute_tolerance) {
            T x_min = half(a) + half(b);
            return optimization_result<T>{x_min, f(x_min), iter, true};
        }

        // Divide interval into three parts
        T m1 = a + (b - a) / T(3);
        T m2 = b - (b - a) / T(3);

        T f1 = f(m1);
        T f2 = f(m2);

        if (f1 < f2) {
            b = m2; // Minimum is in [a, m2]
        } else {
            a = m1; // Minimum is in [m1, b]
        }
    }

    T x_min = half(a) + half(b);
    return optimization_result<T>{x_min, f(x_min), iter, false};
}

// Quadratic interpolation search for smooth functions
template<typename T, typename F>
    requires field<T> && evaluable<F, T>
optimization_result<T> quadratic_interpolation(
    F f,
    T x0,
    T x1,
    T x2,
    tolerance_criterion<T> tolerance = {})
{
    // Ensure x0 < x1 < x2
    if (x0 > x1) std::swap(x0, x1);
    if (x1 > x2) std::swap(x1, x2);
    if (x0 > x1) std::swap(x0, x1);

    T f0 = f(x0);
    T f1 = f(x1);
    T f2 = f(x2);

    size_t iter = 0;
    for (; iter < tolerance.max_iterations; ++iter) {
        // Fit quadratic through three points
        T num = (x1 - x0) * (x1 - x0) * (f1 - f2) -
                (x1 - x2) * (x1 - x2) * (f1 - f0);
        T den = T(2) * ((x1 - x0) * (f1 - f2) - (x1 - x2) * (f1 - f0));

        if (abs(den) < std::numeric_limits<T>::epsilon()) {
            // Denominator too small, fall back to midpoint
            T x_new = half(x0) + half(x2);
            return optimization_result<T>{x_new, f(x_new), iter, false};
        }

        T x_new = x1 - num / den;

        // Ensure x_new is within bounds
        x_new = std::max(x0, std::min(x_new, x2));

        // Check convergence
        if (abs(x_new - x1) < tolerance.absolute_tolerance) {
            return optimization_result<T>{x_new, f(x_new), iter, true};
        }

        T f_new = f(x_new);

        // Update points for next iteration
        if (x_new < x1) {
            if (f_new < f1) {
                x2 = x1;
                f2 = f1;
                x1 = x_new;
                f1 = f_new;
            } else {
                x0 = x_new;
                f0 = f_new;
            }
        } else {
            if (f_new < f1) {
                x0 = x1;
                f0 = f1;
                x1 = x_new;
                f1 = f_new;
            } else {
                x2 = x_new;
                f2 = f_new;
            }
        }
    }

    return optimization_result<T>{x1, f1, iter, false};
}

// Brent's method for univariate optimization (combines golden section and parabolic interpolation)
template<typename T, typename F>
    requires ordered_field<T> && evaluable<F, T>
optimization_result<T> brent_minimize(
    F f,
    T a,
    T b,
    tolerance_criterion<T> tolerance = {})
{
    constexpr T GOLDEN = golden_ratio_conjugate<T>();
    constexpr T ZEPS = std::numeric_limits<T>::epsilon() * T(1e-3);

    // Ensure a < b
    if (a > b) std::swap(a, b);

    T x = a + GOLDEN * (b - a);
    T v = x;
    T w = x;
    T fx = f(x);
    T fv = fx;
    T fw = fx;

    T d = T(0);
    T e = T(0);

    size_t iter = 0;
    for (; iter < tolerance.max_iterations; ++iter) {
        T xm = half(a) + half(b);
        T tol1 = tolerance.absolute_tolerance * abs(x) + ZEPS;
        T tol2 = T(2) * tol1;

        // Check for convergence
        if (abs(x - xm) <= tol2 - half(b - a)) {
            return optimization_result<T>{x, fx, iter, true};
        }

        bool golden_section_step = false;

        if (abs(e) > tol1) {
            // Try parabolic interpolation
            T r = (x - w) * (fx - fv);
            T q = (x - v) * (fx - fw);
            T p = (x - v) * q - (x - w) * r;
            q = T(2) * (q - r);

            if (q > T(0)) p = -p;
            q = abs(q);

            T etemp = e;
            e = d;

            // Check if parabolic interpolation is acceptable
            if (abs(p) >= abs(half(q * etemp)) || p <= q * (a - x) || p >= q * (b - x)) {
                golden_section_step = true;
            } else {
                // Take parabolic step
                d = p / q;
                T u = x + d;

                // Don't evaluate too close to a or b
                if (u - a < tol2 || b - u < tol2) {
                    d = (xm < x) ? -tol1 : tol1;
                }
            }
        } else {
            golden_section_step = true;
        }

        if (golden_section_step) {
            // Golden section step
            e = (x >= xm) ? a - x : b - x;
            d = GOLDEN * e;
        }

        // Evaluate new point
        T u = (abs(d) >= tol1) ? x + d : x + ((d > T(0)) ? tol1 : -tol1);
        T fu = f(u);

        // Update bracket
        if (fu <= fx) {
            if (u >= x) a = x; else b = x;
            v = w; w = x; x = u;
            fv = fw; fw = fx; fx = fu;
        } else {
            if (u < x) a = u; else b = u;
            if (fu <= fw || w == x) {
                v = w; w = u;
                fv = fw; fw = fu;
            } else if (fu <= fv || v == x || v == w) {
                v = u;
                fv = fu;
            }
        }
    }

    return optimization_result<T>{x, fx, iter, false};
}

} // namespace stepanov::optimization