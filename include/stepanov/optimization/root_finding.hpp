#pragma once

#include "concepts.hpp"
#include "../concepts.hpp"
#include <optional>
#include <utility>
#include <cmath>

namespace stepanov::optimization {

/**
 * Root finding algorithms for solving f(x) = 0
 *
 * Following generic programming principles, these algorithms work with any
 * type satisfying the required concepts (ordered fields, continuous functions)
 */

// Bisection method for continuous functions on ordered fields
template<typename T, typename F>
    requires ordered_field<T> && evaluable<F, T>
std::optional<T> bisection(
    F f,
    T a,
    T b,
    tolerance_criterion<T> stopping = {})
{
    // Ensure a < b
    if (a > b) std::swap(a, b);

    T fa = f(a);
    T fb = f(b);

    // Check if root is at endpoints
    if (abs(fa) < stopping.absolute_tolerance) return a;
    if (abs(fb) < stopping.absolute_tolerance) return b;

    // Check for sign change
    if (fa * fb > T(0)) {
        return std::nullopt; // No sign change, cannot guarantee root
    }

    size_t iter = 0;
    while (iter < stopping.max_iterations) {
        T c = half(a) + half(b); // Midpoint, avoiding overflow
        T fc = f(c);

        // Check if we found the root
        if (abs(fc) < stopping.absolute_tolerance ||
            abs(b - a) < stopping.absolute_tolerance) {
            return c;
        }

        // Update interval
        if (fa * fc < T(0)) {
            b = c;
            fb = fc;
        } else {
            a = c;
            fa = fc;
        }

        ++iter;
    }

    // Return best approximation
    return half(a) + half(b);
}

// Secant method - does not require derivatives
template<typename T, typename F>
    requires field<T> && evaluable<F, T>
std::optional<T> secant(
    F f,
    T x0,
    T x1,
    tolerance_criterion<T> stopping = {})
{
    T f0 = f(x0);
    T f1 = f(x1);

    for (size_t iter = 0; iter < stopping.max_iterations; ++iter) {
        // Check for convergence
        if (abs(f1) < stopping.absolute_tolerance) {
            return x1;
        }

        // Check if denominator is too small
        T denominator = f1 - f0;
        if (abs(denominator) < std::numeric_limits<T>::epsilon()) {
            return std::nullopt; // Cannot continue
        }

        // Secant step
        T x2 = x1 - f1 * (x1 - x0) / denominator;

        // Check stopping criterion
        if (abs(x2 - x1) < stopping.absolute_tolerance ||
            stopping(x2, x1, iter)) {
            return x2;
        }

        // Update for next iteration
        x0 = x1;
        f0 = f1;
        x1 = x2;
        f1 = f(x2);
    }

    return std::nullopt;
}

// False position (regula falsi) method - guaranteed convergence for continuous functions
template<typename T, typename F>
    requires ordered_field<T> && evaluable<F, T>
std::optional<T> false_position(
    F f,
    T a,
    T b,
    tolerance_criterion<T> stopping = {})
{
    // Ensure a < b
    if (a > b) std::swap(a, b);

    T fa = f(a);
    T fb = f(b);

    // Check for sign change
    if (fa * fb > T(0)) {
        return std::nullopt; // No sign change
    }

    size_t iter = 0;
    T c_prev = a;

    while (iter < stopping.max_iterations) {
        // False position formula
        T c = (a * fb - b * fa) / (fb - fa);
        T fc = f(c);

        // Check for convergence
        if (abs(fc) < stopping.absolute_tolerance ||
            abs(c - c_prev) < stopping.absolute_tolerance) {
            return c;
        }

        // Update interval
        if (fa * fc < T(0)) {
            b = c;
            fb = fc;
        } else {
            a = c;
            fa = fc;
        }

        c_prev = c;
        ++iter;
    }

    return (a * fb - b * fa) / (fb - fa);
}

// Brent's method - combines bisection, secant, and inverse quadratic interpolation
template<typename T, typename F>
    requires ordered_field<T> && evaluable<F, T>
std::optional<T> brent(
    F f,
    T a,
    T b,
    tolerance_criterion<T> stopping = {})
{
    T fa = f(a);
    T fb = f(b);

    // Check for sign change
    if (fa * fb > T(0)) {
        return std::nullopt;
    }

    if (abs(fa) < abs(fb)) {
        std::swap(a, b);
        std::swap(fa, fb);
    }

    T c = a;
    T fc = fa;
    T s = b; // Current approximation
    T fs = fb;
    T d = c; // Previous value of c
    bool mflag = true;

    for (size_t iter = 0; iter < stopping.max_iterations; ++iter) {
        // Check for convergence
        if (abs(fs) < stopping.absolute_tolerance ||
            abs(b - a) < stopping.absolute_tolerance) {
            return s;
        }

        if (abs(fa - fc) > std::numeric_limits<T>::epsilon() &&
            abs(fb - fc) > std::numeric_limits<T>::epsilon()) {
            // Inverse quadratic interpolation
            T term1 = a * fb * fc / ((fa - fb) * (fa - fc));
            T term2 = b * fa * fc / ((fb - fa) * (fb - fc));
            T term3 = c * fa * fb / ((fc - fa) * (fc - fb));
            s = term1 + term2 + term3;
        } else {
            // Secant method
            s = b - fb * (b - a) / (fb - fa);
        }

        // Conditions for accepting interpolation/extrapolation
        T tmp = (T(3) * a + b) / T(4);
        bool condition1 = (s < tmp && s < b) || (s > tmp && s > b);
        bool condition2 = mflag && abs(s - b) >= abs(b - c) / T(2);
        bool condition3 = !mflag && abs(s - b) >= abs(c - d) / T(2);
        bool condition4 = mflag && abs(b - c) < stopping.absolute_tolerance;
        bool condition5 = !mflag && abs(c - d) < stopping.absolute_tolerance;

        if (condition1 || condition2 || condition3 || condition4 || condition5) {
            // Use bisection
            s = half(a) + half(b);
            mflag = true;
        } else {
            mflag = false;
        }

        fs = f(s);
        d = c;  // Update previous value
        c = b;
        fc = fb;

        if (fa * fs < T(0)) {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }

        if (abs(fa) < abs(fb)) {
            std::swap(a, b);
            std::swap(fa, fb);
        }
    }

    return s;
}

// Ridders' method - faster than bisection, more robust than secant
template<typename T, typename F>
    requires ordered_field<T> && evaluable<F, T>
std::optional<T> ridders(
    F f,
    T a,
    T b,
    tolerance_criterion<T> stopping = {})
{
    T fa = f(a);
    T fb = f(b);

    // Check for sign change
    if (fa * fb > T(0)) {
        return std::nullopt;
    }

    for (size_t iter = 0; iter < stopping.max_iterations; ++iter) {
        T c = half(a) + half(b);
        T fc = f(c);

        // Check for convergence
        if (abs(fc) < stopping.absolute_tolerance) {
            return c;
        }

        T s = sqrt(fc * fc - fa * fb);
        if (s == T(0)) return c;

        T sign = (fa - fb) < T(0) ? T(-1) : T(1);
        T x = c + (c - a) * sign * fc / s;
        T fx = f(x);

        if (abs(fx) < stopping.absolute_tolerance) {
            return x;
        }

        // Update brackets
        if (fc * fx < T(0)) {
            if (c < x) {
                a = c;
                fa = fc;
                b = x;
                fb = fx;
            } else {
                a = x;
                fa = fx;
                b = c;
                fb = fc;
            }
        } else if (fa * fx < T(0)) {
            b = x;
            fb = fx;
        } else {
            a = x;
            fa = fx;
        }

        if (abs(b - a) < stopping.absolute_tolerance) {
            return half(a) + half(b);
        }
    }

    return half(a) + half(b);
}

// Fixed-point iteration for solving x = g(x)
template<typename T, typename G>
    requires field<T> && evaluable<G, T>
std::optional<T> fixed_point(
    G g,
    T initial_guess,
    tolerance_criterion<T> stopping = {})
{
    T x = initial_guess;

    for (size_t iter = 0; iter < stopping.max_iterations; ++iter) {
        T x_new = g(x);

        // Check for convergence
        if (abs(x_new - x) < stopping.absolute_tolerance ||
            stopping(x_new, x, iter)) {
            return x_new;
        }

        x = x_new;
    }

    return std::nullopt;
}

// Müller's method for complex roots (works with real numbers too)
template<typename T, typename F>
    requires field<T> && evaluable<F, T>
std::optional<T> muller(
    F f,
    T x0,
    T x1,
    T x2,
    tolerance_criterion<T> stopping = {})
{
    for (size_t iter = 0; iter < stopping.max_iterations; ++iter) {
        T f0 = f(x0);
        T f1 = f(x1);
        T f2 = f(x2);

        // Compute divided differences
        T h1 = x1 - x0;
        T h2 = x2 - x1;
        T delta1 = (f1 - f0) / h1;
        T delta2 = (f2 - f1) / h2;
        T d = (delta2 - delta1) / (h2 + h1);

        // Compute next approximation
        T b = delta2 + h2 * d;
        T discriminant = b * b - T(4) * f2 * d;

        if (discriminant < T(0)) {
            return std::nullopt; // Complex root for real field
        }

        T sqrt_disc = sqrt(discriminant);
        T denominator = abs(b - sqrt_disc) > abs(b + sqrt_disc) ?
                       b - sqrt_disc : b + sqrt_disc;

        T dx = T(-2) * f2 / denominator;
        T x3 = x2 + dx;

        // Check for convergence
        if (abs(dx) < stopping.absolute_tolerance ||
            abs(f(x3)) < stopping.absolute_tolerance) {
            return x3;
        }

        // Shift for next iteration
        x0 = x1;
        x1 = x2;
        x2 = x3;
    }

    return std::nullopt;
}

} // namespace stepanov::optimization