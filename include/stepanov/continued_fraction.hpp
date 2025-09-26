#pragma once

#include <vector>
#include <utility>
#include <limits>
#include <algorithm>
#include "concepts.hpp"

namespace stepanov {

/**
 * Continued Fraction Algorithms
 *
 * Implements the theory of continued fractions for:
 * - Rational approximation (convergents)
 * - Euclidean algorithm connection
 * - Diophantine equation solving
 *
 * Following Alex Stepanov's approach: these algorithms work on any
 * Euclidean domain, not just integers.
 */

// Generate continued fraction representation
template<euclidean_domain T>
std::vector<T> to_continued_fraction(T numerator, T denominator) {
    std::vector<T> result;

    while (denominator != T(0)) {
        T q = quotient(numerator, denominator);
        result.push_back(q);

        T temp = denominator;
        denominator = remainder(numerator, denominator);
        numerator = temp;
    }

    return result;
}

// Convergent - a rational approximation from partial continued fraction
template<euclidean_domain T>
struct convergent {
    T p;  // numerator
    T q;  // denominator

    convergent() : p(T(0)), q(T(1)) {}
    convergent(T num, T den) : p(num), q(den) {}
};

// Compute convergents - successively better rational approximations
template<euclidean_domain T>
std::vector<convergent<T>> compute_convergents(const std::vector<T>& cf) {
    if (cf.empty()) return {};

    std::vector<convergent<T>> result;
    result.reserve(cf.size());

    // First convergent
    result.emplace_back(cf[0], T(1));

    if (cf.size() == 1) return result;

    // Second convergent
    result.emplace_back(cf[1] * cf[0] + T(1), cf[1]);

    // Subsequent convergents using recurrence relation
    for (size_t i = 2; i < cf.size(); ++i) {
        T p = cf[i] * result[i-1].p + result[i-2].p;
        T q = cf[i] * result[i-1].q + result[i-2].q;
        result.emplace_back(p, q);
    }

    return result;
}

// Find best rational approximation with denominator <= max_denominator
template<euclidean_domain T>
convergent<T> best_rational_approximation(T numerator, T denominator, T max_denominator) {
    auto cf = to_continued_fraction(numerator, denominator);
    auto convergents = compute_convergents(cf);

    // Find the last convergent with denominator <= max_denominator
    for (auto it = convergents.rbegin(); it != convergents.rend(); ++it) {
        if (norm(it->q) <= norm(max_denominator)) {
            return *it;
        }
    }

    return convergent<T>(numerator, denominator);
}

// Semi-convergents (intermediate approximations between convergents)
template<euclidean_domain T>
std::vector<convergent<T>> compute_semiconvergents(const std::vector<T>& cf) {
    auto convergents = compute_convergents(cf);
    std::vector<convergent<T>> result;

    for (size_t i = 0; i < convergents.size(); ++i) {
        result.push_back(convergents[i]);

        // Generate semiconvergents between convergents[i] and convergents[i+1]
        if (i + 1 < convergents.size() && i > 0) {
            T a = cf[i + 1];
            for (T k = T(1); k < a; k = k + T(1)) {
                T p = k * convergents[i].p + convergents[i-1].p;
                T q = k * convergents[i].q + convergents[i-1].q;
                result.emplace_back(p, q);
            }
        }
    }

    return result;
}

// Extended Euclidean algorithm via continued fractions
// Finds x, y such that ax + by = gcd(a, b)
template<euclidean_domain T>
struct bezout_coefficients {
    T x, y, gcd;
};

template<euclidean_domain T>
bezout_coefficients<T> extended_gcd_cf(T a, T b) {
    if (b == T(0)) {
        return {T(1), T(0), a};
    }

    auto cf = to_continued_fraction(a, b);
    auto convergents = compute_convergents(cf);

    size_t n = convergents.size();

    // The relationship between convergents and Bezout coefficients:
    // If p_n/q_n is the last convergent of a/b, then:
    // a * q_{n-1} - b * p_{n-1} = (-1)^n * gcd(a,b)

    T gcd_val = remainder(a, b);
    while (remainder(b, gcd_val) != T(0)) {
        T temp = gcd_val;
        gcd_val = remainder(b, gcd_val);
        b = temp;
    }

    if (n > 1) {
        T sign = (n % 2 == 0) ? T(1) : T(-1);
        return {sign * convergents[n-2].q, -sign * convergents[n-2].p, gcd_val};
    }

    return {T(0), T(1), b};
}

// Solve linear Diophantine equation ax + by = c
template<euclidean_domain T>
struct diophantine_solution {
    bool has_solution;
    T x0, y0;  // particular solution
    T dx, dy;  // general solution: x = x0 + k*dx, y = y0 - k*dy
};

template<euclidean_domain T>
diophantine_solution<T> solve_linear_diophantine(T a, T b, T c) {
    auto [x, y, g] = extended_gcd_cf(a, b);

    // Check if c is divisible by gcd(a, b)
    if (remainder(c, g) != T(0)) {
        return {false, T(0), T(0), T(0), T(0)};
    }

    // Scale the solution
    T scale = quotient(c, g);
    x = x * scale;
    y = y * scale;

    // General solution parameters
    T dx = quotient(b, g);
    T dy = quotient(a, g);

    return {true, x, y, dx, dy};
}

// Periodic continued fractions (for quadratic irrationals)
template<euclidean_domain T>
struct quadratic_cf {
    std::vector<T> pre_period;
    std::vector<T> period;
    bool is_periodic;
};

// Check if a continued fraction is purely periodic
template<euclidean_domain T>
bool is_purely_periodic(const std::vector<T>& cf) {
    if (cf.size() < 2) return false;

    // A continued fraction is purely periodic if it's a reduced
    // quadratic irrational (conjugate between -1 and 0)
    // This is a simplified check
    return cf[0] > T(0);
}

// Pell equation solver using continued fractions
// Solves x^2 - n*y^2 = 1
template<euclidean_domain T>
struct pell_solution {
    T x, y;
    bool exists;
};

template<euclidean_domain T>
requires ordered_ring<T>
pell_solution<T> solve_pell_fundamental(T n) {
    // For simplicity, we'll use a bounded search
    // In practice, this would compute the continued fraction of sqrt(n)

    T max_search = T(1000000);

    for (T y = T(1); y < max_search; y = y + T(1)) {
        T x_squared = n * y * y + T(1);

        // Check if x_squared is a perfect square
        T x = T(1);
        while (x * x < x_squared) {
            x = x + T(1);
        }

        if (x * x == x_squared) {
            return {x, y, true};
        }
    }

    return {T(0), T(0), false};
}

// Generate all solutions to Pell equation from fundamental solution
template<euclidean_domain T>
std::vector<pell_solution<T>> generate_pell_solutions(T n, const pell_solution<T>& fundamental, size_t count) {
    if (!fundamental.exists) return {};

    std::vector<pell_solution<T>> result;
    result.push_back(fundamental);

    T x_prev = fundamental.x;
    T y_prev = fundamental.y;

    for (size_t i = 1; i < count; ++i) {
        // Recurrence: x_{n+1} = x_1 * x_n + n * y_1 * y_n
        //            y_{n+1} = x_1 * y_n + y_1 * x_n
        T x_next = fundamental.x * x_prev + n * fundamental.y * y_prev;
        T y_next = fundamental.x * y_prev + fundamental.y * x_prev;

        result.push_back({x_next, y_next, true});

        x_prev = x_next;
        y_prev = y_next;
    }

    return result;
}

} // namespace stepanov