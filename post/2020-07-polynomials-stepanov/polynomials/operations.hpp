#pragma once

/**
 * @file operations.hpp
 * @brief Polynomial division and GCD
 *
 * THIS MODULE TEACHES: Polynomials as a Euclidean domain.
 *
 * The key insight: the GCD algorithm that works for integers also works
 * for polynomials. Both are examples of Euclidean domains - algebraic
 * structures where division with remainder is always possible.
 *
 * For integers: norm(n) = |n|
 * For polynomials: norm(p) = degree(p)
 *
 * The generic algorithm: gcd(a, b) = gcd(b, a mod b)
 */

#include "sparse.hpp"
#include <utility>
#include <stdexcept>

namespace poly {

// =============================================================================
// Polynomial division
// =============================================================================

/**
 * Polynomial division with remainder.
 *
 * Given polynomials a and b (with b != 0), compute q and r such that:
 *   a = b * q + r   and   degree(r) < degree(b)
 *
 * This is the polynomial analog of integer division.
 *
 * @param dividend The polynomial to divide (a)
 * @param divisor The polynomial to divide by (b)
 * @return Pair (quotient, remainder)
 * @throws std::domain_error if divisor is zero
 */
template<typename T>
    requires field<T>
std::pair<polynomial<T>, polynomial<T>> divmod(
    const polynomial<T>& dividend,
    const polynomial<T>& divisor
) {
    if (divisor.is_zero()) {
        throw std::domain_error("Division by zero polynomial");
    }

    polynomial<T> quotient;
    polynomial<T> remainder = dividend;

    T divisor_lc = divisor.leading_coefficient();
    int divisor_deg = divisor.degree();

    while (!remainder.is_zero() && remainder.degree() >= divisor_deg) {
        int deg_diff = remainder.degree() - divisor_deg;
        T coef = remainder.leading_coefficient() / divisor_lc;

        // Add term to quotient
        auto term = polynomial<T>::monomial(coef, deg_diff);
        quotient = quotient + term;

        // Subtract divisor * term from remainder
        remainder = remainder - divisor * term;
    }

    return {quotient, remainder};
}

/**
 * Quotient of polynomial division.
 */
template<typename T>
    requires field<T>
polynomial<T> quotient(const polynomial<T>& a, const polynomial<T>& b) {
    return divmod(a, b).first;
}

/**
 * Remainder of polynomial division.
 */
template<typename T>
    requires field<T>
polynomial<T> remainder(const polynomial<T>& a, const polynomial<T>& b) {
    return divmod(a, b).second;
}

/**
 * Norm for Euclidean domain: degree of polynomial.
 * Returns -1 for zero polynomial.
 */
template<typename T>
int norm(const polynomial<T>& p) {
    return p.degree();
}

// Operator overloads for division

template<typename T>
    requires field<T>
polynomial<T> operator/(const polynomial<T>& a, const polynomial<T>& b) {
    return quotient(a, b);
}

template<typename T>
    requires field<T>
polynomial<T> operator%(const polynomial<T>& a, const polynomial<T>& b) {
    return remainder(a, b);
}

// =============================================================================
// Greatest Common Divisor
// =============================================================================

/**
 * GCD of two polynomials using Euclidean algorithm.
 *
 * THIS IS THE KEY INSIGHT: The exact same algorithm works for integers
 * and polynomials because both are Euclidean domains.
 *
 * @return GCD (monic polynomial if coefficients are from a field)
 */
template<typename T>
    requires field<T>
polynomial<T> gcd(polynomial<T> a, polynomial<T> b) {
    while (!b.is_zero()) {
        polynomial<T> r = remainder(a, b);
        a = b;
        b = r;
    }

    // Make monic (leading coefficient = 1) for canonical form
    if (!a.is_zero()) {
        a = a / a.leading_coefficient();
    }

    return a;
}

/**
 * Extended Euclidean algorithm for polynomials.
 *
 * Computes gcd(a, b) and coefficients s, t such that:
 *   gcd(a, b) = a * s + b * t
 *
 * This is Bezout's identity for polynomials.
 *
 * @return Tuple (gcd, s, t) where gcd = a*s + b*t
 */
template<typename T>
    requires field<T>
std::tuple<polynomial<T>, polynomial<T>, polynomial<T>>
extended_gcd(polynomial<T> a, polynomial<T> b) {
    polynomial<T> old_r = a, r = b;
    polynomial<T> old_s{T(1)}, s{T(0)};
    polynomial<T> old_t{T(0)}, t{T(1)};

    while (!r.is_zero()) {
        auto [q, rem] = divmod(old_r, r);

        polynomial<T> new_r = rem;
        polynomial<T> new_s = old_s - q * s;
        polynomial<T> new_t = old_t - q * t;

        old_r = r; r = new_r;
        old_s = s; s = new_s;
        old_t = t; t = new_t;
    }

    // Make gcd monic
    if (!old_r.is_zero()) {
        T lc = old_r.leading_coefficient();
        old_r = old_r / lc;
        old_s = old_s / lc;
        old_t = old_t / lc;
    }

    return {old_r, old_s, old_t};
}

/**
 * LCM of two polynomials.
 *
 * lcm(a, b) = a * b / gcd(a, b)
 */
template<typename T>
    requires field<T>
polynomial<T> lcm(const polynomial<T>& a, const polynomial<T>& b) {
    if (a.is_zero() || b.is_zero()) {
        return polynomial<T>{};
    }
    return a * (b / gcd(a, b));
}

// =============================================================================
// Divisibility testing
// =============================================================================

/**
 * Check if a divides b (i.e., b % a == 0).
 */
template<typename T>
    requires field<T>
bool divides(const polynomial<T>& a, const polynomial<T>& b) {
    if (a.is_zero()) return b.is_zero();
    return remainder(b, a).is_zero();
}

/**
 * Check if polynomial is irreducible (cannot be factored).
 *
 * Note: This is a simple check for low-degree cases only.
 * General irreducibility testing is much more complex.
 */
template<typename T>
    requires field<T>
bool is_irreducible_simple(const polynomial<T>& p) {
    if (p.degree() <= 1) return true;
    if (p.degree() == 2 || p.degree() == 3) {
        // For degree 2 or 3, irreducible iff no roots
        // This requires root-finding which we'll defer to evaluation.hpp
        return true;  // Placeholder
    }
    return true;  // Placeholder for higher degrees
}

} // namespace poly
