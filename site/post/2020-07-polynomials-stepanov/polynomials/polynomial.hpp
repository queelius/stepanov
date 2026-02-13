#pragma once

/**
 * @file polynomial.hpp
 * @brief Main umbrella header for the polynomials library
 *
 * Include this header to get access to all polynomial functionality.
 *
 * Usage:
 * @code
 * #include <polynomials/polynomial.hpp>
 *
 * using namespace poly;
 *
 * // Create polynomials
 * auto p = polynomial<double>{1, -2, 1};  // 1 - 2x + x^2 = (x-1)^2
 * auto q = polynomial<double>::x() - 1.0; // x - 1
 *
 * // Arithmetic
 * auto sum = p + q;
 * auto prod = p * q;
 * auto [quot, rem] = divmod(p, q);
 *
 * // GCD (the key insight: same algorithm as for integers!)
 * auto g = gcd(p, q);  // Should be (x - 1)
 *
 * // Evaluation
 * double val = evaluate(p, 2.0);  // p(2)
 *
 * // Calculus
 * auto dp = derivative(p);
 * auto roots = find_roots(p, -10.0, 10.0);
 * @endcode
 */

#include "concepts.hpp"
#include "sparse.hpp"
#include "operations.hpp"
#include "evaluation.hpp"
