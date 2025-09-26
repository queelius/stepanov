#pragma once

/**
 * Main header for Stepanov library optimization module
 *
 * This module provides generic optimization algorithms following
 * Alex Stepanov's principles of generic programming.
 *
 * Includes:
 * - Concepts for optimization
 * - Newton's method and variants
 * - Gradient descent algorithms (standard, momentum, Adam, etc.)
 * - Root finding algorithms (bisection, secant, Brent, etc.)
 * - Golden section search and related methods
 * - Simulated annealing for global optimization
 * - Vector operations and linear algebra utilities
 *
 * All algorithms are designed to work with:
 * - Automatic differentiation (autodiff module)
 * - Rational numbers
 * - Fixed decimal types
 * - Any type satisfying the required concepts
 */

#include "optimization/concepts.hpp"
#include "optimization/vector_ops.hpp"
#include "optimization/newton.hpp"
#include "optimization/gradient_descent.hpp"
#include "optimization/root_finding.hpp"
#include "optimization/golden_section.hpp"
#include "optimization/simulated_annealing.hpp"