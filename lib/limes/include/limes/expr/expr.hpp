#pragma once

/**
 * @file expr.hpp
 * @brief Expression layer for composable calculus expressions.
 *
 * This module provides the user-facing API for building, differentiating,
 * and integrating mathematical expressions. Derivatives are computed
 * symbolically via chain rule at compile time; integrals are evaluated
 * numerically with pluggable methods.
 *
 * @section expr_usage Usage
 *
 * @code{.cpp}
 * using namespace limes::expr;
 *
 * auto x = arg<0>;                              // First variable
 * auto f = sin(x*x);                            // sin(x²)
 * auto df = derivative(f).wrt<0>();             // d/dx[sin(x²)] = 2x·cos(x²)
 *
 * auto I = integral(x*x).over<0>(0.0, 1.0);     // ∫₀¹ x² dx
 * auto result = I.eval();                       // ≈ 1/3
 * @endcode
 *
 * @defgroup expr_nodes Expression Nodes
 * @brief Building blocks for mathematical expressions.
 *
 * @defgroup expr_ops Expression Operations
 * @brief Differentiation, integration, and analysis operations.
 */

// Concepts
#include "concepts.hpp"

// Core node types
#include "nodes/const.hpp"
#include "nodes/var.hpp"
#include "nodes/named_var.hpp"
#include "nodes/binary.hpp"
#include "nodes/unary.hpp"
#include "nodes/pow.hpp"
#include "nodes/primitives.hpp"
#include "nodes/binary_func.hpp"
#include "nodes/bound.hpp"
#include "nodes/conditional.hpp"
#include "nodes/sum_product.hpp"

// Operations
#include "derivative.hpp"
#include "derivative_builder.hpp"
#include "antiderivatives.hpp"
#include "integral.hpp"
#include "product_integral.hpp"

// Analysis
#include "analysis.hpp"

// Inspection
#include "to_string.hpp"

/**
 * @namespace limes::expr
 * @brief Expression layer for composable calculus.
 *
 * This namespace contains all the types and functions for building
 * mathematical expressions, computing derivatives, and evaluating integrals.
 *
 * Key types:
 * - Var, NamedVar - Variables
 * - Const, Zero, One - Constants
 * - Binary, Unary - Composite expressions
 * - UnaryFunc - Mathematical functions (sin, cos, exp, log, etc.)
 * - Integral, BoxIntegral - Integration expressions
 * - ProductIntegral - Separable integral composition
 *
 * Key functions:
 * - derivative(f).wrt<D>() - Symbolic differentiation
 * - integral(f).over<D>(a, b) - Definite integration
 * - integral(f).over_box(...) - N-dimensional box integration
 */
namespace limes::expr {

// Re-export commonly used items for convenience
using algorithms::integration_result;

} // namespace limes::expr
