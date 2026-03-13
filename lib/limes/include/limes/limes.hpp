#pragma once

/**
 * @file limes.hpp
 * @brief Main entry point for the limes library.
 *
 * limes is a C++20 header-only library for composable calculus expressions
 * with symbolic differentiation and numerical integration.
 *
 * @code{.cpp}
 * #include <limes/limes.hpp>
 * using namespace limes::expr;
 *
 * auto x = arg<0>;
 * auto f = sin(x * x);
 * auto df = derivative(f).wrt<0>();
 *
 * auto I = integral(f).over<0>(0.0, 1.0);
 * auto result = I.eval();  // ~0.3103
 * @endcode
 *
 * @author Alex Towell (queelius@gmail.com)
 * @see https://metafunctor.com
 * @copyright MIT License
 */

#include "fwd.hpp"
#include "algorithms/algorithms.hpp"
#include "expr/expr.hpp"
