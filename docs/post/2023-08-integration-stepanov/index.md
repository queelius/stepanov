---
title: "Numerical Integration with Generic Concepts"
date: 2023-08-28
draft: false
tags:
  - C++
  - numerical-methods
  - integration
  - quadrature
  - concepts
  - generic-programming
categories:
  - Computer Science
  - Mathematics
series: ["stepanov"]
series_weight: 10
math: true
description: "Numerical integration demonstrates both classical numerical analysis and Stepanov's philosophy: by identifying the minimal algebraic requirements, our quadrature routines work with dual numbers for automatic differentiation under the integral."
---

*Numerical integration (quadrature) for C++20.*

## Overview

The definite integral represents the signed area under a curve:

$$\int_a^b f(x)\,dx$$

Since most functions lack closed-form antiderivatives, we approximate integrals numerically using **quadrature rules**---weighted sums of function evaluations:

$$\int_a^b f(x)\,dx \approx \sum_i w_i f(x_i)$$

Different rules choose different nodes x_i and weights w_i, trading off accuracy and computational cost.

## Quick Start

```cpp
#include <integration/integrate.hpp>
#include <cmath>
#include <iostream>

int main() {
    using namespace integration;

    // integral from 0 to pi of sin(x) dx = 2
    double result = integrate([](double x) { return std::sin(x); }, 0.0, 3.14159265);

    std::cout << "integral of sin(x) dx = " << result << "\n";  // ~2.0
}
```

## The Quadrature Zoo

### Basic Rules

| Rule | Formula | Error | Exact for |
|------|---------|-------|-----------|
| Midpoint | \((b-a)f(m)\) | \(O(h^3)\) | Linear |
| Trapezoidal | \(\frac{b-a}{2}(f(a)+f(b))\) | \(O(h^3)\) | Linear |
| Simpson's | \(\frac{b-a}{6}(f(a)+4f(m)+f(b))\) | \(O(h^5)\) | Cubic |

```cpp
double m = midpoint_rule(f, a, b);
double t = trapezoidal_rule(f, a, b);
double s = simpsons_rule(f, a, b);
```

### Composite Rules

Divide [a,b] into n subintervals and apply the basic rule to each:

```cpp
// Error: O(h^2) where h = (b-a)/n
double m = composite_midpoint(f, a, b, 100);
double t = composite_trapezoidal(f, a, b, 100);

// Error: O(h^4) - much more accurate!
double s = composite_simpsons(f, a, b, 100);  // n must be even
```

### Gauss-Legendre Quadrature

The optimal choice: n points exactly integrate polynomials of degree 2n-1.

```cpp
// 5-point Gauss-Legendre: exact for degree <= 9
double g = gauss_legendre<5>(f, a, b);
```

### Adaptive Integration

Automatically refines where the function is "difficult":

```cpp
// Recommended for general use
double result = integrate(f, a, b);              // Default tolerance 1e-10
double result = integrate(f, a, b, 1e-12);       // Custom tolerance

// With error estimate
auto [value, error, evals] = integrate_with_error(f, a, b);
```

## Deriving Simpson's Rule

Taylor expand \(f(x)\) around the midpoint \(m = (a+b)/2\). Odd powers vanish by symmetry when integrating from \(a\) to \(b\):

$$\int_a^b f(x)\,dx \approx (b-a)f(m) + f''(m)\frac{(b-a)^3}{24} + O(h^5)$$

Simpson's rule is the unique combination of endpoint and midpoint values that cancels the \(h^2\) error:

$$\int_a^b f(x)\,dx = \frac{b-a}{6}\left[f(a) + 4f(m) + f(b)\right] + O(h^5)$$

Remarkably, this also cancels the h^3 term---Simpson gets a "bonus degree."

## Why Gauss-Legendre is Optimal

For \(n\) evaluation points, we have \(2n\) free parameters (\(n\) nodes + \(n\) weights). We can match \(2n\) conditions: exact integration of \(1, x, x^2, \ldots, x^{2n-1}\).

The solution: nodes are roots of the \(n\)-th Legendre polynomial \(P_n(x)\). These orthogonal polynomials arise naturally from the optimization.

## Generic Programming: The Stepanov Way

This module follows the Stepanov principle: **algorithms arise from minimal operations**.

### The Problem with `std::floating_point`

A naive implementation constrains all functions to `std::floating_point`:

```cpp
// BAD: Excludes custom numeric types
template<std::floating_point T, Integrand<T> F>
T midpoint_rule(F&& f, T a, T b);
```

This rejects perfectly valid types:
- `rational<int>` - exact fractions
- `dual<double>` - automatic differentiation
- `interval<double>` - interval arithmetic
- User-defined real number types

### The Stepanov Solution

Ask: "What operations does this algorithm ACTUALLY need?"

For numerical integration:
- Field operations: `+`, `-`, `*`, `/`, unary `-`
- Ordering: `<` (for adaptive methods)
- Small integer construction: `T(0)`, `T(1)`, `T(2)` (for weights)

We capture this as the `ordered_field` concept:

```cpp
template<typename T>
concept ordered_field =
    has_add<T> && has_sub<T> && has_mul<T> && has_div<T> &&
    has_neg<T> && has_order<T> &&
    requires { T(0); T(1); T(2); };

// GOOD: Works with ANY ordered field
template<typename T, Integrand<T> F>
    requires ordered_field<T>
T midpoint_rule(F&& f, T a, T b);
```

### Powerful Application: Differentiation Under the Integral

With dual numbers, you can compute derivatives of integrals automatically:

```cpp
#include <integration/integrate.hpp>
#include <dual/dual.hpp>

// Compute d/da integral from 0 to 1 of a*x^2 dx
using D = dual::dual<double>;

D a = D::variable(1.0);  // Parameter to differentiate
auto f = [&a](D x) { return a * x * x; };

D result = composite_simpsons(f, D::constant(0.0), D::constant(1.0), 100);

// result.value() = 1/3 (the integral)
// result.derivative() = 1/3 (d/da of the integral = integral of x^2 dx)
```

This is Leibniz's rule implemented through pure algebra!

## Choosing a Method

| Situation | Recommended Method |
|-----------|-------------------|
| General smooth function | `integrate()` (adaptive Simpson) |
| Polynomial-like function | `gauss_legendre<5>()` |
| Periodic on full period | `composite_trapezoidal()` |
| Endpoint singularity | `integrate_left_singularity()` |
| Semi-infinite domain | `integrate_semi_infinite()` |
| Need error estimate | `integrate_with_error()` |
