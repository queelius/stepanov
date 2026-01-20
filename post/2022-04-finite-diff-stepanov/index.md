---
author:
  name: "Alex Towell"
  email: "queelius@gmail.com"
  url: "https://metafunctor.com"
title: "Numerical Differentiation"
date: 2022-04-12
draft: false
tags:
  - C++
  - numerical-methods
  - differentiation
  - error-analysis
  - calculus
categories:
  - Computer Science
  - Mathematics
series: ["stepanov"]
series_weight: 8
math: true
description: "The art of numerical differentiation lies in choosing step size h wisely---small enough that the approximation is good, but not so small that floating-point errors dominate."
---

*Numerical differentiation via finite differences for C++20.*

## Overview

The derivative is defined as a limit:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

We cannot take h = 0 on a computer, but we can choose small h. The art of numerical differentiation lies in choosing h wisely---small enough that the approximation is good, but not so small that floating-point errors dominate.

## Quick Start

```cpp
#include <finite-diff/finite_diff.hpp>
#include <cmath>
#include <iostream>

int main() {
    using namespace finite_diff;

    // Derivative of sin(x) at x = 1
    auto f = [](double x) { return std::sin(x); };

    double x = 1.0;
    double df = central_difference(f, x);  // Uses optimal h

    std::cout << "f'(1) = " << df << "\n";         // ~0.5403
    std::cout << "cos(1) = " << std::cos(1.0) << "\n";  // 0.5403...
}
```

## The Two Errors

Every finite difference approximation has two sources of error:

1. **Truncation error**: From the Taylor series approximation. Decreases as \(h \to 0\).
2. **Round-off error**: From floating-point arithmetic. Increases as \(h \to 0\).

$$\text{Total error} = \text{Truncation} + \text{Round-off} = O(h^p) + O(\varepsilon/h)$$

where \(\varepsilon \approx 2.2 \times 10^{-16}\) (machine epsilon for `double`).

The optimal \(h\) minimizes total error:

| Method | Truncation | Optimal \(h\) | Typical accuracy |
|--------|-----------|-----------|------------------|
| Forward | \(O(h)\) | \(\sqrt{\varepsilon}\) | ~8 digits |
| Central | \(O(h^2)\) | \(\varepsilon^{1/3}\) | ~10 digits |
| Five-point | \(O(h^4)\) | \(\varepsilon^{1/5}\) | ~12 digits |

## API Reference

### First Derivatives

```cpp
// Forward difference: (f(x+h) - f(x)) / h
// O(h) accuracy
double df = forward_difference(f, x);

// Backward difference: (f(x) - f(x-h)) / h
// O(h) accuracy
double df = backward_difference(f, x);

// Central difference: (f(x+h) - f(x-h)) / (2h)
// O(h^2) accuracy - the workhorse
double df = central_difference(f, x);

// Five-point stencil: higher accuracy formula
// O(h^4) accuracy
double df = five_point_stencil(f, x);
```

### Second Derivatives

```cpp
// Second derivative: (f(x+h) - 2f(x) + f(x-h)) / h^2
double d2f = second_derivative(f, x);

// Five-point second derivative: O(h^4)
double d2f = second_derivative_five_point(f, x);
```

### Multivariate Functions

```cpp
// Gradient: grad(f)(x) for f: R^n -> R
auto grad = gradient(f, x);

// Directional derivative: D_v f(x)
double Dvf = directional_derivative(f, x, v);

// Hessian matrix: H[i][j] = d^2f/dx_i dx_j
auto H = hessian(f, x);

// Jacobian matrix for f: R^n -> R^m
auto J = jacobian(f, x);
```

## Deriving the Formulas

### Forward Difference

Taylor expand \(f(x+h)\):

$$f(x+h) = f(x) + hf'(x) + \frac{h^2}{2}f''(x) + O(h^3)$$

Solve for \(f'(x)\):

$$f'(x) = \frac{f(x+h) - f(x)}{h} - \frac{h}{2}f''(x) + O(h^2) = \frac{f(x+h) - f(x)}{h} + O(h)$$

### Central Difference

Taylor expand \(f(x+h)\) and \(f(x-h)\):

$$f(x+h) = f(x) + hf'(x) + \frac{h^2}{2}f''(x) + \frac{h^3}{6}f'''(x) + O(h^4)$$

$$f(x-h) = f(x) - hf'(x) + \frac{h^2}{2}f''(x) - \frac{h^3}{6}f'''(x) + O(h^4)$$

Subtract:

$$f(x+h) - f(x-h) = 2hf'(x) + \frac{h^3}{3}f'''(x) + O(h^5)$$

$$f'(x) = \frac{f(x+h) - f(x-h)}{2h} + O(h^2)$$

The even-order error terms cancel, doubling our accuracy!

### Five-Point Stencil

By taking linear combinations of more points, we can cancel more error terms:

$$f'(x) = \frac{-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)}{12h}$$

This achieves \(O(h^4)\) by canceling the \(h^2\), \(h^3\), and \(h^4\) error terms.

## Optimal Step Size Derivation

For central difference with round-off:

$$\text{Error} = \frac{h^2}{6}|f'''(x)| + \frac{\varepsilon}{h}|f(x)|$$

Minimize by taking derivative with respect to \(h\) and setting to zero:

$$\frac{d(\text{Error})}{dh} = \frac{h}{3}|f'''| - \frac{\varepsilon}{h^2}|f| = 0$$

$$h^3 = \frac{3\varepsilon|f|}{|f'''|} \sim \varepsilon \quad \text{(order of magnitude)}$$

$$h_{\text{opt}} \sim \varepsilon^{1/3}$$

For double precision: \(h \approx 6 \times 10^{-6}\).

## Comparison with Automatic Differentiation

| Aspect | Finite Differences | Automatic Differentiation |
|--------|-------------------|--------------------------|
| Accuracy | Limited by eps^(1/p) | Exact to machine precision |
| Code complexity | Works with any function | Requires special types |
| Computational cost | n+1 or 2n evaluations | 2-3x single evaluation |
| Black-box functions | Yes | No |
| Higher derivatives | Error compounds | Straightforward |

**Use finite differences when:**
- Function is a black box (no source code)
- Quick approximation needed
- Validating other methods

**Use automatic differentiation when:**
- Accuracy is critical
- Computing many derivatives
- Source code is available
