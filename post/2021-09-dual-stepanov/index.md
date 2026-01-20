---
author:
  name: "Alex Towell"
  email: "queelius@gmail.com"
  url: "https://metafunctor.com"
title: "Forward-Mode Automatic Differentiation"
date: 2021-09-20
draft: false
tags:
  - C++
  - automatic-differentiation
  - dual-numbers
  - calculus
  - machine-learning
categories:
  - Computer Science
  - Mathematics
series: ["stepanov"]
series_weight: 7
math: true
description: "Dual numbers extend our number system with an infinitesimal epsilon where epsilon^2 = 0. Evaluating f(x + epsilon) yields f(x) + epsilon * f'(x)---the derivative emerges automatically from the algebra."
---

*Forward-mode automatic differentiation via dual numbers for C++20.*

## Overview

Dual numbers are a simple yet powerful technique for computing exact derivatives. The key insight: if we extend our number system with an element `epsilon` where `epsilon^2 = 0`, then evaluating `f(x + epsilon)` yields `f(x) + epsilon * f'(x)`. The derivative emerges automatically from the algebra.

## Quick Start

```cpp
#include <dual/dual.hpp>
#include <iostream>

int main() {
    using namespace dual;

    // Create a dual variable at x = 2
    auto x = dual<double>::variable(2.0);

    // Compute f(x) = x^3 - 3x + 1
    auto f = x*x*x - 3.0*x + 1.0;

    std::cout << "f(2) = " << f.value() << "\n";       // 3.0
    std::cout << "f'(2) = " << f.derivative() << "\n"; // 9.0
}
```

## The Mathematics

A dual number has the form \(a + b\varepsilon\) where \(\varepsilon^2 = 0\). Arithmetic follows naturally:

$$(a + b\varepsilon) + (c + d\varepsilon) = (a+c) + (b+d)\varepsilon$$

$$(a + b\varepsilon)(c + d\varepsilon) = ac + (ad + bc)\varepsilon + bd\varepsilon^2 = ac + (ad + bc)\varepsilon$$

Notice how the \(bd\varepsilon^2\) term vanishes because \(\varepsilon^2 = 0\).

For a function \(f\), Taylor expansion gives:

$$f(a + b\varepsilon) = f(a) + bf'(a)\varepsilon + \frac{b^2}{2}f''(a)\varepsilon^2 + \cdots = f(a) + bf'(a)\varepsilon$$

If we set \(b = 1\) (marking \(x\) as "the variable we're differentiating with respect to"), then:

$$f(x + \varepsilon) = f(x) + f'(x)\varepsilon$$

The derivative appears as the coefficient of epsilon!

## API Reference

### dual<T>

The core dual number type.

```cpp
// Create a variable for differentiation
auto x = dual<double>::variable(3.0);  // x = 3, dx = 1

// Create a constant
auto c = dual<double>::constant(2.0);  // c = 2, dc = 0

// Access values
double val = x.value();       // 3.0
double deriv = x.derivative(); // 1.0

// Arithmetic operators: +, -, *, /
auto y = sin(x*x) + exp(-x);

// Convenience function
auto [value, deriv] = differentiate([](auto x) { return x*x; }, 3.0);
```

### Mathematical Functions

All standard math functions are supported with correct derivative propagation:

- **Basic**: `sqrt`, `cbrt`, `abs`
- **Exponential**: `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`
- **Trigonometric**: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`
- **Hyperbolic**: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
- **Power**: `pow`, `hypot`
- **Special**: `erf`, `erfc`

### Higher-Order Derivatives

**Second derivatives with dual2:**

```cpp
auto result = differentiate2([](auto x) { return sin(x); }, 1.0);
// result.value  = sin(1)
// result.first  = cos(1)
// result.second = -sin(1)
```

**Arbitrary order with jets:**

```cpp
// Compute f, f', f'', f''', f'''' at x = 1
auto derivs = derivatives<4>([](auto x) { return exp(x); }, 1.0);
// All derivatives of e^x at x=1 equal e
```

## Forward vs Reverse Mode

This library implements **forward mode** AD:

- **Pros**: Simple, no memory overhead, exact derivatives
- **Cons**: Cost is O(n) for n input variables

For functions with many inputs and few outputs (like neural network loss functions), **reverse mode** (backpropagation) is more efficient.

| Use Case | Best Mode |
|----------|-----------|
| f: R -> R^n (one input, many outputs) | Forward |
| f: R^n -> R (many inputs, one output) | Reverse |
| Jacobian-vector products | Forward |
| Vector-Jacobian products | Reverse |

## Why Dual Numbers Matter

Forward-mode AD via dual numbers is:

1. **Exact**: No truncation error like finite differences
2. **Simple**: Just overload arithmetic operators
3. **Composable**: Chain rule is automatic via operator overloading
4. **Generic**: Works with any function written in terms of overloaded operators

The algebraic structure (dual numbers form a ring) determines the algorithm's correctness. This is Stepanov's insight applied to calculus.
