---
title: "Polynomials as Euclidean Domains"
date: 2020-07-14
draft: false
tags:
  - C++
  - generic-programming
  - algebra
  - polynomials
  - euclidean-domain
categories:
  - Computer Science
  - Mathematics
series: ["stepanov"]
series_weight: 5
math: true
description: "The same GCD algorithm works for integers and polynomials because both are Euclidean domains. This profound insight shows how algebraic structure determines algorithmic applicability."
---

*Polynomial arithmetic teaches a profound truth: **the same GCD algorithm works for integers and polynomials**.*

## Overview

Both integers and polynomials are examples of Euclidean domains---algebraic structures where division with remainder is always possible.

```cpp
// For integers: gcd(48, 18) = 6
// For polynomials: gcd(x^3 - 1, x^2 - 1) = x - 1

// Same algorithm, different types!
template<euclidean_domain E>
E gcd(E a, E b) {
    while (b != E(0)) {
        a = std::exchange(b, a % b);
    }
    return a;
}
```

## Quick Start

```cpp
#include <polynomials/polynomial.hpp>
#include <iostream>

using namespace poly;

int main() {
    // Create polynomial x^2 - 1 = (x-1)(x+1)
    auto p = polynomial<double>{-1, 0, 1};

    // Create polynomial x^3 - 1 = (x-1)(x^2+x+1)
    auto q = polynomial<double>{-1, 0, 0, 1};

    // GCD should be (x - 1)
    auto g = gcd(p, q);

    std::cout << "gcd(x^2-1, x^3-1) has degree " << g.degree() << "\n";  // 1

    // Find roots of x^2 - 1
    auto roots = find_roots(p, -10.0, 10.0);
    for (double r : roots) {
        std::cout << "Root: " << r << "\n";  // -1 and 1
    }
}
```

## API Reference

### Creating Polynomials

```cpp
// From dense coefficients (a[i] = coefficient of x^i)
polynomial<double> p{1, -2, 1};  // 1 - 2x + x^2

// Monomial: coefficient * x^degree
auto m = polynomial<double>::monomial(3.0, 4);  // 3x^4

// The variable x
auto x = polynomial<double>::x();  // x

// Constant
polynomial<double> c{5.0};  // 5
```

### Arithmetic

```cpp
auto sum = p + q;
auto diff = p - q;
auto prod = p * q;
auto [quot, rem] = divmod(p, q);  // Division with remainder
auto quot_only = p / q;
auto rem_only = p % q;
```

### GCD and Related

```cpp
auto g = gcd(p, q);                    // Greatest common divisor
auto [g, s, t] = extended_gcd(p, q);   // Bezout: g = p*s + q*t
auto l = lcm(p, q);                    // Least common multiple
bool d = divides(p, q);                // Does p divide q?
```

### Evaluation and Calculus

```cpp
double val = evaluate(p, x);           // p(x)
auto dp = derivative(p);               // p'(x)
auto integral = antiderivative(p);     // integral of p
auto roots = find_roots(p, -10, 10);   // All real roots in interval
auto crit = stationary_points(p, -10, 10);  // Where p'(x) = 0
```

## The Euclidean Domain Insight

What makes polynomials special is that they form a **Euclidean domain**:

| Property | Integers | Polynomials |
|----------|----------|-------------|
| Norm | abs(n) | degree(p) |
| Division | a = b*q + r, abs(r) < abs(b) | a = b*q + r, deg(r) < deg(b) |
| GCD | gcd(48, 18) = 6 | gcd(x^2-1, x-1) = x-1 |

The same Euclidean algorithm works for both because they share this structure.

## Sparse Representation

Polynomials are stored as sorted (degree, coefficient) pairs. This is efficient for sparse polynomials like `x^1000 + 1` (only 2 terms stored, not 1001).

## Why This Matters

The Euclidean domain insight exemplifies Stepanov's philosophy: **algorithms arise from algebraic structure**. When you recognize that polynomials and integers share the same abstract structure, you immediately know that:

- GCD works
- Extended GCD works (Bezout's identity)
- Unique factorization holds
- The Chinese Remainder Theorem applies

One structure, many types, same algorithms.
