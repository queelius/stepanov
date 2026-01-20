---
author:
  name: "Alex Towell"
  email: "queelius@gmail.com"
  url: "https://metafunctor.com"
title: "Differentiation: Three Ways"
date: 2025-01-15
draft: false
tags:
  - C++
  - automatic-differentiation
  - numerical-methods
  - calculus
  - machine-learning
  - dual-numbers
categories:
  - Computer Science
  - Mathematics
series: ["stepanov"]
series_weight: 12
math: true
description: "Three approaches to computing derivatives---forward-mode AD, reverse-mode AD, and finite differences---each with different trade-offs. Understanding when to use each is essential for numerical computing and machine learning."
featured: true
---

*A synthesis of three earlier posts, comparing forward-mode AD, reverse-mode AD, and numerical differentiation.*

Computing derivatives is fundamental to optimization, machine learning, physics simulation, and numerical analysis. This series has explored three distinct approaches:

1. **Forward-mode AD** via dual numbers
2. **Reverse-mode AD** via computational graphs
3. **Numerical differentiation** via finite differences

Each has different strengths. Understanding when to use each is essential for practical numerical computing.

## The Landscape

| Method | Accuracy | Cost for \(f: \mathbb{R}^n \to \mathbb{R}\) | Cost for \(f: \mathbb{R} \to \mathbb{R}^m\) | Memory |
|--------|----------|----------------------|----------------------|--------|
| Forward AD | Exact | \(O(n)\) passes | \(O(1)\) pass | \(O(1)\) |
| Reverse AD | Exact | \(O(1)\) pass | \(O(m)\) passes | \(O(\text{ops})\) |
| Finite Diff | \(O(h^p)\) | \(O(n)\) evaluations | \(O(n)\) evaluations | \(O(1)\) |

The key insight: **problem structure determines the best method**.

## Forward-Mode AD: Dual Numbers

Forward-mode AD extends numbers with an infinitesimal \(\varepsilon\) where \(\varepsilon^2 = 0\). The derivative emerges from arithmetic:

```cpp
// f(x) = x^3 - 3x + 1
// f'(x) = 3x^2 - 3

auto x = dual<double>::variable(2.0);  // x = 2, dx = 1
auto f = x*x*x - 3.0*x + 1.0;

std::cout << f.value() << "\n";       // 3.0
std::cout << f.derivative() << "\n";  // 9.0
```

**Strengths:**
- Simple implementation (operator overloading)
- No memory overhead
- Naturally composable for higher derivatives
- Works with any function of overloaded operators

**When to use:**
- Single input variable (or few inputs)
- Computing Jacobian-vector products
- Higher-order derivatives via nesting
- Sensitivity analysis along one direction

**Complexity**: One forward pass per input variable. For f: R^n -> R^m, computing the full Jacobian requires n passes.

## Reverse-Mode AD: Computational Graphs

Reverse-mode AD builds a computational graph during the forward pass, then propagates gradients backward via the chain rule:

```cpp
auto f = [](const auto& x) {
    return sum(pow(x, 2.0));  // f(x) = sum(x^2)
};

auto df = grad(f);  // Returns gradient function
auto gradient = df(x);  // One backward pass for all partials
```

**Strengths:**
- O(1) backward passes regardless of input dimension
- Powers modern deep learning (backpropagation)
- Efficient for loss functions: f: R^n -> R

**When to use:**
- Many inputs, scalar output (neural networks)
- Computing vector-Jacobian products
- Optimization where you need full gradient

**Complexity**: One forward pass to build graph, one backward pass to compute all gradients. Memory scales with number of operations (must store intermediate values).

## Numerical Differentiation: Finite Differences

Approximate the derivative using the limit definition:

```cpp
// Central difference: f'(x) ~ (f(x+h) - f(x-h)) / 2h
double df = central_difference(f, x);
```

**Strengths:**
- Works with black-box functions
- No special types required
- Simple to implement and understand

**Limitations:**
- Approximate (truncation error + round-off error)
- Requires careful step size selection
- O(n) function evaluations for gradient

**When to use:**
- Black-box functions (no source code)
- Quick approximations
- **Validating AD implementations**

## Decision Tree

```
Need derivatives?
|
+-- Source code available?
|   |
|   +-- Yes: Use AD
|   |   |
|   |   +-- f: R -> R^m (few inputs, many outputs)?
|   |   |   --> Forward-mode AD
|   |   |
|   |   +-- f: R^n -> R (many inputs, scalar output)?
|   |   |   --> Reverse-mode AD
|   |   |
|   |   +-- f: R^n -> R^m (both)?
|   |       --> Forward for m < n, Reverse for n < m
|   |
|   +-- No: Black-box function
|       --> Finite differences
|
+-- Validating correctness?
    --> Finite differences (ground truth)
```

## Side-by-Side Comparison

Consider f(x,y,z) = x^2 + y^2 + z^2, evaluated at (1, 2, 3).

### Forward-Mode (3 passes for full gradient)

```cpp
// Pass 1: differentiate w.r.t. x
auto dx = dual<double>::variable(1.0);
auto y1 = dual<double>::constant(2.0);
auto z1 = dual<double>::constant(3.0);
auto f1 = dx*dx + y1*y1 + z1*z1;
// f1.derivative() = 2 (df/dx)

// Pass 2: differentiate w.r.t. y
// Pass 3: differentiate w.r.t. z
```

### Reverse-Mode (1 pass for full gradient)

```cpp
auto f = [](const auto& v) {
    return v(0)*v(0) + v(1)*v(1) + v(2)*v(2);
};
auto gradient = grad(f)(vector{1.0, 2.0, 3.0});
// gradient = {2, 4, 6} in one backward pass
```

### Finite Differences (3 evaluations for gradient)

```cpp
// df/dx ~ (f(1+h,2,3) - f(1-h,2,3)) / 2h
// df/dy ~ (f(1,2+h,3) - f(1,2-h,3)) / 2h
// df/dz ~ (f(1,2,3+h) - f(1,2,3-h)) / 2h
auto gradient = finite_diff::gradient(f, {1.0, 2.0, 3.0});
```

## Combining Methods: Validation

The gold standard for testing AD: compare against finite differences.

```cpp
auto ad_grad = grad(f)(x);
auto fd_grad = finite_diff::gradient(f, x);

// Should match within tolerance
REQUIRE(approx_equal(ad_grad, fd_grad, 1e-4, 1e-6));
```

Finite differences are slow and approximate, but they don't lie. If your AD implementation disagrees with finite differences, the AD is wrong.

## Complexity Analysis

For \(f: \mathbb{R}^n \to \mathbb{R}^m\) with \(p\) elementary operations:

| Method | Time | Space | Function Evals |
|--------|------|-------|----------------|
| Forward AD | \(O(np)\) | \(O(1)\) | \(n\) |
| Reverse AD | \(O(mp)\) | \(O(p)\) | \(m\) |
| Finite Diff | \(O(n)\) | \(O(1)\) | \(2n\) (central) |

The crossover point:
- **\(n < m\)**: Forward-mode wins
- **\(n > m\)**: Reverse-mode wins
- **\(n \approx m\)**: Either works; consider memory constraints

For neural networks with millions of parameters and scalar loss, reverse mode is the only practical choice.

## Integration with Numerical Quadrature

A beautiful application: differentiation under the integral sign.

```cpp
// Compute d/da of integral from 0 to 1 of a*sin(x) dx
using D = dual<double>;

D a = D::variable(2.0);
auto integrand = [&a](D x) { return a * sin(x); };

D result = integrate(integrand, D::constant(0.0), D::constant(1.0));

// result.value() = 2 * (1 - cos(1)) ~ 0.919
// result.derivative() = (1 - cos(1)) ~ 0.459
```

Forward-mode AD composes naturally with generic numerical algorithms because dual numbers form a ring---they satisfy the algebraic requirements.

## Conclusion

Three methods, three use cases:

1. **Forward-mode AD**: Simple, memory-efficient, ideal for few inputs
2. **Reverse-mode AD**: Powers deep learning, ideal for many inputs
3. **Finite differences**: Universal fallback, validation tool

The Stepanov perspective: each method exploits different structure. Forward AD uses the ring structure of dual numbers. Reverse AD uses the DAG structure of computation. Finite differences use only function evaluation.

Understanding this structure helps you choose wisely---and implement correctly.

## Further Reading

- [Forward-Mode Automatic Differentiation](/computer-science/stepanov/2021-09-dual/) - Dual numbers in detail
- [Reverse-Mode Automatic Differentiation](/computer-science/stepanov/2023-01-autodiff/) - Building a computational graph
- [Numerical Differentiation](/computer-science/stepanov/2022-04-finite-diff/) - Finite difference schemes
- [Numerical Integration with Generic Concepts](/computer-science/stepanov/2023-08-integration/) - Leibniz rule via dual numbers
