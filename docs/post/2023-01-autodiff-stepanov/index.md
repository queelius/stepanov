---
title: "Reverse-Mode Automatic Differentiation"
date: 2023-01-17
draft: false
tags:
  - C++
  - automatic-differentiation
  - backpropagation
  - machine-learning
  - gradients
categories:
  - Computer Science
  - Mathematics
series: ["stepanov"]
series_weight: 9
math: true
description: "Reverse-mode automatic differentiation powers modern machine learning. Understanding how it works demystifies PyTorch, JAX, and TensorFlow---it's just the chain rule applied systematically."
---

*Demystifying automatic differentiation: building a reverse-mode AD system in C++20.*

Automatic differentiation powers modern machine learning. PyTorch's `autograd`, JAX's `jit`, TensorFlow's `GradientTape`---all rely on the same fundamental insight: we can compute exact derivatives by mechanically applying the chain rule.

## The Functional API

The core interface is a single function: `grad`. It takes a function and returns its gradient function:

```cpp
auto f = [](const auto& x) {
    return sum(pow(x, 2.0));  // f(x) = sum(x^2)
};

auto df = grad(f);  // df(x) = 2x

matrix<double> x{{1.0}, {2.0}, {3.0}};
auto gradient = df(x);  // Returns {{2.0}, {4.0}, {6.0}}
```

Because `grad` returns a callable, you can compose it:

```cpp
auto d2f = grad(grad(f));  // Second derivative
```

No classes to instantiate, no global state to manage. Just functions transforming functions.

## How Reverse-Mode AD Works

The `grad` function reveals the algorithm:

```cpp
template <typename F>
auto grad(F&& f) {
    return [f = std::forward<F>(f)]<typename T>(const T& input) {
        graph g;                          // 1. Fresh graph
        auto x = g.make_var(input);       // 2. Create input variable
        auto y = f(x);                    // 3. Forward pass (builds graph)
        g.backward<output_type>(y);       // 4. Backward pass
        return g.gradient(x);             // 5. Extract gradient
    };
}
```

**Forward pass**: When `f(x)` executes, each operation creates a node in the graph. The node stores:
- The computed value
- References to parent nodes
- A *backward function* implementing that operation's Jacobian

**Backward pass**: Starting from the output, we traverse the graph in reverse topological order. Each node's backward function receives the upstream gradient and propagates it to parents via the chain rule.

## Implementing Jacobians

Each differentiable operation provides a backward function. Let's examine three matrix calculus results.

### Matrix Multiplication: C = AB

If the loss is L, and C appears in L, we need dL/dA and dL/dB given dL/dC.

The chain rule gives us:
- \(\partial L/\partial A = (\partial L/\partial C) B^T\)
- \(\partial L/\partial B = A^T (\partial L/\partial C)\)

### Determinant: \(d = \det(A)\)

This is a classic matrix calculus result:

$$\frac{\partial \det(A)}{\partial A} = \det(A) \cdot A^{-T}$$

where \(A^{-T}\) denotes the transpose of the inverse.

### Matrix Inverse: \(B = A^{-1}\)

Starting from \(A \cdot A^{-1} = I\) and differentiating implicitly:

$$dA \cdot A^{-1} + A \cdot dA^{-1} = 0$$
$$dA^{-1} = -A^{-1} \cdot dA \cdot A^{-1}$$

Applying the chain rule to a scalar loss \(L\) through \(B = A^{-1}\):

$$\frac{\partial L}{\partial A} = -A^{-T} \left(\frac{\partial L}{\partial B}\right) A^{-T}$$

## Gradient Accumulation

When a variable appears multiple times in a computation, gradients accumulate:

```cpp
void accumulate_gradient(node_id id, const T& grad) {
    auto& n = nodes_.at(id.index);
    if (!n.gradient.has_value()) {
        n.gradient = grad;
    } else {
        auto& existing = std::any_cast<T&>(n.gradient);
        existing = existing + grad;
    }
}
```

For example, if `y = x + x`, then dy/dx = 2, not 1. Each use of `x` contributes its gradient.

## Testing AD with Finite Differences

How do we know our Jacobians are correct? We verify against finite differences---the computational equivalent of the limit definition of a derivative:

```
f'(x) ~ [f(x + h) - f(x - h)] / 2h
```

This central difference formula has O(h^2) error, better than the one-sided difference.

Tests compare AD gradients against finite differences:

```cpp
TEST_CASE("matmul gradient") {
    auto f = [](const auto& A) {
        return sum(matmul(A, A));  // f(A) = sum(A^2)
    };

    matrix<double> A{{1, 2}, {3, 4}};
    auto ad_grad = grad(f)(A);
    auto fd_grad = finite_diff_gradient([&](const auto& x) {
        return sum(elementa::matmul(x, x));
    }, A);

    REQUIRE(approx_equal(ad_grad, fd_grad, 1e-4, 1e-6));
}
```

Finite differences are slow (O(n) function evaluations for n parameters) and approximate. AD is exact and efficient (O(1) backward passes). But finite differences don't lie---they're the ground truth for testing.

## Practical Example: Gradient Descent

Here's logistic regression training:

```cpp
// Binary cross-entropy loss
auto bce_loss = [&X, &y](const auto& beta) {
    auto logits = matmul(X, beta);
    auto p = sigmoid(logits);
    return -mean(y * log(p) + (ones_like(y) - y) * log(ones_like(p) - p));
};

auto grad_bce = grad(bce_loss);

// Training loop
matrix<double> beta(features, 1, 0.0);  // Initialize weights
for (int iter = 0; iter < 1000; ++iter) {
    auto gradient = grad_bce(beta);
    beta = beta - gradient * learning_rate;
}
```

The gradient computation is automatic. Change the loss function, and `grad` adapts.

## Conclusion

Automatic differentiation is not magic. It's systematic application of the chain rule, implemented through:

1. **Graph construction**: Operations build a DAG during the forward pass
2. **Backward traversal**: Reverse topological order ensures correct dependency ordering
3. **Gradient accumulation**: Variables used multiple times sum their gradients

Understanding this makes PyTorch and JAX less mysterious. When `.backward()` runs, it's doing exactly what we've described: traversing the graph, calling backward functions, accumulating gradients.
