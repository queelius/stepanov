# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Stepanov is a collection of pedagogical blog posts exploring generic programming and algorithmic mathematics, inspired by Alex Stepanov's principles. Each post is a self-contained article with minimal, elegant C++ code demonstrating one beautiful idea.

## Build Commands

```bash
# Configure and build
cd post
cmake -B build
cmake --build build

# Run all tests
ctest --test-dir build --output-on-failure

# Run a single test
./build/test_peasant
./build/test_primality
./build/test_dual
```

## Posts

All posts live in `post/`. Each has `index.md` (article), `*.hpp` (implementation), and tests.

| Post | Topic |
|------|-------|
| `peasant/` | Russian peasant algorithm, exponentiation, 15 monoid examples |
| `miller-rabin/` | Probabilistic primality testing |
| `rational/` | Exact fraction arithmetic, GCD |
| `modular/` | Integers mod N as rings |
| `type-erasure/` | Sean Parent's value-semantic polymorphism |
| `elementa/` | Pedagogical linear algebra, expression templates |
| `dual/` | Forward-mode autodiff via dual numbers |
| `autodiff/` | Reverse-mode autodiff (depends on elementa) |
| `polynomials/` | Polynomial arithmetic, Euclidean domains |
| `finite-diff/` | Numerical differentiation |
| `integration/` | Numerical quadrature (works with dual numbers) |

## Core Design Pattern

Algorithms arise from **algebraic structure**. Types satisfy concepts by providing free functions discoverable via ADL:

```cpp
// Operations for the algebraic concept in peasant.hpp
zero(x)       // additive identity
one(x)        // multiplicative identity
twice(x)      // doubling
half(x)       // halving
even(x)       // parity test
```

Any type providing these operations works with `power()`, `product()`, etc.

## Code Style

- **Identifiers**: `snake_case` for all names
- **Template parameters**: `PascalCase` (e.g., `EuclideanDomain`)
- **Generic algorithms**: Prefer free functions over member functions
- **Operations**: Types expose algebraic operations as free functions for ADL
- **Constexpr**: Use `constexpr` where mathematically meaningful
- **C++ Standard**: C++23

## Philosophy

- **Minimal**: Each implementation is ~100-400 lines
- **Pedagogical**: Code teaches principles, not production patterns
- **Self-contained**: Each post directory stands alone
- **Algebraic**: Structure determines algorithms (Stepanov's insight)
