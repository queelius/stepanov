# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Stepanov is a pedagogical monorepo exploring generic programming and algorithmic mathematics, inspired by Alex Stepanov's principle: **algorithms arise from algebraic structure**. It has two layers:

1. **`post/`** — Self-contained blog posts, each demonstrating one idea with minimal C++ code
2. **`lib/`** — Extracted libraries from those posts, organized as a proper C++20 monorepo

## Build Commands

There are **two independent build systems**:

### Blog posts (`post/`)
```bash
cmake -B post/build -S post
cmake --build post/build
ctest --test-dir post/build --output-on-failure

# Single test
./post/build/test_peasant
./post/build/test_dual
```

### Libraries (`lib/`)
```bash
cmake -B build
cmake --build build
ctest --test-dir build --output-on-failure

# Options
cmake -B build -DSTEPANOV_BUILD_TESTS=OFF
cmake -B build -DSTEPANOV_ENABLE_OPENMP=ON  # for limes
```

## Architecture

### Shared concepts layer (`concepts/`)

Pure C++20 concept definitions for algebraic structures, shared by all libraries. No implementation — just type constraints:

- `algebra.hpp` — AdditiveMonoid, Ring, Field, EuclideanDomain, etc.
- `matrix.hpp` — Matrix concept
- `random.hpp` — Random engine concept

Operations are ADL-discoverable free functions: `zero(x)`, `one(x)`, `twice(x)`, `half(x)`, `even(x)`.

### Libraries (`lib/`)

| Library | Description | Depends on |
|---------|-------------|------------|
| **elementa** | Linear algebra, expression templates, matrix concepts | — |
| **gradator** | Reverse-mode autodiff via computational graphs | elementa |
| **polynomials** | Polynomial arithmetic, Euclidean domain algorithms | — |
| **limes** | Numerical analysis: integration, differentiation, symbolic expressions | — |
| **alea** | Probability and random sampling | elementa |
| **alga** | Algebraic parser combinators | — |

Libraries use **Catch2 v3** (elementa, gradator, alea, alga) or **GoogleTest** (polynomials, limes).

### Blog posts (`post/`)

Post directories use the naming convention `YYYY-MM-topic-stepanov/`. Each has `index.md` (article), `*.hpp` (implementation), and test files. Posts use **C++23** and **GoogleTest**.

| Directory | Topic | Dependencies |
|-----------|-------|--------------|
| `2019-03-peasant-stepanov/` | Russian peasant algorithm, 15 monoid examples | — |
| `2019-06-modular-stepanov/` | Integers mod N as rings | — |
| `2019-09-miller-rabin-stepanov/` | Probabilistic primality testing | — |
| `2020-02-rational-stepanov/` | Exact fraction arithmetic, GCD | — |
| `2020-07-polynomials-stepanov/` | Polynomial arithmetic, Euclidean domains | — |
| `2021-03-elementa-stepanov/` | Pedagogical linear algebra | — |
| `2021-09-dual-stepanov/` | Forward-mode autodiff via dual numbers | — |
| `2022-04-finite-diff-stepanov/` | Numerical differentiation | — |
| `2023-01-autodiff-stepanov/` | Reverse-mode autodiff (gradator) | elementa |
| `2023-08-integration-stepanov/` | Numerical quadrature | dual |
| `2024-02-type-erasure-stepanov/` | Sean Parent's value-semantic polymorphism | — |
| `2025-01-differentiation-stepanov/` | Article only (no code) | — |
| `2026-01-18-synthesis-stepanov/` | Article only (no code) | — |
| `2026-01-19-duality-stepanov/` | Article only (no code) | — |
| `2026-03-accumulator-stepanov/` | Online statistics as monoids, parallel composition | — |
| `2026-03-semiring-stepanov/` | Semirings, graph algorithms via matrix power | — |
| `2026-03-lattice-stepanov/` | Lattices, fixed-point iteration, abstract interpretation | — |
| `2026-03-homomorphism-stepanov/` | Structure-preserving maps, fold as universal homomorphism | — |
| `2026-03-free-algebra-stepanov/` | Free monoids, universal property, fold as theorem | — |

## Core Design Pattern

Types satisfy concepts by providing free functions discoverable via ADL:

```cpp
zero(x)       // additive identity
one(x)        // multiplicative identity
twice(x)      // doubling
half(x)       // halving
even(x)       // parity test
```

Any type providing these operations works with `power()`, `product()`, etc. This pattern recurs across the entire repo — from peasant multiplication to polynomial GCD to dual number differentiation.

## Code Style

- **Identifiers**: `snake_case` for all names
- **Template parameters**: `PascalCase` (e.g., `EuclideanDomain`)
- **Generic algorithms**: Free functions over member functions, discoverable via ADL
- **Constexpr**: Where mathematically meaningful
- **C++ standard**: C++20 for `lib/`, C++23 for `post/`

## Philosophy

- **Minimal**: Each implementation is ~100-400 lines
- **Pedagogical**: Code teaches principles, not production patterns
- **Self-contained**: Each post directory stands alone (except noted dependencies)
- **Algebraic**: Structure determines algorithms (Stepanov's insight)
