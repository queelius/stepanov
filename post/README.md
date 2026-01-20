# C++ Blog

Pedagogical explorations in generic programming and algorithmic mathematics.

Each post is a self-contained article with minimal, elegant C++ code (~100-300 lines) demonstrating one beautiful idea. Inspired by Alex Stepanov's approach: discover algorithms from the minimal operations a type provides.

## Posts

### Core Algorithms

#### [peasant/](peasant/) — One Algorithm, Infinite Powers

The Russian peasant algorithm computes multiplication using only `twice`, `half`, and `even`. The same pattern gives us exponentiation via repeated squaring. With 15 examples showing how one algorithm works across matrices (Fibonacci), quaternions (3D rotations), tropical semirings (shortest paths), and more.

**Key files:** `peasant.hpp` (~200 lines), 15 example monoids

---

#### [miller-rabin/](miller-rabin/) — Is It Prime?

Probabilistic primality testing with rigorous error bounds. Parameterized by maximum false positive rate rather than iteration count—ask for "error < 10⁻¹²" and the algorithm figures out how many witnesses to test.

**Key files:** `primality.hpp` (~100 lines)

---

### Algebraic Types

#### [rational/](rational/) — Exact Rational Arithmetic

Fractions that never lose precision. Unlike floating-point, 1/3 stays exactly 1/3. Demonstrates how GCD naturally arises for maintaining canonical form, and introduces the mediant operation for Stern-Brocot trees.

**Key files:** `rational.hpp` (~120 lines)

---

#### [modular/](modular/) — Finite Rings

Integers modulo N form a ring. When N is prime, it's a field—every non-zero element has an inverse. Demonstrates Fermat's little theorem for computing inverses via exponentiation.

**Key files:** `mod.hpp` (~100 lines)

---

#### [polynomials/](polynomials/) — Polynomial Arithmetic as a Euclidean Domain

The same GCD algorithm works for integers and polynomials—both are Euclidean domains. Sparse representation, polynomial division, and root finding.

**Key files:** `polynomial.hpp` (~400 lines)

---

### Linear Algebra & Autodiff

#### [elementa/](elementa/) — Pedagogical Linear Algebra

A generic matrix library where the Matrix concept defines what matrices can do. Expression templates for lazy evaluation, specialized structures (symmetric, triangular, diagonal), and algorithms that work across all matrix types.

**Key files:** `elementa.hpp` (~900 lines)

---

#### [dual/](dual/) — Forward-Mode Automatic Differentiation

Dual numbers are pairs (a, b) where ε² = 0. Evaluating f(x + ε) yields f(x) + ε·f'(x)—the derivative emerges from the algebra. Supports nested duals for higher derivatives and Taylor jets for arbitrary order.

**Key files:** `dual.hpp` (~450 lines across 7 files)

---

#### [autodiff/](autodiff/) — Reverse-Mode Automatic Differentiation

Build a computational graph, then backpropagate. More efficient than forward mode when computing gradients of scalar outputs with many inputs (like neural network loss functions).

**Key files:** `gradator.hpp` (~1000 lines)

---

### Numerical Methods

#### [finite-diff/](finite-diff/) — Numerical Differentiation

Finite difference formulas for derivatives when you only have function values. Forward, central, five-point stencils with optimal step sizes derived from error analysis. Gradients, Jacobians, and Hessians for multivariate functions.

**Key files:** `finite_diff.hpp` (~300 lines)

---

#### [integration/](integration/) — Numerical Quadrature

Approximate integrals as weighted sums of function values. From the humble trapezoidal rule to Gauss-Legendre (optimal polynomial integration) and adaptive Simpson's. Handles singularities and infinite intervals.

**Key files:** `integrate.hpp` (~370 lines)

---

### Design Patterns

#### [type-erasure/](type-erasure/) — Runtime Polymorphism Without Inheritance

Sean Parent's technique for value-semantic polymorphism. Store any "regular" type in a single container without requiring inheritance. Includes `any_regular` (general values) and `any_vector` (vector spaces).

**Key files:** `any_regular.hpp` (~150 lines), `any_vector.hpp` (~400 lines)

---

## Building

```bash
cd cpp-blog
cmake -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Philosophy

- **Minimal**: Each implementation is ~100-400 lines, no dependencies beyond the standard library
- **Pedagogical**: Code is written to teach, not to be production-ready
- **Self-contained**: Each post directory can be understood independently
- **Algebraic**: Algorithms arise from algebraic structure, not ad-hoc coding

## Future Posts

Ideas from the archive, retrievable via `git show 514f6ca:include/stepanov/<file>`:

- `continued_fraction.hpp` — Best rational approximations
- `fft.hpp` — FFT for any ring (NTT)
- `fenwick_tree.hpp` — The elegant Fenwick tree
- `gcd.hpp` — Binary GCD (Stein's algorithm)
