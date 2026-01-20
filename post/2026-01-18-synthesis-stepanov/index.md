---
author:
  name: "Alex Towell"
  email: "queelius@gmail.com"
  url: "https://metafunctor.com"
title: "Seeing Structure First"
date: 2026-01-18
draft: false
tags:
  - C++
  - generic-programming
  - algorithms
  - monoids
  - algebraic-structures
  - stepanov
categories:
  - Computer Science
  - Mathematics
series: ["stepanov"]
series_weight: 0
math: true
description: "A reflection on eleven explorations in generic programming—how algorithms arise from algebraic structure."
featured: true
---

*A reflection on eleven explorations in generic programming*

## The Question Behind the Code

What do these computations have in common?

- Computing the millionth Fibonacci number
- Finding the shortest path between cities in a weighted graph
- Calculating compound interest over thirty years
- Composing ten 3D rotations into one
- Repeating a string n times

The answer: they're all computed by the same twenty lines of code. Here it is:

```cpp
template<typename T>
constexpr T power(T const& base, T exp) {
    if (exp == zero(exp)) return one(exp);
    if (exp == one(exp))  return base;

    return even(exp)
        ? square(power(base, half(exp)))
        : product(base, power(base, decrement(exp)));
}
```

This shouldn't work. Fibonacci numbers involve integer sequences. Shortest paths involve graphs. Rotations involve 3D geometry. These are different domains with different mathematics.

Yet they share structure. Once you see it, a single algorithm serves them all.

This collection of eleven blog posts is an extended meditation on one idea: **algorithms arise from algebraic structure**. The posts explore different domains—number theory, calculus, linear algebra, polymorphism—but they circle the same insight. Recognize the structure; the algorithm follows.

---

## The Principle

Alex Stepanov articulated this most clearly in *Elements of Programming*: "Generic programming is about abstracting and classifying algorithms and data structures." But the deeper point is *how* to abstract. Not by common syntax or superficial similarity, but by the algebraic laws a type obeys.

Why does structure appear *everywhere*? Because reality has structure. The algebraic structures we discover in programming---groups, rings, monoids---are the same structures physicists discover in nature. Rotations form a group. Spacetime transformations form a group. This isn't coincidence; we're uncovering patterns that exist.

Noether's theorem makes this precise: every continuous symmetry corresponds to a conservation law. Time-translation symmetry gives conservation of energy. Space-translation symmetry gives conservation of momentum. Rotational symmetry gives conservation of angular momentum. The symmetry groups of physics *are* algebraic structures.

When we recognize "this is a monoid" in our code, we're tapping into the same mathematical substrate that governs physical law. The algorithms follow because the structure constrains what's possible---both in computation and in nature.

Consider the `power()` function above. What does it require?

- An associative binary operation (so we can regroup: \((a \cdot b) \cdot c = a \cdot (b \cdot c)\))
- An identity element (so \(1 \cdot x = x \cdot 1 = x\))
- Halving and parity testing on the exponent

That's it. Any type providing these operations—with these laws—can use this algorithm. The requirements are *algebraic*, not syntactic.

There's a deeper way to understand this: **structure is compressed information**. When you say "this type is a monoid," you're transmitting an enormous amount of data in a few bits. A monoid guarantees associativity---that's infinitely many facts about every possible triple of elements. It guarantees an identity---infinitely many facts about every element's interaction with it.

The algorithm exploits this compressed information. It doesn't need to be told that \((a \cdot b) \cdot c = a \cdot (b \cdot c)\) for your particular type; it knows because you declared the structure. Minimal interfaces are powerful precisely *because* they're maximally informative per requirement. Each law you can assume is exponentially more constraint you don't have to check.

This shifts how we think about programming:

| Traditional OOP | Stepanov's Approach |
|-----------------|---------------------|
| What *is* this type? | What can I *do* with it? |
| Inheritance hierarchy | Algebraic concepts |
| Shared base class | Shared operations |
| Runtime polymorphism | Compile-time genericity |

The question isn't "is this a number?" but "does this form a monoid under some operation?" If yes, `power()` applies.

This collection explores what happens when you take this principle seriously across eleven different domains.

---

## Universality: One Algorithm, Many Faces

The [peasant](/post/2019-03-peasant-stepanov/) post demonstrates the principle most dramatically. The Russian peasant algorithm—ancient technique for multiplication without times tables—generalizes to any monoid. The same code computes:

| Type | Operation | Identity | `power(x, n)` computes... |
|------|-----------|----------|---------------------------|
| Integer | × | 1 | \(x^n\) |
| 2×2 Matrix | × | I | \(F(n)\) (Fibonacci via matrix form) |
| Quaternion | × | 1 | n-fold rotation composition |
| Tropical (min,+) | min, + | 0, ∞ | n-hop shortest path |
| String | concat | "" | n-fold string repetition |
| Permutation | ∘ | id | n-fold permutation |
| Affine transform | ∘ | id | n periods of compound interest |
| Polynomial | × | 1 | Polynomial powers |
| Modular integer | × | 1 | Modular exponentiation |
| Boolean matrix | ∧, ∨ | I | n-step reachability |

Fifteen examples in total. Each requires about fifty lines to define the type and its operations. Then `power()` works unchanged.

The pattern repeats in [polynomials](/post/2020-07-polynomials-stepanov/). The GCD algorithm—Euclid's method from 300 BCE—works identically on integers and polynomials:

```cpp
// Same algorithm for both!
template<euclidean_domain E>
E gcd(E a, E b) {
    while (b != E(0)) {
        a = std::exchange(b, a % b);
    }
    return a;
}
```

For integers: \(\gcd(48, 18) = 6\)

For polynomials: \(\gcd(x^3 - 1, x^2 - 1) = x - 1\)

Both are Euclidean domains—algebraic structures where division with remainder is always possible, with the remainder "smaller" than the divisor. Integers use absolute value as the size; polynomials use degree. The algorithm doesn't care which. It only needs the Euclidean property.

---

## Minimal Interfaces

Each algorithm requires *exactly* what it needs. No more, no less.

The peasant algorithms need:

```cpp
zero(x)       // additive identity
one(x)        // multiplicative identity
twice(x)      // doubling
half(x)       // halving
even(x)       // parity test
```

The [integration](/post/2023-08-integration-stepanov/) module needs an ordered field:

```cpp
template<typename T>
concept ordered_field =
    has_add<T> && has_sub<T> && has_mul<T> && has_div<T> &&
    has_neg<T> && has_order<T> &&
    requires { T(0); T(1); T(2); };
```

The [elementa](/post/2021-03-elementa-stepanov/) matrix library needs:

```cpp
template<typename M>
concept Matrix = requires(M m, const M cm, std::size_t i, std::size_t j) {
    typename M::scalar_type;
    { cm.rows() } -> std::same_as<std::size_t>;
    { cm.cols() } -> std::same_as<std::size_t>;
    { m(i, j) } -> std::same_as<typename M::scalar_type&>;
    { cm(i, j) } -> std::same_as<const typename M::scalar_type&>;
};
```

C++20 concepts make these requirements explicit and checkable. But the deeper point is *designing* algorithms around minimal requirements. When you ask "what does this algorithm actually need?", you discover its true scope.

The [rational](/post/2020-02-rational-stepanov/) post illustrates this beautifully. A rational number needs:
- Numerator and denominator (integers)
- GCD (to reduce to lowest terms)
- The usual arithmetic

But *which* integer type? The implementation is templated:

```cpp
template<std::integral T>
class rat {
    T num_;  // numerator (carries sign)
    T den_;  // denominator (always positive)
};
```

Use `int` for small fractions. Use `int64_t` for larger ones. Plug in an arbitrary-precision integer type and get exact rational arithmetic with no overflow. The algorithm doesn't change.

---

## ADL-Based Free Functions

A design pattern runs through every post: types expose algebraic operations as *free functions*, not member methods.

```cpp
// Not this:
class my_type {
    my_type twice() const;
    my_type half() const;
    bool is_even() const;
};

// But this:
my_type twice(my_type x);
my_type half(my_type x);
bool even(my_type x);
```

Why? Argument-dependent lookup (ADL). When generic code calls `twice(x)`, the compiler searches for `twice` in the namespace where `x`'s type is defined. No inheritance required. No base class needed. The type just provides the operations in its namespace.

This pattern enables the full universality. You can't inherit `int` from a base class, but you can provide `twice(int x)` as a free function. Suddenly built-in types participate equally with user-defined types.

The [modular](/post/2019-06-modular-stepanov/) post shows this clearly:

```cpp
template<int64_t N>
constexpr mod_int<N> zero(mod_int<N>) { return mod_int<N>(0); }

template<int64_t N>
constexpr mod_int<N> one(mod_int<N>) { return mod_int<N>(1); }

template<int64_t N>
constexpr mod_int<N> twice(mod_int<N> x) { return x + x; }

template<int64_t N>
constexpr mod_int<N> half(mod_int<N> x) {
    return x * mod_int<N>(2).inverse();  // Requires prime N
}
```

Now `power()` works with modular integers. The same twenty lines compute modular exponentiation—the workhorse of public-key cryptography.

---

## Composition Through Algebra

When two algebraic types compose, their algorithms compose too. This is where the approach becomes genuinely powerful.

### Dual Numbers + Integration = Leibniz's Rule

The [dual](/post/2021-09-dual-stepanov/) post introduces dual numbers for automatic differentiation:

```cpp
dual<double> x = dual<double>::variable(2.0);
auto f = x*x*x - 3.0*x + 1.0;
// f.value() = 3.0
// f.derivative() = 9.0  (the derivative f'(2) = 3x² - 3 = 9)
```

The derivative emerges automatically from the algebra. No symbolic manipulation. No finite differences. Just arithmetic with a different number type.

The [integration](/post/2023-08-integration-stepanov/) module computes numerical integrals:

```cpp
double result = integrate([](double x) { return std::sin(x); }, 0.0, M_PI);
// result ≈ 2.0
```

Now here's the magic. Because `integrate` is written generically—requiring only an ordered field—it works with dual numbers:

```cpp
using D = dual::dual<double>;

D a = D::variable(1.0);  // Parameter to differentiate
auto f = [&a](D x) { return a * x * x; };

D result = composite_simpsons(f, D::constant(0.0), D::constant(1.0), 100);

// result.value() = 1/3 (the integral ∫₀¹ ax² dx = a/3)
// result.derivative() = 1/3 (∂/∂a of the integral = ∫₀¹ x² dx)
```

This is Leibniz's integral rule—the derivative of an integral with respect to a parameter—implemented through *pure algebra*. No special cases. No separate implementation. The composition emerges automatically because both components respect the same algebraic structure.

### Autodiff + Elementa = Matrix Calculus

The [autodiff](/post/2023-01-autodiff-stepanov/) post builds reverse-mode automatic differentiation on top of [elementa](/post/2021-03-elementa-stepanov/)'s matrix concept:

```cpp
auto f = [](const auto& A) {
    return sum(matmul(A, A));  // f(A) = sum(A²)
};

matrix<double> A{{1, 2}, {3, 4}};
auto gradient = grad(f)(A);  // ∂f/∂A
```

The gradient of matrix functions—computed exactly. The chain rule applied through a computational graph. Classic results from matrix calculus (like \(\partial\det(A)/\partial A = \det(A) \cdot A^{-T}\)) appear as backward functions in the graph.

Because autodiff builds on the Matrix concept, not a concrete type, it would work with any conforming matrix implementation: dense, sparse, symbolic, GPU-accelerated. The algorithm is decoupled from the representation.

### Type Erasure + Ring Axioms = Runtime Polymorphism with Guarantees

The [type-erasure](/post/2024-02-type-erasure-stepanov/) post takes Sean Parent's technique and applies it to algebraic structures:

```cpp
class any_ring {
    struct concept_t {
        virtual std::unique_ptr<concept_t> add(concept_t const&) const = 0;
        virtual std::unique_ptr<concept_t> multiply(concept_t const&) const = 0;
        virtual std::unique_ptr<concept_t> negate() const = 0;
        virtual std::unique_ptr<concept_t> zero() const = 0;
        virtual std::unique_ptr<concept_t> one() const = 0;
        // ...
    };
};
```

Now `any_ring` can hold integers, polynomials, matrices—any ring. You can do arithmetic at runtime without knowing the concrete type at compile time. The algebraic laws still hold because every wrapped type satisfies them.

This is Stepanov's approach extended to runtime polymorphism: the interface is algebraic (ring operations), not arbitrary methods.

---

## Correctness from Structure

When the algebra is right, the algorithm cannot be wrong. Not "unlikely to be wrong"—*cannot*.

### Miller-Rabin: Error Bounds from Group Theory

The [miller-rabin](/post/2019-09-miller-rabin-stepanov/) post implements probabilistic primality testing. The algorithm has a beautiful error analysis: for any composite n, at least 3/4 of potential witnesses will detect it.

Where does 3/4 come from? Not from empirical testing. From the structure of multiplicative groups modulo n. The witnesses that fail to detect a composite form a subgroup; subgroups divide the group order; the analysis bounds the subgroup size.

$$
P(\text{false positive}) \leq \left(\frac{1}{4}\right)^k
$$

With \(k = 20\) witnesses: error \(< 10^{-12}\). Not "probably correct"—provably bounded error, derived from algebra.

### Rationals: Canonical Form as Theorem

The [rational](/post/2020-02-rational-stepanov/) post maintains fractions in lowest terms:

```cpp
void reduce() {
    T g = std::gcd(abs(num_), den_);
    num_ /= g;
    den_ /= g;
}
```

Every rational has a unique reduced representation. This isn't a policy decision; it's a theorem about greatest common divisors. The canonical form follows from the algebraic structure of integers (a unique factorization domain) and the definition of GCD.

Because of this, equality testing is trivial:

```cpp
bool operator==(rat const& rhs) const {
    return num_ == rhs.num_ && den_ == rhs.den_;
}
```

No tolerance. No floating-point fuzziness. Exact equality, guaranteed by algebraic structure.

### Dual Numbers: Correctness by Construction

The [dual](/post/2021-09-dual-stepanov/) post computes derivatives through the algebra of dual numbers, where \(\varepsilon^2 = 0\):

$$
f(x + \varepsilon) = f(x) + \varepsilon f'(x)
$$

This isn't an approximation. It's exact. The derivative emerges from the definition of dual number arithmetic:

```cpp
dual operator*(dual const& other) const {
    // (a + bε)(c + dε) = ac + (ad + bc)ε + bdε² = ac + (ad + bc)ε
    return {val * other.val, val * other.der + der * other.val};
}
```

The product rule is *built into* the multiplication. If the algebra is implemented correctly, derivatives are correct by construction.

---

## The Meta-Pattern: How to See This Way

After reading these eleven posts, patterns emerge. Here's how to develop this perspective:

**Ask: What operations does this algorithm actually need?**

Not "what type am I working with?" but "what can I do with it?" List the operations. Check which ones you actually call. That's your concept.

**Ask: What laws must hold?**

Associativity? Commutativity? Distributivity? An identity element? Inverses? These laws determine which algorithms apply. If you need \(a \cdot (b \cdot c) = (a \cdot b) \cdot c\), you need a semigroup. Add an identity, you have a monoid. Add inverses, a group.

| Structure | Laws | Enables |
|-----------|------|---------|
| Semigroup | Associativity | Parallel reduction |
| Monoid | + Identity | Power by squaring |
| Group | + Inverses | Division, subtraction |
| Ring | Monoid (×), Group (+), Distributive | Polynomial arithmetic |
| Field | Ring + multiplicative inverses | Division, linear algebra |
| Euclidean domain | Ring + division with remainder | GCD, fraction reduction |

**Design the concept as the minimal contract.**

Don't require what you don't need. Each additional requirement excludes types that might work. The goal is maximum generality with guaranteed correctness.

**Express operations as free functions.**

This enables ADL, which enables working with types you don't control. Built-in types, third-party libraries, legacy code—all can participate.

**Let composition emerge from shared structure.**

When component A requires concept X, and component B produces concept X, they compose. No explicit integration needed. The algebra mediates.

**Turn laws into tests.**

If you know something is a monoid, you can automatically generate property-based tests. Associativity becomes `assert(op(op(a,b),c) == op(a,op(b,c)))` for random a, b, c. Identity becomes `assert(op(e,x) == x && op(x,e) == x)` for random x.

This isn't new: Claessen and Hughes's QuickCheck (2000) pioneered property-based testing; Wadler's "Theorems for Free" (1989) showed that parametric polymorphism implies algebraic properties. But the connection to Stepanov-style programming is direct: declaring structure gives you tests for free. The laws become executable specifications.

**The open question: discovering structure.**

This collection teaches you to *use* structures once recognized---but how do you recognize them in the first place? That creative act remains underexplored.

Sometimes the process is clear: you have an algorithm, you analyze what operations and laws it requires, and you name the resulting concept. But sometimes you have a messy domain and wonder: is there hidden structure here? Could I reformulate this problem so a known structure applies?

Pattern-matching against known structures helps. Working backward from desired algorithms helps. But no systematic method exists for *discovering* new useful structures. That gap is worth acknowledging---and perhaps worth exploring.

---

## The Posts: A Dependency Graph

The eleven posts build on each other in a directed acyclic graph:

```
peasant (monoids, power by squaring)
├── miller-rabin (modular exponentiation for primality)
├── modular (ring/field structure, Fermat's theorem)
└── type-erasure (algebraic concepts at runtime)

rational (fields, GCD, unique factorization)
└── polynomials (Euclidean domains—same GCD algorithm)

dual (forward-mode autodiff via algebra)
├── finite-diff (verification, error analysis comparison)
├── integration (Leibniz's rule through composition)
└── autodiff (reverse-mode for scalar outputs)
    └── elementa (Matrix concept foundation)
```

**Entry points by interest:**

- *Number theory*: Start with [peasant](/post/2019-03-peasant-stepanov/), then [modular](/post/2019-06-modular-stepanov/), then [miller-rabin](/post/2019-09-miller-rabin-stepanov/)
- *Calculus and optimization*: Start with [dual](/post/2021-09-dual-stepanov/), then [finite-diff](/post/2022-04-finite-diff-stepanov/), then [integration](/post/2023-08-integration-stepanov/)
- *Linear algebra*: Start with [elementa](/post/2021-03-elementa-stepanov/), then [autodiff](/post/2023-01-autodiff-stepanov/)
- *Generic programming patterns*: Start with [peasant](/post/2019-03-peasant-stepanov/), then [type-erasure](/post/2024-02-type-erasure-stepanov/)
- *Algebraic structures*: Start with [rational](/post/2020-02-rational-stepanov/), then [polynomials](/post/2020-07-polynomials-stepanov/)

---

## Appendix: Reading Guide

### peasant/
**One Algorithm, Infinite Powers**

The Russian peasant multiplication algorithm generalizes to any monoid. Computes \(x^n\) in \(O(\log n)\) operations using only associativity and identity. Fifteen worked examples: integers, matrices (Fibonacci), quaternions (rotations), tropical semiring (shortest paths), strings, permutations, affine transforms (compound interest), and more.

*Key insight*: The same code structure appears everywhere because it depends only on monoid axioms, not the specific domain.

### miller-rabin/
**Is It Prime?**

Probabilistic primality testing with quantifiable error bounds. Uses modular exponentiation from peasant. The 3/4 witness bound comes from group theory. Handles Carmichael numbers that fool simpler tests.

*Key insight*: Error analysis follows from algebraic structure, not empirical observation.

### rational/
**Exact Rational Arithmetic**

Fractions that never lose precision. GCD reduction ensures unique representation. Connects to continued fractions and the Stern-Brocot tree.

*Key insight*: Canonical form is a theorem about algebraic structure, not a design choice.

### modular/
**Modular Arithmetic as Rings**

Integers mod N form rings; when N is prime, they form fields. Fermat's little theorem gives multiplicative inverses. Type-safe: `mod_int<7>` and `mod_int<11>` are incompatible types.

*Key insight*: Understanding ring/field structure tells you which algorithms apply.

### polynomials/
**Polynomial Arithmetic as Euclidean Domains**

The same GCD algorithm works for integers and polynomials because both are Euclidean domains. Sparse representation handles high-degree polynomials efficiently.

*Key insight*: Shared algebraic structure enables algorithm reuse across apparently different domains.

### type-erasure/
**Runtime Polymorphism Without Inheritance**

Sean Parent's technique: wrap any type satisfying an interface, without requiring inheritance. Extended to algebraic interfaces: `any_ring`, `any_vector_space`, `any_matrix`.

*Key insight*: Algebraic concepts can be erased at runtime while preserving guarantees.

### dual/
**Forward-Mode Automatic Differentiation**

Dual numbers (\(a + b\varepsilon\) where \(\varepsilon^2 = 0\)) compute derivatives through arithmetic. Evaluate \(f(x + \varepsilon)\) to get \(f(x) + \varepsilon f'(x)\). Supports higher-order derivatives via nesting.

*Key insight*: The derivative emerges automatically from the algebra of dual numbers.

### finite-diff/
**Numerical Differentiation**

Finite difference formulas with optimal step sizes derived from error analysis. Forward, central, and five-point stencils. Useful for verification and black-box functions.

*Key insight*: Two error sources (truncation and round-off) determine optimal step size.

### integration/
**Numerical Quadrature**

Simpson's rule, Gauss-Legendre, adaptive methods. Written generically for any ordered field—enabling composition with dual numbers (Leibniz's rule).

*Key insight*: Generic implementation enables unexpected compositions.

### elementa/
**Pedagogical Linear Algebra**

Matrix concept in C++20. LU decomposition powers determinants, inverses, and linear system solving. Cache-conscious multiplication. Value semantics throughout.

*Key insight*: A concept defines what matrices *are*, enabling algorithms independent of representation.

### autodiff/
**Reverse-Mode Automatic Differentiation**

Gradient computation via computational graphs. Builds on elementa's Matrix concept. Implements matrix calculus Jacobians (determinant, inverse, multiplication). Verified against finite differences.

*Key insight*: Systematic chain rule application, automated through graph traversal.

---

## Where the Principle Struggles

No principle applies universally. Here's where "structure first, algorithm second" meets friction:

**Generic overhead.** Sometimes a specialized algorithm beats the generic one. SIMD-optimized matrix multiplication exploits specific hardware layouts in ways a generic `Matrix` concept can't express. The structure is there, but capturing it costs more than hard-coding it.

**Approximate fit.** Floating-point numbers *almost* form a field, but not quite. Addition isn't associative: \((a + b) + c \neq a + (b + c)\) in general, due to rounding. Algorithms that assume exact associativity may accumulate errors or produce inconsistent results. The structure is a useful fiction, not a guarantee.

**Missing concepts.** Some useful algorithms lack clean algebraic characterization. What structure does "topological sort" require? "Strongly connected components"? These algorithms have preconditions (acyclic graph, directed graph), but they don't fit neatly into the algebraic hierarchy.

**Impedance mismatch.** Stateful, imperative code---caches, mutation, side effects---doesn't always fit the algebraic mold. You *can* model state transformations as monoid actions, but the encoding may obscure rather than clarify.

This isn't failure. It's knowing when the tool applies. The principle illuminates a wide territory; recognizing its boundaries makes you more effective within them.

---

## An Invitation

These eleven posts are entry points, not destinations. The principle—*structure first, algorithm second*—extends far beyond the domains explored here.

Consider data structures: a balanced binary search tree, a skip list, and a B-tree all implement the same abstract interface (ordered set). Algorithms written to that interface work on all three.

Consider I/O: streams, files, network sockets, and string buffers can satisfy the same concept. Generic code handles them uniformly.

Consider concurrency: different synchronization primitives satisfy different algebraic laws. Understanding those laws clarifies which patterns are safe.

The principle is simple. Seeing it takes practice. These posts offer exercises.

Write the algorithm once. State its requirements precisely. Let it work on anything that satisfies those requirements. That's generic programming in the Stepanov tradition.

When you recognize "this is a ring," you immediately know which algorithms apply. When you recognize "this is a monoid," `power()` is available. When you see structure first, algorithms follow.

---

## Further Reading

- Stepanov & Rose, *From Mathematics to Generic Programming* — accessible introduction
- Stepanov & McJones, *Elements of Programming* — the definitive treatment
- Sean Parent, "Inheritance Is The Base Class of Evil" (GoingNative 2013) — type erasure
- Ireland & Rosen, *A Classical Introduction to Modern Number Theory* — modular arithmetic
- Graham, Knuth, Patashnik, *Concrete Mathematics* — rational numbers, recurrences
