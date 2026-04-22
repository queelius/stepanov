---
title: "Algorithms Arise from Algebraic Structure"
description: "A pedagogical series exploring generic programming through the lens of abstract algebra"
---

# Algorithms Arise from Algebraic Structure

This is a collection of nineteen posts and one [interactive explorer](explorer.md) about a single idea: **the algebraic structure of a type determines which algorithms apply to it**.

The idea is Stepanov's. I've been working through it for seven years, and these posts are the record. Each one takes an algebraic structure (monoid, ring, lattice, free algebra), implements it in ~200 lines of C++, and shows what algorithm falls out.

The implementations are pedagogical. They teach principles, not production patterns. The code reads like a textbook that happens to compile.

## The Principle

When you say "this type is a monoid," you're transmitting an enormous amount of information in a few bits. A monoid guarantees associativity, which is infinitely many facts about every triple of elements. It guarantees an identity, which is infinitely many facts about every element's interaction with it.

Algorithms exploit this compressed information. `power()` doesn't need to be told that $(a \cdot b) \cdot c = a \cdot (b \cdot c)$ for your particular type. It knows because you declared the structure. Each law you can assume is exponentially more constraint you don't have to check.

This is why the same twenty lines of code compute integer powers, Fibonacci numbers (via matrices), compound interest (via affine transforms), shortest paths (via the tropical semiring), and string repetition. They share monoid structure. The algorithm follows.

## The Hierarchy

Each algebraic structure adds laws, and each law unlocks algorithms:

| Structure | What it adds | What it enables |
|---|---|---|
| **Semigroup** | Associativity | Parallel reduction |
| **Monoid** | + identity | Power by squaring, O(log n) |
| **Group** | + inverses | Division, negative powers |
| **Semiring** | Two monoids + distributivity | Graph algorithms via matrix power |
| **Ring** | + subtraction | Polynomial arithmetic |
| **Field** | + division | Linear algebra, Gaussian elimination |
| **Euclidean domain** | + division with remainder | GCD (same algorithm for integers and polynomials) |
| **Ordered field** | + total order | Numerical integration |
| **Lattice** | Meet and join | Fixed-point iteration, abstract interpretation |

Minimal requirements maximize applicability. The peasant algorithm doesn't require commutativity, so quaternions and matrices work. Integration doesn't require `std::floating_point`, so dual numbers and rationals work. Every unnecessary requirement excludes types that could participate.

## Beyond Stepanov: Structure of the Maps

Stepanov's original insight covers the structures and their algorithms. But there's a second layer I've been exploring in the later posts: the *maps between structures* tell you just as much as the structures themselves.

A **homomorphism** is a function that preserves structure: $f(a \oplus b) = f(a) \oplus f(b)$. This isn't an abstract definition. It's a concrete property of functions you already use:

- `length(s1 + s2) = length(s1) + length(s2)` (length preserves concatenation)
- `log(a * b) = log(a) + log(b)` (log converts multiplication to addition)
- `sum(xs ++ ys) = sum(xs) + sum(ys)` (sum preserves list concatenation)

These aren't coincidences. They're homomorphisms. And the reason `fold` works on any monoid is that fold *is* the universal homomorphism from the free monoid (lists) to any target monoid. That's a theorem, not a design pattern.

This connects to parallelism. If $f$ is a homomorphism, you can split the input, apply $f$ to each piece, and combine the results. The homomorphism property guarantees correctness. Parallelism isn't a feature you add. It's a consequence of algebraic structure.

## What "Free" Means

Lists are the **free monoid**: the most general monoid on a set of generators. No equations hold except those the axioms force. $[a, b]$ is never equal to $[b, a]$ (no commutativity). $[a, a]$ is never equal to $[a]$ (no idempotency). The structure is as general as possible.

The universal property says: for any function $f: T \to M$ where $M$ is a monoid, there exists a unique homomorphism from the free monoid on $T$ to $M$. That homomorphism is `fold`. The universal property says fold is the only correct way to interpret a list in a monoid.

Polynomials are the free commutative ring. This is why the [polynomial post](post/2020-07-polynomials-stepanov/index.md) and the [free algebra post](post/2026-03-free-algebra-stepanov/index.md) are connected: they're both about free algebraic structures, just at different levels of the hierarchy.

## Six Principles

After nineteen posts, I can state the principles that recur:

1. **Structure determines algorithms.** Recognize the structure, the algorithm follows. This is Stepanov's original insight.

2. **Laws determine complexity.** Associativity gives O(log n). The Euclidean property gives GCD. Lattice monotonicity gives convergent iteration. Each law buys a specific computational capability.

3. **Homomorphisms explain composition.** When components preserve structure, they compose correctly. fold, polynomial evaluation, and parallel reduction are all homomorphisms. This is why they work.

4. **Free algebras explain universality.** Lists and polynomials appear everywhere in programming because they're the most general instances of their algebraic kind. The universal property says why fold is the right operation on lists.

5. **Minimal requirements maximize applicability.** Don't require commutativity if you don't use it. Don't require `std::floating_point` if `ordered_field` suffices. Each unnecessary constraint excludes types that could participate.

6. **Structure is compressed information.** Declaring "this is a monoid" transmits infinitely many facts. Algorithms exploit this compression. Property-based tests verify it.

## The Posts

### Foundations

- [One Algorithm, Infinite Powers](post/2019-03-peasant-stepanov/index.md): The Russian peasant algorithm on fifteen monoids
- [Modular Arithmetic as Rings](post/2019-06-modular-stepanov/index.md): Finite rings and fields, Fermat's theorem
- [Is It Prime?](post/2019-09-miller-rabin-stepanov/index.md): Error bounds from group theory
- [Exact Rational Arithmetic](post/2020-02-rational-stepanov/index.md): Canonical form as theorem
- [Polynomial Arithmetic](post/2020-07-polynomials-stepanov/index.md): Euclidean domains, same GCD for integers and polynomials

### Applied Mathematics

- [Pedagogical Linear Algebra](post/2021-03-elementa-stepanov/index.md): Matrix concept, LU decomposition
- [Forward-Mode Autodiff](post/2021-09-dual-stepanov/index.md): Dual numbers, derivatives from algebra
- [Numerical Differentiation](post/2022-04-finite-diff-stepanov/index.md): Truncation vs. round-off
- [Reverse-Mode Autodiff](post/2023-01-autodiff-stepanov/index.md): Computational graphs, chain rule
- [Numerical Quadrature](post/2023-08-integration-stepanov/index.md): Generic integration, Leibniz rule via composition

### Programming Patterns

- [Runtime Polymorphism](post/2024-02-type-erasure-stepanov/index.md): Sean Parent's technique on algebraic structures
- [Streaming Statistics](post/2026-03-accumulator-stepanov/index.md): Accumulators as monoids, product monoid composition
- [Graph Algorithms](post/2026-03-semiring-stepanov/index.md): One matrix power, six graph problems via semirings
- [Fixed Points](post/2026-03-lattice-stepanov/index.md): Lattices, Tarski's theorem, abstract interpretation

### Algebraic Foundations

- [The Maps Between Structures](post/2026-03-homomorphism-stepanov/index.md): Homomorphisms, fold as universal map
- [Why Lists and Polynomials Are Universal](post/2026-03-free-algebra-stepanov/index.md): Free algebras, universal property

### Reflections

- [Differentiation: Three Ways](post/2025-01-differentiation-stepanov/index.md): Forward AD, reverse AD, finite differences compared
- [Seeing Structure First](post/2026-01-18-synthesis-stepanov/index.md): The meta-pattern across eleven posts
- [Duality](post/2026-01-19-duality-stepanov/index.md): Mathematical opposites and what they reveal

## Interactive Explorer

The [explorer](explorer.md) is a visual map of the algebraic hierarchy and how each post connects to it. Click a structure to see its laws, concept definition, and associated algorithms. Click a post to see its core insight and code.

## Further Reading

- Stepanov & Rose, *From Mathematics to Generic Programming*
- Stepanov & McJones, *Elements of Programming*
- Sean Parent, "Inheritance Is The Base Class of Evil" (GoingNative 2013)
- Chan, Golub & LeVeque, "Updating Formulae and a Pairwise Algorithm for Computing Sample Variances"
- Cousot & Cousot, "Abstract Interpretation: A Unified Lattice Model for Static Analysis of Programs"
