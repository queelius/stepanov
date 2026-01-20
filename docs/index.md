# Stepanov: Generic Programming in C++

These posts explore the philosophy and practice of *generic programming* as taught by Alex Stepanov, architect of the C++ Standard Template Library.

The central insight: **algorithms arise from algebraic structure**. The same algorithm that works on integers works on matrices, polynomials, and modular integers---not by accident, but because they share algebraic properties.

Each post demonstrates one beautiful idea with minimal, pedagogical C++ code (~100-300 lines).

## Core Topics

- **Monoids and exponentiation**: The Russian peasant algorithm reveals the universal structure of repeated operations
- **Rings and fields**: Modular arithmetic teaches which algorithms apply where
- **Euclidean domains**: The same GCD algorithm works for integers and polynomials
- **Generic concepts**: C++20 concepts express algebraic requirements as compile-time contracts

## Reading Order

**Foundations**: Peasant → Modular → Miller-Rabin → Rational → Polynomials

**Linear Algebra Track**: Elementa → Type Erasure

**Calculus Track**: Dual → Finite Differences → Autodiff → Integration

**Philosophy**: Synthesis → Duality

## Further Reading

- Stepanov & Rose, *From Mathematics to Generic Programming*
- Stepanov & McJones, *Elements of Programming*
- [Stepanov's collected papers](http://stepanovpapers.com/)

---

*By [Alex Towell](https://metafunctor.com). These posts also appear on [metafunctor.com](https://metafunctor.com/computer-science/stepanov/).*
