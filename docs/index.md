---
associations:
  featured:
  - slug: stepanov
    type: project
  links:
  - name: Elements of Programming
    url: http://elementsofprogramming.com/
  - name: From Mathematics to Generic Programming
    url: https://www.informit.com/store/from-mathematics-to-generic-programming-9780321942043
  papers: []
  projects:
  - stepanov
  - elementa
  - dual
  - gradator
  writing: []
title: "Stepanov: Generic Programming in C++"
description: "A series exploring Alex Stepanov's insight that algorithms arise from algebraic structure"
---

These posts explore the philosophy and practice of *generic programming* as taught by Alex Stepanov, architect of the C++ Standard Template Library.

The central insight: **algorithms arise from algebraic structure**. The same algorithm that works on integers works on matrices, polynomials, and modular integers---not by accident, but because they share algebraic properties. When you recognize "this is a monoid" or "this is a Euclidean domain," you immediately know which algorithms apply.

Each post demonstrates one beautiful idea with minimal, pedagogical C++ code (~100-300 lines). The implementations prioritize clarity over cleverness---code that reads like a textbook that happens to compile.

## Core Topics

- **Monoids and exponentiation**: The Russian peasant algorithm reveals the universal structure of repeated operations
- **Rings and fields**: Modular arithmetic teaches which algorithms apply where
- **Euclidean domains**: The same GCD algorithm works for integers and polynomials
- **Generic concepts**: C++20 concepts express algebraic requirements as compile-time contracts

## Further Reading

- Stepanov & Rose, *From Mathematics to Generic Programming*
- Stepanov & McJones, *Elements of Programming*
