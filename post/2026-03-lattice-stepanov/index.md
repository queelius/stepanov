---
title: "Lattices: Fixed Points and Iteration"
date: 2026-03-13
draft: false
tags:
- C++
- generic-programming
- algorithms
- lattices
- abstract-interpretation
categories:
- Computer Science
- Mathematics
series:
- stepanov
series_weight: 17
math: true
description: A lattice has two operations, meet and join, satisfying absorption laws. Tarski's theorem gives a generic fixed-point algorithm. Lattice structure determines the iteration, just as monoid structure determines power-by-squaring.
linked_project:
- stepanov
---
*Lattices have two operations of a different kind than rings. The structure determines a fixed-point algorithm.*

## Two Operations, Different Rules

Monoids have one binary operation. Rings have two (addition and multiplication) linked by distributivity. Lattices also have two operations, but with different laws entirely.

A **lattice** is a set with two operations:
- **meet** (\(\wedge\)): greatest lower bound
- **join** (\(\vee\)): least upper bound

Both are idempotent, commutative, and associative. And they satisfy the **absorption laws**:

$$a \wedge (a \vee b) = a \qquad a \vee (a \wedge b) = a$$

Absorption is what distinguishes lattices from a pair of unrelated monoids. It ties meet and join together: knowing one constrains the other.

A **bounded lattice** adds a least element (bottom, \(\bot\)) and a greatest element (top, \(\top\)). Bottom is the identity for join, top is the identity for meet.

In C++20 concepts, with ADL free functions:

```cpp
template<typename L>
concept Lattice = std::semiregular<L> &&
    requires(L a, L b) {
        { meet(a, b) } -> std::convertible_to<L>;
        { join(a, b) } -> std::convertible_to<L>;
    };

template<typename L>
concept BoundedLattice = Lattice<L> &&
    requires(L a) {
        { bottom(a) } -> std::convertible_to<L>;
        { top(a) } -> std::convertible_to<L>;
    };
```

## Four Examples

**Sign lattice.** Abstract signs of integers: bottom (unreachable), negative, zero, positive, top (unknown). Meet is greatest lower bound, join is least upper bound in the Hasse diagram. This is the classic abstract interpretation domain. You can define abstract arithmetic on it: `pos * neg = neg`, `neg + neg = neg`, `pos + neg = top`.

**Intervals.** Closed intervals \([a, b]\) ordered by inclusion. Meet is intersection. Join is the smallest enclosing interval. Bottom is the empty interval. Top is the full range. This is the foundation of interval arithmetic.

**Divisors.** Positive integers ordered by divisibility. Meet is gcd, join is lcm. Bottom is 1 (divides everything), top is 0 (everything divides 0). Lattice structure appearing in number theory.

**Power sets.** Subsets of \(\{0, \ldots, N-1\}\). Meet is intersection (bitwise AND), join is union (bitwise OR). Bottom is the empty set, top is the full set.

All four satisfy `BoundedLattice`. All four satisfy the same laws. The concept constrains the interface; the laws constrain the semantics.

## The Algorithm: Tarski's Fixed-Point Theorem

Here is the payoff. Tarski's theorem: any monotone function on a complete lattice has a least fixed point, computable by iterating from bottom.

Start at \(\bot\). Apply \(f\). Join the result with the current value. Repeat until nothing changes. This is Kleene iteration:

$$x_0 = \bot, \quad x_{n+1} = x_n \vee f(x_n)$$

The sequence \(x_0 \leq x_1 \leq x_2 \leq \cdots\) is ascending. In a finite lattice, it must terminate.

```cpp
template<BoundedLattice L, typename F>
L least_fixed_point(F f, std::size_t max_iter = 1000) {
    L x = bottom(L{});
    for (std::size_t i = 0; i < max_iter; ++i) {
        L next = join(x, f(x));
        if (next == x) return x;
        x = next;
    }
    return x;
}
```

This is the lattice analog of `power()` from the peasant post. There, monoid structure gave us repeated squaring. Here, lattice structure gives us fixed-point iteration. Different algebra, different algorithm, same principle: the structure determines the computation.

## Application: Abstract Interpretation

The sign lattice is not just a mathematical curiosity. It is the simplest example of **abstract interpretation**, a technique for reasoning about programs without running them.

Consider a program variable `x`. Instead of tracking its concrete value, track its abstract sign. An assignment `x = 3` gives `x = positive`. An operation `x = a * b` where `a` is negative and `b` is positive gives `x = negative`.

For loops, we need a fixed point. If a loop body adds a positive number to a variable starting at zero, what sign does the variable have after the loop? We compute the least fixed point of the transfer function:

1. Start: `x = bot` (unreachable)
2. Join with initial condition: `join(bot, zero) = zero`
3. Apply transfer: `zero + positive = positive`
4. Join: `join(zero, positive) = top`
5. Apply transfer: `top + positive = top`
6. Join: `join(top, top) = top`. Fixed point reached.

The result is `top`: the variable could be zero or positive (or, conservatively, anything). The sign lattice is too coarse to distinguish "non-negative" from "unknown". A richer lattice (like intervals) would give a tighter answer.

The point is that the algorithm is the same regardless of the lattice. Swap `sign_lattice` for `interval<int>` or `powerset<N>`, and `least_fixed_point` still works. The lattice structure determines the iteration.

## The Connection

In the peasant post, the observation was: any monoid supports efficient exponentiation. In the accumulator post: any monoid supports composable streaming computation. Here: any bounded lattice supports fixed-point iteration.

Each algebraic structure comes with its own generic algorithm. Monoid structure gives `power()`. Lattice structure gives `least_fixed_point()`. The pattern repeats: define the algebra, get the algorithm.

Algorithms arise from algebraic structure.

## Further Reading

- Davey & Priestley, *Introduction to Lattices and Order*
- Cousot & Cousot, "Abstract Interpretation: A Unified Lattice Model"
- Tarski, "A Lattice-Theoretical Fixpoint Theorem and its Applications"
- Stepanov & Rose, *From Mathematics to Generic Programming*
