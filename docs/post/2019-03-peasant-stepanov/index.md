---
title: "One Algorithm, Infinite Powers"
date: 2019-03-15
draft: false
tags:
  - C++
  - generic-programming
  - algorithms
  - monoids
  - number-theory
categories:
  - Computer Science
  - Mathematics
series: ["stepanov"]
series_weight: 1
math: true
description: "The Russian peasant algorithm teaches us that one algorithm can compute products, powers, Fibonacci numbers, and more---once we see the underlying algebraic structure."
---

*How the Russian peasant algorithm reveals the universal structure of exponentiation*

## The Algorithm

Russian peasants had a clever method for multiplication that doesn't require memorizing times tables. To compute 23 x 17:

```
23    17
11    34     (halve, double)
 5    68
 2   136
 1   272
```

Add the right column wherever the left is odd: 17 + 34 + 68 + 272 = 391. That's 23 x 17.

Why does this work? Because we're really computing:
```
23 x 17 = (16 + 4 + 2 + 1) x 17 = 16x17 + 4x17 + 2x17 + 17
```

The algorithm only needs three operations on the multiplier:
- `half(n)` --- integer division by 2
- `even(n)` --- test if divisible by 2
- Addition on the result

## From Multiplication to Exponentiation

Here's the insight that makes this interesting: the *same algorithm* computes powers.

Replace "add to accumulator" with "multiply into accumulator" and "double the multiplicand" with "square the base":

```cpp
T power(T base, int exp) {
    T result = 1;
    while (exp > 0) {
        if (!even(exp)) result = result * base;
        base = base * base;
        exp = half(exp);
    }
    return result;
}
```

This is O(log n) multiplications instead of O(n). Computing 2^1000 takes about 10 multiplications, not 1000.

## The Monoid Connection

The peasant algorithm works whenever you have:
1. An associative binary operation `*`
2. An identity element `1` where `1 * x = x * 1 = x`

This structure is called a **monoid**. The algorithm computes `x * x * ... * x` (n times) using O(log n) operations.

What makes this powerful is that *many things* form monoids:

| Type | Operation | Identity | Computing x^n gives you... |
|------|-----------|----------|---------------------------|
| Integers | x | 1 | Powers |
| Matrices | x | I | Matrix powers |
| Strings | concat | "" | String repetition |
| Functions | compose | id | Function iteration |
| Permutations | compose | id | Permutation powers |
| Quaternions | x | 1 | Rotation composition |

## Why Associativity Unlocks Efficiency

Why does the peasant algorithm achieve O(log n) instead of O(n)? The answer lies in a single algebraic law: **associativity**.

Associativity says \((a \cdot b) \cdot c = a \cdot (b \cdot c)\). This looks innocuous, but it means we can *restructure* computation without changing results. Consider computing \(a^8\):

```
Naive:     a x a x a x a x a x a x a x a     (7 multiplications)
Peasant:   ((a^2)^2)^2                        (3 multiplications)
```

Both produce the same answer because we can freely regroup. The peasant algorithm exploits this freedom systematically: instead of accumulating one factor at a time, it squares intermediate results and combines them.

This restructuring **is** the source of logarithmic complexity---not a clever trick, but an inevitable consequence of the law. Given associativity, you're *permitted* to compute \(a^8 = (a^4)^2 = ((a^2)^2)^2\). Without associativity, this rewriting would be invalid; the expressions would mean different things.

Each algebraic law unlocks specific computational freedoms:

| Law | What it permits |
|-----|-----------------|
| Associativity | Restructuring evaluation order (enables O(log n) via squaring) |
| Identity | Base case for recursion (\(x^0 = 1\)) |
| Commutativity | Reordering operands (we don't require this!) |
| Inverses | Computing negative powers |

Note what we *don't* require: commutativity. If we needed \(a \cdot b = b \cdot a\), then quaternions and matrices couldn't participate---but they do, because the algorithm never reorders operands. It only regroups them.

## What We Don't Require

The absence of a law is information: it tells you what you *can't* do.

Quaternion multiplication is non-commutative: \(q_1 \cdot q_2 \neq q_2 \cdot q_1\) in general. The peasant algorithm works anyway because it never swaps operand order. When computing \(q^5\), we use \(q \cdot q \cdot q \cdot q \cdot q\)---the q's appear in the same sequence regardless of how we group them.

If we *had* required commutativity, we'd have excluded quaternions, matrices, permutations, and function composition---losing most of the interesting examples. Minimal requirements maximize applicability.

What *would* commutativity enable? Parallel reduction with arbitrary partitioning. If \(a \cdot b = b \cdot a\), you can split a sequence into chunks, reduce each chunk independently, then combine---regardless of which elements ended up in which chunk. Without commutativity, chunk boundaries must respect operand order.

## Examples in Code

### Fibonacci via Matrix Exponentiation

The Fibonacci recurrence F(n) = F(n-1) + F(n-2) can be encoded as matrix multiplication:

```
[F(n+1)]   [1 1]^n   [1]
[F(n)  ] = [1 0]   x [0]
```

Computing F(1,000,000) takes about 20 matrix multiplications:

```cpp
mat2 fib_matrix{1, 1, 1, 0};
mat2 result = power(fib_matrix, 1000000);
// result.b is F(1,000,000)
```

### Quaternion Rotations

A rotation by angle theta around axis (x,y,z) is a unit quaternion. Composing rotations is quaternion multiplication. To rotate by theta x n:

```cpp
auto rot = rotation_z(theta);
auto rot_n = power(rot, n);  // O(log n) multiplications
```

### Shortest Paths via Tropical Semiring

In the tropical semiring, "addition" is min and "multiplication" is +. Matrix "multiplication" computes path lengths. Powers find multi-hop paths:

```cpp
trop_matrix adj = /* adjacency matrix with edge weights */;
auto paths_k = power(adj, k);  // paths_k[i][j] = shortest k-hop path i->j
```

### Compound Interest

An affine transformation f(x) = ax + b under composition:
```
(a1*x + b1) compose (a2*x + b2) = a1(a2*x + b2) + b1 = (a1*a2)x + (a1*b2 + b1)
```

Compound interest with rate r and deposit d is f(x) = (1+r)x + d. After n years:

```cpp
affine yearly = {1.05, 100};  // 5% interest, $100 deposit
affine after_30_years = power(yearly, 30);
double final_balance = after_30_years(1000);  // Starting with $1000
```

## The Minimal Interface

The implementation uses C++20 concepts to express exactly what's needed:

```cpp
template<typename T>
concept algebraic = requires(T a) {
    { zero(a) } -> convertible_to<T>;
    { one(a) } -> convertible_to<T>;
    { twice(a) } -> convertible_to<T>;
    { half(a) } -> convertible_to<T>;
    { even(a) } -> convertible_to<bool>;
    { a + a } -> convertible_to<T>;
};
```

Any type satisfying this concept works with `power()`. The examples demonstrate 15 different monoids, each with about 50 lines of code.

## Why This Matters

Alex Stepanov's key insight: **algorithms arise from algebraic structure**. The peasant algorithm isn't really "about" integers or matrices---it's about monoids. Once you see this, you find the same pattern everywhere.

This is generic programming: write the algorithm once, state its requirements precisely, and let it work on anything that satisfies those requirements.

## Further Reading

- Stepanov & Rose, *From Mathematics to Generic Programming*
- Stepanov, *Elements of Programming*
