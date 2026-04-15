---
title: "Homomorphisms: The Maps Between Structures"
date: 2026-03-13
draft: false
tags:
- C++
- generic-programming
- algorithms
- monoids
- homomorphisms
categories:
- Computer Science
- Mathematics
series:
- stepanov
series_weight: 17
math: true
description: A homomorphism preserves structure. fold is the universal homomorphism from the free monoid. This is the algebraic reason that fold, evaluation, and parallelism work.
linked_project:
- stepanov
---
*A homomorphism is a function that preserves algebraic structure. This post shows that fold, sum, length, and even the logarithm are all the same idea.*

## Structures and Maps

The series so far has built up algebraic structures: monoids in the [peasant post]({{< ref "/post/2019-03-peasant-stepanov" >}}), rings in the [modular post]({{< ref "/post/2019-06-modular-stepanov" >}}), Euclidean domains in the [polynomial post]({{< ref "/post/2020-07-polynomials-stepanov" >}}), product monoids in the [accumulator post]({{< ref "/post/2026-03-accumulator-stepanov" >}}). But structures alone are half the story. The other half is the maps between them.

A **homomorphism** is a function \\(f: A \to B\\) between two structures of the same kind that preserves the operation:

$$f(a \oplus b) = f(a) \oplus f(b)$$

The operation on the left is in \\(A\\). The operation on the right is in \\(B\\). The function \\(f\\) "commutes" with the operation. That is the entire definition.

## The Concept

We need a monoid concept for this post. A monoid has an associative binary operation and an identity element. As always, the operations are ADL free functions:

```cpp
template<typename M>
concept Monoid = std::semiregular<M> &&
    requires(M a, M b) {
        { op(a, b) } -> std::convertible_to<M>;
        { identity(a) } -> std::convertible_to<M>;
    };
```

And a runtime check that a function preserves the monoid structure:

```cpp
template<Monoid A, Monoid B, typename F>
bool is_homomorphism(F f, const A& a1, const A& a2) {
    return f(op(a1, a2)) == op(f(a1), f(a2));
}
```

This tests one pair of inputs. It is not a proof, but it catches violations.

## Examples Everywhere

**Length.** The string monoid (concatenation, empty string) maps to the integers under addition. The map is `length`. And length is a homomorphism:

$$\text{length}(s_1 + s_2) = \text{length}(s_1) + \text{length}(s_2)$$

The length of a concatenation is the sum of the lengths. This is not a coincidence. It is the homomorphism property.

**Sum.** Lists of integers under concatenation form a monoid. The integers under addition form a monoid. The map `sum` is a homomorphism:

$$\text{sum}(xs \mathbin{+\!\!+} ys) = \text{sum}(xs) + \text{sum}(ys)$$

**Product.** Same source monoid, different target. Now the integers under multiplication:

$$\text{prod}(xs \mathbin{+\!\!+} ys) = \text{prod}(xs) \times \text{prod}(ys)$$

**Logarithm.** The positive reals under multiplication form a monoid. The reals under addition form a monoid. The logarithm maps one to the other:

$$\log(a \times b) = \log(a) + \log(b)$$

This is the defining property of the logarithm, stated as algebra. The logarithm is a homomorphism from \\((\mathbb{R}^+, \times, 1)\\) to \\((\mathbb{R}, +, 0)\\).

**Count.** For any type \\(T\\), the function that sends a list to its length is a homomorphism from the list monoid to \\((\mathbb{Z}, +, 0)\\). This is the same as the string length example, generalized.

These are not arbitrary functions. They are structure-preserving maps.

## Fold: The Universal Homomorphism

Here is the deepest example. Lists are the **free monoid**: the most general monoid on a set of generators. Given any set \\(T\\), the lists over \\(T\\) form a monoid under concatenation, and this monoid is "free" in a precise sense.

The universal property: for any monoid \\(M\\) and any function \\(f: T \to M\\), there is exactly one homomorphism from the free monoid on \\(T\\) to \\(M\\) that extends \\(f\\). That homomorphism is `fold`.

```cpp
template<Monoid M, typename T, typename F>
    requires std::invocable<F, T>
          && std::convertible_to<std::invoke_result_t<F, T>, M>
M fold(F f, const std::vector<T>& xs) {
    M result = identity(M{});
    for (const auto& x : xs)
        result = op(result, std::invoke(f, x));
    return result;
}
```

This is `fold` stated as algebra. The function \\(f\\) maps generators to the target monoid. `fold` extends that map to entire lists by applying the monoid operation. The result is a homomorphism: `fold(f, xs ++ ys) = op(fold(f, xs), fold(f, ys))`.

Sum is `fold` where \\(f\\) sends each integer to itself in the additive monoid. Product is `fold` where \\(f\\) sends each integer to itself in the multiplicative monoid. String concatenation is `fold` where \\(f\\) is the identity on the string monoid.

Every time you write a loop that accumulates a result by combining elements one at a time, you are computing a fold. And every fold is a homomorphism.

## The Peasant Connection

The [peasant post]({{< ref "/post/2019-03-peasant-stepanov" >}}) computed \\(x^n\\) by repeated squaring. The binary representation of \\(n\\) is a sequence of bits: a word in the free monoid on \\(\{0, 1\}\\). The function `power()` reads this word bit by bit, squaring when it sees a 0 and multiplying by \\(x\\) when it sees a 1.

This is a homomorphism. The free monoid on \\(\{0, 1\}\\) (bit strings under concatenation) maps to the endomorphism monoid (functions from the monoid to itself, under composition). Each bit maps to either "square" or "square and multiply." `power()` folds this sequence of operations into a single result.

The binary representation decomposes the exponent. The homomorphism reassembles it in the target monoid. The correctness of repeated squaring is a consequence of this algebraic structure.

## What Homomorphisms Buy You

If \\(f\\) is a homomorphism and you can split the input into parts, you can compute \\(f\\) on each part independently and combine the results:

$$f(a \oplus b \oplus c) = f(a) \oplus f(b) \oplus f(c)$$

This is the algebraic reason parallelism works. The [accumulator post]({{< ref "/post/2026-03-accumulator-stepanov" >}}) showed that accumulators are monoids and that `fold` over accumulators gives parallel statistics. The reason the partitioning works, the reason you can split data across threads and merge results, is that fold is a homomorphism. Associativity alone lets you re-parenthesize. The homomorphism property lets you distribute across parts.

Not every function is a homomorphism. The square function \\(x \mapsto x^2\\) is not a homomorphism from \\((\mathbb{Z}, +)\\) to \\((\mathbb{Z}, +)\\), because \\((a+b)^2 \neq a^2 + b^2\\). Functions that fail the homomorphism property cannot be parallelized by splitting and combining. The algebra tells you which functions admit this decomposition and which do not.

## The Pattern

Every homomorphism in this post has the same shape:

1. Two monoids, a source and a target.
2. A function between them.
3. The function commutes with the operation.

When condition 3 holds, you get composability (homomorphisms compose), parallelism (split, compute, combine), and correctness (the algebra guarantees it).

Algorithms arise from algebraic structure. Homomorphisms are the maps that carry that structure from one domain to another.

## Further Reading

- Mac Lane, *Categories for the Working Mathematician*, Ch. 1 (free objects and universal properties)
- Bird & de Moor, *Algebra of Programming* (homomorphisms and program derivation)
- Stepanov & Rose, *From Mathematics to Generic Programming*
