---
title: "Free Algebras: Why Lists and Polynomials Are Universal"
date: 2026-03-13
draft: false
tags:
- C++
- generic-programming
- algorithms
- monoids
- category-theory
categories:
- Computer Science
- Mathematics
series:
- stepanov
series_weight: 19
math: true
description: The free monoid on a set is the type of lists over that set. The universal property says fold is the unique homomorphism from lists to any monoid. This explains why lists, multisets, and polynomials appear everywhere.
linked_project:
- stepanov
---
*Lists are everywhere in programming. Not because they are convenient. Because they are algebraically universal.*

## Why Lists?

Arrays are more cache-friendly. Hash maps have better lookup. Yet lists (sequences, vectors, streams) remain the default container in nearly every language. The standard explanation is convention, or ease of construction. The real explanation is algebraic.

A list is the **free monoid**. It is the most general monoid you can build from a set of generators. And the universal property of free monoids says that fold, the operation that processes a list element by element, is not a design pattern. It is a theorem.

## The Free Monoid

Start with a set \\(S\\) of generators. The free monoid on \\(S\\) is the set of all finite sequences of elements from \\(S\\), with concatenation as the operation and the empty sequence as the identity.

"Free" means: no equations hold except those forced by the monoid axioms (associativity and identity). In particular:

- \\([a, b] \neq [b, a]\\). Commutativity is not imposed.
- \\([a, a] \neq [a]\\). Idempotency is not imposed.
- \\([a, b, c] = [a] \cdot [b] \cdot [c]\\). Every sequence is a product of singletons.

In C++:

```cpp
template<typename T>
class free_monoid {
    std::vector<T> elements_;
public:
    free_monoid() = default;
    explicit free_monoid(T x) : elements_{std::move(x)} {}
    // ...
};

// Monoid operations via ADL
template<typename T>
free_monoid<T> op(const free_monoid<T>& a, const free_monoid<T>& b);  // concatenation

template<typename T>
free_monoid<T> identity(const free_monoid<T>&);  // empty sequence
```

The `free_monoid<int>` is the type of finite sequences of integers. Its operation is concatenation. It satisfies the `Monoid` concept. And it is the most general monoid on `int`: no structure beyond associativity and identity.

## The Universal Property

Here is the key fact. Given any function \\(f: S \to M\\) where \\(M\\) is a monoid, there exists a **unique** monoid homomorphism \\(\overline{f}: \text{Free}(S) \to M\\) extending \\(f\\). This homomorphism is defined by:

$$\overline{f}([a_1, a_2, \ldots, a_n]) = f(a_1) \cdot f(a_2) \cdot \ldots \cdot f(a_n)$$

In code:

```cpp
template<Monoid M, typename T, typename F>
M extend(F f, const free_monoid<T>& xs) {
    M result = identity(M{});
    for (const auto& x : xs.elements())
        result = op(result, f(x));
    return result;
}
```

This is fold. The universal property says fold is the **only** structure-preserving way to interpret a list in a monoid. Any function that respects the monoid structure must agree with fold.

## Fold Is a Theorem

When you write `std::accumulate` or `std::reduce`, you are invoking the universal property. The homomorphism condition:

$$\overline{f}(\text{op}(xs, ys)) = \text{op}(\overline{f}(xs), \overline{f}(ys))$$

is not something you need to verify. It follows from the construction. Fold preserves concatenation because it is the unique homomorphism.

This also explains why fold over an empty list returns the identity element. Not by convention, but by the homomorphism property: \\(\overline{f}([]) = e_M\\), because homomorphisms preserve identity.

## Concrete Examples

Every standard reduction is an instance of `extend`:

| Target monoid | Operation | Identity | `extend` computes |
|---------------|-----------|----------|-------------------|
| `additive<int>` | + | 0 | sum |
| `multiplicative<int>` | * | 1 | product |
| `max_monoid<int>` | max | lowest | maximum |
| `string_concat` | + | "" | concatenation |

Length is also an instance: map every element to 1 in `additive<int>`, then fold. The count of elements in a list is a monoid homomorphism from the free monoid to \\((\mathbb{Z}, +, 0)\\).

## Free Commutative Monoids: Multisets

The free monoid imposes no commutativity. What if we do?

The free **commutative** monoid on \\(S\\) is the set of multisets (bags) over \\(S\\). Order is forgotten, but multiplicity is preserved. \\(\{a, b\} = \{b, a\}\\), but \\(\{a, a\} \neq \{a\}\\).

```cpp
template<typename T>
class free_commutative_monoid {
    std::map<T, std::size_t> counts_;
public:
    // Operation: add counts. Identity: empty multiset.
};
```

Adding more axioms yields more specialized structures:

| Free algebra | Axioms imposed | Elements are |
|-------------|----------------|--------------|
| Free monoid | associativity, identity | sequences (lists) |
| Free commutative monoid | + commutativity | multisets (bags) |
| Free abelian group | + inverses | signed multisets |
| Free commutative ring | ring axioms | polynomials |

Each row adds equations. Each addition collapses more elements into equivalence. The free monoid is the most general; polynomials are the most constrained in this table.

## The Polynomial Connection

The [polynomial post]({{< ref "/post/2020-07-polynomials-stepanov" >}}) built polynomial arithmetic as a Euclidean domain. But a polynomial in one variable over a ring \\(R\\) is an element of the free commutative \\(R\\)-algebra on one generator. Polynomial evaluation at a point \\(a\\):

$$p(x) = c_0 + c_1 x + c_2 x^2 + \cdots \mapsto c_0 + c_1 a + c_2 a^2 + \cdots$$

is the universal homomorphism from the free algebra to \\(R\\). The polynomial post was working with a free algebra all along. Evaluation is `extend`, with the generator \\(x\\) mapped to the value \\(a\\).

## Why This Matters

Free algebras explain why certain data structures recur across programming languages and mathematical disciplines. Lists appear everywhere because the free monoid is universal: any computation that respects associativity and identity factors through it. Multisets appear in combinatorics because the free commutative monoid is universal for commutative computations. Polynomials appear in algebra because the free ring is universal for ring computations.

These structures are not design choices. They are forced by the axioms. The free algebra on a set of axioms is the inevitable starting point, the structure you get before imposing any problem-specific equations.

Algorithms arise from algebraic structure. Free algebras explain which structures are the starting points.

## Further Reading

- Mac Lane, *Categories for the Working Mathematician*, ch. IV (free algebras)
- Stepanov & Rose, *From Mathematics to Generic Programming*
- Awodey, *Category Theory*, ch. 2 (free/forgetful adjunction)
