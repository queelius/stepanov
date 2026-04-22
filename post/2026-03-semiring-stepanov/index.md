---
title: "Semirings: One Algorithm, Six Graph Problems"
date: 2026-03-13
draft: false
tags:
- C++
- generic-programming
- algorithms
- semirings
- graph-theory
categories:
- Computer Science
- Mathematics
series:
- stepanov
series_weight: 16
math: true
description: A semiring has two monoidal operations linked by distributivity. Matrix multiplication over different semirings gives shortest paths, longest paths, widest paths, reachability, and path counting, all from the same code.
linked_project:
- stepanov
---
*The peasant post showed that power() works on any monoid. But what happens when you have two operations instead of one?*

## From Monoids to Semirings

A monoid is one operation with an identity element. The peasant algorithm exploits this: give it any monoid and it computes powers by repeated squaring. The accumulator post used the same structure for streaming statistics.

A semiring is two monoids on the same set, linked by a compatibility condition. Formally, a semiring \\((S, +, \times, 0, 1)\\) satisfies:

1. \\((S, +, 0)\\) is a commutative monoid
2. \\((S, \times, 1)\\) is a monoid
3. \\(\times\\) distributes over \\(+\\): \\(a \times (b + c) = a \times b + a \times c\\)
4. \\(0\\) annihilates: \\(0 \times a = a \times 0 = 0\\)

In C++20, using the same ADL free functions as the peasant post:

```cpp
template<typename S>
concept Semiring = std::semiregular<S> &&
    requires(S a, S b) {
        { a + b } -> std::convertible_to<S>;
        { a * b } -> std::convertible_to<S>;
        { zero(a) } -> std::convertible_to<S>;
        { one(a) } -> std::convertible_to<S>;
    };
```

The concept captures syntax. The axioms (associativity, distributivity, annihilation) are semantic requirements that the programmer must ensure.

## Five Semirings

The ordinary integers are a semiring. But there are others, and each one corresponds to a different graph problem.

| Semiring | \\(+\\) | \\(\times\\) | \\(0\\) | \\(1\\) | Graph problem |
|----------|---------|--------------|---------|---------|---------------|
| Boolean | or | and | false | true | Reachability |
| Tropical min | min | plus | \\(\infty\\) | 0 | Shortest paths |
| Tropical max | max | plus | \\(-\infty\\) | 0 | Longest paths |
| Bottleneck | max | min | \\(-\infty\\) | \\(\infty\\) | Widest paths |
| Counting | plus | times | 0 | 1 | Number of paths |

The naming of the tropical semirings is counterintuitive but standard. In the tropical min semiring, the "addition" is min and the "multiplication" is ordinary addition. This matters because matrix multiplication uses both operations, and we need the algebraic structure to be correct: the inner product of a row and column computes the best path through an intermediate node.

Each semiring is a small struct with `operator+`, `operator*`, and ADL functions `zero()` and `one()`:

```cpp
struct boolean_semiring {
    bool val;
    constexpr boolean_semiring operator+(boolean_semiring rhs) const {
        return boolean_semiring(val || rhs.val);
    }
    constexpr boolean_semiring operator*(boolean_semiring rhs) const {
        return boolean_semiring(val && rhs.val);
    }
};
constexpr boolean_semiring zero(boolean_semiring) { return boolean_semiring(false); }
constexpr boolean_semiring one(boolean_semiring)  { return boolean_semiring(true); }
```

## Matrices Over a Semiring

Matrix multiplication requires addition and multiplication. That is exactly what a semiring provides. If \\(S\\) is a semiring, then \\(n \times n\\) matrices over \\(S\\) form a semiring too, with entry-wise addition and the usual row-times-column product (using \\(S\\)'s operations).

```cpp
template<Semiring S, std::size_t N>
class matrix {
    S data_[N][N];
public:
    constexpr matrix operator*(const matrix& rhs) const {
        matrix result;
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t k = 0; k < N; ++k)
                for (std::size_t j = 0; j < N; ++j)
                    result(i, j) = result(i, j) + data_[i][k] * rhs(k, j);
        return result;
    }
};
```

This is ordinary matrix multiplication. The only difference is that `+` and `*` are the semiring operations, not necessarily arithmetic addition and multiplication.

## The Insight

Represent a graph as an adjacency matrix over a semiring. Entry \\(A(i,j)\\) holds the edge weight from node \\(i\\) to node \\(j\\), or the semiring zero if there is no edge.

Now compute \\(A^k\\). Entry \\(A^k(i,j)\\) combines, over all \\(k\\)-hop paths from \\(i\\) to \\(j\\), the "product" of edge weights along each path, then takes the "sum" across all such paths. What "product" and "sum" mean depends on the semiring:

- **Boolean**: \\(A^k(i,j)\\) is true iff there exists a path of length \\(k\\). Product is AND (all edges must exist), sum is OR (any path suffices).
- **Tropical min**: \\(A^k(i,j)\\) is the shortest \\(k\\)-edge path. Product is addition (path length is the sum of edge weights), sum is min (we want the shortest).
- **Tropical max**: \\(A^k(i,j)\\) is the longest \\(k\\)-edge path. Same idea, but sum is max.
- **Bottleneck**: \\(A^k(i,j)\\) is the widest \\(k\\)-edge path. Product is min (capacity is the narrowest edge), sum is max (we want the widest such path).
- **Counting**: \\(A^k(i,j)\\) is the number of \\(k\\)-edge paths. Product is multiplication (each intermediate edge multiplies the count), sum is addition (paths from different intermediaries add up).

Five different graph algorithms. One matrix power computation. The semiring determines the semantics.

## The Algorithm

To find all-pairs answers considering paths of any length, compute the closure: \\(I + A + A^2 + \cdots + A^{N-1}\\), where \\(N\\) is the number of nodes. For a graph with no negative cycles, \\(N-1\\) hops suffice to cover all simple paths.

```cpp
template<Semiring S, std::size_t N>
constexpr matrix<S, N> closure(const matrix<S, N>& adj, std::size_t max_hops = N - 1) {
    auto result = matrix<S, N>::identity();
    auto ak = matrix<S, N>::identity();
    for (std::size_t k = 1; k <= max_hops; ++k) {
        ak = ak * adj;
        result = result + ak;
    }
    return result;
}
```

The individual matrix powers use the same repeated-squaring algorithm from the peasant post:

```cpp
template<Semiring S, std::size_t N>
constexpr matrix<S, N> power(matrix<S, N> base, std::size_t exp) {
    auto result = matrix<S, N>::identity();
    while (exp > 0) {
        if (exp & 1) result = result * base;
        base = base * base;
        exp >>= 1;
    }
    return result;
}
```

Same algorithm. Richer algebraic structure. The power function does not know whether it is computing reachability, shortest paths, or path counts. It only knows it has a semiring.

## Example

Consider a 4-node graph:

```
0 --2--> 1 --3--> 3
|        |
5        1
v        v
2 --4--> 3
```

Build the adjacency matrix once, then swap the semiring:

```cpp
// Shortest paths (tropical min)
auto dist = closure(adjacency<tropical_min<double>, 4>({
    {0, 1, tropical_min(2.0)},
    {0, 2, tropical_min(5.0)},
    {1, 3, tropical_min(3.0)},
    {1, 2, tropical_min(1.0)},
    {2, 3, tropical_min(4.0)},
}));
// dist(0, 3).val == 5.0  (path 0->1->3, cost 2+3)
// dist(0, 2).val == 3.0  (path 0->1->2, cost 2+1, not direct 5)

// Number of paths (counting)
auto paths = closure(adjacency<counting, 4>({
    {0, 1, counting(1)}, {0, 2, counting(1)},
    {1, 3, counting(1)}, {1, 2, counting(1)},
    {2, 3, counting(1)},
}));
// paths(0, 3).val == 3  (three paths: 0->1->3, 0->2->3, 0->1->2->3)
```

The graph topology is the same. The semiring determines what question gets answered.

## The Connection

The peasant post established the pattern: identify the algebraic structure, then write the algorithm once. There, the structure was a monoid and the algorithm was `power()`. Here, the structure is a semiring and the algorithm is still `power()`, applied to matrices.

The accumulator post composed monoids via the product construction. Here, we compose two monoids via distributivity to get a semiring. Same principle at a different level of abstraction.

Algorithms arise from algebraic structure. The more structure you identify, the more algorithms you get from the same code.

## Further Reading

- Gondran & Minoux, *Graphs, Dioids and Semirings*
- Dolan, "Fun with Semirings" (Haskell Symposium 2013)
- Stepanov & Rose, *From Mathematics to Generic Programming*
