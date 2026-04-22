---
title: "Disjoint Intervals: Where 1D Buys You a Boolean Algebra"
date: 2019-12-15
draft: false
tags:
- C++
- generic-programming
- algorithms
- boolean-algebra
- computational-geometry
categories:
- Computer Science
- Mathematics
series:
- stepanov
series_weight: 4
math: true
description: A single interval is closed under intersection but not union or complement. A sorted, merged, canonical set of intervals is closed under all three. The closure is a property of one dimension and does not survive the move to two.
linked_project:
- stepanov
---
*A single interval is barely a thing. The right structure is a canonical set of disjoint intervals, and it turns out to be a Boolean algebra. This post builds it, checks the axioms, and names the trap that 1D lets you escape and 2D does not.*

## The Interval, Alone, Is Disappointing

Take a totally ordered set \\(T\\) (the integers, the reals, anything linearly ordered) and let \\(I\\) be a closed interval \\([a, b]\\) in \\(T\\). Two facts jump out immediately:

- The intersection of two intervals is another interval (possibly empty). So \\(I\\) is closed under \\(\cap\\).
- The union of two intervals is usually *not* an interval. \\([0, 3] \cup [5, 8]\\) has a gap, and no single interval represents it.
- The complement of an interval is usually *not* an interval either. The complement of \\([0, 10]\\) over \\(\mathbb{R}\\) is \\((-\infty, 0) \cup (10, \infty)\\), which is two intervals.

So \\(I\\) is not closed under union or complement. That closes the door on a Boolean algebra of individual intervals.

We escape by changing the object we work with. Instead of a single interval, take a *set* of intervals. The set
$$S = \lbrace [0, 3], [5, 8], [12, 20] \rbrace$$
is the natural answer. But we need to be careful about how we represent \\(S\\): a `std::vector<interval>` where someone could put \\([0, 3]\\) and \\([2, 10]\\) and \\([5, 8]\\) all in is not a useful representation. Three vectors that all describe the same underlying set would compare unequal.

The cure is a **canonical form**: one representation per value. Sort the intervals by lower endpoint. Merge any that overlap or touch. Drop the empties. The result is a sorted, non-overlapping, maximally-merged sequence, and it is the *only* such sequence describing the given union of points. Two canonical forms compare equal iff the underlying sets of points are equal.

That is the core invariant. Everything that follows is a consequence.

## The Concept

```cpp
template<typename T>
concept endpoint = std::regular<T> && std::totally_ordered<T>;
```

We need the underlying value type to be **regular** (copyable, equality-comparable) and **totally ordered**. No other assumptions. No metric, no infinity, no arithmetic. Just order.

For infinity handling we provide `lowest_endpoint<T>()` and `highest_endpoint<T>()` that return \\(\pm\infty\\) when `std::numeric_limits<T>::has_infinity` is true, and the extreme representable value otherwise. Integer endpoints use `lowest()` and `max()`; floating-point endpoints use actual \\(\pm\infty\\).

## Endpoints Are Data, Not Decoration

The single most bug-prone thing about intervals is the handling of open and closed endpoints. It is tempting to treat open-vs-closed as a minor UI detail. It is not: the openness flags participate in the algebra.

Concretely, consider \\([0, 5]\\) and \\((5, 10]\\). At the point \\(5\\), the first interval contains it, the second does not. The union is \\([0, 10]\\), closed at both ends, because \\(5\\) is in the result via the first operand.

Now consider \\([0, 5)\\) and \\((5, 10]\\). The point \\(5\\) is in *neither*. These two intervals touch at \\(5\\) but are separated by the singleton gap \\(\lbrace 5 \rbrace\\). The union is \\([0, 5) \cup (5, 10]\\), two separate intervals.

The merge rule is: two intervals that touch at a point \\(x\\) merge iff at least one of them is closed at \\(x\\). If both are open there, they do not merge.

And the complement rule: at every finite boundary the openness *flips*. The complement of \\([3, 7]\\) is \\((-\infty, 3) \cup (7, \infty)\\): open at \\(3\\), open at \\(7\\), because the originals were closed there, and the endpoint is now excluded from the complement rather than included in the original. If you forget the flip, you end up with a "complement" that still contains the boundary points, and \\(A \cap \neg A\\) is no longer empty.

These two rules are what make the Boolean algebra work, and they are where every hand-rolled interval library I have ever read has a bug.

## The Sweep-Line Merge, and a Trap

Canonicalizing a bag of intervals is a sweep-line algorithm: sort by lower endpoint, then walk left-to-right extending a running `(lower, upper, lc, rc)` quadruple until you hit a gap, then emit and restart.

The trap is in the sort. The natural partial order on intervals is **containment**: \\(A \le B\\) iff \\(A \subseteq B\\). It is tempting to hand that to `std::sort` as a `std::less<interval>` specialization. This is Undefined Behavior.

`std::sort` requires a **strict weak ordering**: a total, antisymmetric, transitive order where "incomparable" is an equivalence relation. Containment is a partial order: two disjoint intervals \\([1, 3]\\) and \\([5, 7]\\) are incomparable and their "incomparability" is not transitive in the way `std::sort` requires. Feed containment to `std::sort` and it will, depending on implementation, loop forever, corrupt memory, or silently produce a wrong answer.

The fix: sort by a proper linear key. We sort by lower endpoint, with a tie-breaker that puts closed-left before open-left so the wider interval contributes the leftmost boundary:

```cpp
std::sort(xs.begin(), xs.end(), [](const interval<T>& a, const interval<T>& b) {
    const T al = *a.lower_bound(), bl = *b.lower_bound();
    if (al != bl) return al < bl;
    return a.left_closed() && !b.left_closed();
});
```

This is a strict weak order. Sorting is safe. The containment relation still exists and is useful (for the subset test \\(\subseteq\\)), but it is *not* the `operator<` used for sorting. That distinction is the post's Stepanov moment: **one type can participate in multiple orders, and you have to pick the right one for the algorithm**.

## The Boolean Algebra

Once the canonical form is a maintained invariant, the set operations fall out:

$$A \cup B \qquad A \cap B \qquad \neg A \qquad A \setminus B \qquad A \oplus B$$

Union is "concatenate the two canonical vectors and re-canonicalize". Intersection is a linear scan over both sequences, producing the pairwise intersection of each overlapping pair. Complement walks left-to-right, emitting the gaps between input intervals with endpoint openness flipped. Difference is \\(A \cap \neg B\\). Symmetric difference is \\((A \cup B) \setminus (A \cap B)\\).

Every one of these produces a canonical-form output, so the algebra is closed. And every one runs in \\(O(n + m)\\) on sorted input. That linearity is not a coincidence: it is what canonical form buys you.

The laws:

| Law | Statement |
|---|---|
| Commutativity | \\(A \cup B = B \cup A,\ A \cap B = B \cap A\\) |
| Associativity | \\((A \cup B) \cup C = A \cup (B \cup C),\ (A \cap B) \cap C = A \cap (B \cap C)\\) |
| Distributivity | \\(A \cap (B \cup C) = (A \cap B) \cup (A \cap C),\ A \cup (B \cap C) = (A \cup B) \cap (A \cup C)\\) |
| Identity | \\(A \cup \emptyset = A,\ A \cap U = A\\) |
| Complement | \\(A \cup \neg A = U,\ A \cap \neg A = \emptyset\\) |
| Idempotence | \\(A \cup A = A,\ A \cap A = A\\) |
| Absorption | \\(A \cup (A \cap B) = A,\ A \cap (A \cup B) = A\\) |
| De Morgan | \\(\neg(A \cup B) = \neg A \cap \neg B,\ \neg(A \cap B) = \neg A \cup \neg B\\) |

The tests exercise each of these against concrete values. Because canonical form is unique, equality is structural: `a == b` iff `a.intervals_ == b.intervals_`. Every failed law would show up immediately.

## Why One Dimension Is Special

Try the same construction in two dimensions. Replace each interval with an axis-aligned rectangle: two intervals, one per axis. Intersection still works (intersect each axis). But union does not.

The union of two overlapping rectangles is an L-shape, and an L-shape is not a rectangle. There is no canonical form "one rectangle per subset you want to represent". You can *decompose* the L-shape into a finite set of disjoint rectangles, but the decomposition is not unique (you can split it left-to-right or top-to-bottom or diagonally). Without uniqueness, equality is not structural, and the Boolean algebra collapses.

The complement is worse. The complement of a rectangle in 2D is an unbounded region with one rectangular hole, and a region with a hole cannot be written as any finite union of rectangles. You need a more expressive representation: slab decompositions, trapezoidal maps, BSP trees. Each of those has a proper academic literature behind it, and all of them have \\(O(n \log n)\\) or worse Boolean operations. That is Boost.Geometry territory, not a header-only toy.

So the punchline is:

> **1D admits a unique canonical form for any finite union of intervals; 2D does not.** That single fact is why `disjoint_intervals<T>` is a Boolean algebra and `disjoint_rectangles<T>` is not.

## The Code in One Page

The full implementation is `intervals.hpp` in this post's directory. It is around 300 lines, including docstrings. The key pieces:

```cpp
// A pair of endpoints and two bits of openness.
template<endpoint T>
class interval { ... };

// Canonicalize (sort + sweep + merge) is the only non-trivial algorithm.
template<endpoint T>
std::vector<interval<T>> canonicalize(std::vector<interval<T>> xs);

// A container that maintains canonical form as an invariant.
template<endpoint T>
class disjoint_intervals { ... };

// The Boolean algebra, as free functions with ADL.
template<endpoint T> disjoint_intervals<T> unite(...);
template<endpoint T> disjoint_intervals<T> intersect(...);
template<endpoint T> disjoint_intervals<T> complement(...);
template<endpoint T> disjoint_intervals<T> difference(...);
template<endpoint T> disjoint_intervals<T> symmetric_difference(...);

// Operator sugar.
template<endpoint T> auto operator|(...);   // union
template<endpoint T> auto operator&(...);   // intersection
template<endpoint T> auto operator~(...);   // complement
template<endpoint T> auto operator-(...);   // difference
template<endpoint T> auto operator^(...);   // symmetric difference
```

No parser, no formatter, no I/O. The point is the algebra, not the string handling.

## Things This Gives You

The practical payoff of a Boolean algebra is composition. If you need to express "all time slots that are not lunch, on weekdays, and when the conference room is free", you write

```cpp
(weekdays & not_lunch) - room_booked
```

and the canonical form takes care of every boundary condition for you. No manual merge loop, no comparison of endpoints, no "did I handle the case where two intervals touch at a point?" Same for IP allowlist/blocklist composition, sensor coverage gap analysis, test parameter coverage, and any other "union, intersection, complement, compose" workflow.

What you do *not* get is efficient point-query when the set is large. `contains(s, x)` is linear in the number of intervals; a real spatial structure would be logarithmic. But for modest sizes the vector layout beats any tree on constant factors, and the constant factors are what usually matter.

## Coda

Canonical form is one of Stepanov's recurring themes. Reduced fractions. Normalized floating-point numbers. Sorted and deduped sequences. In every case the move is the same: one representation per value, invariant maintained by every operation, and equality becomes structural.

Disjoint intervals are a small, self-contained case of the same pattern, and because they land in one dimension they reach all the way to a Boolean algebra. They are the deepest thing 1D gives you, and the reason to stay in 1D when you can.
