// Tests for the disjoint-intervals post.
//
// The suite is organized in three layers:
//
//   1. interval<T>: the building block.
//   2. disjoint_intervals<T>: the canonical-form container.
//   3. Boolean algebra laws: the whole point of the post.
//
// Each algebraic axiom we claim in the article is checked here against a
// small handful of concrete values. That is not a proof, but the tests are
// designed to fail if anything in the sweep-line merge or complement loses
// track of the open/closed flags; that is the most failure-prone area.

#include "intervals.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <vector>

using stepanov::interval;
using stepanov::disjoint_intervals;

// =============================================================================
// interval<T>: construction, queries, intersection
// =============================================================================

TEST(Interval, DefaultIsEmpty) {
    interval<double> i{};
    EXPECT_TRUE(empty(i));
    EXPECT_FALSE(i.lower_bound().has_value());
    EXPECT_FALSE(i.upper_bound().has_value());
}

TEST(Interval, NamedConstructors) {
    EXPECT_TRUE(contains(interval<double>::closed(0, 10),    5.0));
    EXPECT_TRUE(contains(interval<double>::closed(0, 10),    0.0));
    EXPECT_TRUE(contains(interval<double>::closed(0, 10),    10.0));
    EXPECT_FALSE(contains(interval<double>::open(0, 10),     0.0));
    EXPECT_FALSE(contains(interval<double>::open(0, 10),     10.0));
    EXPECT_TRUE(contains(interval<double>::left_open(0, 10), 10.0));
    EXPECT_FALSE(contains(interval<double>::left_open(0, 10), 0.0));
    EXPECT_TRUE(contains(interval<double>::right_open(0, 10), 0.0));
    EXPECT_FALSE(contains(interval<double>::right_open(0, 10), 10.0));
    EXPECT_TRUE(contains(interval<double>::point(7),         7.0));
    EXPECT_FALSE(contains(interval<double>::point(7),        7.0000001));
}

TEST(Interval, DegenerateIsNormalizedToEmpty) {
    // Reversed bounds.
    EXPECT_TRUE(empty(interval<int>{10, 5}));
    // Same point but with an open endpoint.
    EXPECT_TRUE(empty(interval<int>::open(3, 3)));
    EXPECT_TRUE(empty(interval<int>::left_open(3, 3)));
    EXPECT_TRUE(empty(interval<int>::right_open(3, 3)));
    // Only the fully closed singleton is non-empty.
    EXPECT_FALSE(empty(interval<int>::point(3)));
}

TEST(Interval, IntersectionOfIntervals) {
    auto a = interval<double>::closed(0, 10);
    auto b = interval<double>::closed(5, 15);
    auto x = intersect(a, b);
    EXPECT_EQ(*x.lower_bound(), 5.0);
    EXPECT_EQ(*x.upper_bound(), 10.0);
    EXPECT_TRUE(x.left_closed());
    EXPECT_TRUE(x.right_closed());
}

TEST(Interval, IntersectionOpenClosedFlagsAtTie) {
    // [0, 10] ∩ (10, 20) has no point in common.
    auto a = interval<double>::closed(0, 10);
    auto b = interval<double>::left_open(10, 20);
    EXPECT_TRUE(empty(intersect(a, b)));

    // [0, 10) ∩ [10, 20] also has no point in common (10 excluded left).
    auto c = interval<double>::right_open(0, 10);
    auto d = interval<double>::closed(10, 20);
    EXPECT_TRUE(empty(intersect(c, d)));

    // [0, 10] ∩ [10, 20] keeps the shared endpoint.
    auto e = interval<double>::closed(0, 10);
    auto f = interval<double>::closed(10, 20);
    EXPECT_EQ(intersect(e, f), interval<double>::point(10));
}

// =============================================================================
// disjoint_intervals<T>: canonicalization
// =============================================================================

TEST(DisjointIntervals, EmptyInputsAreDropped) {
    disjoint_intervals<int> s{
        interval<int>::closed(1, 3),
        interval<int>{10, 5},            // reversed, empty
        interval<int>::closed(7, 9),
        interval<int>::open(4, 4)        // degenerate, empty
    };
    EXPECT_EQ(s.size(), 2u);
}

TEST(DisjointIntervals, OverlapsMerge) {
    disjoint_intervals<double> s{
        interval<double>::closed(0, 5),
        interval<double>::closed(3, 8),
        interval<double>::closed(20, 30)
    };
    EXPECT_EQ(s.size(), 2u);
    EXPECT_EQ(*s.begin()->lower_bound(), 0.0);
    EXPECT_EQ(*s.begin()->upper_bound(), 8.0);
}

TEST(DisjointIntervals, TouchingClosedMerges) {
    // [1, 3] and [3, 5] share the point 3 (both closed there): merge.
    disjoint_intervals<int> s{
        interval<int>::closed(1, 3),
        interval<int>::closed(3, 5)
    };
    EXPECT_EQ(s.size(), 1u);
    EXPECT_EQ(*s.begin()->upper_bound(), 5);
}

TEST(DisjointIntervals, TouchingBothOpenDoesNotMerge) {
    // [1, 3) and (3, 5] have a gap at 3: do NOT merge.
    disjoint_intervals<int> s{
        interval<int>::right_open(1, 3),
        interval<int>::left_open(3, 5)
    };
    EXPECT_EQ(s.size(), 2u);
}

TEST(DisjointIntervals, TouchingOneClosedMerges) {
    // [1, 3] ∪ (3, 5]. The point 3 is in the first, not the second,
    // but there is no gap: the result is [1, 5].
    disjoint_intervals<int> s{
        interval<int>::closed(1, 3),
        interval<int>::left_open(3, 5)
    };
    EXPECT_EQ(s.size(), 1u);
    EXPECT_EQ(*s.begin()->lower_bound(), 1);
    EXPECT_EQ(*s.begin()->upper_bound(), 5);
    EXPECT_TRUE(s.begin()->left_closed());
    EXPECT_TRUE(s.begin()->right_closed());
}

TEST(DisjointIntervals, MergeOfMixedOpenness) {
    // (1, 3] ∪ [3, 5): touches closed at 3, merges to (1, 5).
    disjoint_intervals<int> s{
        interval<int>::left_open(1, 3),
        interval<int>::right_open(3, 5)
    };
    EXPECT_EQ(s.size(), 1u);
    auto merged = *s.begin();
    EXPECT_EQ(*merged.lower_bound(), 1);
    EXPECT_EQ(*merged.upper_bound(), 5);
    EXPECT_FALSE(merged.left_closed());   // from (1, ...
    EXPECT_FALSE(merged.right_closed());  // from ..., 5)
}

// =============================================================================
// Point-set queries
// =============================================================================

TEST(DisjointIntervals, ContainsScalar) {
    disjoint_intervals<int> s{
        interval<int>::closed(0, 10),
        interval<int>::closed(20, 30)
    };
    EXPECT_TRUE(contains(s, 5));
    EXPECT_TRUE(contains(s, 25));
    EXPECT_FALSE(contains(s, 15));
    EXPECT_FALSE(contains(s, -1));
}

TEST(DisjointIntervals, Measure) {
    disjoint_intervals<double> s{
        interval<double>::closed(0, 10),
        interval<double>::closed(20, 30)
    };
    EXPECT_DOUBLE_EQ(measure(s), 20.0);
}

// =============================================================================
// Boolean algebra laws
// =============================================================================
//
// We fix three concrete values and check each axiom. The test is not a
// proof, but if the sweep or complement forgets open/closed flags these
// checks detect it: the canonical form would differ.

namespace {

const disjoint_intervals<double> A{
    interval<double>::closed(0, 5),
    interval<double>::closed(10, 15)
};
const disjoint_intervals<double> B{
    interval<double>::closed(3, 12)
};
const disjoint_intervals<double> C{
    interval<double>::closed(-2, 2),
    interval<double>::closed(20, 25)
};

}  // namespace

TEST(Algebra, UnionCommutative) {
    EXPECT_EQ(A | B, B | A);
    EXPECT_EQ(A | C, C | A);
}

TEST(Algebra, IntersectionCommutative) {
    EXPECT_EQ(A & B, B & A);
    EXPECT_EQ(A & C, C & A);
}

TEST(Algebra, UnionAssociative) {
    EXPECT_EQ((A | B) | C, A | (B | C));
}

TEST(Algebra, IntersectionAssociative) {
    EXPECT_EQ((A & B) & C, A & (B & C));
}

TEST(Algebra, Distributive) {
    // A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C)
    EXPECT_EQ(A & (B | C), (A & B) | (A & C));
    // A ∪ (B ∩ C) = (A ∪ B) ∩ (A ∪ C)
    EXPECT_EQ(A | (B & C), (A | B) & (A | C));
}

TEST(Algebra, Idempotent) {
    EXPECT_EQ(A | A, A);
    EXPECT_EQ(A & A, A);
}

TEST(Algebra, Absorption) {
    EXPECT_EQ(A | (A & B), A);
    EXPECT_EQ(A & (A | B), A);
}

TEST(Algebra, DoubleComplement) {
    EXPECT_EQ(~~A, A);
}

TEST(Algebra, DeMorgan) {
    EXPECT_EQ(~(A | B), ~A & ~B);
    EXPECT_EQ(~(A & B), ~A | ~B);
}

TEST(Algebra, ComplementExcludesBoundary) {
    // ~[3, 7] should be (-inf, 3) ∪ (7, +inf). The endpoints 3 and 7 are
    // in [3, 7] and therefore NOT in its complement.
    disjoint_intervals<double> s{interval<double>::closed(3, 7)};
    auto c = ~s;
    EXPECT_FALSE(contains(c, 3.0));
    EXPECT_FALSE(contains(c, 7.0));
    EXPECT_TRUE(contains(c, 2.9999));
    EXPECT_TRUE(contains(c, 7.0001));
    // And ~(3, 7) includes 3 and 7.
    disjoint_intervals<double> o{interval<double>::open(3, 7)};
    auto co = ~o;
    EXPECT_TRUE(contains(co, 3.0));
    EXPECT_TRUE(contains(co, 7.0));
}

TEST(Algebra, DifferenceMatchesDefinition) {
    // A - B should equal A ∩ ~B by the identity used in the implementation.
    EXPECT_EQ(A - B, A & ~B);
}

TEST(Algebra, SymmetricDifferenceMatchesDefinition) {
    // A ^ B = (A - B) ∪ (B - A) = (A ∪ B) - (A ∩ B)
    EXPECT_EQ(A ^ B, (A - B) | (B - A));
    EXPECT_EQ(A ^ B, (A | B) - (A & B));
}

// =============================================================================
// Stepanov trap: the sort comparator pitfall
// =============================================================================
//
// This test does not directly exercise the comparator; it canonicalizes
// many overlapping intervals of mixed openness. If the sweep feeds a
// non-strict-weak-ordering to std::sort it will loop or crash. If it
// forgets open/closed flags the output equality check will fail.

TEST(Canonicalize, StressWithMixedOpenness) {
    std::vector<interval<double>> xs;
    for (int i = 0; i < 20; ++i) {
        xs.push_back(interval<double>::closed(i,     i + 2));
        xs.push_back(interval<double>::open(i + 1,   i + 3));
        xs.push_back(interval<double>::left_open(i,  i + 1));
        xs.push_back(interval<double>::right_open(i, i + 2));
    }
    auto s = disjoint_intervals<double>(std::move(xs));
    // The union collapses to a single interval spanning [0, 22].
    ASSERT_EQ(s.size(), 1u);
    EXPECT_EQ(*s.begin()->lower_bound(), 0.0);
    EXPECT_EQ(*s.begin()->upper_bound(), 22.0);
    EXPECT_TRUE(s.begin()->left_closed());
    EXPECT_FALSE(s.begin()->right_closed());  // last interval is open at 22
}

TEST(Canonicalize, EmptyInputProducesEmpty) {
    disjoint_intervals<int> s{};
    EXPECT_TRUE(s.empty());
    EXPECT_TRUE((~~s) == s);
}
