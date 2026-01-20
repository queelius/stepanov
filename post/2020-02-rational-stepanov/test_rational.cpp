#include <gtest/gtest.h>
#include "rational.hpp"

using namespace rational;

// =============================================================================
// Construction and reduction
// =============================================================================

TEST(RationalTest, DefaultConstruction) {
    rat<int> r;
    EXPECT_EQ(r.numerator(), 0);
    EXPECT_EQ(r.denominator(), 1);
}

TEST(RationalTest, IntegerConstruction) {
    rat<int> r(5);
    EXPECT_EQ(r.numerator(), 5);
    EXPECT_EQ(r.denominator(), 1);
}

TEST(RationalTest, FractionConstruction) {
    rat<int> r(3, 4);
    EXPECT_EQ(r.numerator(), 3);
    EXPECT_EQ(r.denominator(), 4);
}

TEST(RationalTest, AutomaticReduction) {
    rat<int> r(6, 8);  // Should reduce to 3/4
    EXPECT_EQ(r.numerator(), 3);
    EXPECT_EQ(r.denominator(), 4);
}

TEST(RationalTest, NegativeDenominator) {
    rat<int> r(3, -4);  // Should become -3/4
    EXPECT_EQ(r.numerator(), -3);
    EXPECT_EQ(r.denominator(), 4);
}

TEST(RationalTest, BothNegative) {
    rat<int> r(-3, -4);  // Should become 3/4
    EXPECT_EQ(r.numerator(), 3);
    EXPECT_EQ(r.denominator(), 4);
}

TEST(RationalTest, ZeroNumerator) {
    rat<int> r(0, 5);  // Should become 0/1
    EXPECT_EQ(r.numerator(), 0);
    EXPECT_EQ(r.denominator(), 1);
}

TEST(RationalTest, DivisionByZero) {
    EXPECT_THROW(rat<int>(1, 0), std::domain_error);
}

// =============================================================================
// Arithmetic
// =============================================================================

TEST(RationalTest, Addition) {
    rat<int> a(1, 2);
    rat<int> b(1, 3);
    auto c = a + b;
    EXPECT_EQ(c.numerator(), 5);
    EXPECT_EQ(c.denominator(), 6);
}

TEST(RationalTest, Subtraction) {
    rat<int> a(1, 2);
    rat<int> b(1, 3);
    auto c = a - b;
    EXPECT_EQ(c.numerator(), 1);
    EXPECT_EQ(c.denominator(), 6);
}

TEST(RationalTest, Multiplication) {
    rat<int> a(2, 3);
    rat<int> b(3, 4);
    auto c = a * b;
    EXPECT_EQ(c.numerator(), 1);
    EXPECT_EQ(c.denominator(), 2);
}

TEST(RationalTest, Division) {
    rat<int> a(2, 3);
    rat<int> b(4, 5);
    auto c = a / b;
    EXPECT_EQ(c.numerator(), 5);
    EXPECT_EQ(c.denominator(), 6);
}

TEST(RationalTest, Negation) {
    rat<int> a(3, 4);
    auto b = -a;
    EXPECT_EQ(b.numerator(), -3);
    EXPECT_EQ(b.denominator(), 4);
}

// =============================================================================
// Comparison
// =============================================================================

TEST(RationalTest, Equality) {
    EXPECT_EQ(rat<int>(1, 2), rat<int>(2, 4));
    EXPECT_EQ(rat<int>(3, 4), rat<int>(6, 8));
    EXPECT_NE(rat<int>(1, 2), rat<int>(1, 3));
}

TEST(RationalTest, Ordering) {
    EXPECT_LT(rat<int>(1, 3), rat<int>(1, 2));
    EXPECT_GT(rat<int>(2, 3), rat<int>(1, 2));
    EXPECT_LE(rat<int>(1, 2), rat<int>(1, 2));
    EXPECT_GE(rat<int>(1, 2), rat<int>(1, 2));
}

TEST(RationalTest, NegativeComparison) {
    EXPECT_LT(rat<int>(-1, 2), rat<int>(1, 2));
    EXPECT_LT(rat<int>(-2, 3), rat<int>(-1, 3));
}

// =============================================================================
// Properties
// =============================================================================

TEST(RationalTest, Predicates) {
    EXPECT_TRUE(rat<int>(0).is_zero());
    EXPECT_FALSE(rat<int>(1, 2).is_zero());

    EXPECT_TRUE(rat<int>(5).is_integer());
    EXPECT_FALSE(rat<int>(1, 2).is_integer());

    EXPECT_TRUE(rat<int>(1, 2).is_positive());
    EXPECT_TRUE(rat<int>(-1, 2).is_negative());
}

TEST(RationalTest, Reciprocal) {
    rat<int> a(3, 4);
    auto b = a.reciprocal();
    EXPECT_EQ(b.numerator(), 4);
    EXPECT_EQ(b.denominator(), 3);

    EXPECT_THROW(rat<int>(0).reciprocal(), std::domain_error);
}

TEST(RationalTest, Conversion) {
    rat<int> a(1, 2);
    EXPECT_DOUBLE_EQ(static_cast<double>(a), 0.5);

    rat<int> b(1, 3);
    EXPECT_NEAR(static_cast<double>(b), 0.333333, 0.0001);
}

// =============================================================================
// Mediant
// =============================================================================

TEST(RationalTest, Mediant) {
    // Mediant of 0/1 and 1/1 is 1/2
    auto m = mediant(rat<int>(0, 1), rat<int>(1, 1));
    EXPECT_EQ(m.numerator(), 1);
    EXPECT_EQ(m.denominator(), 2);

    // Mediant of 1/2 and 1/1 is 2/3
    auto m2 = mediant(rat<int>(1, 2), rat<int>(1, 1));
    EXPECT_EQ(m2.numerator(), 2);
    EXPECT_EQ(m2.denominator(), 3);
}

TEST(RationalTest, MediantInequality) {
    // For a/b < c/d, we have a/b < mediant < c/d
    rat<int> a(1, 3);
    rat<int> b(1, 2);
    auto m = mediant(a, b);

    EXPECT_LT(a, m);
    EXPECT_LT(m, b);
}

// =============================================================================
// Algebraic identities
// =============================================================================

TEST(RationalTest, AdditiveIdentity) {
    rat<int> a(3, 4);
    rat<int> zero(0);
    EXPECT_EQ(a + zero, a);
    EXPECT_EQ(zero + a, a);
}

TEST(RationalTest, MultiplicativeIdentity) {
    rat<int> a(3, 4);
    rat<int> one(1);
    EXPECT_EQ(a * one, a);
    EXPECT_EQ(one * a, a);
}

TEST(RationalTest, AdditiveInverse) {
    rat<int> a(3, 4);
    EXPECT_EQ(a + (-a), rat<int>(0));
}

TEST(RationalTest, MultiplicativeInverse) {
    rat<int> a(3, 4);
    EXPECT_EQ(a * a.reciprocal(), rat<int>(1));
}

TEST(RationalTest, Commutativity) {
    rat<int> a(1, 2);
    rat<int> b(2, 3);
    EXPECT_EQ(a + b, b + a);
    EXPECT_EQ(a * b, b * a);
}

TEST(RationalTest, Associativity) {
    rat<int> a(1, 2);
    rat<int> b(2, 3);
    rat<int> c(3, 4);
    EXPECT_EQ((a + b) + c, a + (b + c));
    EXPECT_EQ((a * b) * c, a * (b * c));
}

TEST(RationalTest, Distributivity) {
    rat<int> a(1, 2);
    rat<int> b(2, 3);
    rat<int> c(3, 4);
    EXPECT_EQ(a * (b + c), a * b + a * c);
}
