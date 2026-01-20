#include <gtest/gtest.h>
#include "../peasant.hpp"

using namespace peasant;

// =============================================================================
// Concept Tests
// =============================================================================

TEST(PeasantTest, ConceptsSatisfied) {
    static_assert(has_zero<int>);
    static_assert(has_one<int>);
    static_assert(has_twice<int>);
    static_assert(has_half<int>);
    static_assert(has_even<int>);
    static_assert(has_increment<int>);
    static_assert(has_decrement<int>);
    static_assert(algebraic<int>);
    static_assert(algebraic<long>);
    static_assert(algebraic<long long>);
}

TEST(PeasantTest, AdlFunctions) {
    EXPECT_EQ(zero(42), 0);
    EXPECT_EQ(one(42), 1);
    EXPECT_EQ(twice(5), 10);
    EXPECT_EQ(half(10), 5);
    EXPECT_TRUE(even(4));
    EXPECT_FALSE(even(5));
    EXPECT_EQ(increment(5), 6);
    EXPECT_EQ(decrement(5), 4);
    EXPECT_EQ(quotient(17, 5), 3);
    EXPECT_EQ(remainder(17, 5), 2);
}

// =============================================================================
// Product Tests (Russian Peasant)
// =============================================================================

TEST(PeasantTest, ProductBasic) {
    EXPECT_EQ(product(5, 3), 15);
    EXPECT_EQ(product(7, 8), 56);
    EXPECT_EQ(product(12, 12), 144);
}

TEST(PeasantTest, ProductIdentities) {
    EXPECT_EQ(product(42, 0), 0);
    EXPECT_EQ(product(0, 42), 0);
    EXPECT_EQ(product(42, 1), 42);
    EXPECT_EQ(product(1, 42), 42);
}

TEST(PeasantTest, ProductNegative) {
    EXPECT_EQ(product(5, -3), -15);
    EXPECT_EQ(product(-5, 3), -15);
    EXPECT_EQ(product(-5, -3), 15);
}

TEST(PeasantTest, ProductCommutative) {
    for (int a = -10; a <= 10; ++a) {
        for (int b = -10; b <= 10; ++b) {
            EXPECT_EQ(product(a, b), product(b, a));
        }
    }
}

// =============================================================================
// Square Tests
// =============================================================================

TEST(PeasantTest, Square) {
    EXPECT_EQ(square(0), 0);
    EXPECT_EQ(square(1), 1);
    EXPECT_EQ(square(5), 25);
    EXPECT_EQ(square(-5), 25);
    EXPECT_EQ(square(12), 144);
}

// =============================================================================
// Power Tests (Repeated Squaring)
// =============================================================================

TEST(PeasantTest, PowerBasic) {
    EXPECT_EQ(power(2, 0), 1);
    EXPECT_EQ(power(2, 1), 2);
    EXPECT_EQ(power(2, 10), 1024);
    EXPECT_EQ(power(3, 4), 81);
    EXPECT_EQ(power(5, 3), 125);
}

TEST(PeasantTest, PowerZeroBase) {
    EXPECT_EQ(power(0, 1), 0);
    EXPECT_EQ(power(0, 5), 0);
}

TEST(PeasantTest, PowerNegativeBase) {
    EXPECT_EQ(power(-2, 2), 4);
    EXPECT_EQ(power(-2, 3), -8);
    EXPECT_EQ(power(-3, 3), -27);
}

TEST(PeasantTest, PowerLaws) {
    // a^(m+n) = a^m * a^n
    for (int base = 2; base <= 4; ++base) {
        for (int m = 0; m <= 4; ++m) {
            for (int n = 0; n <= 4; ++n) {
                EXPECT_EQ(power(base, m + n), product(power(base, m), power(base, n)));
            }
        }
    }
}

// =============================================================================
// GCD Tests
// =============================================================================

TEST(PeasantTest, GcdBasic) {
    EXPECT_EQ(gcd(10, 5), 5);
    EXPECT_EQ(gcd(12, 8), 4);
    EXPECT_EQ(gcd(17, 13), 1);
    EXPECT_EQ(gcd(54, 24), 6);
}

TEST(PeasantTest, GcdWithZero) {
    EXPECT_EQ(gcd(0, 5), 5);
    EXPECT_EQ(gcd(5, 0), 5);
    EXPECT_EQ(gcd(0, 0), 0);
}

TEST(PeasantTest, GcdCommutative) {
    EXPECT_EQ(gcd(48, 18), gcd(18, 48));
    EXPECT_EQ(gcd(100, 75), gcd(75, 100));
}

TEST(PeasantTest, GcdNegative) {
    EXPECT_EQ(gcd(-10, 5), 5);
    EXPECT_EQ(gcd(10, -5), 5);
    EXPECT_EQ(gcd(-10, -5), 5);
}

// =============================================================================
// Extended GCD Tests
// =============================================================================

TEST(PeasantTest, ExtendedGcd) {
    auto [g, x, y] = extended_gcd(30, 18);
    EXPECT_EQ(g, 6);
    EXPECT_EQ(30 * x + 18 * y, g);  // Bezout's identity
}

TEST(PeasantTest, ExtendedGcdCoprime) {
    auto [g, x, y] = extended_gcd(17, 13);
    EXPECT_EQ(g, 1);
    EXPECT_EQ(17 * x + 13 * y, 1);
}

// =============================================================================
// LCM Tests
// =============================================================================

TEST(PeasantTest, Lcm) {
    EXPECT_EQ(lcm(4, 6), 12);
    EXPECT_EQ(lcm(3, 5), 15);
    EXPECT_EQ(lcm(12, 18), 36);
    EXPECT_EQ(lcm(0, 5), 0);
}

TEST(PeasantTest, LcmGcdRelation) {
    // lcm(a,b) * gcd(a,b) = a * b
    for (int a = 1; a <= 20; ++a) {
        for (int b = 1; b <= 20; ++b) {
            EXPECT_EQ(lcm(a, b) * gcd(a, b), a * b);
        }
    }
}

// =============================================================================
// Coprime Tests
// =============================================================================

TEST(PeasantTest, Coprime) {
    EXPECT_TRUE(coprime(5, 7));
    EXPECT_TRUE(coprime(9, 16));
    EXPECT_FALSE(coprime(6, 9));
    EXPECT_FALSE(coprime(10, 15));
}

// =============================================================================
// Constexpr Tests
// =============================================================================

TEST(PeasantTest, Constexpr) {
    constexpr int p = product(6, 7);
    constexpr int s = square(5);
    constexpr int pw = power(2, 8);
    constexpr int g = gcd(48, 18);
    constexpr int l = lcm(4, 6);
    constexpr bool c = coprime(7, 11);

    EXPECT_EQ(p, 42);
    EXPECT_EQ(s, 25);
    EXPECT_EQ(pw, 256);
    EXPECT_EQ(g, 6);
    EXPECT_EQ(l, 12);
    EXPECT_TRUE(c);
}
