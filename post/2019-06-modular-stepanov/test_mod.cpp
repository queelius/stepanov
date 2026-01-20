#include <gtest/gtest.h>
#include "mod.hpp"

using namespace modular;

// =============================================================================
// Construction and normalization
// =============================================================================

TEST(ModIntTest, DefaultConstruction) {
    mod_int<7> a;
    EXPECT_EQ(a.v, 0);
}

TEST(ModIntTest, PositiveConstruction) {
    mod_int<7> a(5);
    EXPECT_EQ(a.v, 5);

    mod_int<7> b(10);  // 10 mod 7 = 3
    EXPECT_EQ(b.v, 3);
}

TEST(ModIntTest, NegativeConstruction) {
    mod_int<7> a(-1);  // -1 mod 7 = 6
    EXPECT_EQ(a.v, 6);

    mod_int<7> b(-8);  // -8 mod 7 = -1 mod 7 = 6
    EXPECT_EQ(b.v, 6);
}

// =============================================================================
// Addition
// =============================================================================

TEST(ModIntTest, Addition) {
    mod_int<7> a(3), b(5);
    EXPECT_EQ((a + b).v, 1);  // 3 + 5 = 8 mod 7 = 1
}

TEST(ModIntTest, AdditionIdentity) {
    mod_int<7> a(4), z(0);
    EXPECT_EQ((a + z).v, 4);
    EXPECT_EQ((z + a).v, 4);
}

TEST(ModIntTest, AdditiveInverse) {
    mod_int<7> a(3);
    EXPECT_EQ((a + (-a)).v, 0);
}

// =============================================================================
// Subtraction
// =============================================================================

TEST(ModIntTest, Subtraction) {
    mod_int<7> a(3), b(5);
    EXPECT_EQ((a - b).v, 5);  // 3 - 5 = -2 mod 7 = 5
}

// =============================================================================
// Multiplication
// =============================================================================

TEST(ModIntTest, Multiplication) {
    mod_int<7> a(3), b(4);
    EXPECT_EQ((a * b).v, 5);  // 12 mod 7 = 5
}

TEST(ModIntTest, MultiplicationIdentity) {
    mod_int<7> a(4), one(1);
    EXPECT_EQ((a * one).v, 4);
    EXPECT_EQ((one * a).v, 4);
}

TEST(ModIntTest, MultiplicationByZero) {
    mod_int<7> a(4), z(0);
    EXPECT_EQ((a * z).v, 0);
}

// =============================================================================
// Power
// =============================================================================

TEST(ModIntTest, PowerBasic) {
    mod_int<7> a(2);
    EXPECT_EQ(a.pow(0).v, 1);
    EXPECT_EQ(a.pow(1).v, 2);
    EXPECT_EQ(a.pow(2).v, 4);
    EXPECT_EQ(a.pow(3).v, 1);  // 8 mod 7 = 1
}

TEST(ModIntTest, FermatLittleTheorem) {
    // a^(p-1) ≡ 1 (mod p) for prime p, a ≠ 0
    for (int64_t a = 1; a < 7; ++a) {
        mod_int<7> x(a);
        EXPECT_EQ(x.pow(6).v, 1);
    }
}

TEST(ModIntTest, LargePower) {
    mod_1e9_7 a(2);
    EXPECT_EQ(a.pow(10).v, 1024);  // 2^10 = 1024 < 10^9+7
}

// =============================================================================
// Inverse and Division
// =============================================================================

TEST(ModIntTest, Inverse) {
    // For prime p, a * a^(-1) ≡ 1 (mod p)
    for (int64_t a = 1; a < 7; ++a) {
        mod_int<7> x(a);
        mod_int<7> inv = x.inverse();
        EXPECT_EQ((x * inv).v, 1);
    }
}

TEST(ModIntTest, Division) {
    mod_int<7> a(3), b(2);
    // 3 / 2 ≡ 3 * 4 ≡ 12 ≡ 5 (mod 7) since 2 * 4 ≡ 1
    EXPECT_EQ((a / b).v, 5);

    // Verify: 5 * 2 = 10 ≡ 3 (mod 7) ✓
    EXPECT_EQ((mod_int<7>(5) * b).v, 3);
}

// =============================================================================
// Ring axioms
// =============================================================================

TEST(ModIntTest, Commutativity) {
    mod_int<7> a(3), b(5);
    EXPECT_EQ(a + b, b + a);
    EXPECT_EQ(a * b, b * a);
}

TEST(ModIntTest, Associativity) {
    mod_int<7> a(2), b(3), c(4);
    EXPECT_EQ((a + b) + c, a + (b + c));
    EXPECT_EQ((a * b) * c, a * (b * c));
}

TEST(ModIntTest, Distributivity) {
    mod_int<7> a(2), b(3), c(4);
    EXPECT_EQ(a * (b + c), a * b + a * c);
}

// =============================================================================
// Algebraic operations for generic algorithms
// =============================================================================

TEST(ModIntTest, AlgebraicOps) {
    mod_int<7> a(3);
    EXPECT_EQ(zero(a).v, 0);
    EXPECT_EQ(one(a).v, 1);
    EXPECT_EQ(twice(a).v, 6);  // 2 * 3 = 6
}

TEST(ModIntTest, Half) {
    mod_int<7> a(6);
    EXPECT_EQ(half(a).v, 3);  // 6 / 2 = 3

    // half(5) where 2^(-1) = 4: 5 * 4 = 20 ≡ 6 (mod 7)
    mod_int<7> b(5);
    EXPECT_EQ(half(b).v, 6);
}

TEST(ModIntTest, Even) {
    EXPECT_TRUE(even(mod_int<7>(0)));
    EXPECT_TRUE(even(mod_int<7>(2)));
    EXPECT_TRUE(even(mod_int<7>(4)));
    EXPECT_FALSE(even(mod_int<7>(1)));
    EXPECT_FALSE(even(mod_int<7>(3)));
}

// =============================================================================
// Large modulus
// =============================================================================

TEST(ModIntTest, LargeModulus) {
    mod_1e9_7 a(1000000006);
    mod_1e9_7 b(5);

    // a + b wraps around
    EXPECT_EQ((a + b).v, 4);

    // Large power
    EXPECT_EQ(mod_1e9_7(2).pow(30).v, 1073741824 % 1000000007);
}
