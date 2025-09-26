#include <gtest/gtest.h>
#include <stepanov/builtin_adaptors.hpp>
#include <stepanov/math.hpp>  // For square function
#include <limits>
#include <cmath>
#include <random>

using namespace stepanov;

class BuiltinAdaptorsTest : public ::testing::Test {
protected:
    std::mt19937 rng{42}; // Fixed seed for reproducibility
};

// ============== Integral Type Adaptors ==============

TEST_F(BuiltinAdaptorsTest, IntegralEven) {
    // Test even() for various integral types
    EXPECT_TRUE(even(0));
    EXPECT_TRUE(even(2));
    EXPECT_TRUE(even(4));
    EXPECT_TRUE(even(-2));
    EXPECT_TRUE(even(-4));
    EXPECT_TRUE(even(100));
    EXPECT_TRUE(even(-100));

    EXPECT_FALSE(even(1));
    EXPECT_FALSE(even(3));
    EXPECT_FALSE(even(-1));
    EXPECT_FALSE(even(-3));
    EXPECT_FALSE(even(99));
    EXPECT_FALSE(even(-99));

    // Test with different integral types
    EXPECT_TRUE(even(2L));
    EXPECT_TRUE(even(2LL));
    EXPECT_TRUE(even(static_cast<short>(2)));
    EXPECT_TRUE(even(static_cast<unsigned>(2)));

    EXPECT_FALSE(even(3L));
    EXPECT_FALSE(even(3LL));
    EXPECT_FALSE(even(static_cast<short>(3)));
    EXPECT_FALSE(even(static_cast<unsigned>(3)));
}

TEST_F(BuiltinAdaptorsTest, IntegralTwice) {
    EXPECT_EQ(twice(0), 0);
    EXPECT_EQ(twice(1), 2);
    EXPECT_EQ(twice(5), 10);
    EXPECT_EQ(twice(100), 200);
    EXPECT_EQ(twice(-3), -6);
    EXPECT_EQ(twice(-50), -100);

    // Test with different integral types
    EXPECT_EQ(twice(5L), 10L);
    EXPECT_EQ(twice(5LL), 10LL);
    EXPECT_EQ(twice(static_cast<short>(5)), static_cast<short>(10));
    EXPECT_EQ(twice(5u), 10u);

    // Test edge cases (avoid overflow in tests)
    int max_half = std::numeric_limits<int>::max() / 2;
    EXPECT_EQ(twice(max_half), max_half * 2);
}

TEST_F(BuiltinAdaptorsTest, IntegralHalf) {
    EXPECT_EQ(half(0), 0);
    EXPECT_EQ(half(2), 1);
    EXPECT_EQ(half(10), 5);
    EXPECT_EQ(half(100), 50);
    EXPECT_EQ(half(-6), -3);
    EXPECT_EQ(half(-100), -50);

    // Test odd numbers (floor division)
    EXPECT_EQ(half(1), 0);
    EXPECT_EQ(half(3), 1);
    EXPECT_EQ(half(5), 2);
    EXPECT_EQ(half(99), 49);
    EXPECT_EQ(half(-1), -1);  // Note: -1 >> 1 = -1 (arithmetic shift)
    EXPECT_EQ(half(-3), -2);

    // Test with different integral types
    EXPECT_EQ(half(10L), 5L);
    EXPECT_EQ(half(10LL), 5LL);
    EXPECT_EQ(half(static_cast<short>(10)), static_cast<short>(5));
    EXPECT_EQ(half(10u), 5u);
}

TEST_F(BuiltinAdaptorsTest, IntegralIncrement) {
    EXPECT_EQ(increment(0), 1);
    EXPECT_EQ(increment(5), 6);
    EXPECT_EQ(increment(-1), 0);
    EXPECT_EQ(increment(-5), -4);
    EXPECT_EQ(increment(99), 100);

    // Test with different integral types
    EXPECT_EQ(increment(5L), 6L);
    EXPECT_EQ(increment(5LL), 6LL);
    EXPECT_EQ(increment(static_cast<short>(5)), static_cast<short>(6));
    EXPECT_EQ(increment(5u), 6u);
}

TEST_F(BuiltinAdaptorsTest, IntegralDecrement) {
    EXPECT_EQ(decrement(1), 0);
    EXPECT_EQ(decrement(5), 4);
    EXPECT_EQ(decrement(0), -1);
    EXPECT_EQ(decrement(-4), -5);
    EXPECT_EQ(decrement(100), 99);

    // Test with different integral types
    EXPECT_EQ(decrement(6L), 5L);
    EXPECT_EQ(decrement(6LL), 5LL);
    EXPECT_EQ(decrement(static_cast<short>(6)), static_cast<short>(5));
    EXPECT_EQ(decrement(6u), 5u);
}

// ============== Floating Point Type Adaptors ==============

TEST_F(BuiltinAdaptorsTest, FloatingPointEven) {
    // Test even() for floating point types
    EXPECT_TRUE(even(0.0));
    EXPECT_TRUE(even(2.0));
    EXPECT_TRUE(even(4.0));
    EXPECT_TRUE(even(-2.0));
    EXPECT_TRUE(even(-4.0));
    EXPECT_TRUE(even(100.0));

    EXPECT_FALSE(even(1.0));
    EXPECT_FALSE(even(3.0));
    EXPECT_FALSE(even(-1.0));
    EXPECT_FALSE(even(-3.0));

    // Test fractional values
    EXPECT_FALSE(even(2.5));  // 2.5 % 2 = 0.5 != 0
    EXPECT_FALSE(even(3.7));
    EXPECT_TRUE(even(4.0));

    // Test with float type
    EXPECT_TRUE(even(2.0f));
    EXPECT_FALSE(even(3.0f));
}

TEST_F(BuiltinAdaptorsTest, FloatingPointTwice) {
    EXPECT_DOUBLE_EQ(twice(0.0), 0.0);
    EXPECT_DOUBLE_EQ(twice(1.0), 2.0);
    EXPECT_DOUBLE_EQ(twice(5.5), 11.0);
    EXPECT_DOUBLE_EQ(twice(-3.5), -7.0);
    EXPECT_DOUBLE_EQ(twice(100.25), 200.5);

    // Test with float type
    EXPECT_FLOAT_EQ(twice(2.5f), 5.0f);
    EXPECT_FLOAT_EQ(twice(-1.5f), -3.0f);

    // Test special values
    EXPECT_TRUE(std::isinf(twice(std::numeric_limits<double>::infinity())));
    EXPECT_TRUE(std::isnan(twice(std::numeric_limits<double>::quiet_NaN())));
}

TEST_F(BuiltinAdaptorsTest, FloatingPointHalf) {
    EXPECT_DOUBLE_EQ(half(0.0), 0.0);
    EXPECT_DOUBLE_EQ(half(2.0), 1.0);
    EXPECT_DOUBLE_EQ(half(10.0), 5.0);
    EXPECT_DOUBLE_EQ(half(11.0), 5.5);
    EXPECT_DOUBLE_EQ(half(-6.0), -3.0);
    EXPECT_DOUBLE_EQ(half(-7.0), -3.5);

    // Test with float type
    EXPECT_FLOAT_EQ(half(5.0f), 2.5f);
    EXPECT_FLOAT_EQ(half(-3.0f), -1.5f);

    // Test special values
    EXPECT_TRUE(std::isinf(half(std::numeric_limits<double>::infinity())));
    EXPECT_TRUE(std::isnan(half(std::numeric_limits<double>::quiet_NaN())));
}

TEST_F(BuiltinAdaptorsTest, FloatingPointIncrement) {
    EXPECT_DOUBLE_EQ(increment(0.0), 1.0);
    EXPECT_DOUBLE_EQ(increment(5.5), 6.5);
    EXPECT_DOUBLE_EQ(increment(-1.5), -0.5);
    EXPECT_DOUBLE_EQ(increment(99.9), 100.9);

    // Test with float type
    EXPECT_FLOAT_EQ(increment(5.5f), 6.5f);
    EXPECT_FLOAT_EQ(increment(-1.5f), -0.5f);
}

TEST_F(BuiltinAdaptorsTest, FloatingPointDecrement) {
    EXPECT_DOUBLE_EQ(decrement(1.0), 0.0);
    EXPECT_DOUBLE_EQ(decrement(5.5), 4.5);
    EXPECT_DOUBLE_EQ(decrement(0.0), -1.0);
    EXPECT_DOUBLE_EQ(decrement(-3.5), -4.5);

    // Test with float type
    EXPECT_FLOAT_EQ(decrement(6.5f), 5.5f);
    EXPECT_FLOAT_EQ(decrement(-0.5f), -1.5f);
}

// ============== Euclidean Domain Operations ==============

TEST_F(BuiltinAdaptorsTest, IntegralQuotient) {
    EXPECT_EQ(quotient(10, 3), 3);
    EXPECT_EQ(quotient(15, 4), 3);
    EXPECT_EQ(quotient(20, 5), 4);
    EXPECT_EQ(quotient(7, 7), 1);
    EXPECT_EQ(quotient(0, 5), 0);

    // Test negative values
    EXPECT_EQ(quotient(-10, 3), -3);
    EXPECT_EQ(quotient(10, -3), -3);
    EXPECT_EQ(quotient(-10, -3), 3);

    // Test with different integral types
    EXPECT_EQ(quotient(10L, 3L), 3L);
    EXPECT_EQ(quotient(10LL, 3LL), 3LL);
}

TEST_F(BuiltinAdaptorsTest, IntegralRemainder) {
    EXPECT_EQ(remainder(10, 3), 1);
    EXPECT_EQ(remainder(15, 4), 3);
    EXPECT_EQ(remainder(20, 5), 0);
    EXPECT_EQ(remainder(7, 7), 0);
    EXPECT_EQ(remainder(0, 5), 0);

    // Test negative values (behavior depends on implementation)
    int a = -10, b = 3;
    int q = quotient(a, b);
    int r = remainder(a, b);
    EXPECT_EQ(a, q * b + r);  // Verify division identity

    // Test with different integral types
    EXPECT_EQ(remainder(10L, 3L), 1L);
    EXPECT_EQ(remainder(10LL, 3LL), 1LL);
}

TEST_F(BuiltinAdaptorsTest, FloorHalf) {
    // Floor half for integral types
    EXPECT_EQ(floor_half(0), 0);
    EXPECT_EQ(floor_half(1), 0);
    EXPECT_EQ(floor_half(2), 1);
    EXPECT_EQ(floor_half(3), 1);
    EXPECT_EQ(floor_half(10), 5);
    EXPECT_EQ(floor_half(11), 5);

    // Test negative values
    EXPECT_EQ(floor_half(-1), -1);
    EXPECT_EQ(floor_half(-2), -1);
    EXPECT_EQ(floor_half(-3), -2);
}

// Note: square function tests moved to test_math.cpp as square is defined in math.hpp

TEST_F(BuiltinAdaptorsTest, Norm) {
    // Test norm for integral types
    EXPECT_EQ(norm(5), 5);
    EXPECT_EQ(norm(-5), 5);
    EXPECT_EQ(norm(0), 0);
    EXPECT_EQ(norm(100), 100);
    EXPECT_EQ(norm(-100), 100);

    // Test with different integral types
    EXPECT_EQ(norm(5L), 5L);
    EXPECT_EQ(norm(-5L), 5L);
    EXPECT_EQ(norm(5LL), 5LL);
    EXPECT_EQ(norm(-5LL), 5LL);

    // Test norm for unsigned types (should return same value)
    EXPECT_EQ(norm(5u), 5u);
    EXPECT_EQ(norm(100u), 100u);

    // Test norm for floating point types
    EXPECT_DOUBLE_EQ(norm(5.5), 5.5);
    EXPECT_DOUBLE_EQ(norm(-5.5), 5.5);
    EXPECT_FLOAT_EQ(norm(3.14f), 3.14f);
    EXPECT_FLOAT_EQ(norm(-3.14f), 3.14f);
}

// ============== Properties and Invariants ==============

TEST_F(BuiltinAdaptorsTest, TwiceHalfInvariant) {
    // Test that twice(half(x)) recovers even x
    for (int x = -100; x <= 100; x += 2) {  // Even numbers
        EXPECT_EQ(twice(half(x)), x) << "Failed for x=" << x;
    }

    // For odd numbers, twice(half(x)) = x - 1
    for (int x = -99; x <= 99; x += 2) {  // Odd numbers
        EXPECT_EQ(twice(half(x)), x - 1) << "Failed for x=" << x;
    }

    // Test with floating point (exact recovery)
    for (double x = -10.0; x <= 10.0; x += 0.5) {
        EXPECT_DOUBLE_EQ(twice(half(x)), x) << "Failed for x=" << x;
    }
}

TEST_F(BuiltinAdaptorsTest, IncrementDecrementInvariant) {
    // Test that increment and decrement are inverses
    for (int x = -100; x <= 100; ++x) {
        EXPECT_EQ(decrement(increment(x)), x) << "Failed for x=" << x;
        EXPECT_EQ(increment(decrement(x)), x) << "Failed for x=" << x;
    }

    // Test with floating point
    // Use EXPECT_NEAR with a small tolerance due to floating-point precision
    for (double x = -10.0; x <= 10.0; x += 0.1) {
        EXPECT_NEAR(decrement(increment(x)), x, 1e-10) << "Failed for x=" << x;
        EXPECT_NEAR(increment(decrement(x)), x, 1e-10) << "Failed for x=" << x;
    }
}

TEST_F(BuiltinAdaptorsTest, QuotientRemainderIdentity) {
    // Test division identity: a = quotient(a, b) * b + remainder(a, b)
    std::uniform_int_distribution<int> dist_a(-1000, 1000);
    std::uniform_int_distribution<int> dist_b(1, 100);

    for (int i = 0; i < 100; ++i) {
        int a = dist_a(rng);
        int b = dist_b(rng);
        if (dist_a(rng) % 2 == 0) b = -b;  // Sometimes use negative divisor

        int q = quotient(a, b);
        int r = remainder(a, b);
        EXPECT_EQ(a, q * b + r)
            << "Failed for a=" << a << ", b=" << b
            << ", q=" << q << ", r=" << r;
    }
}

TEST_F(BuiltinAdaptorsTest, EvenOddPartition) {
    // Test that even() correctly partitions integers
    for (int x = -100; x <= 100; ++x) {
        if (even(x)) {
            EXPECT_EQ(x % 2, 0) << "x=" << x << " reported as even but x%2=" << (x % 2);
        } else {
            EXPECT_NE(x % 2, 0) << "x=" << x << " reported as odd but x%2=" << (x % 2);
        }
    }
}

// ============== Constexpr Tests ==============

TEST_F(BuiltinAdaptorsTest, ConstexprOperations) {
    // Test that operations are constexpr
    constexpr int ce_twice = twice(5);
    constexpr int ce_half = half(10);
    constexpr bool ce_even = even(4);
    constexpr int ce_inc = increment(5);
    constexpr int ce_dec = decrement(5);
    constexpr int ce_quot = quotient(10, 3);
    constexpr int ce_rem = remainder(10, 3);
    // Note: square is tested in test_math.cpp

    EXPECT_EQ(ce_twice, 10);
    EXPECT_EQ(ce_half, 5);
    EXPECT_TRUE(ce_even);
    EXPECT_EQ(ce_inc, 6);
    EXPECT_EQ(ce_dec, 4);
    EXPECT_EQ(ce_quot, 3);
    EXPECT_EQ(ce_rem, 1);

    // Test with floating point
    constexpr double ce_twice_d = twice(5.5);
    constexpr double ce_half_d = half(11.0);
    constexpr bool ce_even_d = even(4.0);
    constexpr double ce_inc_d = increment(5.5);
    constexpr double ce_dec_d = decrement(5.5);

    EXPECT_DOUBLE_EQ(ce_twice_d, 11.0);
    EXPECT_DOUBLE_EQ(ce_half_d, 5.5);
    EXPECT_TRUE(ce_even_d);
    EXPECT_DOUBLE_EQ(ce_inc_d, 6.5);
    EXPECT_DOUBLE_EQ(ce_dec_d, 4.5);
}

// ============== Type Deduction Tests ==============

TEST_F(BuiltinAdaptorsTest, TypeDeduction) {
    // Test that operations preserve types correctly
    auto result_int = twice(5);
    static_assert(std::is_same_v<decltype(result_int), int>);

    auto result_long = twice(5L);
    static_assert(std::is_same_v<decltype(result_long), long>);

    auto result_double = twice(5.0);
    static_assert(std::is_same_v<decltype(result_double), double>);

    auto result_float = twice(5.0f);
    static_assert(std::is_same_v<decltype(result_float), float>);

    // Test that even() returns bool for all types
    auto even_int = even(4);
    auto even_double = even(4.0);
    static_assert(std::is_same_v<decltype(even_int), bool>);
    static_assert(std::is_same_v<decltype(even_double), bool>);
}

// ============== Edge Cases ==============

TEST_F(BuiltinAdaptorsTest, EdgeCases) {
    // Test with minimum and maximum values (be careful with overflow)
    constexpr int int_max = std::numeric_limits<int>::max();
    constexpr int int_min = std::numeric_limits<int>::min();

    // Half of max/min values
    EXPECT_EQ(half(int_max), int_max / 2);
    EXPECT_EQ(half(int_min), int_min / 2);

    // Even/odd at boundaries
    EXPECT_EQ(even(int_max), (int_max % 2) == 0);
    EXPECT_EQ(even(int_min), (int_min % 2) == 0);

    // Norm at boundaries
    EXPECT_EQ(norm(int_max), int_max);
    // Note: norm(int_min) might overflow for signed types

    // Test with zero
    EXPECT_EQ(twice(0), 0);
    EXPECT_EQ(half(0), 0);
    EXPECT_TRUE(even(0));
    EXPECT_EQ(increment(0), 1);
    EXPECT_EQ(decrement(0), -1);
    EXPECT_EQ(norm(0), 0);
}