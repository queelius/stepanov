#include <gtest/gtest.h>
#include <stepanov/builtin_adaptors.hpp>  // Must come before math.hpp for ADL
#include <stepanov/math.hpp>
#include <limits>
#include <vector>
#include <random>

using namespace stepanov;

// Test fixture for Math operations
class MathTest : public ::testing::Test {
protected:
    std::mt19937 rng{42}; // Fixed seed for reproducibility

    template<typename T>
    T random_value(T min_val, T max_val) {
        if constexpr (std::is_integral_v<T>) {
            std::uniform_int_distribution<T> dist(min_val, max_val);
            return dist(rng);
        } else {
            std::uniform_real_distribution<T> dist(min_val, max_val);
            return dist(rng);
        }
    }
};

// ============== Product Function Tests ==============

TEST_F(MathTest, ProductBasicCases) {
    // Test with integers
    EXPECT_EQ(product(5, 3), 15);
    EXPECT_EQ(product(7, 8), 56);
    EXPECT_EQ(product(-3, 4), -12);
    EXPECT_EQ(product(-5, -6), 30);

    // Test identity elements
    EXPECT_EQ(product(42, 0), 0);
    EXPECT_EQ(product(0, 42), 0);
    EXPECT_EQ(product(42, 1), 42);
    EXPECT_EQ(product(1, 42), 42);
}

TEST_F(MathTest, ProductEdgeCases) {
    // Test zero multiplication
    EXPECT_EQ(product(0, 0), 0);

    // Test large numbers
    EXPECT_EQ(product(1000, 1000), 1000000);

    // Test negative numbers
    EXPECT_EQ(product(-1, 1), -1);
    EXPECT_EQ(product(-1, -1), 1);
}

TEST_F(MathTest, ProductPowerOfTwo) {
    // Test efficient path for powers of 2
    EXPECT_EQ(product(7, 2), 14);
    EXPECT_EQ(product(7, 4), 28);
    EXPECT_EQ(product(7, 8), 56);
    EXPECT_EQ(product(7, 16), 112);
}

TEST_F(MathTest, ProductOddNumbers) {
    // Test recursive path for odd numbers
    EXPECT_EQ(product(6, 3), 18);
    EXPECT_EQ(product(6, 5), 30);
    EXPECT_EQ(product(6, 7), 42);
    EXPECT_EQ(product(6, 9), 54);
}

TEST_F(MathTest, ProductCommutative) {
    // Test commutativity: a * b = b * a
    for (int i = 0; i < 20; ++i) {
        int a = random_value(-100, 100);
        int b = random_value(-100, 100);
        EXPECT_EQ(product(a, b), product(b, a))
            << "Failed for a=" << a << ", b=" << b;
    }
}

TEST_F(MathTest, ProductAssociative) {
    // Test associativity: (a * b) * c = a * (b * c)
    for (int i = 0; i < 20; ++i) {
        int a = random_value(-20, 20);
        int b = random_value(-20, 20);
        int c = random_value(-20, 20);
        EXPECT_EQ(product(product(a, b), c), product(a, product(b, c)))
            << "Failed for a=" << a << ", b=" << b << ", c=" << c;
    }
}

// ============== Square Function Tests ==============

TEST_F(MathTest, SquareBasicCases) {
    EXPECT_EQ(square(0), 0);
    EXPECT_EQ(square(1), 1);
    EXPECT_EQ(square(2), 4);
    EXPECT_EQ(square(3), 9);
    EXPECT_EQ(square(10), 100);
    EXPECT_EQ(square(-5), 25);
    EXPECT_EQ(square(-1), 1);
}

TEST_F(MathTest, SquareVsProduct) {
    // Verify square(x) = product(x, x)
    for (int i = -50; i <= 50; ++i) {
        EXPECT_EQ(square(i), product(i, i)) << "Failed for i=" << i;
    }
}

// ============== Power Function Tests ==============

TEST_F(MathTest, PowerBasicCases) {
    EXPECT_EQ(power(2, 0), 1);  // Any number to power 0 is 1
    EXPECT_EQ(power(2, 1), 2);  // Any number to power 1 is itself
    EXPECT_EQ(power(2, 2), 4);
    EXPECT_EQ(power(2, 3), 8);
    EXPECT_EQ(power(2, 10), 1024);
    EXPECT_EQ(power(3, 4), 81);
    EXPECT_EQ(power(5, 3), 125);
}

TEST_F(MathTest, PowerZeroBase) {
    EXPECT_EQ(power(0, 1), 0);
    EXPECT_EQ(power(0, 2), 0);
    EXPECT_EQ(power(0, 10), 0);
}

TEST_F(MathTest, PowerNegativeBase) {
    EXPECT_EQ(power(-2, 0), 1);
    EXPECT_EQ(power(-2, 1), -2);
    EXPECT_EQ(power(-2, 2), 4);
    EXPECT_EQ(power(-2, 3), -8);
    EXPECT_EQ(power(-2, 4), 16);
    EXPECT_EQ(power(-3, 3), -27);
}

TEST_F(MathTest, PowerEfficientPath) {
    // Test efficient path through repeated squaring
    // 2^16 should use 4 squarings instead of 16 multiplications
    EXPECT_EQ(power(2, 16), 65536);
    EXPECT_EQ(power(3, 8), 6561);
}

TEST_F(MathTest, PowerLaws) {
    // Test power laws: a^(m+n) = a^m * a^n
    for (int i = 0; i < 10; ++i) {
        int base = random_value(2, 5);
        int m = random_value(1, 5);
        int n = random_value(1, 5);
        EXPECT_EQ(power(base, m + n), product(power(base, m), power(base, n)))
            << "Failed for base=" << base << ", m=" << m << ", n=" << n;
    }
}

// ============== Sum Function Tests ==============

TEST_F(MathTest, SumBasicCases) {
    EXPECT_EQ(sum(0, 0), 0);
    EXPECT_EQ(sum(1, 0), 1);
    EXPECT_EQ(sum(0, 1), 1);
    EXPECT_EQ(sum(1, 1), 2);
    EXPECT_EQ(sum(5, 3), 8);
    EXPECT_EQ(sum(10, 20), 30);
}

TEST_F(MathTest, SumNegativeNumbers) {
    EXPECT_EQ(sum(-5, 3), -2);
    EXPECT_EQ(sum(5, -3), 2);
    EXPECT_EQ(sum(-5, -3), -8);
}

TEST_F(MathTest, SumIdentity) {
    // Test additive identity: a + 0 = a
    for (int i = -50; i <= 50; ++i) {
        EXPECT_EQ(sum(i, 0), i) << "Failed for i=" << i;
        EXPECT_EQ(sum(0, i), i) << "Failed for i=" << i;
    }
}

TEST_F(MathTest, SumCommutative) {
    // Test commutativity: a + b = b + a
    for (int i = 0; i < 20; ++i) {
        int a = random_value(-100, 100);
        int b = random_value(-100, 100);
        EXPECT_EQ(sum(a, b), sum(b, a))
            << "Failed for a=" << a << ", b=" << b;
    }
}

TEST_F(MathTest, SumAssociative) {
    // Test associativity: (a + b) + c = a + (b + c)
    for (int i = 0; i < 20; ++i) {
        int a = random_value(-50, 50);
        int b = random_value(-50, 50);
        int c = random_value(-50, 50);
        EXPECT_EQ(sum(sum(a, b), c), sum(a, sum(b, c)))
            << "Failed for a=" << a << ", b=" << b << ", c=" << c;
    }
}

TEST_F(MathTest, SumEvenNumbers) {
    // Test efficient path for even numbers
    EXPECT_EQ(sum(4, 6), 10);
    EXPECT_EQ(sum(8, 12), 20);
    EXPECT_EQ(sum(100, 200), 300);
}

TEST_F(MathTest, SumOddNumbers) {
    // Test path with odd numbers
    EXPECT_EQ(sum(3, 5), 8);
    EXPECT_EQ(sum(7, 11), 18);
    EXPECT_EQ(sum(13, 17), 30);
}

TEST_F(MathTest, SumMixedParity) {
    // Test mixing even and odd
    EXPECT_EQ(sum(3, 4), 7);
    EXPECT_EQ(sum(6, 7), 13);
    EXPECT_EQ(sum(10, 15), 25);
}

// ============== Power Accumulate Tests ==============

TEST_F(MathTest, PowerAccumulateBasic) {
    EXPECT_EQ(power_accumulate(2, 0, 1), 1);
    EXPECT_EQ(power_accumulate(2, 1, 1), 2);
    EXPECT_EQ(power_accumulate(2, 3, 1), 8);
    EXPECT_EQ(power_accumulate(2, 10, 1), 1024);
    EXPECT_EQ(power_accumulate(3, 4, 1), 81);
}

TEST_F(MathTest, PowerAccumulateWithInitialValue) {
    EXPECT_EQ(power_accumulate(2, 3, 5), 8 * 5);  // 2^3 * 5
    EXPECT_EQ(power_accumulate(3, 2, 10), 9 * 10); // 3^2 * 10
}

TEST_F(MathTest, PowerAccumulateEdgeCases) {
    EXPECT_EQ(power_accumulate(0, 5, 1), 0);
    EXPECT_EQ(power_accumulate(1, 100, 1), 1);
    EXPECT_EQ(power_accumulate(10, 0, 7), 7);
}

// ============== Power Mod Tests ==============

TEST_F(MathTest, PowerModBasic) {
    EXPECT_EQ(power_mod(2, 3, 5), 3);   // 2^3 = 8, 8 % 5 = 3
    EXPECT_EQ(power_mod(3, 4, 7), 4);   // 3^4 = 81, 81 % 7 = 4
    EXPECT_EQ(power_mod(5, 3, 13), 8);  // 5^3 = 125, 125 % 13 = 8
}

TEST_F(MathTest, PowerModLargeExponents) {
    // Test with large exponents where direct computation would overflow
    EXPECT_EQ(power_mod(2, 100, 7), 2);
    EXPECT_EQ(power_mod(3, 100, 7), 4);
    EXPECT_EQ(power_mod(7, 50, 11), 1);  // 7^50 mod 11 = 1
}

TEST_F(MathTest, PowerModFermatLittleTheorem) {
    // Test Fermat's little theorem: a^(p-1) ≡ 1 (mod p) for prime p
    EXPECT_EQ(power_mod(2, 10, 11), 1);  // 2^10 mod 11 = 1
    EXPECT_EQ(power_mod(3, 12, 13), 1);  // 3^12 mod 13 = 1
    EXPECT_EQ(power_mod(5, 6, 7), 1);    // 5^6 mod 7 = 1
}

// ============== Multiply Accumulate Tests ==============

TEST_F(MathTest, MultiplyAccumulateBasic) {
    EXPECT_EQ(multiply_accumulate(3, 4, 5), 17);    // 3*4 + 5 = 17
    EXPECT_EQ(multiply_accumulate(2, 7, 10), 24);   // 2*7 + 10 = 24
    EXPECT_EQ(multiply_accumulate(5, 6, 0), 30);    // 5*6 + 0 = 30
}

TEST_F(MathTest, MultiplyAccumulateNegative) {
    EXPECT_EQ(multiply_accumulate(-3, 4, 5), -7);   // -3*4 + 5 = -7
    EXPECT_EQ(multiply_accumulate(3, -4, 5), -7);   // 3*-4 + 5 = -7
    EXPECT_EQ(multiply_accumulate(3, 4, -5), 7);    // 3*4 + -5 = 7
}

// ============== Inner Product Tests ==============

TEST_F(MathTest, InnerProductBasic) {
    std::vector<int> v1 = {1, 2, 3};
    std::vector<int> v2 = {4, 5, 6};
    EXPECT_EQ(inner_product(v1.begin(), v1.end(), v2.begin(), 0), 32);
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
}

TEST_F(MathTest, InnerProductWithInitialValue) {
    std::vector<int> v1 = {1, 2, 3};
    std::vector<int> v2 = {4, 5, 6};
    EXPECT_EQ(inner_product(v1.begin(), v1.end(), v2.begin(), 10), 42);
    // 10 + 1*4 + 2*5 + 3*6 = 10 + 32 = 42
}

TEST_F(MathTest, InnerProductEmptyVectors) {
    std::vector<int> v1, v2;
    EXPECT_EQ(inner_product(v1.begin(), v1.end(), v2.begin(), 5), 5);
}

TEST_F(MathTest, InnerProductSingleElement) {
    std::vector<int> v1 = {3};
    std::vector<int> v2 = {7};
    EXPECT_EQ(inner_product(v1.begin(), v1.end(), v2.begin(), 0), 21);
}

TEST_F(MathTest, InnerProductOrthogonal) {
    std::vector<int> v1 = {1, 0, 0};
    std::vector<int> v2 = {0, 1, 0};
    EXPECT_EQ(inner_product(v1.begin(), v1.end(), v2.begin(), 0), 0);
}

// ============== Abs Function Tests ==============

TEST_F(MathTest, AbsBasic) {
    EXPECT_EQ(abs(5), 5);
    EXPECT_EQ(abs(-5), 5);
    EXPECT_EQ(abs(0), 0);
    EXPECT_EQ(abs(100), 100);
    EXPECT_EQ(abs(-100), 100);
}

TEST_F(MathTest, AbsEdgeCases) {
    EXPECT_EQ(abs(1), 1);
    EXPECT_EQ(abs(-1), 1);

    // Test with different types
    EXPECT_EQ(abs(5L), 5L);
    EXPECT_EQ(abs(-5L), 5L);
}

// ============== Min/Max Function Tests ==============

TEST_F(MathTest, MinBasic) {
    EXPECT_EQ(min(3, 5), 3);
    EXPECT_EQ(min(5, 3), 3);
    EXPECT_EQ(min(-3, 5), -3);
    EXPECT_EQ(min(-5, -3), -5);
    EXPECT_EQ(min(0, 0), 0);
}

TEST_F(MathTest, MaxBasic) {
    EXPECT_EQ(max(3, 5), 5);
    EXPECT_EQ(max(5, 3), 5);
    EXPECT_EQ(max(-3, 5), 5);
    EXPECT_EQ(max(-5, -3), -3);
    EXPECT_EQ(max(0, 0), 0);
}

TEST_F(MathTest, MinMaxProperties) {
    // Test idempotence: min(a, a) = a, max(a, a) = a
    for (int i = -10; i <= 10; ++i) {
        EXPECT_EQ(min(i, i), i);
        EXPECT_EQ(max(i, i), i);
    }

    // Test that min(a, b) + max(a, b) = a + b
    for (int i = 0; i < 20; ++i) {
        int a = random_value(-50, 50);
        int b = random_value(-50, 50);
        EXPECT_EQ(min(a, b) + max(a, b), a + b)
            << "Failed for a=" << a << ", b=" << b;
    }
}

// ============== Remainder/Quotient Tests ==============

TEST_F(MathTest, RemainderBasic) {
    EXPECT_EQ(remainder(10, 3), 1);
    EXPECT_EQ(remainder(15, 4), 3);
    EXPECT_EQ(remainder(20, 5), 0);
    EXPECT_EQ(remainder(7, 7), 0);
}

TEST_F(MathTest, QuotientBasic) {
    EXPECT_EQ(quotient(10, 3), 3);
    EXPECT_EQ(quotient(15, 4), 3);
    EXPECT_EQ(quotient(20, 5), 4);
    EXPECT_EQ(quotient(7, 7), 1);
}

TEST_F(MathTest, DivisionIdentity) {
    // Test: a = quotient(a, b) * b + remainder(a, b)
    for (int i = 0; i < 20; ++i) {
        int a = random_value(1, 100);
        int b = random_value(1, 50);
        EXPECT_EQ(a, quotient(a, b) * b + remainder(a, b))
            << "Failed for a=" << a << ", b=" << b;
    }
}

// ============== Floating Point Tests ==============

TEST_F(MathTest, FloatingPointOperations) {
    // Test with doubles
    double a = 3.5, b = 2.5;

    // Basic operations should work with floating point
    EXPECT_DOUBLE_EQ(sum(a, b), 6.0);
    EXPECT_DOUBLE_EQ(product(a, 2.0), 7.0);
    EXPECT_DOUBLE_EQ(square(3.0), 9.0);

    // Test abs with floating point
    EXPECT_DOUBLE_EQ(abs(-3.14), 3.14);
    EXPECT_DOUBLE_EQ(abs(2.718), 2.718);

    // Test min/max with floating point
    EXPECT_DOUBLE_EQ(min(3.14, 2.718), 2.718);
    EXPECT_DOUBLE_EQ(max(3.14, 2.718), 3.14);
}

// ============== Performance/Stress Tests ==============

TEST_F(MathTest, StressTestLargePowers) {
    // Test that large powers don't cause stack overflow
    // Using repeated squaring should handle this efficiently
    int64_t result = power_accumulate(2LL, 30, 1LL);
    EXPECT_EQ(result, 1073741824LL);
}

TEST_F(MathTest, StressTestLargeModularPowers) {
    // Test large modular exponentiation
    int64_t result = power_mod(12345LL, 67890, 99991LL);
    EXPECT_GT(result, 0);
    EXPECT_LT(result, 99991LL);
}

// ============== Type Trait Tests ==============

TEST_F(MathTest, ConceptsSatisfied) {
    // Verify that built-in types satisfy our concepts
    static_assert(stepanov::has_twice<int>);
    static_assert(stepanov::has_half<int>);
    static_assert(stepanov::has_even<int>);
    static_assert(stepanov::has_increment<int>);
    static_assert(stepanov::has_decrement<int>);

    static_assert(stepanov::regular<int>);
    static_assert(stepanov::ring<int>);
    static_assert(stepanov::ordered_ring<int>);
}