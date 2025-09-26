#include <gtest/gtest.h>
#include <stepanov/builtin_adaptors.hpp>
#include <stepanov/mod.hpp>
#include <vector>
#include <random>

using namespace stepanov;

class ModTest : public ::testing::Test {
protected:
    std::mt19937 rng{42}; // Fixed seed for reproducibility
};

// ============== Basic Modular Arithmetic Tests ==============

TEST_F(ModTest, ModAddBasicCases) {
    // Basic addition modulo
    EXPECT_EQ(mod_add(5, 7, 10), 2);     // (5 + 7) % 10 = 2
    EXPECT_EQ(mod_add(8, 9, 10), 7);     // (8 + 9) % 10 = 7
    EXPECT_EQ(mod_add(3, 4, 7), 0);      // (3 + 4) % 7 = 0
    EXPECT_EQ(mod_add(15, 17, 13), 6);   // (15 + 17) % 13 = 6

    // Identity element
    EXPECT_EQ(mod_add(5, 0, 10), 5);
    EXPECT_EQ(mod_add(0, 7, 10), 7);

    // Addition with wrap-around
    EXPECT_EQ(mod_add(9, 9, 10), 8);
    EXPECT_EQ(mod_add(6, 6, 7), 5);
}

TEST_F(ModTest, ModAddLargeNumbers) {
    // Test with larger moduli to avoid overflow issues
    EXPECT_EQ(mod_add(1000000, 2000000, 2500000), 500000);
    EXPECT_EQ(mod_add(999999, 999999, 1000000), 999998);

    // Numbers already larger than modulus
    EXPECT_EQ(mod_add(25, 30, 10), 5);  // (25 % 10 + 30 % 10) % 10
}

TEST_F(ModTest, ModSubBasicCases) {
    // Basic subtraction modulo
    EXPECT_EQ(mod_sub(7, 3, 10), 4);     // (7 - 3) % 10 = 4
    EXPECT_EQ(mod_sub(3, 7, 10), 6);     // (3 - 7 + 10) % 10 = 6
    EXPECT_EQ(mod_sub(8, 8, 10), 0);     // (8 - 8) % 10 = 0

    // Subtraction with negative results
    EXPECT_EQ(mod_sub(2, 5, 7), 4);      // (2 - 5 + 7) % 7 = 4
    EXPECT_EQ(mod_sub(0, 1, 10), 9);     // (0 - 1 + 10) % 10 = 9
}

TEST_F(ModTest, ModMulBasicCases) {
    // Basic multiplication modulo
    EXPECT_EQ(mod_mul(3, 4, 10), 2);     // (3 * 4) % 10 = 2
    EXPECT_EQ(mod_mul(5, 7, 11), 2);     // (5 * 7) % 11 = 2
    EXPECT_EQ(mod_mul(6, 6, 7), 1);      // (6 * 6) % 7 = 1

    // Multiplication by zero
    EXPECT_EQ(mod_mul(0, 100, 10), 0);
    EXPECT_EQ(mod_mul(100, 0, 10), 0);

    // Multiplication by one
    EXPECT_EQ(mod_mul(7, 1, 10), 7);
    EXPECT_EQ(mod_mul(1, 8, 10), 8);
}

TEST_F(ModTest, ModMulLargeNumbers) {
    // Test with large numbers to ensure no overflow
    int64_t a = 1000000007;
    int64_t b = 1000000009;
    int64_t m = 1000000021;

    // Compute using mod_mul to avoid overflow
    auto result = mod_mul(a, b, m);
    EXPECT_GE(result, 0);
    EXPECT_LT(result, m);
}

TEST_F(ModTest, ModPowBasicCases) {
    // Basic modular exponentiation
    EXPECT_EQ(mod_pow(2, 3, 5), 3);      // 2^3 % 5 = 8 % 5 = 3
    EXPECT_EQ(mod_pow(3, 4, 7), 4);      // 3^4 % 7 = 81 % 7 = 4
    EXPECT_EQ(mod_pow(5, 3, 13), 8);     // 5^3 % 13 = 125 % 13 = 8

    // Special cases
    EXPECT_EQ(mod_pow(0, 100, 10), 0);   // 0^n = 0
    EXPECT_EQ(mod_pow(1, 1000000, 10), 1); // 1^n = 1
    EXPECT_EQ(mod_pow(10, 0, 7), 1);     // n^0 = 1
}

TEST_F(ModTest, ModPowFermatLittleTheorem) {
    // Fermat's little theorem: a^(p-1) ≡ 1 (mod p) for prime p
    EXPECT_EQ(mod_pow(2, 10, 11), 1);    // 2^10 % 11 = 1
    EXPECT_EQ(mod_pow(3, 6, 7), 1);      // 3^6 % 7 = 1
    EXPECT_EQ(mod_pow(5, 12, 13), 1);    // 5^12 % 13 = 1
    EXPECT_EQ(mod_pow(7, 16, 17), 1);    // 7^16 % 17 = 1
}

TEST_F(ModTest, ModPowLargeExponents) {
    // Test with very large exponents
    EXPECT_EQ(mod_pow(2, 1000000, 1000), mod_pow(2, 1000000 % 400, 1000));
    // Using Euler's theorem: φ(1000) = 400

    // Large base and exponent
    int64_t result = mod_pow(12345, 67890, 99991);
    EXPECT_GE(result, 0);
    EXPECT_LT(result, 99991);
}

// ============== Modular Inverse Tests ==============

TEST_F(ModTest, ModInverseBasicCases) {
    // Basic modular inverse cases
    EXPECT_EQ(mod_inverse(3, 7), 5);     // 3 * 5 ≡ 1 (mod 7)
    EXPECT_EQ(mod_inverse(5, 7), 3);     // 5 * 3 ≡ 1 (mod 7)
    EXPECT_EQ(mod_inverse(2, 5), 3);     // 2 * 3 ≡ 1 (mod 5)
    EXPECT_EQ(mod_inverse(4, 9), 7);     // 4 * 7 ≡ 1 (mod 9)

    // Self-inverse elements
    EXPECT_EQ(mod_inverse(1, 10), 1);    // 1 is always self-inverse
    EXPECT_EQ(mod_inverse(6, 7), 6);     // 6 * 6 = 36 ≡ 1 (mod 7)
}

TEST_F(ModTest, ModInverseVerification) {
    // Verify that a * inverse(a) ≡ 1 (mod m)
    std::vector<std::pair<int, int>> test_cases = {
        {3, 11}, {7, 13}, {9, 17}, {11, 19}, {13, 23}
    };

    for (auto [a, m] : test_cases) {
        auto inv = mod_inverse(a, m);
        EXPECT_EQ(mod_mul(a, inv, m), 1)
            << "Failed for a=" << a << ", m=" << m << ", inv=" << inv;
    }
}

TEST_F(ModTest, ModInverseNoInverse) {
    // Cases where inverse doesn't exist (gcd(a, m) != 1)
    EXPECT_EQ(mod_inverse(2, 4), 0);     // gcd(2, 4) = 2
    EXPECT_EQ(mod_inverse(6, 9), 0);     // gcd(6, 9) = 3
    EXPECT_EQ(mod_inverse(10, 15), 0);   // gcd(10, 15) = 5
}

// ============== Properties Tests ==============

TEST_F(ModTest, ModAddCommutative) {
    // Test commutativity: (a + b) % m = (b + a) % m
    std::uniform_int_distribution<int> dist(0, 100);
    for (int i = 0; i < 20; ++i) {
        int a = dist(rng);
        int b = dist(rng);
        int m = dist(rng) + 1; // Avoid m = 0

        EXPECT_EQ(mod_add(a, b, m), mod_add(b, a, m))
            << "Failed for a=" << a << ", b=" << b << ", m=" << m;
    }
}

TEST_F(ModTest, ModAddAssociative) {
    // Test associativity: ((a + b) + c) % m = (a + (b + c)) % m
    std::uniform_int_distribution<int> dist(0, 50);
    for (int i = 0; i < 20; ++i) {
        int a = dist(rng);
        int b = dist(rng);
        int c = dist(rng);
        int m = dist(rng) + 1;

        auto left = mod_add(mod_add(a, b, m), c, m);
        auto right = mod_add(a, mod_add(b, c, m), m);

        EXPECT_EQ(left, right)
            << "Failed for a=" << a << ", b=" << b << ", c=" << c << ", m=" << m;
    }
}

TEST_F(ModTest, ModMulCommutative) {
    // Test commutativity: (a * b) % m = (b * a) % m
    std::uniform_int_distribution<int> dist(0, 100);
    for (int i = 0; i < 20; ++i) {
        int a = dist(rng);
        int b = dist(rng);
        int m = dist(rng) + 1;

        EXPECT_EQ(mod_mul(a, b, m), mod_mul(b, a, m))
            << "Failed for a=" << a << ", b=" << b << ", m=" << m;
    }
}

TEST_F(ModTest, ModMulAssociative) {
    // Test associativity: ((a * b) * c) % m = (a * (b * c)) % m
    std::uniform_int_distribution<int> dist(0, 30);
    for (int i = 0; i < 20; ++i) {
        int a = dist(rng);
        int b = dist(rng);
        int c = dist(rng);
        int m = dist(rng) + 1;

        auto left = mod_mul(mod_mul(a, b, m), c, m);
        auto right = mod_mul(a, mod_mul(b, c, m), m);

        EXPECT_EQ(left, right)
            << "Failed for a=" << a << ", b=" << b << ", c=" << c << ", m=" << m;
    }
}

TEST_F(ModTest, ModDistributive) {
    // Test distributivity: (a * (b + c)) % m = ((a * b) + (a * c)) % m
    std::uniform_int_distribution<int> dist(0, 30);
    for (int i = 0; i < 20; ++i) {
        int a = dist(rng);
        int b = dist(rng);
        int c = dist(rng);
        int m = dist(rng) + 1;

        auto left = mod_mul(a, mod_add(b, c, m), m);
        auto right = mod_add(mod_mul(a, b, m), mod_mul(a, c, m), m);

        EXPECT_EQ(left, right)
            << "Failed for a=" << a << ", b=" << b << ", c=" << c << ", m=" << m;
    }
}

// ============== Chinese Remainder Theorem Tests ==============

TEST_F(ModTest, ChineseRemainderTheoremBasic) {
    // x ≡ 2 (mod 3) and x ≡ 3 (mod 5)
    // Solution: x ≡ 8 (mod 15)
    auto x = chinese_remainder({2, 3}, {3, 5});
    EXPECT_EQ(x % 3, 2);
    EXPECT_EQ(x % 5, 3);
    EXPECT_EQ(x, 8);
}

TEST_F(ModTest, ChineseRemainderTheoremMultiple) {
    // x ≡ 1 (mod 3), x ≡ 2 (mod 5), x ≡ 3 (mod 7)
    // Solution: x ≡ 52 (mod 105)
    auto x = chinese_remainder({1, 2, 3}, {3, 5, 7});
    EXPECT_EQ(x % 3, 1);
    EXPECT_EQ(x % 5, 2);
    EXPECT_EQ(x % 7, 3);
    EXPECT_EQ(x, 52);
}

// ============== Constexpr Tests ==============

TEST_F(ModTest, ConstexprOperations) {
    // Test that operations are constexpr
    constexpr auto add_result = mod_add(7, 8, 10);
    constexpr auto sub_result = mod_sub(7, 3, 10);
    constexpr auto mul_result = mod_mul(3, 4, 10);
    constexpr auto pow_result = mod_pow(2, 3, 5);

    EXPECT_EQ(add_result, 5);
    EXPECT_EQ(sub_result, 4);
    EXPECT_EQ(mul_result, 2);
    EXPECT_EQ(pow_result, 3);
}

// ============== Edge Cases ==============

TEST_F(ModTest, EdgeCases) {
    // Modulo 1 (everything becomes 0)
    EXPECT_EQ(mod_add(100, 200, 1), 0);
    EXPECT_EQ(mod_mul(100, 200, 1), 0);
    EXPECT_EQ(mod_pow(100, 200, 1), 0);

    // Large moduli
    int64_t large_mod = 1000000007; // Common prime modulus
    EXPECT_LT(mod_add(999999999, 999999999, large_mod), large_mod);
    EXPECT_LT(mod_mul(999999, 999999, large_mod), large_mod);

    // Power of 2 moduli (useful for bit operations)
    EXPECT_EQ(mod_add(15, 17, 16), 0);  // (15 + 17) % 16 = 0
    EXPECT_EQ(mod_mul(7, 9, 8), 7);     // (7 * 9) % 8 = 7
}

// ============== Performance Tests ==============

TEST_F(ModTest, PerformanceLargePowers) {
    // Test that large powers are computed efficiently
    // This should use repeated squaring, not naive multiplication
    auto start = std::chrono::high_resolution_clock::now();

    int64_t result = mod_pow(2, 1000000, 1000000007);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Should complete in reasonable time (< 100ms)
    EXPECT_LT(duration.count(), 100);
    EXPECT_GE(result, 0);
    EXPECT_LT(result, 1000000007);
}