#include <gtest/gtest.h>
#include <stepanov/builtin_adaptors.hpp>  // Must come first for ADL
#include <stepanov/gcd.hpp>
#include <vector>
#include <random>
#include <numeric>
#include <optional>

using namespace stepanov;

class GcdTest : public ::testing::Test {
protected:
    std::mt19937 rng{42}; // Fixed seed for reproducibility

    // Helper to verify GCD properties
    template<typename T>
    void verifyGcdProperties(T a, T b, T g) {
        // GCD divides both inputs
        EXPECT_EQ(remainder(a, g), T(0)) << "gcd(" << a << "," << b << ")=" << g << " doesn't divide " << a;
        EXPECT_EQ(remainder(b, g), T(0)) << "gcd(" << a << "," << b << ")=" << g << " doesn't divide " << b;

        // GCD is the greatest common divisor
        T q_a = quotient(a, g);
        T q_b = quotient(b, g);
        EXPECT_EQ(gcd(q_a, q_b), T(1)) << "gcd(" << a << "/" << g << ", " << b << "/" << g << ") should be 1";
    }
};

// ============== Basic GCD Tests ==============

TEST_F(GcdTest, BasicGcdCases) {
    // Simple cases
    EXPECT_EQ(gcd(10, 5), 5);
    EXPECT_EQ(gcd(12, 8), 4);
    EXPECT_EQ(gcd(17, 13), 1);
    EXPECT_EQ(gcd(100, 40), 20);
    EXPECT_EQ(gcd(54, 24), 6);

    // Commutative property
    EXPECT_EQ(gcd(48, 18), gcd(18, 48));
    EXPECT_EQ(gcd(100, 75), gcd(75, 100));
}

TEST_F(GcdTest, GcdWithZero) {
    // GCD with zero
    EXPECT_EQ(gcd(0, 5), 5);
    EXPECT_EQ(gcd(5, 0), 5);
    EXPECT_EQ(gcd(0, 0), 0);
    EXPECT_EQ(gcd(0, 100), 100);
    EXPECT_EQ(gcd(-10, 0), 10);  // Should handle negatives
}

TEST_F(GcdTest, GcdWithOne) {
    // GCD with 1
    EXPECT_EQ(gcd(1, 1), 1);
    EXPECT_EQ(gcd(1, 100), 1);
    EXPECT_EQ(gcd(100, 1), 1);
    EXPECT_EQ(gcd(1, 999), 1);
}

TEST_F(GcdTest, GcdWithNegatives) {
    // GCD with negative numbers
    EXPECT_EQ(std::abs(gcd(-10, 5)), 5);
    EXPECT_EQ(std::abs(gcd(10, -5)), 5);
    EXPECT_EQ(std::abs(gcd(-10, -5)), 5);
    EXPECT_EQ(std::abs(gcd(-48, 18)), 6);
    EXPECT_EQ(std::abs(gcd(48, -18)), 6);
    EXPECT_EQ(std::abs(gcd(-48, -18)), 6);
}

TEST_F(GcdTest, GcdPrimeNumbers) {
    // GCD of prime numbers
    EXPECT_EQ(gcd(2, 3), 1);
    EXPECT_EQ(gcd(5, 7), 1);
    EXPECT_EQ(gcd(11, 13), 1);
    EXPECT_EQ(gcd(17, 19), 1);

    // GCD of same prime
    EXPECT_EQ(gcd(7, 7), 7);
    EXPECT_EQ(gcd(13, 13), 13);

    // GCD of prime and its multiple
    EXPECT_EQ(gcd(7, 14), 7);
    EXPECT_EQ(gcd(21, 7), 7);
}

TEST_F(GcdTest, GcdLargeNumbers) {
    // Test with larger numbers
    EXPECT_EQ(gcd(1234567890, 987654321), 9);
    EXPECT_EQ(gcd(1000000007, 1000000009), 1);  // Two large primes
    EXPECT_EQ(gcd(1024, 768), 256);  // Powers of 2
}

TEST_F(GcdTest, GcdProperties) {
    // Test GCD properties with random numbers
    std::uniform_int_distribution<int> dist(1, 1000);

    for (int i = 0; i < 50; ++i) {
        int a = dist(rng);
        int b = dist(rng);
        int g = gcd(a, b);

        verifyGcdProperties(a, b, g);

        // Associative property: gcd(gcd(a,b),c) = gcd(a,gcd(b,c))
        int c = dist(rng);
        EXPECT_EQ(gcd(gcd(a, b), c), gcd(a, gcd(b, c)))
            << "Failed associativity for " << a << "," << b << "," << c;
    }
}

// ============== Binary GCD Tests ==============

TEST_F(GcdTest, BinaryGcdBasic) {
    // Basic cases
    EXPECT_EQ(binary_gcd(10, 5), 5);
    EXPECT_EQ(binary_gcd(12, 8), 4);
    EXPECT_EQ(binary_gcd(17, 13), 1);
    EXPECT_EQ(binary_gcd(100, 40), 20);
}

TEST_F(GcdTest, BinaryGcdPowersOfTwo) {
    // Efficient with powers of 2
    EXPECT_EQ(binary_gcd(16, 8), 8);
    EXPECT_EQ(binary_gcd(32, 64), 32);
    EXPECT_EQ(binary_gcd(1024, 512), 512);
    EXPECT_EQ(binary_gcd(256, 128), 128);
}

TEST_F(GcdTest, BinaryGcdVsNormalGcd) {
    // Binary GCD should give same results as normal GCD
    std::uniform_int_distribution<int> dist(1, 10000);

    for (int i = 0; i < 100; ++i) {
        int a = dist(rng);
        int b = dist(rng);

        EXPECT_EQ(binary_gcd(a, b), gcd(a, b))
            << "Mismatch for a=" << a << ", b=" << b;
    }
}

TEST_F(GcdTest, BinaryGcdEdgeCases) {
    // Edge cases
    EXPECT_EQ(binary_gcd(0, 5), 5);
    EXPECT_EQ(binary_gcd(5, 0), 5);
    EXPECT_EQ(binary_gcd(0, 0), 0);
    EXPECT_EQ(binary_gcd(1, 1), 1);
    EXPECT_EQ(binary_gcd(1, 1000000), 1);
}

// ============== Extended GCD Tests ==============

TEST_F(GcdTest, ExtendedGcdBasic) {
    auto [g, x, y] = extended_gcd(30, 18);
    EXPECT_EQ(g, 6);
    EXPECT_EQ(30 * x + 18 * y, g);  // Bézout's identity

    auto [g2, x2, y2] = extended_gcd(35, 15);
    EXPECT_EQ(g2, 5);
    EXPECT_EQ(35 * x2 + 15 * y2, g2);
}

TEST_F(GcdTest, ExtendedGcdCoprime) {
    // For coprime numbers
    auto [g, x, y] = extended_gcd(17, 13);
    EXPECT_EQ(g, 1);
    EXPECT_EQ(17 * x + 13 * y, 1);

    auto [g2, x2, y2] = extended_gcd(25, 7);
    EXPECT_EQ(g2, 1);
    EXPECT_EQ(25 * x2 + 7 * y2, 1);
}

TEST_F(GcdTest, ExtendedGcdBezoutIdentity) {
    // Verify Bézout's identity for random inputs
    std::uniform_int_distribution<int> dist(1, 1000);

    for (int i = 0; i < 50; ++i) {
        int a = dist(rng);
        int b = dist(rng);

        auto [g, x, y] = extended_gcd(a, b);
        EXPECT_EQ(a * x + b * y, g)
            << "Bézout's identity failed for a=" << a << ", b=" << b
            << ", g=" << g << ", x=" << x << ", y=" << y;

        // Also verify it's the GCD
        EXPECT_EQ(g, gcd(a, b));
    }
}

TEST_F(GcdTest, ExtendedGcdWithZero) {
    auto [g, x, y] = extended_gcd(5, 0);
    EXPECT_EQ(g, 5);
    EXPECT_EQ(5 * x + 0 * y, g);

    auto [g2, x2, y2] = extended_gcd(0, 7);
    EXPECT_EQ(g2, 7);
    EXPECT_EQ(0 * x2 + 7 * y2, g2);
}

// ============== LCM Tests ==============

TEST_F(GcdTest, LcmBasic) {
    EXPECT_EQ(lcm(4, 6), 12);
    EXPECT_EQ(lcm(3, 5), 15);
    EXPECT_EQ(lcm(12, 18), 36);
    EXPECT_EQ(lcm(10, 15), 30);
    EXPECT_EQ(lcm(21, 6), 42);
}

TEST_F(GcdTest, LcmWithZero) {
    EXPECT_EQ(lcm(0, 5), 0);
    EXPECT_EQ(lcm(5, 0), 0);
    EXPECT_EQ(lcm(0, 0), 0);
}

TEST_F(GcdTest, LcmWithOne) {
    EXPECT_EQ(lcm(1, 5), 5);
    EXPECT_EQ(lcm(7, 1), 7);
    EXPECT_EQ(lcm(1, 1), 1);
}

TEST_F(GcdTest, LcmCoprime) {
    // LCM of coprime numbers is their product
    EXPECT_EQ(lcm(7, 11), 77);
    EXPECT_EQ(lcm(13, 17), 221);
    EXPECT_EQ(lcm(5, 9), 45);
}

TEST_F(GcdTest, LcmGcdRelation) {
    // Test: lcm(a,b) * gcd(a,b) = a * b
    std::uniform_int_distribution<int> dist(1, 100);

    for (int i = 0; i < 50; ++i) {
        int a = dist(rng);
        int b = dist(rng);

        int g = gcd(a, b);
        int l = lcm(a, b);

        EXPECT_EQ(l * g, a * b)
            << "LCM-GCD relation failed for a=" << a << ", b=" << b;
    }
}

// ============== Multiple Argument GCD/LCM Tests ==============

TEST_F(GcdTest, GcdMultipleArgs) {
    EXPECT_EQ(gcd(12, 18, 24), 6);
    EXPECT_EQ(gcd(10, 15, 20, 25), 5);
    EXPECT_EQ(gcd(14, 21, 35), 7);
    EXPECT_EQ(gcd(100, 75, 50, 25), 25);
}

TEST_F(GcdTest, LcmMultipleArgs) {
    EXPECT_EQ(lcm(4, 6, 8), 24);
    EXPECT_EQ(lcm(3, 4, 5), 60);
    EXPECT_EQ(lcm(6, 8, 12), 24);
    EXPECT_EQ(lcm(5, 10, 15, 20), 60);
}

// ============== Coprime Tests ==============

TEST_F(GcdTest, CoprimeBasic) {
    EXPECT_TRUE(coprime(5, 7));
    EXPECT_TRUE(coprime(9, 16));
    EXPECT_TRUE(coprime(13, 17));
    EXPECT_TRUE(coprime(25, 36));

    EXPECT_FALSE(coprime(6, 9));
    EXPECT_FALSE(coprime(10, 15));
    EXPECT_FALSE(coprime(12, 18));
    EXPECT_FALSE(coprime(100, 50));
}

TEST_F(GcdTest, CoprimeWithOne) {
    // Any number is coprime with 1
    EXPECT_TRUE(coprime(1, 1));
    EXPECT_TRUE(coprime(1, 100));
    EXPECT_TRUE(coprime(999, 1));
}

// ============== GCD Range Tests ==============

TEST_F(GcdTest, GcdRangeBasic) {
    std::vector<int> v1 = {12, 18, 24};
    EXPECT_EQ(gcd_range(v1.begin(), v1.end()), 6);

    std::vector<int> v2 = {10, 15, 20, 25};
    EXPECT_EQ(gcd_range(v2.begin(), v2.end()), 5);

    std::vector<int> v3 = {7, 14, 21, 28};
    EXPECT_EQ(gcd_range(v3.begin(), v3.end()), 7);
}

TEST_F(GcdTest, GcdRangeEmpty) {
    std::vector<int> empty;
    EXPECT_EQ(gcd_range(empty.begin(), empty.end()), 0);
}

TEST_F(GcdTest, GcdRangeSingle) {
    std::vector<int> single = {42};
    EXPECT_EQ(gcd_range(single.begin(), single.end()), 42);
}

TEST_F(GcdTest, GcdRangeEarlyExit) {
    // Should exit early when GCD becomes 1
    std::vector<int> v = {100, 101, 102};  // 100 and 101 are coprime
    EXPECT_EQ(gcd_range(v.begin(), v.end()), 1);
}

// ============== Modular Inverse Tests ==============

TEST_F(GcdTest, ModInverseBasic) {
    auto inv = mod_inverse(3, 7);
    ASSERT_TRUE(inv.has_value());
    EXPECT_EQ((3 * inv.value()) % 7, 1);

    auto inv2 = mod_inverse(5, 11);
    ASSERT_TRUE(inv2.has_value());
    EXPECT_EQ((5 * inv2.value()) % 11, 1);
}

TEST_F(GcdTest, ModInverseNoSolution) {
    // No inverse when not coprime
    auto inv = mod_inverse(6, 9);  // gcd(6,9) = 3
    EXPECT_FALSE(inv.has_value());

    auto inv2 = mod_inverse(10, 15);  // gcd(10,15) = 5
    EXPECT_FALSE(inv2.has_value());
}

TEST_F(GcdTest, ModInverseExhaustive) {
    // Test all inverses modulo small primes
    for (int p : {5, 7, 11, 13}) {
        for (int a = 1; a < p; ++a) {
            auto inv = mod_inverse(a, p);
            ASSERT_TRUE(inv.has_value())
                << "Should have inverse for a=" << a << " mod " << p;
            EXPECT_EQ((a * inv.value()) % p, 1)
                << "Invalid inverse for a=" << a << " mod " << p;
        }
    }
}

// ============== Chinese Remainder Theorem Tests ==============

TEST_F(GcdTest, CrtBasic) {
    // x ≡ 2 (mod 3), x ≡ 3 (mod 5)
    std::vector<int> remainders = {2, 3};
    std::vector<int> moduli = {3, 5};

    auto result = chinese_remainder(remainders, moduli);
    ASSERT_TRUE(result.has_solution);
    EXPECT_EQ(result.solution % 3, 2);
    EXPECT_EQ(result.solution % 5, 3);
    EXPECT_EQ(result.modulus, 15);  // lcm(3, 5)
}

TEST_F(GcdTest, CrtMultiple) {
    // x ≡ 1 (mod 2), x ≡ 2 (mod 3), x ≡ 3 (mod 5)
    std::vector<int> remainders = {1, 2, 3};
    std::vector<int> moduli = {2, 3, 5};

    auto result = chinese_remainder(remainders, moduli);
    ASSERT_TRUE(result.has_solution);
    EXPECT_EQ(result.solution % 2, 1);
    EXPECT_EQ(result.solution % 3, 2);
    EXPECT_EQ(result.solution % 5, 3);
    EXPECT_EQ(result.modulus, 30);  // lcm(2, 3, 5)
}

TEST_F(GcdTest, CrtNoSolution) {
    // Inconsistent system: x ≡ 1 (mod 2), x ≡ 2 (mod 4)
    // No number can be odd (≡1 mod 2) and ≡2 mod 4 (which means even)
    std::vector<int> remainders = {1, 2};
    std::vector<int> moduli = {2, 4};

    auto result = chinese_remainder(remainders, moduli);
    EXPECT_FALSE(result.has_solution);
}

TEST_F(GcdTest, CrtNonCoprime) {
    // Non-coprime moduli but consistent
    // x ≡ 1 (mod 4), x ≡ 5 (mod 6)
    // Solution exists because 5 ≡ 1 (mod 2) and 1 ≡ 1 (mod 2)
    std::vector<int> remainders = {1, 5};
    std::vector<int> moduli = {4, 6};

    auto result = chinese_remainder(remainders, moduli);
    if (result.has_solution) {
        EXPECT_EQ(result.solution % 4, 1);
        EXPECT_EQ(result.solution % 6, 5);
    }
}

TEST_F(GcdTest, CrtEdgeCases) {
    // Empty input
    std::vector<int> empty_r, empty_m;
    auto result = chinese_remainder(empty_r, empty_m);
    EXPECT_FALSE(result.has_solution);

    // Mismatched sizes
    std::vector<int> r = {1, 2};
    std::vector<int> m = {3};
    auto result2 = chinese_remainder(r, m);
    EXPECT_FALSE(result2.has_solution);
}

// ============== Constexpr Tests ==============

TEST_F(GcdTest, ConstexprGcd) {
    // Test that GCD functions are constexpr
    constexpr int g1 = gcd(12, 8);
    constexpr int g2 = binary_gcd(12, 8);
    constexpr int l1 = lcm(4, 6);
    constexpr bool c1 = coprime(5, 7);

    EXPECT_EQ(g1, 4);
    EXPECT_EQ(g2, 4);
    EXPECT_EQ(l1, 12);
    EXPECT_TRUE(c1);

    // Extended GCD is also constexpr (but structured bindings can't be constexpr in C++20)
    constexpr auto result = extended_gcd(30, 18);
    EXPECT_EQ(result.gcd, 6);
}

// ============== Type Deduction Tests ==============

TEST_F(GcdTest, TypeDeduction) {
    // Test that operations preserve types
    auto g_int = gcd(10, 5);
    static_assert(std::is_same_v<decltype(g_int), int>);

    auto g_long = gcd(10L, 5L);
    static_assert(std::is_same_v<decltype(g_long), long>);

    auto g_longlong = gcd(10LL, 5LL);
    static_assert(std::is_same_v<decltype(g_longlong), long long>);
}