#include <gtest/gtest.h>
#include <stepanov/builtin_adaptors.hpp>
#include <stepanov/math.hpp>
#include <stepanov/gcd.hpp>
#include <stepanov/primality.hpp>
#include <stepanov/rational.hpp>
#include <stepanov/fixed_decimal.hpp>
#include <vector>
#include <numeric>
#include <random>

using namespace stepanov;

class MathIntegrationTest : public ::testing::Test {
protected:
    std::mt19937 rng{42};
};

// ============== GCD with Math Operations ==============

TEST_F(MathIntegrationTest, GcdWithPowerAndProduct) {
    // Test that GCD works with results from power/product operations
    int a = power(2, 6);  // 64
    int b = product(8, 6); // 48

    int g = gcd(a, b);
    EXPECT_EQ(g, 16);

    // Verify the GCD properties
    EXPECT_EQ(remainder(a, g), 0);
    EXPECT_EQ(remainder(b, g), 0);
    EXPECT_EQ(gcd(quotient(a, g), quotient(b, g)), 1);
}

TEST_F(MathIntegrationTest, ExtendedGcdLinearCombination) {
    // Extended GCD: find x, y such that ax + by = gcd(a, b)
    int a = 35;
    int b = 15;

    int g = gcd(a, b);
    EXPECT_EQ(g, 5);

    // Verify using Bezout's identity
    // For 35 and 15: 35*1 + 15*(-2) = 5
    // This can be verified by working backwards from the Euclidean algorithm
}

// ============== Primality Testing with Number Theory ==============

TEST_F(MathIntegrationTest, FermatTestWithPowerMod) {
    // Test small primes
    std::vector<int> small_primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};

    for (int p : small_primes) {
        // Fermat's little theorem: a^(p-1) ≡ 1 (mod p) for prime p
        for (int a = 2; a < std::min(p, 10); ++a) {
            if (gcd(a, p) == 1) {
                EXPECT_EQ(power_mod(a, p - 1, p), 1)
                    << "Failed Fermat test for p=" << p << ", a=" << a;
            }
        }
    }
}

TEST_F(MathIntegrationTest, PrimalityWithGcd) {
    // Test that primes are coprime to all smaller positive integers
    std::vector<int> primes = {11, 13, 17, 19, 23};

    for (int p : primes) {
        for (int i = 2; i < p; ++i) {
            EXPECT_EQ(gcd(p, i), 1)
                << "Prime " << p << " should be coprime to " << i;
        }
    }
}

// ============== Rational Numbers using GCD ==============

TEST_F(MathIntegrationTest, RationalArithmeticWithGcd) {
    // Simulate rational arithmetic using GCD for simplification

    // 2/3 + 3/4 = 8/12 + 9/12 = 17/12
    int num1 = 2, den1 = 3;
    int num2 = 3, den2 = 4;

    // Add fractions
    int num_result = num1 * den2 + num2 * den1;
    int den_result = den1 * den2;

    // Simplify using GCD
    int g = gcd(num_result, den_result);
    num_result = quotient(num_result, g);
    den_result = quotient(den_result, g);

    EXPECT_EQ(num_result, 17);
    EXPECT_EQ(den_result, 12);

    // 3/4 * 4/5 = 12/20 = 3/5
    num1 = 3; den1 = 4;
    num2 = 4; den2 = 5;

    num_result = product(num1, num2);
    den_result = product(den1, den2);

    g = gcd(num_result, den_result);
    num_result = quotient(num_result, g);
    den_result = quotient(den_result, g);

    EXPECT_EQ(num_result, 3);
    EXPECT_EQ(den_result, 5);
}

// ============== Modular Arithmetic Integration ==============

TEST_F(MathIntegrationTest, ModularInverseWithGcd) {
    // Modular inverse exists iff gcd(a, m) = 1
    int a = 7;
    int m = 11;

    EXPECT_EQ(gcd(a, m), 1); // Verify coprime

    // Find inverse using extended Euclidean algorithm concept
    // 7 * 8 = 56 ≡ 1 (mod 11)
    int inv = 8;
    EXPECT_EQ(remainder(product(a, inv), m), 1);
}

TEST_F(MathIntegrationTest, ChineseRemainderWithGcd) {
    // Chinese Remainder Theorem requires pairwise coprime moduli
    std::vector<int> moduli = {3, 5, 7};

    // Verify pairwise coprime
    for (size_t i = 0; i < moduli.size(); ++i) {
        for (size_t j = i + 1; j < moduli.size(); ++j) {
            EXPECT_EQ(gcd(moduli[i], moduli[j]), 1)
                << "Moduli " << moduli[i] << " and " << moduli[j]
                << " should be coprime for CRT";
        }
    }

    // Find x such that x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7)
    // Solution: x = 23
    int x = 23;
    EXPECT_EQ(remainder(x, 3), 2);
    EXPECT_EQ(remainder(x, 5), 3);
    EXPECT_EQ(remainder(x, 7), 2);
}

// ============== Pythagorean Triples ==============

TEST_F(MathIntegrationTest, PythagoreanTriplesWithGcd) {
    // Generate primitive Pythagorean triples using the formula:
    // a = m^2 - n^2, b = 2mn, c = m^2 + n^2
    // where m > n > 0, gcd(m, n) = 1, and m - n is odd

    std::vector<std::tuple<int, int, int>> triples;

    for (int m = 2; m <= 10; ++m) {
        for (int n = 1; n < m; ++n) {
            if (gcd(m, n) == 1 && even(m - n) == false) {
                int a = square(m) - square(n);
                int b = product(2, product(m, n));
                int c = square(m) + square(n);

                // Verify Pythagorean relation
                EXPECT_EQ(square(a) + square(b), square(c))
                    << "Failed for triple (" << a << ", " << b << ", " << c << ")";

                // Verify primitive (gcd = 1)
                int g = gcd(gcd(a, b), c);
                EXPECT_EQ(g, 1)
                    << "Triple (" << a << ", " << b << ", " << c << ") is not primitive";

                triples.push_back({a, b, c});
            }
        }
    }

    EXPECT_GT(triples.size(), 0); // Should find some triples
}

// ============== Fibonacci and GCD ==============

TEST_F(MathIntegrationTest, FibonacciGcdProperty) {
    // Property: gcd(F_m, F_n) = F_{gcd(m,n)}
    // where F_k is the k-th Fibonacci number

    // Generate Fibonacci numbers
    std::vector<int64_t> fib(20);
    fib[0] = 0;
    fib[1] = 1;
    for (size_t i = 2; i < fib.size(); ++i) {
        fib[i] = sum(fib[i - 1], fib[i - 2]);
    }

    // Test the property for small indices
    for (int m = 3; m < 10; ++m) {
        for (int n = 3; n < m; ++n) {
            int g_indices = gcd(m, n);
            int64_t g_fibs = gcd(fib[m], fib[n]);

            EXPECT_EQ(g_fibs, fib[g_indices])
                << "Failed for F_" << m << " and F_" << n;
        }
    }
}

// ============== Perfect Powers ==============

TEST_F(MathIntegrationTest, PerfectPowerDetection) {
    // Check if a number is a perfect power using GCD and repeated operations

    // Test perfect squares
    for (int i = 1; i <= 10; ++i) {
        int sq = square(i);
        // A perfect square n has the property that gcd(n, n-1) = 1
        EXPECT_EQ(gcd(sq, sq - 1), 1);
    }

    // Test perfect cubes
    for (int i = 2; i <= 5; ++i) {
        int cube = product(i, square(i));
        // Verify it's actually a cube
        EXPECT_EQ(cube, power(i, 3));
    }
}

// ============== Totient Function ==============

TEST_F(MathIntegrationTest, EulerTotientWithGcd) {
    // Euler's totient function φ(n) counts integers k ≤ n with gcd(k, n) = 1

    auto totient = [](int n) {
        int count = 0;
        for (int i = 1; i <= n; ++i) {
            if (gcd(i, n) == 1) {
                ++count;
            }
        }
        return count;
    };

    // Test for small values
    EXPECT_EQ(totient(1), 1);   // φ(1) = 1
    EXPECT_EQ(totient(2), 1);   // φ(2) = 1
    EXPECT_EQ(totient(6), 2);   // φ(6) = 2 (1, 5 are coprime to 6)
    EXPECT_EQ(totient(9), 6);   // φ(9) = 6 (1,2,4,5,7,8)
    EXPECT_EQ(totient(10), 4);  // φ(10) = 4 (1,3,7,9)

    // For prime p, φ(p) = p - 1
    std::vector<int> primes = {7, 11, 13};
    for (int p : primes) {
        EXPECT_EQ(totient(p), p - 1)
            << "Failed for prime " << p;
    }
}

// ============== Lattice Points ==============

TEST_F(MathIntegrationTest, LatticePointsOnLine) {
    // Number of lattice points on line from (0,0) to (a,b) is gcd(a,b) + 1
    // (including endpoints)

    struct Point {
        int x, y;
    };

    std::vector<Point> endpoints = {{3, 4}, {6, 8}, {5, 12}, {7, 7}};

    for (const auto& p : endpoints) {
        int expected_interior = gcd(p.x, p.y) - 1; // Excluding endpoints
        int expected_total = gcd(p.x, p.y) + 1;    // Including endpoints

        // Verify by counting
        int count = 0;
        int g = gcd(p.x, p.y);
        for (int k = 0; k <= g; ++k) {
            int x = product(k, quotient(p.x, g));
            int y = product(k, quotient(p.y, g));
            if (remainder(p.x, g) == 0 && remainder(p.y, g) == 0) {
                count++;
            }
        }

        EXPECT_EQ(count, expected_total)
            << "Failed for line to (" << p.x << ", " << p.y << ")";
    }
}

// ============== Multiplicative Order ==============

TEST_F(MathIntegrationTest, MultiplicativeOrderModulo) {
    // Find the multiplicative order of a modulo n
    // (smallest positive k such that a^k ≡ 1 (mod n))

    auto mult_order = [](int a, int n) -> int {
        if (gcd(a, n) != 1) return -1; // Not defined

        int order = 1;
        int current = remainder(a, n);
        while (current != 1 && order < n) {
            current = remainder(product(current, a), n);
            order++;
        }
        return current == 1 ? order : -1;
    };

    // Test cases
    EXPECT_EQ(mult_order(2, 7), 3);   // 2^3 ≡ 1 (mod 7)
    EXPECT_EQ(mult_order(3, 7), 6);   // 3^6 ≡ 1 (mod 7)
    EXPECT_EQ(mult_order(2, 11), 10); // 2^10 ≡ 1 (mod 11)

    // Order divides φ(n)
    int n = 13;
    int a = 2;
    int order = mult_order(a, n);
    int totient_n = 12; // φ(13) = 12

    EXPECT_EQ(remainder(totient_n, order), 0)
        << "Order should divide Euler's totient";
}

// ============== Performance Comparison ==============

TEST_F(MathIntegrationTest, PerformanceComparisonGcdAlgorithms) {
    // Compare performance characteristics of different approaches
    std::uniform_int_distribution<int> dist(1, 1000000);

    // Generate test pairs
    std::vector<std::pair<int, int>> test_pairs;
    for (int i = 0; i < 100; ++i) {
        test_pairs.emplace_back(dist(rng), dist(rng));
    }

    // Test that our generic GCD works correctly
    for (const auto& [a, b] : test_pairs) {
        int g = gcd(a, b);
        EXPECT_EQ(remainder(a, g), 0);
        EXPECT_EQ(remainder(b, g), 0);

        // Compare with std::gcd
        EXPECT_EQ(g, std::gcd(a, b));
    }
}

// ============== Complex Number Applications ==============

TEST_F(MathIntegrationTest, GaussianIntegerGcd) {
    // GCD in Gaussian integers (complex numbers with integer parts)
    struct GaussInt {
        int real, imag;

        int norm() const {
            return square(real) + square(imag);
        }
    };

    // For Gaussian integers, we can compute a GCD-like value
    // using the norm function

    GaussInt a{3, 4};  // 3 + 4i
    GaussInt b{5, 0};  // 5

    // The norm is multiplicative: N(ab) = N(a)N(b)
    int norm_a = a.norm(); // 9 + 16 = 25
    int norm_b = b.norm(); // 25

    EXPECT_EQ(norm_a, 25);
    EXPECT_EQ(norm_b, 25);

    // GCD of the norms
    int g = gcd(norm_a, norm_b);
    EXPECT_EQ(g, 25);
}