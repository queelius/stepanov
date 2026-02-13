/**
 * @file test_gcd.cpp
 * @brief Tests for polynomial GCD and Euclidean domain operations
 *
 * THIS IS THE KEY INSIGHT: The same GCD algorithm works for integers and polynomials.
 */

#include <polynomials/polynomial.hpp>
#include <gtest/gtest.h>
#include <cmath>

using namespace poly;

constexpr double tol = 1e-10;

// Helper to check if two polynomials are scalar multiples of each other
bool are_associates(const polynomial<double>& p, const polynomial<double>& q) {
    if (p.is_zero() && q.is_zero()) return true;
    if (p.is_zero() || q.is_zero()) return false;
    if (p.degree() != q.degree()) return false;

    // Check if q = c * p for some constant c
    double c = q.leading_coefficient() / p.leading_coefficient();
    for (const auto& [deg, coef] : p) {
        if (std::abs(q[deg] - c * coef) > tol) return false;
    }
    return true;
}

// =============================================================================
// GCD tests
// =============================================================================

TEST(GCDTest, CoprimePairLinear) {
    // gcd(x, x+1) = 1 (coprime)
    auto x = polynomial<double>::x();
    auto x_plus_1 = x + polynomial<double>{1.0};

    auto g = gcd(x, x_plus_1);

    EXPECT_EQ(g.degree(), 0);  // Constant (unit)
}

TEST(GCDTest, CommonLinearFactor) {
    // p = (x-1)(x+1) = x^2 - 1
    // q = (x-1)(x+2) = x^2 + x - 2
    // gcd = x - 1

    polynomial<double> p{-1.0, 0.0, 1.0};     // x^2 - 1
    polynomial<double> q{-2.0, 1.0, 1.0};     // x^2 + x - 2

    auto g = gcd(p, q);

    // g should be monic (x - 1), i.e., degree 1 with leading coef 1
    EXPECT_EQ(g.degree(), 1);
    EXPECT_NEAR(g.leading_coefficient(), 1.0, tol);
    // Check that g divides both p and q
    EXPECT_TRUE(remainder(p, g).is_zero());
    EXPECT_TRUE(remainder(q, g).is_zero());
}

TEST(GCDTest, PolynomialAndItsDerivative) {
    // p = (x-1)^2 = x^2 - 2x + 1
    // p' = 2x - 2 = 2(x-1)
    // gcd(p, p') = x - 1 (the repeated root)

    polynomial<double> p{1.0, -2.0, 1.0};     // (x-1)^2
    auto dp = derivative(p);                   // 2x - 2

    auto g = gcd(p, dp);

    // g should be associate to (x-1)
    EXPECT_EQ(g.degree(), 1);
    EXPECT_TRUE(divides(g, p));
    EXPECT_TRUE(divides(g, dp));
}

TEST(GCDTest, GCDWithZero) {
    polynomial<double> p{1.0, 2.0, 1.0};
    polynomial<double> zero;

    auto g1 = gcd(p, zero);
    auto g2 = gcd(zero, p);

    // gcd(p, 0) = p (normalized)
    EXPECT_TRUE(are_associates(g1, p));
    EXPECT_TRUE(are_associates(g2, p));
}

TEST(GCDTest, IntegerAnalogy) {
    // Just like gcd(12, 18) = 6
    // gcd(x^3 - x, x^2 - 1) should give common factor (x-1)(x+1) = x^2-1... no wait
    // x^3 - x = x(x-1)(x+1)
    // x^2 - 1 = (x-1)(x+1)
    // gcd = x^2 - 1

    polynomial<double> p{0.0, -1.0, 0.0, 1.0};  // x^3 - x
    polynomial<double> q{-1.0, 0.0, 1.0};       // x^2 - 1

    auto g = gcd(p, q);

    EXPECT_EQ(g.degree(), 2);
    EXPECT_TRUE(divides(g, p));
    EXPECT_TRUE(divides(g, q));
}

// =============================================================================
// Extended GCD tests
// =============================================================================

TEST(ExtendedGCDTest, BezoutIdentity) {
    // Verify: gcd(a, b) = a*s + b*t

    polynomial<double> a{-1.0, 0.0, 1.0};     // x^2 - 1
    polynomial<double> b{-2.0, 1.0, 1.0};     // x^2 + x - 2

    auto [g, s, t] = extended_gcd(a, b);

    // Check Bezout identity
    auto sum = a * s + b * t;

    // g and sum should be equal (or associates)
    EXPECT_TRUE(are_associates(g, sum));
}

TEST(ExtendedGCDTest, CoprimePair) {
    // For coprime polynomials, gcd = 1 and we get Bezout coefficients

    auto x = polynomial<double>::x();
    auto x_plus_1 = x + polynomial<double>{1.0};

    auto [g, s, t] = extended_gcd(x, x_plus_1);

    EXPECT_EQ(g.degree(), 0);  // gcd = constant

    // Verify identity
    auto sum = x * s + x_plus_1 * t;
    EXPECT_TRUE(are_associates(g, sum));
}

// =============================================================================
// LCM tests
// =============================================================================

TEST(LCMTest, CommonFactor) {
    // p = (x-1)(x+1), q = (x-1)(x+2)
    // lcm = (x-1)(x+1)(x+2)

    polynomial<double> p{-1.0, 0.0, 1.0};     // (x-1)(x+1)
    polynomial<double> q{-2.0, 1.0, 1.0};     // (x-1)(x+2)

    auto l = lcm(p, q);

    // lcm should have degree 3: (x-1)(x+1)(x+2)
    EXPECT_EQ(l.degree(), 3);

    // Both p and q should divide lcm
    EXPECT_TRUE(divides(p, l));
    EXPECT_TRUE(divides(q, l));
}

TEST(LCMTest, RelationWithGCD) {
    // lcm(p, q) * gcd(p, q) = p * q (up to units)

    polynomial<double> p{-1.0, 0.0, 1.0};     // x^2 - 1
    polynomial<double> q{-2.0, 1.0, 1.0};     // x^2 + x - 2

    auto g = gcd(p, q);
    auto l = lcm(p, q);

    // l * g should be associate to p * q
    auto lg = l * g;
    auto pq = p * q;

    EXPECT_TRUE(are_associates(lg, pq));
}

// =============================================================================
// Divisibility tests
// =============================================================================

TEST(DivisibilityTest, ExactDivision) {
    polynomial<double> p{-1.0, 0.0, 1.0};  // x^2 - 1
    polynomial<double> q{-1.0, 1.0};       // x - 1

    EXPECT_TRUE(divides(q, p));  // (x-1) divides (x^2-1)
    EXPECT_FALSE(divides(p, q)); // (x^2-1) does not divide (x-1)
}

TEST(DivisibilityTest, SelfDivision) {
    polynomial<double> p{1.0, 2.0, 3.0};
    EXPECT_TRUE(divides(p, p));
}

TEST(DivisibilityTest, ConstantDivision) {
    polynomial<double> p{2.0, 4.0, 6.0};
    polynomial<double> two{2.0};

    EXPECT_TRUE(divides(two, p));  // 2 divides all coefficients
}

// =============================================================================
// The fundamental insight: same algorithm as integers
// =============================================================================

TEST(InsightTest, EuclideanAlgorithmStructure) {
    // The GCD algorithm is:
    //   while b != 0: a, b = b, a % b
    //   return a
    //
    // This works because both integers and polynomials form Euclidean domains:
    // - Integers: norm(n) = |n|, and |a % b| < |b|
    // - Polynomials: norm(p) = degree(p), and degree(a % b) < degree(b)

    // Let's verify the degree decreases at each step
    polynomial<double> a{1.0, 0.0, 0.0, 0.0, 1.0};  // x^4 + 1
    polynomial<double> b{1.0, 0.0, 1.0};            // x^2 + 1

    int prev_deg = b.degree();
    polynomial<double> r = remainder(a, b);

    while (!r.is_zero()) {
        // The key property: degree strictly decreases
        EXPECT_LT(r.degree(), prev_deg);

        prev_deg = r.degree();
        a = b;
        b = r;
        r = remainder(a, b);
    }
}
