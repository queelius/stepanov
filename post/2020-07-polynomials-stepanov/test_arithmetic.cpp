/**
 * @file test_arithmetic.cpp
 * @brief Tests for polynomial arithmetic operations
 */

#include <polynomials/polynomial.hpp>
#include <gtest/gtest.h>

using namespace poly;

// =============================================================================
// Construction tests
// =============================================================================

TEST(PolynomialTest, DefaultConstruction) {
    polynomial<double> p;
    EXPECT_TRUE(p.is_zero());
    EXPECT_EQ(p.degree(), -1);
}

TEST(PolynomialTest, ConstantConstruction) {
    polynomial<double> p{5.0};
    EXPECT_FALSE(p.is_zero());
    EXPECT_EQ(p.degree(), 0);
    EXPECT_DOUBLE_EQ(p[0], 5.0);
}

TEST(PolynomialTest, ZeroConstantConstruction) {
    polynomial<double> p{0.0};
    EXPECT_TRUE(p.is_zero());
    EXPECT_EQ(p.degree(), -1);
}

TEST(PolynomialTest, DenseConstruction) {
    // 1 + 2x + 3x^2
    polynomial<double> p{1.0, 2.0, 3.0};
    EXPECT_EQ(p.degree(), 2);
    EXPECT_DOUBLE_EQ(p[0], 1.0);
    EXPECT_DOUBLE_EQ(p[1], 2.0);
    EXPECT_DOUBLE_EQ(p[2], 3.0);
    EXPECT_DOUBLE_EQ(p[3], 0.0);  // Missing coefficient
}

TEST(PolynomialTest, MonomialConstruction) {
    // 5x^3
    auto m = polynomial<double>::monomial(5.0, 3);
    EXPECT_EQ(m.degree(), 3);
    EXPECT_DOUBLE_EQ(m[3], 5.0);
    EXPECT_DOUBLE_EQ(m[0], 0.0);
}

TEST(PolynomialTest, VariableX) {
    auto x = polynomial<double>::x();
    EXPECT_EQ(x.degree(), 1);
    EXPECT_DOUBLE_EQ(x[0], 0.0);
    EXPECT_DOUBLE_EQ(x[1], 1.0);
}

// =============================================================================
// Accessor tests
// =============================================================================

TEST(PolynomialTest, LeadingCoefficient) {
    polynomial<double> p{1.0, 2.0, 3.0};  // 1 + 2x + 3x^2
    EXPECT_DOUBLE_EQ(p.leading_coefficient(), 3.0);
}

TEST(PolynomialTest, ConstantTerm) {
    polynomial<double> p{5.0, 2.0, 1.0};
    EXPECT_DOUBLE_EQ(p.constant_term(), 5.0);
}

TEST(PolynomialTest, NumTerms) {
    polynomial<double> p{1.0, 0.0, 3.0};  // 1 + 3x^2 (sparse)
    EXPECT_EQ(p.num_terms(), 2);
}

TEST(PolynomialTest, IsConstant) {
    polynomial<double> constant{5.0};
    polynomial<double> zero{};
    polynomial<double> linear{1.0, 2.0};

    EXPECT_TRUE(constant.is_constant());
    EXPECT_TRUE(zero.is_constant());
    EXPECT_FALSE(linear.is_constant());
}

// =============================================================================
// Addition tests
// =============================================================================

TEST(PolynomialTest, Addition) {
    polynomial<double> p{1.0, 2.0, 3.0};  // 1 + 2x + 3x^2
    polynomial<double> q{4.0, 5.0};       // 4 + 5x

    auto sum = p + q;
    EXPECT_EQ(sum.degree(), 2);
    EXPECT_DOUBLE_EQ(sum[0], 5.0);
    EXPECT_DOUBLE_EQ(sum[1], 7.0);
    EXPECT_DOUBLE_EQ(sum[2], 3.0);
}

TEST(PolynomialTest, AdditionCancellation) {
    polynomial<double> p{1.0, 2.0, 3.0};
    polynomial<double> q{0.0, -2.0};  // -2x

    auto sum = p + q;
    EXPECT_EQ(sum.degree(), 2);
    EXPECT_DOUBLE_EQ(sum[0], 1.0);
    EXPECT_DOUBLE_EQ(sum[1], 0.0);  // Cancellation
    EXPECT_DOUBLE_EQ(sum[2], 3.0);
}

TEST(PolynomialTest, AdditionToZero) {
    polynomial<double> p{1.0, 2.0};
    polynomial<double> q{-1.0, -2.0};

    auto sum = p + q;
    EXPECT_TRUE(sum.is_zero());
}

// =============================================================================
// Subtraction tests
// =============================================================================

TEST(PolynomialTest, Subtraction) {
    polynomial<double> p{5.0, 3.0, 2.0};
    polynomial<double> q{1.0, 1.0, 1.0};

    auto diff = p - q;
    EXPECT_EQ(diff.degree(), 2);
    EXPECT_DOUBLE_EQ(diff[0], 4.0);
    EXPECT_DOUBLE_EQ(diff[1], 2.0);
    EXPECT_DOUBLE_EQ(diff[2], 1.0);
}

TEST(PolynomialTest, Negation) {
    polynomial<double> p{1.0, -2.0, 3.0};
    auto neg = -p;

    EXPECT_DOUBLE_EQ(neg[0], -1.0);
    EXPECT_DOUBLE_EQ(neg[1], 2.0);
    EXPECT_DOUBLE_EQ(neg[2], -3.0);
}

// =============================================================================
// Multiplication tests
// =============================================================================

TEST(PolynomialTest, Multiplication) {
    polynomial<double> p{1.0, 1.0};        // 1 + x
    polynomial<double> q{1.0, -1.0};       // 1 - x

    auto prod = p * q;  // (1+x)(1-x) = 1 - x^2
    EXPECT_EQ(prod.degree(), 2);
    EXPECT_DOUBLE_EQ(prod[0], 1.0);
    EXPECT_DOUBLE_EQ(prod[1], 0.0);
    EXPECT_DOUBLE_EQ(prod[2], -1.0);
}

TEST(PolynomialTest, MultiplicationLinear) {
    polynomial<double> p{1.0, 2.0};   // 1 + 2x
    polynomial<double> q{3.0, 4.0};   // 3 + 4x

    auto prod = p * q;  // (1+2x)(3+4x) = 3 + 10x + 8x^2
    EXPECT_EQ(prod.degree(), 2);
    EXPECT_DOUBLE_EQ(prod[0], 3.0);
    EXPECT_DOUBLE_EQ(prod[1], 10.0);
    EXPECT_DOUBLE_EQ(prod[2], 8.0);
}

TEST(PolynomialTest, MultiplicationByZero) {
    polynomial<double> p{1.0, 2.0, 3.0};
    polynomial<double> zero;

    auto prod = p * zero;
    EXPECT_TRUE(prod.is_zero());
}

TEST(PolynomialTest, ScalarMultiplication) {
    polynomial<double> p{1.0, 2.0, 3.0};

    auto doubled = 2.0 * p;
    EXPECT_DOUBLE_EQ(doubled[0], 2.0);
    EXPECT_DOUBLE_EQ(doubled[1], 4.0);
    EXPECT_DOUBLE_EQ(doubled[2], 6.0);
}

// =============================================================================
// Division tests
// =============================================================================

TEST(PolynomialTest, DivisionExact) {
    // (x^2 - 1) / (x - 1) = x + 1
    polynomial<double> p{-1.0, 0.0, 1.0};  // x^2 - 1
    polynomial<double> q{-1.0, 1.0};       // x - 1

    auto [quot, rem] = divmod(p, q);

    EXPECT_EQ(quot.degree(), 1);
    EXPECT_DOUBLE_EQ(quot[0], 1.0);  // x + 1
    EXPECT_DOUBLE_EQ(quot[1], 1.0);
    EXPECT_TRUE(rem.is_zero());
}

TEST(PolynomialTest, DivisionWithRemainder) {
    // (x^3 + 1) / (x + 1) = x^2 - x + 1 remainder 0
    // Actually: x^3 + 1 = (x + 1)(x^2 - x + 1)
    polynomial<double> p{1.0, 0.0, 0.0, 1.0};  // x^3 + 1
    polynomial<double> q{1.0, 1.0};            // x + 1

    auto [quot, rem] = divmod(p, q);

    // x^2 - x + 1
    EXPECT_EQ(quot.degree(), 2);
    EXPECT_DOUBLE_EQ(quot[0], 1.0);
    EXPECT_DOUBLE_EQ(quot[1], -1.0);
    EXPECT_DOUBLE_EQ(quot[2], 1.0);
    EXPECT_TRUE(rem.is_zero());
}

TEST(PolynomialTest, DivisionByZeroThrows) {
    polynomial<double> p{1.0, 2.0};
    polynomial<double> zero;

    EXPECT_THROW(divmod(p, zero), std::domain_error);
}

TEST(PolynomialTest, DivisionProperty) {
    // For any a, b with b != 0: a = b*q + r
    polynomial<double> a{1.0, 2.0, 3.0, 4.0};  // 1 + 2x + 3x^2 + 4x^3
    polynomial<double> b{1.0, 1.0};            // 1 + x

    auto [q, r] = divmod(a, b);

    // Verify: a = b*q + r
    auto reconstructed = b * q + r;
    EXPECT_EQ(reconstructed, a);

    // Verify: deg(r) < deg(b) or r = 0
    EXPECT_TRUE(r.is_zero() || r.degree() < b.degree());
}

// =============================================================================
// Equality tests
// =============================================================================

TEST(PolynomialTest, Equality) {
    polynomial<double> p{1.0, 2.0, 3.0};
    polynomial<double> q{1.0, 2.0, 3.0};
    polynomial<double> r{1.0, 2.0};

    EXPECT_TRUE(p == q);
    EXPECT_FALSE(p == r);
    EXPECT_TRUE(p != r);
}

TEST(PolynomialTest, EqualityZero) {
    polynomial<double> p;
    polynomial<double> q{0.0};

    EXPECT_TRUE(p == q);
}
