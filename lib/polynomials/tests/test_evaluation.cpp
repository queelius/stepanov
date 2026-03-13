/**
 * @file test_evaluation.cpp
 * @brief Tests for polynomial evaluation and calculus operations
 */

#include <polynomials/polynomial.hpp>
#include <gtest/gtest.h>
#include <cmath>

using namespace poly;

constexpr double tol = 1e-10;

// =============================================================================
// Evaluation tests
// =============================================================================

TEST(EvaluationTest, Constant) {
    polynomial<double> p{5.0};
    EXPECT_DOUBLE_EQ(evaluate(p, 0.0), 5.0);
    EXPECT_DOUBLE_EQ(evaluate(p, 100.0), 5.0);
}

TEST(EvaluationTest, Linear) {
    // p(x) = 2x + 3
    polynomial<double> p{3.0, 2.0};
    EXPECT_DOUBLE_EQ(evaluate(p, 0.0), 3.0);
    EXPECT_DOUBLE_EQ(evaluate(p, 1.0), 5.0);
    EXPECT_DOUBLE_EQ(evaluate(p, -1.0), 1.0);
}

TEST(EvaluationTest, Quadratic) {
    // p(x) = x^2 - 2x + 1 = (x-1)^2
    polynomial<double> p{1.0, -2.0, 1.0};
    EXPECT_DOUBLE_EQ(evaluate(p, 0.0), 1.0);
    EXPECT_DOUBLE_EQ(evaluate(p, 1.0), 0.0);  // Root
    EXPECT_DOUBLE_EQ(evaluate(p, 2.0), 1.0);
}

TEST(EvaluationTest, Zero) {
    polynomial<double> p;
    EXPECT_DOUBLE_EQ(evaluate(p, 42.0), 0.0);
}

TEST(EvaluationTest, EvaluateMany) {
    polynomial<double> p{0.0, 0.0, 1.0};  // x^2
    std::vector<double> points = {0.0, 1.0, 2.0, 3.0};

    auto values = evaluate_many(p, points);

    EXPECT_DOUBLE_EQ(values[0], 0.0);
    EXPECT_DOUBLE_EQ(values[1], 1.0);
    EXPECT_DOUBLE_EQ(values[2], 4.0);
    EXPECT_DOUBLE_EQ(values[3], 9.0);
}

TEST(EvaluationTest, HornerVsNaive) {
    // Both methods should give the same result
    polynomial<double> p{1.0, -2.0, 3.0, -4.0, 5.0};  // 1 - 2x + 3x^2 - 4x^3 + 5x^4

    double x = 2.5;
    double val1 = evaluate(p, x);
    double val2 = evaluate_horner(p, x);

    EXPECT_NEAR(val1, val2, tol);
}

// =============================================================================
// Derivative tests
// =============================================================================

TEST(DerivativeTest, Constant) {
    polynomial<double> p{5.0};
    auto dp = derivative(p);
    EXPECT_TRUE(dp.is_zero());
}

TEST(DerivativeTest, Linear) {
    // d/dx (3 + 2x) = 2
    polynomial<double> p{3.0, 2.0};
    auto dp = derivative(p);

    EXPECT_EQ(dp.degree(), 0);
    EXPECT_DOUBLE_EQ(dp[0], 2.0);
}

TEST(DerivativeTest, Quadratic) {
    // d/dx (1 - 2x + 3x^2) = -2 + 6x
    polynomial<double> p{1.0, -2.0, 3.0};
    auto dp = derivative(p);

    EXPECT_EQ(dp.degree(), 1);
    EXPECT_DOUBLE_EQ(dp[0], -2.0);
    EXPECT_DOUBLE_EQ(dp[1], 6.0);
}

TEST(DerivativeTest, PowerRule) {
    // d/dx (x^n) = n*x^(n-1)
    auto p = polynomial<double>::monomial(1.0, 5);  // x^5
    auto dp = derivative(p);

    EXPECT_EQ(dp.degree(), 4);
    EXPECT_DOUBLE_EQ(dp[4], 5.0);  // 5x^4
}

TEST(DerivativeTest, Chain) {
    // d^2/dx^2 (x^4) = 12x^2
    auto p = polynomial<double>::monomial(1.0, 4);
    auto dp = derivative(p);
    auto ddp = derivative(dp);

    EXPECT_EQ(ddp.degree(), 2);
    EXPECT_DOUBLE_EQ(ddp[2], 12.0);
}

// =============================================================================
// Antiderivative tests
// =============================================================================

TEST(AntiderivativeTest, Constant) {
    // integral of 3 = 3x + C
    polynomial<double> p{3.0};
    auto integral = antiderivative(p, 5.0);  // C = 5

    EXPECT_EQ(integral.degree(), 1);
    EXPECT_DOUBLE_EQ(integral[0], 5.0);
    EXPECT_DOUBLE_EQ(integral[1], 3.0);
}

TEST(AntiderivativeTest, Linear) {
    // integral of 2x = x^2 + C
    polynomial<double> p{0.0, 2.0};  // 2x
    auto integral = antiderivative(p);

    EXPECT_EQ(integral.degree(), 2);
    EXPECT_DOUBLE_EQ(integral[2], 1.0);
}

TEST(AntiderivativeTest, PowerRule) {
    // integral of x^n = x^(n+1)/(n+1)
    auto p = polynomial<double>::monomial(1.0, 3);  // x^3
    auto integral = antiderivative(p);

    EXPECT_EQ(integral.degree(), 4);
    EXPECT_DOUBLE_EQ(integral[4], 0.25);  // x^4/4
}

TEST(AntiderivativeTest, FundamentalTheorem) {
    // derivative(antiderivative(p)) = p
    polynomial<double> p{1.0, -2.0, 3.0};
    auto integral = antiderivative(p);
    auto back = derivative(integral);

    EXPECT_EQ(back, p);
}

// =============================================================================
// Definite integral tests
// =============================================================================

TEST(DefiniteIntegralTest, Basic) {
    // integral from 0 to 1 of x dx = 0.5
    polynomial<double> p{0.0, 1.0};  // x
    double result = definite_integral(p, 0.0, 1.0);

    EXPECT_NEAR(result, 0.5, tol);
}

TEST(DefiniteIntegralTest, Quadratic) {
    // integral from 0 to 2 of x^2 dx = 8/3
    auto p = polynomial<double>::monomial(1.0, 2);  // x^2
    double result = definite_integral(p, 0.0, 2.0);

    EXPECT_NEAR(result, 8.0/3.0, tol);
}

// =============================================================================
// Root finding tests
// =============================================================================

TEST(RootFindingTest, LinearRoot) {
    // x - 2 = 0 has root at x = 2
    polynomial<double> p{-2.0, 1.0};

    auto root = find_root_newton(p, 0.0);

    ASSERT_TRUE(root.has_value());
    EXPECT_NEAR(*root, 2.0, tol);
}

TEST(RootFindingTest, QuadraticRoots) {
    // x^2 - 4 = (x-2)(x+2) has roots at -2 and 2
    polynomial<double> p{-4.0, 0.0, 1.0};

    auto roots = find_roots(p, -10.0, 10.0);

    ASSERT_EQ(roots.size(), 2);
    EXPECT_NEAR(roots[0], -2.0, tol);
    EXPECT_NEAR(roots[1], 2.0, tol);
}

TEST(RootFindingTest, VerifyRoots) {
    // Roots should satisfy p(root) = 0
    polynomial<double> p{-6.0, 1.0, 1.0};  // x^2 + x - 6 = (x-2)(x+3)

    auto roots = find_roots(p, -10.0, 10.0);

    for (double r : roots) {
        EXPECT_NEAR(evaluate(p, r), 0.0, tol);
    }
}

TEST(RootFindingTest, NoRootsInInterval) {
    // x^2 + 1 has no real roots
    polynomial<double> p{1.0, 0.0, 1.0};

    auto roots = find_roots(p, -10.0, 10.0);

    EXPECT_TRUE(roots.empty());
}

// =============================================================================
// Synthetic division tests
// =============================================================================

TEST(SyntheticDivisionTest, Basic) {
    // (x^2 - 1) / (x - 1) using synthetic division at c = 1
    polynomial<double> p{-1.0, 0.0, 1.0};

    auto [quotient, rem] = synthetic_division(p, 1.0);

    // Quotient should be x + 1
    EXPECT_EQ(quotient.degree(), 1);
    EXPECT_NEAR(quotient[0], 1.0, tol);
    EXPECT_NEAR(quotient[1], 1.0, tol);

    // Remainder should be 0 (since 1 is a root)
    EXPECT_NEAR(rem, 0.0, tol);
}

TEST(SyntheticDivisionTest, NonRoot) {
    // (x^2 - 1) divided by (x - 0) = x^2 - 1 = x * x + (-1)
    // So quotient is x, remainder is -1
    polynomial<double> p{-1.0, 0.0, 1.0};

    auto [quotient, rem] = synthetic_division(p, 0.0);

    EXPECT_EQ(quotient.degree(), 1);
    EXPECT_NEAR(quotient[1], 1.0, tol);  // x
    EXPECT_NEAR(rem, -1.0, tol);
}

TEST(SyntheticDivisionTest, RemainderTheorem) {
    // p(c) = remainder when dividing by (x - c)
    polynomial<double> p{1.0, -2.0, 3.0, -1.0};  // -1 + 3x - 2x^2 + x^3

    double c = 2.0;
    auto [quotient, rem] = synthetic_division(p, c);

    EXPECT_NEAR(rem, evaluate(p, c), tol);
}

// =============================================================================
// Stationary and inflection points
// =============================================================================

TEST(CriticalPointsTest, StationaryPoints) {
    // p(x) = x^3 - 3x has derivative 3x^2 - 3 = 3(x^2 - 1)
    // Stationary points at x = -1 and x = 1
    polynomial<double> p{0.0, -3.0, 0.0, 1.0};

    auto stats = stationary_points(p, -10.0, 10.0);

    ASSERT_EQ(stats.size(), 2);
    EXPECT_NEAR(stats[0], -1.0, tol);
    EXPECT_NEAR(stats[1], 1.0, tol);
}

TEST(CriticalPointsTest, InflectionPoints) {
    // p(x) = x^3 has second derivative 6x
    // Inflection point at x = 0
    polynomial<double> p{0.0, 0.0, 0.0, 1.0};

    auto inflects = inflection_points(p, -10.0, 10.0);

    ASSERT_EQ(inflects.size(), 1);
    EXPECT_NEAR(inflects[0], 0.0, tol);
}

// =============================================================================
// Deflation tests
// =============================================================================

TEST(DeflationTest, RemoveRoot) {
    // p = x^2 - 3x + 2 = (x-1)(x-2)
    // Deflate by root 1 should give (x-2)
    polynomial<double> p{2.0, -3.0, 1.0};

    auto deflated = deflate(p, 1.0);

    EXPECT_EQ(deflated.degree(), 1);
    // Should be x - 2
    EXPECT_NEAR(deflated[0], -2.0, tol);
    EXPECT_NEAR(deflated[1], 1.0, tol);
}
