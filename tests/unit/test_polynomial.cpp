#include <gtest/gtest.h>
#include <stepanov/builtin_adaptors.hpp>
#include <stepanov/polynomial.hpp>
#include <vector>
#include <random>
#include <complex>
#include <cmath>

using namespace stepanov;

class PolynomialTest : public ::testing::Test {
protected:
    std::mt19937 rng{42}; // Fixed seed for reproducibility

    // Helper to create polynomial from coefficients
    template<typename T>
    polynomial<T> from_coeffs(std::initializer_list<T> coeffs) {
        polynomial<T> p;
        int degree = 0;
        for (auto c : coeffs) {
            if (c != T{0}) {
                p[degree] = c;
            }
            degree++;
        }
        return p;
    }

    // Helper to compare polynomials
    template<typename T>
    bool poly_equal(const polynomial<T>& a, const polynomial<T>& b, T eps = T{}) {
        if (a.degree() != b.degree()) return false;

        for (const auto& [deg, coeff] : a) {
            auto it = b.find(deg);
            if (it == b.end()) {
                if (std::abs(coeff) > eps) return false;
            } else {
                if (std::abs(coeff - it->second) > eps) return false;
            }
        }

        for (const auto& [deg, coeff] : b) {
            auto it = a.find(deg);
            if (it == a.end()) {
                if (std::abs(coeff) > eps) return false;
            }
        }

        return true;
    }
};

// ============== Basic Construction Tests ==============

TEST_F(PolynomialTest, ConstructionDefault) {
    polynomial<int> p;
    EXPECT_EQ(p.degree(), -1); // Empty polynomial has degree -1
    EXPECT_TRUE(p.empty());
}

TEST_F(PolynomialTest, ConstructionFromCoefficients) {
    // p(x) = 3x^2 + 2x + 1
    auto p = from_coeffs<int>({1, 2, 3});
    EXPECT_EQ(p.degree(), 2);
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 2);
    EXPECT_EQ(p[2], 3);
}

TEST_F(PolynomialTest, ConstructionSparse) {
    // p(x) = x^5 + x^2 + 1
    polynomial<int> p;
    p[0] = 1;
    p[2] = 1;
    p[5] = 1;

    EXPECT_EQ(p.degree(), 5);
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[5], 1);
    EXPECT_EQ(p[1], 0); // Non-existent coefficients are 0
    EXPECT_EQ(p[3], 0);
}

// ============== Evaluation Tests ==============

TEST_F(PolynomialTest, EvaluationBasic) {
    // p(x) = 2x^2 + 3x + 1
    auto p = from_coeffs<int>({1, 3, 2});

    EXPECT_EQ(p(0), 1);     // p(0) = 1
    EXPECT_EQ(p(1), 6);     // p(1) = 2 + 3 + 1 = 6
    EXPECT_EQ(p(2), 15);    // p(2) = 8 + 6 + 1 = 15
    EXPECT_EQ(p(-1), 0);    // p(-1) = 2 - 3 + 1 = 0
}

TEST_F(PolynomialTest, EvaluationFloatingPoint) {
    // p(x) = x^2 - 2
    auto p = from_coeffs<double>({-2.0, 0.0, 1.0});

    EXPECT_DOUBLE_EQ(p(0.0), -2.0);
    EXPECT_DOUBLE_EQ(p(1.0), -1.0);
    EXPECT_NEAR(p(std::sqrt(2.0)), 0.0, 1e-10); // Root at sqrt(2)
}

TEST_F(PolynomialTest, EvaluationComplex) {
    using complex = std::complex<double>;

    // p(z) = z^2 + 1 (roots at ±i)
    auto p = from_coeffs<complex>({complex(1, 0), complex(0, 0), complex(1, 0)});

    complex i(0, 1);
    auto result = p(i);
    EXPECT_NEAR(result.real(), 0.0, 1e-10);
    EXPECT_NEAR(result.imag(), 0.0, 1e-10);
}

// ============== Arithmetic Operations Tests ==============

TEST_F(PolynomialTest, AdditionBasic) {
    // p(x) = x^2 + 2x + 1
    auto p = from_coeffs<int>({1, 2, 1});
    // q(x) = x + 3
    auto q = from_coeffs<int>({3, 1, 0});

    auto sum = p + q;
    EXPECT_EQ(sum.degree(), 2);
    EXPECT_EQ(sum[0], 4); // 1 + 3
    EXPECT_EQ(sum[1], 3); // 2 + 1
    EXPECT_EQ(sum[2], 1); // 1 + 0
}

TEST_F(PolynomialTest, AdditionDifferentDegrees) {
    // p(x) = x^5 + 1
    polynomial<int> p;
    p[0] = 1;
    p[5] = 1;

    // q(x) = x^2 + x
    polynomial<int> q;
    q[1] = 1;
    q[2] = 1;

    auto sum = p + q;
    EXPECT_EQ(sum.degree(), 5);
    EXPECT_EQ(sum[0], 1);
    EXPECT_EQ(sum[1], 1);
    EXPECT_EQ(sum[2], 1);
    EXPECT_EQ(sum[5], 1);
}

TEST_F(PolynomialTest, SubtractionBasic) {
    // p(x) = x^2 + 3x + 2
    auto p = from_coeffs<int>({2, 3, 1});
    // q(x) = x + 1
    auto q = from_coeffs<int>({1, 1, 0});

    auto diff = p - q;
    EXPECT_EQ(diff.degree(), 2);
    EXPECT_EQ(diff[0], 1);  // 2 - 1
    EXPECT_EQ(diff[1], 2);  // 3 - 1
    EXPECT_EQ(diff[2], 1);  // 1 - 0
}

TEST_F(PolynomialTest, MultiplicationBasic) {
    // p(x) = x + 1
    auto p = from_coeffs<int>({1, 1});
    // q(x) = x - 1
    auto q = from_coeffs<int>({-1, 1});

    // (x + 1)(x - 1) = x^2 - 1
    auto prod = p * q;
    EXPECT_EQ(prod.degree(), 2);
    EXPECT_EQ(prod[0], -1);
    EXPECT_EQ(prod[1], 0);
    EXPECT_EQ(prod[2], 1);
}

TEST_F(PolynomialTest, MultiplicationExpansion) {
    // p(x) = x^2 + 2x + 1 = (x + 1)^2
    auto p = from_coeffs<int>({1, 2, 1});
    // q(x) = x + 1
    auto q = from_coeffs<int>({1, 1});

    // (x + 1)^3 = x^3 + 3x^2 + 3x + 1
    auto prod = p * q;
    EXPECT_EQ(prod.degree(), 3);
    EXPECT_EQ(prod[0], 1);
    EXPECT_EQ(prod[1], 3);
    EXPECT_EQ(prod[2], 3);
    EXPECT_EQ(prod[3], 1);
}

TEST_F(PolynomialTest, ScalarMultiplication) {
    // p(x) = 2x^2 + 3x + 1
    auto p = from_coeffs<int>({1, 3, 2});

    auto scaled = p * 3;
    EXPECT_EQ(scaled[0], 3);
    EXPECT_EQ(scaled[1], 9);
    EXPECT_EQ(scaled[2], 6);
}

// ============== Division Tests ==============

TEST_F(PolynomialTest, DivisionExact) {
    // p(x) = x^2 - 1 = (x + 1)(x - 1)
    auto p = from_coeffs<int>({-1, 0, 1});
    // q(x) = x - 1
    auto q = from_coeffs<int>({-1, 1});

    auto [quotient, remainder] = p.divide(q);

    // Quotient should be x + 1
    EXPECT_EQ(quotient.degree(), 1);
    EXPECT_EQ(quotient[0], 1);
    EXPECT_EQ(quotient[1], 1);

    // Remainder should be 0
    EXPECT_EQ(remainder.degree(), -1);
}

TEST_F(PolynomialTest, DivisionWithRemainder) {
    // p(x) = x^3 + 2x + 1
    auto p = from_coeffs<int>({1, 2, 0, 1});
    // q(x) = x^2 + 1
    auto q = from_coeffs<int>({1, 0, 1});

    auto [quotient, remainder] = p.divide(q);

    // x^3 + 2x + 1 = (x^2 + 1) * x + (x + 1)
    EXPECT_EQ(quotient.degree(), 1);
    EXPECT_EQ(quotient[0], 0);
    EXPECT_EQ(quotient[1], 1);

    EXPECT_EQ(remainder.degree(), 1);
    EXPECT_EQ(remainder[0], 1);
    EXPECT_EQ(remainder[1], 1);
}

// ============== Derivative Tests ==============

TEST_F(PolynomialTest, DerivativeBasic) {
    // p(x) = x^3 + 3x^2 + 2x + 1
    auto p = from_coeffs<int>({1, 2, 3, 1});

    // p'(x) = 3x^2 + 6x + 2
    auto dp = p.derivative();
    EXPECT_EQ(dp.degree(), 2);
    EXPECT_EQ(dp[0], 2);
    EXPECT_EQ(dp[1], 6);
    EXPECT_EQ(dp[2], 3);
}

TEST_F(PolynomialTest, DerivativeConstant) {
    // p(x) = 5
    auto p = from_coeffs<int>({5});

    // p'(x) = 0
    auto dp = p.derivative();
    EXPECT_EQ(dp.degree(), -1); // Zero polynomial
}

TEST_F(PolynomialTest, DerivativeHighOrder) {
    // p(x) = x^5
    polynomial<int> p;
    p[5] = 1;

    // p'(x) = 5x^4
    auto dp1 = p.derivative();
    EXPECT_EQ(dp1.degree(), 4);
    EXPECT_EQ(dp1[4], 5);

    // p''(x) = 20x^3
    auto dp2 = dp1.derivative();
    EXPECT_EQ(dp2.degree(), 3);
    EXPECT_EQ(dp2[3], 20);
}

// ============== Root Finding Tests ==============

TEST_F(PolynomialTest, NewtonMethodLinear) {
    // p(x) = 2x - 4, root at x = 2
    auto p = from_coeffs<double>({-4.0, 2.0});

    double root = p.newton_root(1.0); // Start from x = 1
    EXPECT_NEAR(root, 2.0, 1e-10);
}

TEST_F(PolynomialTest, NewtonMethodQuadratic) {
    // p(x) = x^2 - 2, roots at ±sqrt(2)
    auto p = from_coeffs<double>({-2.0, 0.0, 1.0});

    double root1 = p.newton_root(1.0);  // Converges to positive root
    double root2 = p.newton_root(-1.0); // Converges to negative root

    EXPECT_NEAR(std::abs(root1), std::sqrt(2.0), 1e-10);
    EXPECT_NEAR(std::abs(root2), std::sqrt(2.0), 1e-10);
    EXPECT_TRUE(root1 * root2 < 0); // Opposite signs
}

TEST_F(PolynomialTest, NewtonMethodCubic) {
    // p(x) = x^3 - x - 2, has root near x = 1.52
    auto p = from_coeffs<double>({-2.0, -1.0, 0.0, 1.0});

    double root = p.newton_root(1.0);
    EXPECT_NEAR(p(root), 0.0, 1e-10); // Verify it's actually a root
}

// ============== Composition Tests ==============

TEST_F(PolynomialTest, CompositionBasic) {
    // p(x) = x^2 + 1
    auto p = from_coeffs<int>({1, 0, 1});
    // q(x) = x + 2
    auto q = from_coeffs<int>({2, 1});

    // p(q(x)) = (x + 2)^2 + 1 = x^2 + 4x + 5
    auto comp = p.compose(q);
    EXPECT_EQ(comp.degree(), 2);
    EXPECT_EQ(comp[0], 5);
    EXPECT_EQ(comp[1], 4);
    EXPECT_EQ(comp[2], 1);
}

// ============== GCD Tests ==============

TEST_F(PolynomialTest, GcdBasic) {
    // p(x) = x^2 - 1 = (x-1)(x+1)
    auto p = from_coeffs<double>({-1.0, 0.0, 1.0});
    // q(x) = x - 1
    auto q = from_coeffs<double>({-1.0, 1.0});

    auto g = polynomial<double>::gcd(p, q);

    // GCD should be proportional to (x - 1)
    EXPECT_EQ(g.degree(), 1);
    double ratio = g[1] / 1.0;
    EXPECT_NEAR(g[0] / ratio, -1.0, 1e-10);
}

// ============== Properties Tests ==============

TEST_F(PolynomialTest, AlgebraicProperties) {
    auto p = from_coeffs<int>({1, 2, 3});
    auto q = from_coeffs<int>({4, 5, 6});
    auto r = from_coeffs<int>({7, 8, 9});

    // Commutativity of addition
    EXPECT_TRUE(poly_equal(p + q, q + p));

    // Associativity of addition
    EXPECT_TRUE(poly_equal((p + q) + r, p + (q + r)));

    // Commutativity of multiplication
    EXPECT_TRUE(poly_equal(p * q, q * p));

    // Distributivity
    EXPECT_TRUE(poly_equal(p * (q + r), p * q + p * r));
}

TEST_F(PolynomialTest, IdentityElements) {
    auto p = from_coeffs<int>({1, 2, 3});
    polynomial<int> zero;
    polynomial<int> one;
    one[0] = 1;

    // Additive identity
    EXPECT_TRUE(poly_equal(p + zero, p));
    EXPECT_TRUE(poly_equal(zero + p, p));

    // Multiplicative identity
    EXPECT_TRUE(poly_equal(p * one, p));
    EXPECT_TRUE(poly_equal(one * p, p));

    // Multiplicative zero
    EXPECT_TRUE(poly_equal(p * zero, zero));
    EXPECT_TRUE(poly_equal(zero * p, zero));
}

// ============== Special Polynomials Tests ==============

TEST_F(PolynomialTest, ChebyshevPolynomial) {
    // T_0(x) = 1
    polynomial<int> T0;
    T0[0] = 1;

    // T_1(x) = x
    polynomial<int> T1;
    T1[1] = 1;

    // T_2(x) = 2x^2 - 1
    polynomial<int> T2;
    T2[0] = -1;
    T2[2] = 2;

    // Verify recurrence: T_{n+1}(x) = 2x*T_n(x) - T_{n-1}(x)
    polynomial<int> two_x;
    two_x[1] = 2;

    auto T2_computed = two_x * T1 - T0;
    EXPECT_TRUE(poly_equal(T2, T2_computed));
}

// ============== Edge Cases ==============

TEST_F(PolynomialTest, EmptyPolynomial) {
    polynomial<int> p;

    EXPECT_EQ(p.degree(), -1);
    EXPECT_EQ(p(0), 0);
    EXPECT_EQ(p(100), 0);

    auto dp = p.derivative();
    EXPECT_EQ(dp.degree(), -1);
}

TEST_F(PolynomialTest, LargeDegree) {
    polynomial<int> p;
    p[1000] = 1;

    EXPECT_EQ(p.degree(), 1000);
    EXPECT_EQ(p(2), 1 << 1000); // This will overflow, but tests the computation

    auto dp = p.derivative();
    EXPECT_EQ(dp.degree(), 999);
    EXPECT_EQ(dp[999], 1000);
}

// ============== String Representation Tests ==============

TEST_F(PolynomialTest, StringOutput) {
    // p(x) = x^2 + 2x + 3
    auto p = from_coeffs<int>({3, 2, 1});

    std::stringstream ss;
    ss << p;
    std::string str = ss.str();

    // Should contain these terms (exact format may vary)
    EXPECT_NE(str.find("x^2"), std::string::npos);
    EXPECT_NE(str.find("2x"), std::string::npos);
    EXPECT_NE(str.find("3"), std::string::npos);
}