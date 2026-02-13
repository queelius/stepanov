// test_gradator.cpp - Tests for gradator automatic differentiation library
//
// Tests verify analytical gradients against finite difference approximations.
// This is the gold standard for testing AD implementations.

#include <gtest/gtest.h>
#include <gradator.hpp>
#include <cmath>

using namespace gradator;
using namespace elementa;

// Tolerance for finite difference comparisons
constexpr double RTOL = 1e-4;
constexpr double ATOL = 1e-6;

// Helper to check if two matrices are approximately equal
template <typename T>
bool matrices_approx_equal(const matrix<T>& a, const matrix<T>& b, T rtol = RTOL, T atol = ATOL) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < a.cols(); ++j) {
            auto diff = std::abs(a(i, j) - b(i, j));
            auto tol = atol + rtol * std::abs(b(i, j));
            if (diff > tol) return false;
        }
    }
    return true;
}

// =============================================================================
// Basic Variable and Graph Tests
// =============================================================================

TEST(GraphVar, CreateFromMatrix) {
    graph g;
    matrix<double> m{{1, 2}, {3, 4}};
    auto x = g.make_var(m);

    EXPECT_EQ(x.value().rows(), 2);
    EXPECT_EQ(x.value().cols(), 2);
    EXPECT_EQ(x.value()(0, 0), 1);
    EXPECT_EQ(g.size(), 1);
}

TEST(GraphVar, MultipleVariables) {
    graph g;
    auto x = g.make_var(matrix<double>{{1, 2}});
    auto y = g.make_var(matrix<double>{{3, 4}});

    EXPECT_EQ(g.size(), 2);
    EXPECT_EQ(x.id().index, 0);
    EXPECT_EQ(y.id().index, 1);
}

TEST(GraphVar, ValConstant) {
    val<matrix<double>> c(matrix<double>{{1, 2}});
    EXPECT_EQ(c.value()(0, 0), 1);
}

// =============================================================================
// Arithmetic Operation Tests
// =============================================================================

TEST(GradNegation, Basic) {
    auto f = [](const auto& x) { return -x; };

    matrix<double> input{{1, 2}, {3, 4}};
    auto analytical = grad(f)(input);
    auto expected = -ones<double>(2, 2);

    EXPECT_TRUE(matrices_approx_equal(analytical, expected));
}

TEST(GradAddition, VarPlusVar) {
    graph g;
    matrix<double> m1{{1, 2}};
    matrix<double> m2{{3, 4}};

    auto x = g.make_var(m1);
    auto y = g.make_var(m2);
    auto z = x + y;

    g.backward<matrix<double>>(z);

    auto dx = g.gradient(x);
    auto dy = g.gradient(y);

    EXPECT_TRUE(matrices_approx_equal(dx, ones<double>(1, 2)));
    EXPECT_TRUE(matrices_approx_equal(dy, ones<double>(1, 2)));
}

TEST(GradAddition, VarPlusVal) {
    auto f = [](const auto& x) {
        val<matrix<double>> c(matrix<double>{{10, 20}});
        return x + c;
    };

    matrix<double> input{{1, 2}};
    auto analytical = grad(f)(input);

    EXPECT_TRUE(matrices_approx_equal(analytical, ones<double>(1, 2)));
}

TEST(GradSubtraction, Basic) {
    graph g;
    matrix<double> m1{{5, 6}};
    matrix<double> m2{{1, 2}};

    auto x = g.make_var(m1);
    auto y = g.make_var(m2);
    auto z = x - y;

    g.backward<matrix<double>>(z);

    auto dx = g.gradient(x);
    auto dy = g.gradient(y);

    EXPECT_TRUE(matrices_approx_equal(dx, ones<double>(1, 2)));
    EXPECT_TRUE(matrices_approx_equal(dy, -ones<double>(1, 2)));
}

TEST(GradScalarMul, Basic) {
    auto f = [](const auto& x) { return 3.0 * x; };

    matrix<double> input{{1, 2}, {3, 4}};
    auto analytical = grad(f)(input);
    auto expected = ones<double>(2, 2) * 3.0;

    EXPECT_TRUE(matrices_approx_equal(analytical, expected));
}

// =============================================================================
// Matrix Operation Tests
// =============================================================================

TEST(GradMatmul, VarTimesVar) {
    graph g;
    matrix<double> A{{1, 2}, {3, 4}};
    matrix<double> B{{5, 6}, {7, 8}};

    auto x = g.make_var(A);
    auto y = g.make_var(B);
    auto z = matmul(x, y);
    auto s = sum(z);

    g.backward<double>(s);

    auto dA = g.gradient(x);
    auto dB = g.gradient(y);

    auto f_A = [&B](const matrix<double>& A_) {
        return elementa::sum(elementa::matmul(A_, B));
    };
    auto f_B = [&A](const matrix<double>& B_) {
        return elementa::sum(elementa::matmul(A, B_));
    };

    auto numerical_dA = finite_diff_gradient(f_A, A);
    auto numerical_dB = finite_diff_gradient(f_B, B);

    EXPECT_TRUE(matrices_approx_equal(dA, numerical_dA));
    EXPECT_TRUE(matrices_approx_equal(dB, numerical_dB));
}

TEST(GradTranspose, Basic) {
    auto f = [](const auto& x) {
        return sum(transpose(x));
    };

    matrix<double> input{{1, 2, 3}, {4, 5, 6}};
    auto analytical = grad(f)(input);

    auto f_scalar = [](const matrix<double>& x) {
        return elementa::sum(elementa::transpose(x));
    };
    auto numerical = finite_diff_gradient(f_scalar, input);

    EXPECT_TRUE(matrices_approx_equal(analytical, numerical));
}

TEST(GradTrace, Basic) {
    auto f = [](const auto& x) {
        return trace(x);
    };

    matrix<double> input{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto analytical = grad(f)(input);
    auto expected = eye<double>(3);

    EXPECT_TRUE(matrices_approx_equal(analytical, expected));
}

TEST(GradSum, Basic) {
    auto f = [](const auto& x) {
        return sum(x);
    };

    matrix<double> input{{1, 2}, {3, 4}};
    auto analytical = grad(f)(input);
    auto expected = ones<double>(2, 2);

    EXPECT_TRUE(matrices_approx_equal(analytical, expected));
}

// =============================================================================
// Transcendental Operation Tests
// =============================================================================

TEST(GradExp, Basic) {
    auto f = [](const auto& x) {
        return sum(exp(x));
    };

    matrix<double> input{{0.1, 0.2}, {0.3, 0.4}};
    auto analytical = grad(f)(input);
    auto expected = elementa::exp(input);

    EXPECT_TRUE(matrices_approx_equal(analytical, expected));
}

TEST(GradLog, Basic) {
    auto f = [](const auto& x) {
        return sum(log(x));
    };

    matrix<double> input{{1, 2}, {3, 4}};
    auto analytical = grad(f)(input);

    auto f_scalar = [](const matrix<double>& x) {
        return elementa::sum(elementa::log(x));
    };
    auto numerical = finite_diff_gradient(f_scalar, input);

    EXPECT_TRUE(matrices_approx_equal(analytical, numerical));
}

TEST(GradSqrt, Basic) {
    auto f = [](const auto& x) {
        return sum(sqrt(x));
    };

    matrix<double> input{{1, 4}, {9, 16}};
    auto analytical = grad(f)(input);

    auto f_scalar = [](const matrix<double>& x) {
        return elementa::sum(elementa::sqrt(x));
    };
    auto numerical = finite_diff_gradient(f_scalar, input);

    EXPECT_TRUE(matrices_approx_equal(analytical, numerical));
}

TEST(GradPow, Basic) {
    auto f = [](const auto& x) {
        return sum(pow(x, 3.0));
    };

    matrix<double> input{{1, 2}, {3, 4}};
    auto analytical = grad(f)(input);

    auto f_scalar = [](const matrix<double>& x) {
        return elementa::sum(elementa::pow(x, 3.0));
    };
    auto numerical = finite_diff_gradient(f_scalar, input);

    EXPECT_TRUE(matrices_approx_equal(analytical, numerical));
}

// =============================================================================
// Linear Algebra Operation Tests
// =============================================================================

TEST(GradDet, Basic) {
    auto f = [](const auto& x) {
        return det(x);
    };

    matrix<double> A{{2, 1}, {1, 3}};
    auto analytical = grad(f)(A);

    auto f_scalar = [](const matrix<double>& x) {
        return elementa::det(x);
    };
    auto numerical = finite_diff_gradient(f_scalar, A);

    EXPECT_TRUE(matrices_approx_equal(analytical, numerical, 1e-3, 1e-5));
}

TEST(GradLogdet, Basic) {
    auto f = [](const auto& x) {
        return logdet(x);
    };

    matrix<double> A{{4, 1}, {1, 3}};
    auto analytical = grad(f)(A);

    auto f_scalar = [](const matrix<double>& x) {
        auto [sign, ld] = elementa::logdet(x);
        return ld;
    };
    auto numerical = finite_diff_gradient(f_scalar, A);

    EXPECT_TRUE(matrices_approx_equal(analytical, numerical, 1e-3, 1e-5));
}

TEST(GradInverse, Basic) {
    auto f = [](const auto& x) {
        return sum(inverse(x));
    };

    matrix<double> A{{2, 1}, {1, 3}};
    auto analytical = grad(f)(A);

    auto f_scalar = [](const matrix<double>& x) {
        return elementa::sum(elementa::inverse(x));
    };
    auto numerical = finite_diff_gradient(f_scalar, A);

    EXPECT_TRUE(matrices_approx_equal(analytical, numerical, 1e-3, 1e-5));
}

TEST(GradSolve, WrtA) {
    matrix<double> A{{2, 1}, {1, 3}};
    matrix<double> b{{1}, {2}};

    auto f = [&b](const auto& x) {
        val<matrix<double>> bv(b);
        return sum(solve(x, bv));
    };

    auto analytical = grad(f)(A);

    auto f_scalar = [&b](const matrix<double>& A_) {
        return elementa::sum(elementa::solve(A_, b));
    };
    auto numerical = finite_diff_gradient(f_scalar, A);

    EXPECT_TRUE(matrices_approx_equal(analytical, numerical, 1e-3, 1e-5));
}

TEST(GradSolve, WrtB) {
    matrix<double> A{{2, 1}, {1, 3}};
    matrix<double> b{{1}, {2}};

    auto f = [&A](const auto& x) {
        val<matrix<double>> Av(A);
        return sum(solve(Av, x));
    };

    auto analytical = grad(f)(b);

    auto f_scalar = [&A](const matrix<double>& b_) {
        return elementa::sum(elementa::solve(A, b_));
    };
    auto numerical = finite_diff_gradient(f_scalar, b);

    EXPECT_TRUE(matrices_approx_equal(analytical, numerical, 1e-3, 1e-5));
}

// =============================================================================
// Composite Function Tests
// =============================================================================

TEST(GradChain, Composition) {
    auto f = [](const auto& x) {
        return sum(exp(2.0 * x));
    };

    matrix<double> input{{0.1, 0.2}, {0.3, 0.4}};
    auto analytical = grad(f)(input);

    auto f_scalar = [](const matrix<double>& x) {
        return elementa::sum(elementa::exp(2.0 * x));
    };
    auto numerical = finite_diff_gradient(f_scalar, input);

    EXPECT_TRUE(matrices_approx_equal(analytical, numerical));
}

TEST(GradQuadratic, QuadraticForm) {
    matrix<double> A{{2, 1}, {1, 3}};

    auto f = [&A](const auto& x) {
        val<matrix<double>> Av(A);
        return sum(matmul(matmul(transpose(x), Av), x));
    };

    matrix<double> x{{1}, {2}};
    auto analytical = grad(f)(x);

    auto f_scalar = [&A](const matrix<double>& x_) {
        return elementa::sum(elementa::matmul(elementa::matmul(elementa::transpose(x_), A), x_));
    };
    auto numerical = finite_diff_gradient(f_scalar, x);

    EXPECT_TRUE(matrices_approx_equal(analytical, numerical, 1e-3, 1e-5));
}

TEST(GradLogSumExp, SumExp) {
    matrix<double> input{{1, 2}, {3, 4}};

    auto f2 = [](const auto& x) {
        return sum(exp(x));
    };

    auto analytical = grad(f2)(input);

    auto f_scalar = [](const matrix<double>& x) {
        return elementa::sum(elementa::exp(x));
    };
    auto numerical = finite_diff_gradient(f_scalar, input);

    EXPECT_TRUE(matrices_approx_equal(analytical, numerical));
}

// =============================================================================
// Gradient Accumulation Tests
// =============================================================================

TEST(GradAccumulation, VariableUsedTwice) {
    auto f = [](const auto& x) {
        return x + x;
    };

    matrix<double> input{{1, 2}};
    auto analytical = grad(f)(input);
    auto expected = ones<double>(1, 2) * 2.0;

    EXPECT_TRUE(matrices_approx_equal(analytical, expected));
}

TEST(GradAccumulation, DiamondPattern) {
    auto f = [](const auto& x) {
        auto y = x + x;
        return y + y;
    };

    matrix<double> input{{1}};
    auto analytical = grad(f)(input);
    auto expected = matrix<double>{{4.0}};

    EXPECT_TRUE(matrices_approx_equal(analytical, expected));
}

// =============================================================================
// Higher-Order Derivative Tests
// =============================================================================

TEST(Hessian, SumSquared) {
    auto f = [](const auto& x) {
        return sum(pow(x, 2.0));
    };

    matrix<double> input{{1, 2}};
    auto H = hessian(f)(input);
    auto expected = eye<double>(2) * 2.0;

    EXPECT_TRUE(matrices_approx_equal(H, expected, 1e-3, 1e-5));
}

TEST(Hessian, Quadratic) {
    auto f2 = [](const auto& x) {
        auto x1 = 3.0 * pow(x, 2.0);
        return sum(x1);
    };

    matrix<double> input{{1}, {2}};
    auto H = hessian(f2)(input);
    auto expected = eye<double>(2) * 6.0;

    EXPECT_TRUE(matrices_approx_equal(H, expected, 1e-3, 1e-5));
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(GradEdge, OneByOne) {
    auto f = [](const auto& x) {
        return det(x);
    };

    matrix<double> input{{5.0}};
    auto analytical = grad(f)(input);
    auto expected = matrix<double>{{1.0}};

    EXPECT_TRUE(matrices_approx_equal(analytical, expected));
}

TEST(GradEdge, Identity) {
    auto f = [](const auto& x) {
        return x;
    };

    matrix<double> input{{1, 2}, {3, 4}};
    auto analytical = grad(f)(input);
    auto expected = ones<double>(2, 2);

    EXPECT_TRUE(matrices_approx_equal(analytical, expected));
}

// =============================================================================
// Practical Use Case Tests
// =============================================================================

TEST(GradPractical, LinearRegression) {
    matrix<double> A{{1, 2}, {3, 4}, {5, 6}};

    auto loss = [&A](const auto& x) {
        val<matrix<double>> Av(A);
        auto Ax = matmul(Av, x);
        return sum(hadamard(Ax, Ax));
    };

    matrix<double> x{{0.1}, {0.2}};
    auto analytical = grad(loss)(x);

    auto loss_scalar = [&A](const matrix<double>& x_) {
        auto Ax = elementa::matmul(A, x_);
        return elementa::sum(elementa::hadamard(Ax, Ax));
    };
    auto numerical = finite_diff_gradient(loss_scalar, x);

    EXPECT_TRUE(matrices_approx_equal(analytical, numerical, 1e-4, 1e-6));
}

TEST(GradPractical, Logdet) {
    matrix<double> Sigma{{2, 0.5}, {0.5, 1}};

    auto nll = [](const auto& S) {
        return logdet(S);
    };

    auto analytical = grad(nll)(Sigma);

    auto nll_scalar = [](const matrix<double>& S) {
        auto [sign, ld] = elementa::logdet(S);
        return ld;
    };
    auto numerical = finite_diff_gradient(nll_scalar, Sigma, 1e-6);

    EXPECT_TRUE(matrices_approx_equal(analytical, numerical, 1e-3, 1e-4));
}
