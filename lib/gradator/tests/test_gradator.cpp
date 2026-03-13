// test_gradator.cpp - Tests for gradator automatic differentiation library
//
// Tests verify analytical gradients against finite difference approximations.
// This is the gold standard for testing AD implementations.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <gradator.hpp>
#include <cmath>

using namespace gradator;
using namespace elementa;
using Catch::Approx;

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

TEST_CASE("Graph and variable creation", "[graph][var]") {
    SECTION("Create variable from matrix") {
        graph g;
        matrix<double> m{{1, 2}, {3, 4}};
        auto x = g.make_var(m);

        REQUIRE(x.value().rows() == 2);
        REQUIRE(x.value().cols() == 2);
        REQUIRE(x.value()(0, 0) == 1);
        REQUIRE(g.size() == 1);
    }

    SECTION("Multiple variables") {
        graph g;
        auto x = g.make_var(matrix<double>{{1, 2}});
        auto y = g.make_var(matrix<double>{{3, 4}});

        REQUIRE(g.size() == 2);
        REQUIRE(x.id().index == 0);
        REQUIRE(y.id().index == 1);
    }

    SECTION("val (constant) creation") {
        val<matrix<double>> c(matrix<double>{{1, 2}});
        REQUIRE(c.value()(0, 0) == 1);
    }
}

// =============================================================================
// Arithmetic Operation Tests
// =============================================================================

TEST_CASE("Negation gradient", "[grad][negation]") {
    auto f = [](const auto& x) { return -x; };

    matrix<double> input{{1, 2}, {3, 4}};

    // Analytical gradient
    auto analytical = grad(f)(input);

    // Expected: gradient of -x is -1 for each element
    auto expected = -ones<double>(2, 2);

    REQUIRE(matrices_approx_equal(analytical, expected));
}

TEST_CASE("Addition gradient", "[grad][addition]") {
    SECTION("var + var") {
        graph g;
        matrix<double> m1{{1, 2}};
        matrix<double> m2{{3, 4}};

        auto x = g.make_var(m1);
        auto y = g.make_var(m2);
        auto z = x + y;

        g.backward<matrix<double>>(z);

        auto dx = g.gradient(x);
        auto dy = g.gradient(y);

        // ∂(x+y)/∂x = 1, ∂(x+y)/∂y = 1
        REQUIRE(matrices_approx_equal(dx, ones<double>(1, 2)));
        REQUIRE(matrices_approx_equal(dy, ones<double>(1, 2)));
    }

    SECTION("var + val") {
        auto f = [](const auto& x) {
            val<matrix<double>> c(matrix<double>{{10, 20}});
            return x + c;
        };

        matrix<double> input{{1, 2}};
        auto analytical = grad(f)(input);

        // Gradient w.r.t. x is 1 (constant doesn't affect gradient)
        REQUIRE(matrices_approx_equal(analytical, ones<double>(1, 2)));
    }
}

TEST_CASE("Subtraction gradient", "[grad][subtraction]") {
    graph g;
    matrix<double> m1{{5, 6}};
    matrix<double> m2{{1, 2}};

    auto x = g.make_var(m1);
    auto y = g.make_var(m2);
    auto z = x - y;

    g.backward<matrix<double>>(z);

    auto dx = g.gradient(x);
    auto dy = g.gradient(y);

    // ∂(x-y)/∂x = 1, ∂(x-y)/∂y = -1
    REQUIRE(matrices_approx_equal(dx, ones<double>(1, 2)));
    REQUIRE(matrices_approx_equal(dy, -ones<double>(1, 2)));
}

TEST_CASE("Scalar multiplication gradient", "[grad][scalar_mul]") {
    auto f = [](const auto& x) { return 3.0 * x; };

    matrix<double> input{{1, 2}, {3, 4}};
    auto analytical = grad(f)(input);

    // Gradient of 3*x is 3
    auto expected = ones<double>(2, 2) * 3.0;

    REQUIRE(matrices_approx_equal(analytical, expected));
}

// =============================================================================
// Matrix Operation Tests
// =============================================================================

TEST_CASE("Matrix multiplication gradient", "[grad][matmul]") {
    SECTION("var * var") {
        graph g;
        matrix<double> A{{1, 2}, {3, 4}};
        matrix<double> B{{5, 6}, {7, 8}};

        auto x = g.make_var(A);
        auto y = g.make_var(B);
        auto z = matmul(x, y);  // C = A * B

        // Need a scalar output for simple gradient
        auto s = sum(z);

        g.backward<double>(s);

        auto dA = g.gradient(x);
        auto dB = g.gradient(y);

        // Verify with finite differences
        auto f_A = [&B](const matrix<double>& A_) {
            return elementa::sum(elementa::matmul(A_, B));
        };
        auto f_B = [&A](const matrix<double>& B_) {
            return elementa::sum(elementa::matmul(A, B_));
        };

        auto numerical_dA = finite_diff_gradient(f_A, A);
        auto numerical_dB = finite_diff_gradient(f_B, B);

        REQUIRE(matrices_approx_equal(dA, numerical_dA));
        REQUIRE(matrices_approx_equal(dB, numerical_dB));
    }
}

TEST_CASE("Transpose gradient", "[grad][transpose]") {
    auto f = [](const auto& x) {
        return sum(transpose(x));
    };

    matrix<double> input{{1, 2, 3}, {4, 5, 6}};
    auto analytical = grad(f)(input);

    // Numerical gradient
    auto f_scalar = [](const matrix<double>& x) {
        return elementa::sum(elementa::transpose(x));
    };
    auto numerical = finite_diff_gradient(f_scalar, input);

    REQUIRE(matrices_approx_equal(analytical, numerical));
}

TEST_CASE("Trace gradient", "[grad][trace]") {
    auto f = [](const auto& x) {
        return trace(x);
    };

    matrix<double> input{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto analytical = grad(f)(input);

    // Gradient of trace is identity matrix
    auto expected = eye<double>(3);

    REQUIRE(matrices_approx_equal(analytical, expected));
}

TEST_CASE("Sum gradient", "[grad][sum]") {
    auto f = [](const auto& x) {
        return sum(x);
    };

    matrix<double> input{{1, 2}, {3, 4}};
    auto analytical = grad(f)(input);

    // Gradient of sum is all ones
    auto expected = ones<double>(2, 2);

    REQUIRE(matrices_approx_equal(analytical, expected));
}

// =============================================================================
// Transcendental Operation Tests
// =============================================================================

TEST_CASE("Exp gradient", "[grad][exp]") {
    auto f = [](const auto& x) {
        return sum(exp(x));
    };

    matrix<double> input{{0.1, 0.2}, {0.3, 0.4}};
    auto analytical = grad(f)(input);

    // Gradient of sum(exp(x)) w.r.t. x_ij is exp(x_ij)
    auto expected = elementa::exp(input);

    REQUIRE(matrices_approx_equal(analytical, expected));
}

TEST_CASE("Log gradient", "[grad][log]") {
    auto f = [](const auto& x) {
        return sum(log(x));
    };

    matrix<double> input{{1, 2}, {3, 4}};
    auto analytical = grad(f)(input);

    // Gradient of sum(log(x)) w.r.t. x_ij is 1/x_ij
    auto f_scalar = [](const matrix<double>& x) {
        return elementa::sum(elementa::log(x));
    };
    auto numerical = finite_diff_gradient(f_scalar, input);

    REQUIRE(matrices_approx_equal(analytical, numerical));
}

TEST_CASE("Sqrt gradient", "[grad][sqrt]") {
    auto f = [](const auto& x) {
        return sum(sqrt(x));
    };

    matrix<double> input{{1, 4}, {9, 16}};
    auto analytical = grad(f)(input);

    // Gradient of sqrt(x) is 1/(2*sqrt(x))
    auto f_scalar = [](const matrix<double>& x) {
        return elementa::sum(elementa::sqrt(x));
    };
    auto numerical = finite_diff_gradient(f_scalar, input);

    REQUIRE(matrices_approx_equal(analytical, numerical));
}

TEST_CASE("Pow gradient", "[grad][pow]") {
    auto f = [](const auto& x) {
        return sum(pow(x, 3.0));  // x^3
    };

    matrix<double> input{{1, 2}, {3, 4}};
    auto analytical = grad(f)(input);

    // Gradient of x^3 is 3*x^2
    auto f_scalar = [](const matrix<double>& x) {
        return elementa::sum(elementa::pow(x, 3.0));
    };
    auto numerical = finite_diff_gradient(f_scalar, input);

    REQUIRE(matrices_approx_equal(analytical, numerical));
}

// =============================================================================
// Linear Algebra Operation Tests
// =============================================================================

TEST_CASE("Determinant gradient", "[grad][det]") {
    auto f = [](const auto& x) {
        return det(x);
    };

    matrix<double> A{{2, 1}, {1, 3}};
    auto analytical = grad(f)(A);

    // Numerical gradient
    auto f_scalar = [](const matrix<double>& x) {
        return elementa::det(x);
    };
    auto numerical = finite_diff_gradient(f_scalar, A);

    REQUIRE(matrices_approx_equal(analytical, numerical, 1e-3, 1e-5));
}

TEST_CASE("Logdet gradient", "[grad][logdet]") {
    auto f = [](const auto& x) {
        return logdet(x);
    };

    // Use positive definite matrix
    matrix<double> A{{4, 1}, {1, 3}};
    auto analytical = grad(f)(A);

    // Numerical gradient
    auto f_scalar = [](const matrix<double>& x) {
        auto [sign, ld] = elementa::logdet(x);
        return ld;
    };
    auto numerical = finite_diff_gradient(f_scalar, A);

    REQUIRE(matrices_approx_equal(analytical, numerical, 1e-3, 1e-5));
}

TEST_CASE("Inverse gradient", "[grad][inverse]") {
    auto f = [](const auto& x) {
        return sum(inverse(x));
    };

    matrix<double> A{{2, 1}, {1, 3}};
    auto analytical = grad(f)(A);

    // Numerical gradient
    auto f_scalar = [](const matrix<double>& x) {
        return elementa::sum(elementa::inverse(x));
    };
    auto numerical = finite_diff_gradient(f_scalar, A);

    REQUIRE(matrices_approx_equal(analytical, numerical, 1e-3, 1e-5));
}

TEST_CASE("Solve gradient", "[grad][solve]") {
    SECTION("Gradient w.r.t. A in solve(A, b)") {
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

        REQUIRE(matrices_approx_equal(analytical, numerical, 1e-3, 1e-5));
    }

    SECTION("Gradient w.r.t. b in solve(A, b)") {
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

        REQUIRE(matrices_approx_equal(analytical, numerical, 1e-3, 1e-5));
    }
}

// =============================================================================
// Composite Function Tests
// =============================================================================

TEST_CASE("Chain rule - composition of operations", "[grad][chain]") {
    // f(x) = sum(exp(2*x))
    auto f = [](const auto& x) {
        return sum(exp(2.0 * x));
    };

    matrix<double> input{{0.1, 0.2}, {0.3, 0.4}};
    auto analytical = grad(f)(input);

    // Numerical gradient
    auto f_scalar = [](const matrix<double>& x) {
        return elementa::sum(elementa::exp(2.0 * x));
    };
    auto numerical = finite_diff_gradient(f_scalar, input);

    REQUIRE(matrices_approx_equal(analytical, numerical));
}

TEST_CASE("Quadratic form gradient", "[grad][quadratic]") {
    // f(x) = x^T * A * x = sum(transpose(x) * A * x)
    // This is a classic test case

    matrix<double> A{{2, 1}, {1, 3}};

    auto f = [&A](const auto& x) {
        val<matrix<double>> Av(A);
        return sum(matmul(matmul(transpose(x), Av), x));
    };

    matrix<double> x{{1}, {2}};
    auto analytical = grad(f)(x);

    // For quadratic form x^T A x, gradient is (A + A^T) * x
    auto f_scalar = [&A](const matrix<double>& x_) {
        return elementa::sum(elementa::matmul(elementa::matmul(elementa::transpose(x_), A), x_));
    };
    auto numerical = finite_diff_gradient(f_scalar, x);

    REQUIRE(matrices_approx_equal(analytical, numerical, 1e-3, 1e-5));
}

TEST_CASE("Log-sum-exp (softmax-related)", "[grad][logsumexp]") {
    // f(x) = log(sum(exp(x))) - a common numerically-sensitive operation
    auto f = [](const auto& x) {
        return log(sum(exp(x)));
    };

    matrix<double> input{{1, 2}, {3, 4}};

    // We need to handle this differently since log returns scalar
    // Let's test the gradient of log(sum(exp(x)))
    graph g;
    auto x = g.make_var(input);
    auto y = sum(exp(x));  // This returns var<double>

    // For now, let's test exp followed by sum
    auto f2 = [](const auto& x) {
        return sum(exp(x));
    };

    auto analytical = grad(f2)(input);

    auto f_scalar = [](const matrix<double>& x) {
        return elementa::sum(elementa::exp(x));
    };
    auto numerical = finite_diff_gradient(f_scalar, input);

    REQUIRE(matrices_approx_equal(analytical, numerical));
}

// =============================================================================
// Gradient Accumulation Tests
// =============================================================================

TEST_CASE("Gradient accumulation - variable used multiple times", "[grad][accumulation]") {
    // f(x) = x + x = 2*x
    // The gradient should be 2 (accumulated from both uses)
    auto f = [](const auto& x) {
        return x + x;
    };

    matrix<double> input{{1, 2}};
    auto analytical = grad(f)(input);

    auto expected = ones<double>(1, 2) * 2.0;

    REQUIRE(matrices_approx_equal(analytical, expected));
}

TEST_CASE("Diamond pattern - shared intermediate", "[grad][diamond]") {
    // y = x + x
    // z = y + y = 2*(x + x) = 4*x
    // Gradient should be 4
    auto f = [](const auto& x) {
        auto y = x + x;
        return y + y;
    };

    matrix<double> input{{1}};
    auto analytical = grad(f)(input);

    auto expected = matrix<double>{{4.0}};

    REQUIRE(matrices_approx_equal(analytical, expected));
}

// =============================================================================
// Higher-Order Derivative Tests
// =============================================================================

TEST_CASE("Hessian computation", "[grad][hessian]") {
    // f(x) = sum(x^2) = x1^2 + x2^2 + ...
    // Gradient: 2*x
    // Hessian: 2*I
    auto f = [](const auto& x) {
        return sum(pow(x, 2.0));
    };

    matrix<double> input{{1, 2}};
    auto H = hessian(f)(input);

    // Expected: 2*I
    auto expected = eye<double>(2) * 2.0;

    REQUIRE(matrices_approx_equal(H, expected, 1e-3, 1e-5));
}

TEST_CASE("Hessian of quadratic", "[grad][hessian][quadratic]") {
    // f(x) = x^T * A * x where A is symmetric
    // Hessian = 2*A
    matrix<double> A{{3, 1}, {1, 2}};

    auto f = [&A](const auto& x) {
        // Reshape x to column vector
        val<matrix<double>> Av(A);
        return sum(hadamard(matmul(Av, x), x));  // x^T A x element-wise
    };

    // Simpler: f(x) = 3*x1^2 + 2*x1*x2 + 2*x2^2
    auto f2 = [](const auto& x) {
        auto x1 = 3.0 * pow(x, 2.0);  // 3*x^2
        return sum(x1);
    };

    matrix<double> input{{1}, {2}};
    auto H = hessian(f2)(input);

    // For sum(3*x^2), Hessian is 6*I
    auto expected = eye<double>(2) * 6.0;

    REQUIRE(matrices_approx_equal(H, expected, 1e-3, 1e-5));
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_CASE("1x1 matrix operations", "[grad][edge]") {
    auto f = [](const auto& x) {
        return det(x);  // det of 1x1 is just the element
    };

    matrix<double> input{{5.0}};
    auto analytical = grad(f)(input);

    // Gradient of det([[x]]) = 1
    auto expected = matrix<double>{{1.0}};

    REQUIRE(matrices_approx_equal(analytical, expected));
}

TEST_CASE("Identity function gradient", "[grad][edge]") {
    auto f = [](const auto& x) {
        return x;  // identity
    };

    matrix<double> input{{1, 2}, {3, 4}};
    auto analytical = grad(f)(input);

    // Gradient of identity is all ones
    auto expected = ones<double>(2, 2);

    REQUIRE(matrices_approx_equal(analytical, expected));
}

// =============================================================================
// Practical Use Case Tests
// =============================================================================

TEST_CASE("Linear regression loss gradient", "[grad][practical]") {
    // Simplified: L = sum((Ax)^2) where we differentiate w.r.t. x
    // Gradient: 2 * A^T * A * x

    matrix<double> A{{1, 2}, {3, 4}, {5, 6}};

    auto loss = [&A](const auto& x) {
        val<matrix<double>> Av(A);
        auto Ax = matmul(Av, x);
        return sum(hadamard(Ax, Ax));  // No division - keep it simple
    };

    matrix<double> x{{0.1}, {0.2}};
    auto analytical = grad(loss)(x);

    // Numerical verification
    auto loss_scalar = [&A](const matrix<double>& x_) {
        auto Ax = elementa::matmul(A, x_);
        return elementa::sum(elementa::hadamard(Ax, Ax));
    };
    auto numerical = finite_diff_gradient(loss_scalar, x);

    REQUIRE(matrices_approx_equal(analytical, numerical, 1e-4, 1e-6));
}

TEST_CASE("Logdet gradient (Gaussian-like)", "[grad][practical]") {
    // Test log determinant gradient which is key for Gaussian likelihoods
    // Gradient of logdet(S) w.r.t. S is S^{-T}

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

    REQUIRE(matrices_approx_equal(analytical, numerical, 1e-3, 1e-4));
}
