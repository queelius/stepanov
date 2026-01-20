/**
 * @file test_finite_diff.cpp
 * @brief Tests for finite difference numerical differentiation
 */

#include "finite_diff.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <numbers>

using namespace finite_diff;

// =============================================================================
// Helper: Known functions with exact derivatives
// =============================================================================

// f(x) = x², f'(x) = 2x, f''(x) = 2
double square(double x) { return x * x; }
double square_deriv(double x) { return 2 * x; }

// f(x) = x³, f'(x) = 3x², f''(x) = 6x
double cube(double x) { return x * x * x; }
double cube_deriv(double x) { return 3 * x * x; }

// f(x) = sin(x), f'(x) = cos(x), f''(x) = -sin(x)
double sine(double x) { return std::sin(x); }
double sine_deriv(double x) { return std::cos(x); }

// f(x) = e^x, f'(x) = e^x, f''(x) = e^x
double exponential(double x) { return std::exp(x); }

// =============================================================================
// First Derivative Tests
// =============================================================================

TEST(FiniteDiffTest, ForwardDifference) {
    double x = 2.0;
    double fd = forward_difference(square, x);
    double exact = square_deriv(x);

    // Forward difference is O(h), expect ~1e-8 accuracy
    EXPECT_NEAR(fd, exact, 1e-6);
}

TEST(FiniteDiffTest, BackwardDifference) {
    double x = 2.0;
    double bd = backward_difference(square, x);
    double exact = square_deriv(x);

    EXPECT_NEAR(bd, exact, 1e-6);
}

TEST(FiniteDiffTest, CentralDifference) {
    double x = 2.0;
    double cd = central_difference(square, x);
    double exact = square_deriv(x);

    // Central difference is O(h²), should be more accurate
    EXPECT_NEAR(cd, exact, 1e-10);
}

TEST(FiniteDiffTest, CentralDifferenceSine) {
    double x = 1.0;
    double cd = central_difference(sine, x);
    double exact = sine_deriv(x);

    EXPECT_NEAR(cd, exact, 1e-10);
}

TEST(FiniteDiffTest, FivePointStencil) {
    double x = 2.0;
    double fp = five_point_stencil(cube, x);
    double exact = cube_deriv(x);

    // Five-point is O(h⁴), very accurate
    EXPECT_NEAR(fp, exact, 1e-10);
}

TEST(FiniteDiffTest, FivePointStencilExp) {
    double x = 1.0;
    double fp = five_point_stencil(exponential, x);
    double exact = std::exp(x);

    EXPECT_NEAR(fp, exact, 1e-8);
}

// =============================================================================
// Second Derivative Tests
// =============================================================================

TEST(FiniteDiffTest, SecondDerivativePolynomial) {
    double x = 3.0;
    double d2 = second_derivative(square, x);
    double exact = 2.0;  // d²/dx² (x²) = 2

    // Second derivatives have ~O(h²) + O(ε/h²) error
    EXPECT_NEAR(d2, exact, 1e-4);
}

TEST(FiniteDiffTest, SecondDerivativeCubic) {
    double x = 2.0;
    double d2 = second_derivative(cube, x);
    double exact = 6.0 * x;  // d²/dx² (x³) = 6x

    EXPECT_NEAR(d2, exact, 1e-4);
}

TEST(FiniteDiffTest, SecondDerivativeSine) {
    double x = std::numbers::pi / 4;
    double d2 = second_derivative(sine, x);
    double exact = -std::sin(x);  // d²/dx² sin(x) = -sin(x)

    EXPECT_NEAR(d2, exact, 1e-5);
}

TEST(FiniteDiffTest, SecondDerivativeFivePoint) {
    double x = 1.5;
    double d2 = second_derivative_five_point(cube, x);
    double exact = 6.0 * x;

    EXPECT_NEAR(d2, exact, 1e-6);
}

// =============================================================================
// Richardson Extrapolation Tests
// =============================================================================

TEST(FiniteDiffTest, RichardsonExtrapolation) {
    double x = 2.0;
    double h = 0.01;

    // Regular central difference
    double cd = central_difference(cube, x, h);

    // Richardson extrapolation should be more accurate
    double re = richardson_extrapolation(cube, x, h, 2);
    double exact = cube_deriv(x);

    // Richardson should reduce error significantly
    double cd_err = std::abs(cd - exact);
    double re_err = std::abs(re - exact);

    EXPECT_LT(re_err, cd_err);
    EXPECT_NEAR(re, exact, 1e-10);
}

// =============================================================================
// Gradient Tests
// =============================================================================

TEST(FiniteDiffTest, GradientQuadratic) {
    // f(x,y) = x² + y²
    // ∇f = [2x, 2y]
    auto f = [](const std::vector<double>& x) {
        return x[0]*x[0] + x[1]*x[1];
    };

    std::vector<double> point = {3.0, 4.0};
    auto grad = gradient(f, point);

    ASSERT_EQ(grad.size(), 2);
    EXPECT_NEAR(grad[0], 6.0, 1e-8);  // 2*3
    EXPECT_NEAR(grad[1], 8.0, 1e-8);  // 2*4
}

TEST(FiniteDiffTest, GradientRosenbrock) {
    // Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²
    // ∂f/∂x = -2(1-x) - 400x(y-x²)
    // ∂f/∂y = 200(y-x²)
    auto f = [](const std::vector<double>& v) {
        double x = v[0], y = v[1];
        return (1-x)*(1-x) + 100*(y-x*x)*(y-x*x);
    };

    std::vector<double> point = {1.0, 1.0};  // At the minimum
    auto grad = gradient(f, point);

    // At (1,1), gradient should be [0, 0]
    ASSERT_EQ(grad.size(), 2);
    EXPECT_NEAR(grad[0], 0.0, 1e-6);
    EXPECT_NEAR(grad[1], 0.0, 1e-6);
}

// =============================================================================
// Directional Derivative Tests
// =============================================================================

TEST(FiniteDiffTest, DirectionalDerivative) {
    // f(x,y) = x² + y²
    auto f = [](const std::vector<double>& x) {
        return x[0]*x[0] + x[1]*x[1];
    };

    std::vector<double> point = {1.0, 0.0};
    std::vector<double> direction = {1.0, 0.0};  // Unit vector in x direction

    double dd = directional_derivative(f, point, direction);

    // ∇f = [2x, 2y] = [2, 0] at (1,0)
    // D_v f = ∇f · v = [2,0] · [1,0] = 2
    EXPECT_NEAR(dd, 2.0, 1e-8);
}

// =============================================================================
// Hessian Tests
// =============================================================================

TEST(FiniteDiffTest, HessianQuadratic) {
    // f(x,y) = x² + 2xy + 3y²
    // H = [[2, 2], [2, 6]]
    auto f = [](const std::vector<double>& v) {
        double x = v[0], y = v[1];
        return x*x + 2*x*y + 3*y*y;
    };

    std::vector<double> point = {1.0, 2.0};
    auto H = hessian(f, point);

    ASSERT_EQ(H.size(), 2);
    ASSERT_EQ(H[0].size(), 2);

    // Hessian uses second derivatives, which have lower accuracy
    EXPECT_NEAR(H[0][0], 2.0, 1e-4);  // ∂²f/∂x²
    EXPECT_NEAR(H[0][1], 2.0, 1e-4);  // ∂²f/∂x∂y
    EXPECT_NEAR(H[1][0], 2.0, 1e-4);  // ∂²f/∂y∂x (symmetry)
    EXPECT_NEAR(H[1][1], 6.0, 1e-4);  // ∂²f/∂y²
}

// =============================================================================
// Jacobian Tests
// =============================================================================

TEST(FiniteDiffTest, JacobianLinear) {
    // f: R² → R²
    // f(x,y) = [2x + y, x - y]
    // J = [[2, 1], [1, -1]]
    std::function<std::vector<double>(const std::vector<double>&)> f =
        [](const std::vector<double>& v) -> std::vector<double> {
            return {2*v[0] + v[1], v[0] - v[1]};
        };

    std::vector<double> point = {1.0, 1.0};
    auto J = jacobian(f, point);

    ASSERT_EQ(J.size(), 2);
    ASSERT_EQ(J[0].size(), 2);

    EXPECT_NEAR(J[0][0], 2.0, 1e-8);   // ∂f₁/∂x
    EXPECT_NEAR(J[0][1], 1.0, 1e-8);   // ∂f₁/∂y
    EXPECT_NEAR(J[1][0], 1.0, 1e-8);   // ∂f₂/∂x
    EXPECT_NEAR(J[1][1], -1.0, 1e-8);  // ∂f₂/∂y
}

TEST(FiniteDiffTest, JacobianNonlinear) {
    // f: R² → R²
    // f(x,y) = [x*y, x²]
    // J = [[y, x], [2x, 0]]
    std::function<std::vector<double>(const std::vector<double>&)> f =
        [](const std::vector<double>& v) -> std::vector<double> {
            return {v[0]*v[1], v[0]*v[0]};
        };

    std::vector<double> point = {2.0, 3.0};
    auto J = jacobian(f, point);

    EXPECT_NEAR(J[0][0], 3.0, 1e-8);   // ∂(xy)/∂x = y = 3
    EXPECT_NEAR(J[0][1], 2.0, 1e-8);   // ∂(xy)/∂y = x = 2
    EXPECT_NEAR(J[1][0], 4.0, 1e-8);   // ∂(x²)/∂x = 2x = 4
    EXPECT_NEAR(J[1][1], 0.0, 1e-8);   // ∂(x²)/∂y = 0
}

// =============================================================================
// Error Estimation Tests
// =============================================================================

TEST(FiniteDiffTest, ErrorEstimation) {
    double x = 1.5;
    auto result = central_with_error(sine, x);

    double exact = std::cos(x);

    // Check value is close
    EXPECT_NEAR(result.value, exact, 1e-10);

    // Error estimate should be non-negative
    EXPECT_GE(result.error, 0.0);

    // Actual error should be small
    double actual_error = std::abs(result.value - exact);
    EXPECT_LT(actual_error, 1e-10);
}

// =============================================================================
// Comparison Tests
// =============================================================================

TEST(FiniteDiffTest, CompareWithExact) {
    double x = 2.0;
    auto cmp = compare(square, x, square_deriv(x));

    EXPECT_TRUE(cmp.agrees(1e-10));
    EXPECT_LT(cmp.absolute_error, 1e-10);
}

// =============================================================================
// Optimal Step Size Tests
// =============================================================================

TEST(FiniteDiffTest, OptimalStepSizes) {
    // Verify optimal step sizes are reasonable
    EXPECT_GT(optimal_h<double>::forward, 1e-10);
    EXPECT_LT(optimal_h<double>::forward, 1e-6);

    EXPECT_GT(optimal_h<double>::central, 1e-8);
    EXPECT_LT(optimal_h<double>::central, 1e-4);

    EXPECT_GT(optimal_h<double>::five_point, 1e-5);
    EXPECT_LT(optimal_h<double>::five_point, 1e-2);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(FiniteDiffTest, DerivativeAtZero) {
    double x = 0.0;
    double cd = central_difference(square, x);
    EXPECT_NEAR(cd, 0.0, 1e-10);  // f'(0) = 0 for f(x) = x²
}

TEST(FiniteDiffTest, LambdaFunction) {
    auto f = [](double x) { return x * x * x - 2 * x; };
    // f'(x) = 3x² - 2

    double x = 1.5;
    double cd = central_difference(f, x);
    double exact = 3 * x * x - 2;

    EXPECT_NEAR(cd, exact, 1e-10);
}
