/**
 * @file test_integrate.cpp
 * @brief Tests for numerical integration (quadrature)
 */

#include "integrate.hpp"
#include <dual/dual.hpp>  // For dual number tests
#include <gtest/gtest.h>
#include <cmath>
#include <numbers>

using namespace integration;

// =============================================================================
// Test Functions with Known Integrals
// =============================================================================

// ∫[0,1] x dx = 1/2
double linear(double x) { return x; }

// ∫[0,1] x² dx = 1/3
double quadratic(double x) { return x * x; }

// ∫[0,1] x³ dx = 1/4
double cubic(double x) { return x * x * x; }

// ∫[0,π] sin(x) dx = 2
double sine(double x) { return std::sin(x); }

// ∫[0,1] e^x dx = e - 1
double exponential(double x) { return std::exp(x); }

// ∫[0,1] 1/(1+x²) dx = π/4
double arctan_deriv(double x) { return 1.0 / (1.0 + x*x); }

// =============================================================================
// Basic Rule Tests
// =============================================================================

TEST(IntegrationTest, MidpointRuleLinear) {
    // Midpoint is exact for linear functions
    double result = midpoint_rule(linear, 0.0, 1.0);
    EXPECT_NEAR(result, 0.5, 1e-14);
}

TEST(IntegrationTest, TrapezoidalRuleLinear) {
    // Trapezoidal is exact for linear functions
    double result = trapezoidal_rule(linear, 0.0, 1.0);
    EXPECT_NEAR(result, 0.5, 1e-14);
}

TEST(IntegrationTest, SimpsonsRuleQuadratic) {
    // Simpson's is exact for quadratics
    double result = simpsons_rule(quadratic, 0.0, 1.0);
    EXPECT_NEAR(result, 1.0/3.0, 1e-14);
}

TEST(IntegrationTest, SimpsonsRuleCubic) {
    // Simpson's is also exact for cubics (bonus degree)
    double result = simpsons_rule(cubic, 0.0, 1.0);
    EXPECT_NEAR(result, 0.25, 1e-14);
}

// =============================================================================
// Composite Rule Tests
// =============================================================================

TEST(IntegrationTest, CompositeMidpointSine) {
    double result = composite_midpoint(sine, 0.0, std::numbers::pi, 100);
    EXPECT_NEAR(result, 2.0, 1e-4);
}

TEST(IntegrationTest, CompositeTrapezoidalSine) {
    double result = composite_trapezoidal(sine, 0.0, std::numbers::pi, 100);
    EXPECT_NEAR(result, 2.0, 1e-3);  // O(h²) accuracy
}

TEST(IntegrationTest, CompositeSimpsonsSine) {
    double result = composite_simpsons(sine, 0.0, std::numbers::pi, 100);
    EXPECT_NEAR(result, 2.0, 1e-7);  // O(h⁴) accuracy
}

TEST(IntegrationTest, CompositeSimpsonsExponential) {
    double result = composite_simpsons(exponential, 0.0, 1.0, 100);
    EXPECT_NEAR(result, std::exp(1.0) - 1.0, 1e-10);
}

TEST(IntegrationTest, CompositeSimpsonsOddNThrows) {
    EXPECT_THROW(composite_simpsons(sine, 0.0, 1.0, 101), std::invalid_argument);
}

// =============================================================================
// Gauss-Legendre Tests
// =============================================================================

TEST(IntegrationTest, GaussLegendre2Quadratic) {
    // 2-point Gauss-Legendre is exact for degree ≤ 3
    double result = gauss_legendre<2>(quadratic, 0.0, 1.0);
    EXPECT_NEAR(result, 1.0/3.0, 1e-14);
}

TEST(IntegrationTest, GaussLegendre3Quartic) {
    // 3-point Gauss-Legendre is exact for degree ≤ 5
    auto quartic = [](double x) { return x*x*x*x; };  // ∫[0,1] x⁴ dx = 1/5
    double result = gauss_legendre<3>(quartic, 0.0, 1.0);
    EXPECT_NEAR(result, 0.2, 1e-14);
}

TEST(IntegrationTest, GaussLegendre5Sine) {
    double result = gauss_legendre<5>(sine, 0.0, std::numbers::pi);
    EXPECT_NEAR(result, 2.0, 1e-6);  // 5-point rule on non-polynomial
}

TEST(IntegrationTest, GaussLegendre5Exponential) {
    double result = gauss_legendre<5>(exponential, 0.0, 1.0);
    EXPECT_NEAR(result, std::exp(1.0) - 1.0, 1e-10);
}

// =============================================================================
// Adaptive Integration Tests
// =============================================================================

TEST(IntegrationTest, AdaptiveSimpsonsSmooth) {
    double result = adaptive_simpsons(exponential, 0.0, 1.0);
    EXPECT_NEAR(result, std::exp(1.0) - 1.0, 1e-10);
}

TEST(IntegrationTest, AdaptiveSimpsonsSine) {
    double result = adaptive_simpsons(sine, 0.0, std::numbers::pi);
    EXPECT_NEAR(result, 2.0, 1e-10);
}

TEST(IntegrationTest, AdaptiveSimpsonsArctan) {
    double result = adaptive_simpsons(arctan_deriv, 0.0, 1.0);
    EXPECT_NEAR(result, std::numbers::pi / 4.0, 1e-10);
}

TEST(IntegrationTest, AdaptiveSimpsonsOscillatory) {
    // ∫[0,2π] sin(x) dx = 0 (cancellation)
    auto f = [](double x) { return std::sin(x); };
    double result = adaptive_simpsons(f, 0.0, 2.0 * std::numbers::pi);
    EXPECT_NEAR(result, 0.0, 1e-10);
}

// =============================================================================
// Convenience Interface Tests
// =============================================================================

TEST(IntegrationTest, IntegrateSine) {
    double result = integrate(sine, 0.0, std::numbers::pi);
    EXPECT_NEAR(result, 2.0, 1e-10);
}

TEST(IntegrationTest, IntegrateWithError) {
    auto result = integrate_with_error(sine, 0.0, std::numbers::pi);
    EXPECT_NEAR(result.value, 2.0, 1e-10);
    EXPECT_LT(result.error_estimate, 1e-6);
}

// =============================================================================
// Special Integrand Tests
// =============================================================================

TEST(IntegrationTest, LeftSingularity) {
    // ∫[0,1] 1/√x dx = 2
    auto f = [](double x) { return 1.0 / std::sqrt(x); };
    double result = integrate_left_singularity(f, 0.0, 1.0);
    // Slightly lower accuracy due to skipping small region near singularity
    EXPECT_NEAR(result, 2.0, 1e-4);
}

TEST(IntegrationTest, SemiInfinite) {
    // ∫[0,∞) e^{-x} dx = 1
    auto f = [](double x) { return std::exp(-x); };
    double result = integrate_semi_infinite(f, 1e-8);
    EXPECT_NEAR(result, 1.0, 1e-5);
}

TEST(IntegrationTest, SemiInfiniteGaussian) {
    // ∫[0,∞) e^{-x²} dx = √π/2
    auto f = [](double x) { return std::exp(-x*x); };
    double result = integrate_semi_infinite(f, 1e-8);
    EXPECT_NEAR(result, std::sqrt(std::numbers::pi) / 2.0, 1e-4);
}

// =============================================================================
// Convergence Tests
// =============================================================================

TEST(IntegrationTest, TrapezoidalConvergence) {
    // Error should decrease as O(h²)
    double exact = 2.0;
    double err_10 = std::abs(composite_trapezoidal(sine, 0.0, std::numbers::pi, 10) - exact);
    double err_20 = std::abs(composite_trapezoidal(sine, 0.0, std::numbers::pi, 20) - exact);

    // Doubling n should reduce error by factor of ~4 (since O(h²))
    double ratio = err_10 / err_20;
    EXPECT_GT(ratio, 3.5);
    EXPECT_LT(ratio, 4.5);
}

TEST(IntegrationTest, SimpsonsConvergence) {
    // Error should decrease as O(h⁴)
    double exact = 2.0;
    double err_10 = std::abs(composite_simpsons(sine, 0.0, std::numbers::pi, 10) - exact);
    double err_20 = std::abs(composite_simpsons(sine, 0.0, std::numbers::pi, 20) - exact);

    // Doubling n should reduce error by factor of ~16 (since O(h⁴))
    double ratio = err_10 / err_20;
    EXPECT_GT(ratio, 14.0);
    EXPECT_LT(ratio, 18.0);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(IntegrationTest, ZeroLengthInterval) {
    double result = integrate(sine, 1.0, 1.0);
    EXPECT_NEAR(result, 0.0, 1e-14);
}

TEST(IntegrationTest, NegativeInterval) {
    // ∫[b,a] f dx = -∫[a,b] f dx
    double forward = integrate(sine, 0.0, std::numbers::pi);
    double backward = integrate(sine, std::numbers::pi, 0.0);
    EXPECT_NEAR(forward + backward, 0.0, 1e-10);
}

TEST(IntegrationTest, LambdaFunction) {
    auto f = [](double x) { return x*x - 2*x + 1; };  // (x-1)²
    // ∫[0,2] (x-1)² dx = 2/3
    double result = integrate(f, 0.0, 2.0);
    EXPECT_NEAR(result, 2.0/3.0, 1e-10);
}

// =============================================================================
// Generic Type Tests (Stepanov-style)
// =============================================================================

// These tests demonstrate the power of generic programming: the same
// integration algorithms work with dual<double> for automatic differentiation,
// enabling computation of derivatives of integrals.

TEST(GenericIntegrationTest, DualMidpointRule) {
    // Test midpoint_rule with dual numbers
    // f(x) = x² → ∫[0,1] x² dx = 1/3
    using D = dual::dual<double>;

    auto f = [](D x) { return x * x; };
    D result = midpoint_rule(f, D(0.0), D(1.0));

    // Midpoint rule: (b-a) * f((a+b)/2) = 1 * (0.5)² = 0.25
    EXPECT_NEAR(result.value(), 0.25, 1e-14);
}

TEST(GenericIntegrationTest, DualTrapezoidalRule) {
    // Test trapezoidal_rule with dual numbers
    // f(x) = x → ∫[0,1] x dx = 1/2
    using D = dual::dual<double>;

    auto f = [](D x) { return x; };
    D result = trapezoidal_rule(f, D(0.0), D(1.0));

    // Trapezoidal is exact for linear: 0.5
    EXPECT_NEAR(result.value(), 0.5, 1e-14);
}

TEST(GenericIntegrationTest, DualSimpsonsRule) {
    // Test Simpson's rule with dual numbers
    // f(x) = x³ → ∫[0,1] x³ dx = 1/4
    using D = dual::dual<double>;

    auto f = [](D x) { return x * x * x; };
    D result = simpsons_rule(f, D(0.0), D(1.0));

    // Simpson's is exact for cubics
    EXPECT_NEAR(result.value(), 0.25, 1e-14);
}

TEST(GenericIntegrationTest, DualCompositeMidpoint) {
    // Composite midpoint with dual numbers
    using D = dual::dual<double>;

    auto f = [](D x) { return x * x; };  // x²
    D result = composite_midpoint(f, D(0.0), D(1.0), 100);

    // ∫[0,1] x² dx = 1/3
    EXPECT_NEAR(result.value(), 1.0/3.0, 1e-4);
}

TEST(GenericIntegrationTest, DualCompositeTrapezoidal) {
    // Composite trapezoidal with dual numbers
    using D = dual::dual<double>;

    auto f = [](D x) { return x * x; };
    D result = composite_trapezoidal(f, D(0.0), D(1.0), 100);

    EXPECT_NEAR(result.value(), 1.0/3.0, 1e-3);
}

TEST(GenericIntegrationTest, DualCompositeSimpson) {
    // Composite Simpson's with dual numbers
    using D = dual::dual<double>;

    auto f = [](D x) { return x * x; };
    D result = composite_simpsons(f, D(0.0), D(1.0), 100);

    EXPECT_NEAR(result.value(), 1.0/3.0, 1e-10);
}

TEST(GenericIntegrationTest, DerivativeOfIntegralViaParameter) {
    // BEAUTIFUL APPLICATION: Differentiation under the integral sign!
    //
    // Compute ∂/∂a ∫[0,1] a·x² dx using dual numbers.
    // By Leibniz rule: ∂/∂a ∫[0,1] a·x² dx = ∫[0,1] x² dx = 1/3
    //
    // We'll integrate f(x) = a·x² where a = dual::variable(1.0),
    // and the derivative() of the result gives us ∂/∂a of the integral.

    using D = dual::dual<double>;

    // Parameter 'a' as a variable for differentiation
    D a = D::variable(1.0);

    // f(x) = a·x² (a is the parameter we're differentiating w.r.t.)
    auto f = [&a](D x) { return a * x * x; };

    // Integrate - since a carries derivative=1, the result's derivative
    // is ∂/∂a ∫ a·x² dx
    D result = composite_simpsons(f, D::constant(0.0), D::constant(1.0), 100);

    // Value: ∫[0,1] 1·x² dx = 1/3
    EXPECT_NEAR(result.value(), 1.0/3.0, 1e-10);

    // Derivative: ∂/∂a ∫[0,1] a·x² dx = ∫[0,1] x² dx = 1/3
    EXPECT_NEAR(result.derivative(), 1.0/3.0, 1e-10);
}

TEST(GenericIntegrationTest, DerivativeOfIntegralNonlinear) {
    // More complex example: ∂/∂a ∫[0,1] sin(a·x) dx
    //
    // Analytical: ∫ sin(ax) dx = -cos(ax)/a + C
    //             ∫[0,1] sin(ax) dx = (1 - cos(a))/a
    // At a=1:     (1 - cos(1))/1 = 1 - cos(1) ≈ 0.4597
    //
    // Derivative w.r.t. a: d/da[(1-cos(a))/a] = (a·sin(a) - 1 + cos(a))/a²
    // At a=1: (sin(1) - 1 + cos(1))/1 ≈ 0.3818

    using D = dual::dual<double>;

    D a = D::variable(1.0);

    auto f = [&a](D x) {
        // sin(a*x) using dual sin
        return dual::sin(a * x);
    };

    D result = composite_simpsons(f, D::constant(0.0), D::constant(1.0), 200);

    // Value: (1 - cos(1))/1 ≈ 0.4597
    double expected_value = 1.0 - std::cos(1.0);
    EXPECT_NEAR(result.value(), expected_value, 1e-6);

    // Derivative: (sin(1) - 1 + cos(1))/1 ≈ 0.3818
    double expected_deriv = std::sin(1.0) - 1.0 + std::cos(1.0);
    EXPECT_NEAR(result.derivative(), expected_deriv, 1e-4);
}

TEST(GenericIntegrationTest, AdaptiveWithDual) {
    // Test adaptive Simpson's with dual numbers
    using D = dual::dual<double>;

    auto f = [](D x) { return x * x; };

    // Need to provide explicit tolerance since default is only for floating-point
    D tol = D::constant(1e-8);
    D result = adaptive_simpsons(f, D::constant(0.0), D::constant(1.0), tol);

    EXPECT_NEAR(result.value(), 1.0/3.0, 1e-7);
}

TEST(GenericIntegrationTest, IntegrateWithDual) {
    // Test the integrate() convenience function with dual numbers
    using D = dual::dual<double>;

    auto f = [](D x) { return x * x * x; };  // x³

    D tol = D::constant(1e-8);
    D result = integrate(f, D::constant(0.0), D::constant(1.0), tol);

    // ∫[0,1] x³ dx = 1/4
    EXPECT_NEAR(result.value(), 0.25, 1e-7);
}
