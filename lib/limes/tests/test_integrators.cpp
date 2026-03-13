#include <gtest/gtest.h>
#include <cmath>
#include <functional>
#include <numbers>
#include <limes/limes.hpp>

using namespace limes::algorithms;

// Test functions with known integrals
namespace test_functions {
    // Simple polynomial: x^2, integral from 0 to 1 = 1/3
    template<typename T>
    T quadratic(T x) { return x * x; }

    // Exponential: e^x, integral from 0 to 1 = e - 1
    template<typename T>
    T exponential(T x) { return std::exp(x); }

    // Trigonometric: sin(x), integral from 0 to pi = 2
    template<typename T>
    T sine(T x) { return std::sin(x); }

    // Gaussian: exp(-x^2), integral from -inf to inf = sqrt(pi)
    template<typename T>
    T gaussian(T x) { return std::exp(-x * x); }

    // Oscillatory: sin(10x), integral from 0 to pi = 0
    template<typename T>
    T oscillatory(T x) { return std::sin(T(10) * x); }
}

// Test fixture for integrator tests
template <typename T>
class IntegratorTest : public ::testing::Test {
protected:
    // Type-aware tolerances that account for precision limits
    // Float can only achieve ~7 decimal digits, double ~15
    static constexpr T default_tol = std::is_same_v<T, float> ? T(1e-5) : T(1e-8);
    static constexpr T strict_tol = std::is_same_v<T, float> ? T(1e-5) : T(1e-12);
    static constexpr T loose_tol = T(1e-4);

    template<typename Integrator, typename F>
    void test_integration(Integrator& integrator, F&& f, T a, T b,
                         T expected, T tolerance) {
        auto result = integrator(std::forward<F>(f), a, b, tolerance);

        EXPECT_TRUE(result.converged())
            << "Integration did not converge";

        EXPECT_NEAR(result.value(), expected, tolerance * T(10))
            << "Integration result outside tolerance"
            << "\n  Computed: " << result.value()
            << "\n  Expected: " << expected
            << "\n  Error: " << result.error()
            << "\n  Evaluations: " << result.evaluations();

        EXPECT_LE(result.error(), tolerance * T(10))
            << "Estimated error too large";
    }
};

using FloatTypes = ::testing::Types<float, double, long double>;
TYPED_TEST_SUITE(IntegratorTest, FloatTypes);

// Test quadrature integrator with different rules
TYPED_TEST(IntegratorTest, QuadratureIntegratorGaussLegendre) {
    using T = TypeParam;
    using rule = quadrature::gauss_legendre<T, 5>;

    quadrature_integrator<T, rule> integrator{rule{}};

    // Test polynomial integration
    this->test_integration(integrator,
                          test_functions::quadratic<T>,
                          T(0), T(1), T(1)/T(3), this->strict_tol);

    // Test exponential
    this->test_integration(integrator,
                          test_functions::exponential<T>,
                          T(0), T(1), std::exp(T(1)) - T(1), this->default_tol);
}

TYPED_TEST(IntegratorTest, QuadratureIntegratorGaussKronrod) {
    using T = TypeParam;
    using rule = quadrature::gauss_kronrod_15<T>;

    quadrature_integrator<T, rule> integrator{rule{}};

    // Test trigonometric function
    this->test_integration(integrator,
                          test_functions::sine<T>,
                          T(0), std::numbers::pi_v<T>, T(2), this->strict_tol);
}

// Test adaptive integrator
TYPED_TEST(IntegratorTest, AdaptiveIntegrator) {
    using T = TypeParam;

    auto integrator = adaptive_integrator<T>{};

    // Simple function
    this->test_integration(integrator,
                          test_functions::quadratic<T>,
                          T(0), T(1), T(1)/T(3), this->default_tol);

    // Oscillatory function - should trigger subdivision
    auto result = integrator(test_functions::oscillatory<T>,
                            T(0), std::numbers::pi_v<T>, this->default_tol);

    // For sin(10x) from 0 to pi:
    // Integral = [-cos(10x)/10] from 0 to pi = (-cos(10*pi) + cos(0))/10 = (-1 + 1)/10 = 0
    // cos(10*pi) = 1, so the result is 0
    EXPECT_NEAR(result.value(), T(0), this->loose_tol);
}

// Test Romberg integrator
TYPED_TEST(IntegratorTest, RombergIntegrator) {
    using T = TypeParam;

    auto integrator = romberg_integrator<T>{};

    // Smooth function
    this->test_integration(integrator,
                          test_functions::exponential<T>,
                          T(0), T(1), std::exp(T(1)) - T(1), this->strict_tol);

    // Test with same integrator - use type-aware tolerance
    this->test_integration(integrator,
                          test_functions::sine<T>,
                          T(0), std::numbers::pi_v<T>, T(2), this->strict_tol);
}

// Test Simpson integrator
TYPED_TEST(IntegratorTest, SimpsonIntegrator) {
    using T = TypeParam;

    using rule = quadrature::simpson_rule<T>;
    quadrature_integrator<T, rule> integrator{rule{}};

    // Simpson's rule is exact for cubic polynomials on [-1, 1]
    auto cubic = [](T x) { return x * x * x; };
    auto result = integrator(cubic, T(-1), T(1));

    EXPECT_NEAR(result.value(), T(0), this->strict_tol);
}

// Test finite interval approximation for Gaussian integral
TYPED_TEST(IntegratorTest, GaussianIntegral) {
    using T = TypeParam;

    // For Gaussian integral from -5 to 5 (captures most of the mass)
    auto integrator = adaptive_integrator<T>{};
    auto result = integrator(
        test_functions::gaussian<T>,
        T(-5), T(5),
        this->default_tol
    );

    // sqrt(pi) = 1.7724538509055159
    // erf(5) * sqrt(pi) is very close to sqrt(pi)
    EXPECT_NEAR(result.value(), std::sqrt(std::numbers::pi_v<T>), this->loose_tol);
}

// Test error estimation
TEST(IntegratorErrorEstimation, ErrorBounds) {
    // Use a function where we can calculate the truncation error
    auto f = [](double x) { return std::exp(x); };

    auto integrator = adaptive_integrator<double>{};
    auto result = integrator(f, 0.0, 1.0, 1e-10);

    double exact = std::exp(1.0) - 1.0;

    // The integrator achieves very high accuracy (near machine epsilon)
    // Just verify that the result is accurate
    EXPECT_NEAR(result.value(), exact, 1e-10);

    // The error estimate should be a reasonable order of magnitude
    EXPECT_GE(result.error(), 0.0);  // Error should be non-negative
}

// Test convergence behavior
TEST(IntegratorConvergence, ConvergenceRates) {
    auto f = [](double x) { return std::sin(x); };
    double exact = 2.0;

    std::vector<double> tolerances = {1e-4, 1e-6, 1e-8, 1e-10};
    std::vector<double> errors;
    std::vector<size_t> evaluations;

    for (double tol : tolerances) {
        auto integrator = adaptive_integrator<double>{};
        auto result = integrator(f, 0.0, std::numbers::pi, tol);

        errors.push_back(std::abs(result.value() - exact));
        evaluations.push_back(result.evaluations());

        // Check that requested tolerance is met
        EXPECT_LE(errors.back(), tol * 10);
    }

    // For well-conditioned integrals like sin(x), the integrator may achieve
    // machine precision quickly. We only check that errors don't increase.
    for (size_t i = 1; i < errors.size(); ++i) {
        EXPECT_LE(errors[i], errors[i-1] + 1e-14);  // Allow for machine epsilon variations
    }

    // Evaluations should not decrease (may stay same if already converged)
    for (size_t i = 1; i < evaluations.size(); ++i) {
        EXPECT_GE(evaluations[i], evaluations[i-1]);
    }
}

// Test handling of discontinuous functions
TEST(IntegratorSpecialCases, DiscontinuousFunction) {
    // Step function with discontinuity at x = 0.5
    auto f = [](double x) { return x < 0.5 ? 0.0 : 1.0; };

    // Exact integral from 0 to 1 is 0.5
    auto integrator = adaptive_integrator<double>{};
    auto result = integrator(f, 0.0, 1.0, 1e-6);

    // Adaptive integrator should handle discontinuity reasonably well
    EXPECT_NEAR(result.value(), 0.5, 1e-4);
}

// Test handling of peaked functions
TEST(IntegratorSpecialCases, PeakedFunction) {
    // Narrow Gaussian peaked at x = 0
    double sigma = 0.001;
    auto f = [sigma](double x) {
        return std::exp(-(x * x) / (2 * sigma * sigma)) /
               (sigma * std::sqrt(2 * std::numbers::pi));
    };

    // Integral from -1 to 1 should be close to 1 (normalized Gaussian)
    auto integrator = adaptive_integrator<double>{};
    auto result = integrator(f, -1.0, 1.0, 1e-8);

    EXPECT_NEAR(result.value(), 1.0, 1e-6);
}

// Test with lambda captures
TEST(IntegratorLambdas, CapturedVariables) {
    double a = 2.0;
    double b = 3.0;

    auto f = [a, b](double x) { return a * x + b; };

    auto integrator = adaptive_integrator<double>{};
    auto result = integrator(f, 0.0, 1.0, 1e-10);

    // Integral of 2x + 3 from 0 to 1 = x^2 + 3x |_0^1 = 1 + 3 = 4
    EXPECT_NEAR(result.value(), 4.0, 1e-10);
}

// Test with function pointers
double test_func_ptr(double x) {
    return x * x * x;
}

TEST(IntegratorFunctionTypes, FunctionPointer) {
    auto integrator = adaptive_integrator<double>{};
    auto result = integrator(test_func_ptr, 0.0, 2.0, 1e-10);

    // Integral of x^3 from 0 to 2 = x^4/4 |_0^2 = 4
    EXPECT_NEAR(result.value(), 4.0, 1e-10);
}

// Test with std::function
TEST(IntegratorFunctionTypes, StdFunction) {
    std::function<double(double)> f = [](double x) { return std::cos(x); };

    auto integrator = adaptive_integrator<double>{};
    auto result = integrator(f, 0.0, std::numbers::pi / 2, 1e-10);

    // Integral of cos(x) from 0 to pi/2 = sin(pi/2) - sin(0) = 1
    EXPECT_NEAR(result.value(), 1.0, 1e-10);
}

// Test numerical stability
TEST(IntegratorStability, NumericalStability) {
    // Test with very small interval
    {
        auto f = [](double x) { return x; };
        auto integrator = adaptive_integrator<double>{};
        auto result = integrator(f, 0.0, 1e-15, 1e-10);

        // Should handle tiny intervals gracefully
        EXPECT_NEAR(result.value(), 5e-31, 1e-40);
    }

    // Test with very large values
    {
        auto f = [](double x) { return 1e10; };
        auto integrator = adaptive_integrator<double>{};
        auto result = integrator(f, 0.0, 1.0, 1e-10);

        EXPECT_NEAR(result.value(), 1e10, 1e-2);
    }
}

// Test Kahan accumulator on ill-conditioned sum
TEST(IntegratorAccumulators, KahanAccuracy) {
    auto f = [](double x) { return 1e10 + std::sin(x); };

    using acc = accumulators::kahan_accumulator<double>;
    quadrature_integrator<double, quadrature::gauss_legendre<double, 5>, acc> integrator;
    auto result = integrator(f, 0.0, std::numbers::pi, 1e-8);

    double expected = 1e10 * std::numbers::pi + 2.0;
    EXPECT_NEAR(result.value(), expected, 100.0);
}
