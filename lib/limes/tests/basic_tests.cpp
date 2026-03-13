#include <iostream>
#include <cmath>
#include <cassert>
#include <numbers>
#include <limes/limes.hpp>

using namespace limes::algorithms;

template<typename T>
bool approx_equal(T a, T b, T tol = T(1e-10)) {
    return std::abs(a - b) <= tol * std::max(std::abs(a), std::abs(b));
}

void test_simple_polynomial() {
    std::cout << "Testing polynomial integration... ";

    auto f = [](double x) { return x * x; };
    auto integrator = adaptive_integrator<double>{};
    auto result = integrator(f, 0.0, 1.0, 1e-12);

    assert(approx_equal(result.value(), 1.0/3.0, 1e-10));
    std::cout << "PASSED\n";
}

void test_trigonometric() {
    std::cout << "Testing trigonometric integration... ";

    auto f = [](double x) { return std::sin(x); };
    auto integrator = adaptive_integrator<double>{};
    auto result = integrator(f, 0.0, std::numbers::pi, 1e-12);

    assert(approx_equal(result.value(), 2.0, 1e-10));
    std::cout << "PASSED\n";
}

void test_exponential() {
    std::cout << "Testing exponential integration... ";

    auto f = [](double x) { return std::exp(-x); };
    auto integrator = adaptive_integrator<double>{};
    auto result = integrator(f, 0.0, 10.0, 1e-12);

    double exact = 1.0 - std::exp(-10.0);
    assert(approx_equal(result.value(), exact, 1e-10));
    std::cout << "PASSED\n";
}

void test_gaussian_integral() {
    std::cout << "Testing Gaussian integral... ";

    auto f = [](double x) { return std::exp(-x * x); };

    // Integrate from -10 to 10 instead of -inf to inf
    // exp(-100) is negligible so this captures almost all the mass
    auto integrator = adaptive_integrator<double>{};
    auto result = integrator(f, -10.0, 10.0, 1e-8);

    double exact = std::sqrt(std::numbers::pi);
    assert(approx_equal(result.value(), exact, 1e-6));
    std::cout << "PASSED\n";
}

void test_accumulators() {
    std::cout << "Testing different accumulators... ";

    // Test that compensated accumulators improve accuracy
    using T = double;
    constexpr T tiny = 1e-16;
    constexpr std::size_t n = 1000000;

    // Simple accumulator (will have rounding errors)
    accumulators::simple_accumulator<T> simple;
    for (std::size_t i = 0; i < n; ++i) {
        simple += tiny;
    }

    // Kahan accumulator (should be more accurate)
    accumulators::kahan_accumulator<T> kahan;
    for (std::size_t i = 0; i < n; ++i) {
        kahan += tiny;
    }

    T exact = tiny * n;
    T simple_error = std::abs(simple() - exact);
    T kahan_error = std::abs(kahan() - exact);

    // Kahan should be more accurate
    assert(kahan_error < simple_error);
    std::cout << "PASSED\n";
}

void test_quadrature_rules() {
    std::cout << "Testing quadrature rules... ";

    // Test that Gauss quadrature integrates polynomials exactly
    auto f = [](double x) { return x * x * x; };  // Degree 3 polynomial

    quadrature::gauss_legendre<double, 2> gl2;
    quadrature_integrator<double, decltype(gl2)> integrator{gl2};

    auto result = integrator(f, -1.0, 1.0);

    // x^3 from -1 to 1 = 0
    assert(approx_equal(result.value(), 0.0, 1e-14));
    std::cout << "PASSED\n";
}

void test_integration_result_operations() {
    std::cout << "Testing integration result operations... ";

    integration_result<double> r1{1.0, 0.01, 100};
    integration_result<double> r2{2.0, 0.02, 200};

    auto sum = r1 + r2;
    assert(approx_equal(sum.value(), 3.0));
    assert(approx_equal(sum.error(), 0.03));
    assert(sum.evaluations() == 300);

    auto scaled = r1 * 2.0;
    assert(approx_equal(scaled.value(), 2.0));
    assert(approx_equal(scaled.error(), 0.02));

    std::cout << "PASSED\n";
}

void test_romberg_integrator() {
    std::cout << "Testing Romberg integrator... ";

    auto f = [](double x) { return std::exp(x); };
    auto integrator = romberg_integrator<double>{};
    auto result = integrator(f, 0.0, 1.0, 1e-12);

    double exact = std::exp(1.0) - 1.0;
    assert(approx_equal(result.value(), exact, 1e-10));
    std::cout << "PASSED\n";
}

int main() {
    std::cout << "Running limes Library Tests\n";
    std::cout << "============================\n\n";

    try {
        test_simple_polynomial();
        test_trigonometric();
        test_exponential();
        test_gaussian_integral();
        test_accumulators();
        test_quadrature_rules();
        test_integration_result_operations();
        test_romberg_integrator();

        std::cout << "\n============================\n";
        std::cout << "All tests PASSED!\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nTest FAILED: " << e.what() << "\n";
        return 1;
    }
}
