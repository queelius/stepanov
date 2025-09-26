#include <iostream>
#include <cassert>
#include <cmath>
#include <complex>
#include "../../include/stepanov/optimization.hpp"
#include "../../include/stepanov/polynomial.hpp"
#include "../../include/stepanov/rational.hpp"

using namespace stepanov;
using namespace stepanov::optimization;

// Test functions
template<typename T>
T linear_func(T x) {
    return T(2) * x - T(5); // Root at x = 2.5
}

template<typename T>
T quadratic_func(T x) {
    return x * x - T(4); // Roots at x = ±2
}

template<typename T>
T cubic_func(T x) {
    return x * x * x - T(2) * x - T(5); // Root near x = 2.095
}

template<typename T>
T transcendental_func(T x) {
    return std::cos(x) - x; // Root near x = 0.739
}

template<typename T>
T exp_func(T x) {
    return std::exp(x) - T(3) * x; // Multiple roots
}

void test_bisection() {
    std::cout << "Testing bisection method...\n";

    // Linear function
    auto linear_root = bisection(linear_func<double>, 0.0, 5.0);
    assert(linear_root.has_value());
    assert(std::abs(*linear_root - 2.5) < 1e-6);
    std::cout << "  Linear: root = " << *linear_root
              << ", f(root) = " << linear_func(*linear_root) << "\n";

    // Quadratic function
    auto quad_root = bisection(quadratic_func<double>, 0.0, 3.0);
    assert(quad_root.has_value());
    assert(std::abs(*quad_root - 2.0) < 1e-6);
    std::cout << "  Quadratic: root = " << *quad_root
              << ", f(root) = " << quadratic_func(*quad_root) << "\n";

    // Transcendental function
    auto trans_root = bisection(transcendental_func<double>, 0.0, 1.0);
    assert(trans_root.has_value());
    assert(std::abs(transcendental_func(*trans_root)) < 1e-6);
    std::cout << "  Transcendental: root = " << *trans_root
              << ", f(root) = " << transcendental_func(*trans_root) << "\n";

    // Test with no sign change (should return nullopt)
    auto no_root = bisection(quadratic_func<double>, 3.0, 5.0);
    assert(!no_root.has_value());
    std::cout << "  No sign change: correctly returned nullopt\n";

    std::cout << "  Bisection tests passed!\n\n";
}

void test_secant() {
    std::cout << "Testing secant method...\n";

    // Cubic function
    auto cubic_root = secant(cubic_func<double>, 2.0, 3.0);
    assert(cubic_root.has_value());
    assert(std::abs(cubic_func(*cubic_root)) < 1e-6);
    std::cout << "  Cubic: root = " << *cubic_root
              << ", f(root) = " << cubic_func(*cubic_root) << "\n";

    // Exponential function
    auto exp_root = secant(exp_func<double>, 0.0, 1.0);
    if (exp_root.has_value()) {
        std::cout << "  Exponential: root = " << *exp_root
                  << ", f(root) = " << exp_func(*exp_root) << "\n";
    }

    // Fast convergence for smooth functions
    tolerance_criterion<double> tight_tol{1e-12, 1e-12, 20};
    auto precise_root = secant(quadratic_func<double>, 1.5, 2.5, tight_tol);
    assert(precise_root.has_value());
    assert(std::abs(*precise_root - 2.0) < 1e-10);
    std::cout << "  High precision: root = " << *precise_root
              << " (error < 1e-10)\n";

    std::cout << "  Secant tests passed!\n\n";
}

void test_false_position() {
    std::cout << "Testing false position method...\n";

    auto fp_root = false_position(cubic_func<double>, 1.0, 3.0);
    assert(fp_root.has_value());
    assert(std::abs(cubic_func(*fp_root)) < 1e-6);
    std::cout << "  Cubic: root = " << *fp_root
              << ", f(root) = " << cubic_func(*fp_root) << "\n";

    // Compare convergence with bisection
    size_t bisection_iters = 0;
    size_t fp_iters = 0;

    // Count iterations for bisection
    {
        double a = 1.0, b = 3.0;
        while (b - a > 1e-6) {
            double c = (a + b) / 2;
            if (cubic_func(a) * cubic_func(c) < 0)
                b = c;
            else
                a = c;
            bisection_iters++;
        }
    }

    // Count iterations for false position
    {
        double a = 1.0, b = 3.0;
        while (std::abs(b - a) > 1e-6) {
            double c = (a * cubic_func(b) - b * cubic_func(a)) /
                      (cubic_func(b) - cubic_func(a));
            if (cubic_func(a) * cubic_func(c) < 0)
                b = c;
            else
                a = c;
            fp_iters++;
            if (fp_iters > 100) break; // Safety
        }
    }

    std::cout << "  Convergence comparison: bisection = " << bisection_iters
              << " iters, false position = " << fp_iters << " iters\n";

    std::cout << "  False position tests passed!\n\n";
}

void test_brent() {
    std::cout << "Testing Brent's method...\n";

    // Brent's method combines best of bisection, secant, and inverse quadratic
    auto brent_root = brent(cubic_func<double>, 1.0, 3.0);
    assert(brent_root.has_value());
    assert(std::abs(cubic_func(*brent_root)) < 1e-6);
    std::cout << "  Cubic: root = " << *brent_root
              << ", f(root) = " << cubic_func(*brent_root) << "\n";

    // Test on difficult function
    auto difficult_func = [](double x) {
        return std::sin(x) - x / 2;
    };

    auto diff_root = brent(difficult_func, 1.5, 2.0);
    assert(diff_root.has_value());
    std::cout << "  Difficult function: root = " << *diff_root
              << ", f(root) = " << difficult_func(*diff_root) << "\n";

    std::cout << "  Brent's method tests passed!\n\n";
}

void test_ridders() {
    std::cout << "Testing Ridders' method...\n";

    auto ridders_root = ridders(cubic_func<double>, 1.0, 3.0);
    assert(ridders_root.has_value());
    assert(std::abs(cubic_func(*ridders_root)) < 1e-6);
    std::cout << "  Cubic: root = " << *ridders_root
              << ", f(root) = " << cubic_func(*ridders_root) << "\n";

    // Test exponential convergence
    tolerance_criterion<double> tight{1e-14, 1e-14, 10};
    auto precise = ridders(quadratic_func<double>, 1.0, 3.0, tight);
    assert(precise.has_value());
    double error = std::abs(*precise - 2.0);
    std::cout << "  Quadratic (high precision): error = " << error << "\n";
    assert(error < 1e-12);

    std::cout << "  Ridders' method tests passed!\n\n";
}

void test_fixed_point() {
    std::cout << "Testing fixed-point iteration...\n";

    // Solve x = cos(x) by fixed-point iteration
    auto cos_fixed = [](double x) { return std::cos(x); };

    auto fp_result = fixed_point(cos_fixed, 0.5);
    assert(fp_result.has_value());
    assert(std::abs(*fp_result - std::cos(*fp_result)) < 1e-6);
    std::cout << "  cos(x) = x: root = " << *fp_result
              << ", check: " << *fp_result << " = " << std::cos(*fp_result) << "\n";

    // Solve x^2 = 2 using x = 2/x iteration (for x > 0)
    auto sqrt2_iter = [](double x) { return 2.0 / x; };

    // This iteration doesn't converge directly, need averaging
    auto sqrt2_avg = [](double x) { return 0.5 * (x + 2.0 / x); };

    auto sqrt2_result = fixed_point(sqrt2_avg, 1.5);
    assert(sqrt2_result.has_value());
    assert(std::abs(*sqrt2_result - std::sqrt(2.0)) < 1e-6);
    std::cout << "  sqrt(2): result = " << *sqrt2_result
              << ", error = " << std::abs(*sqrt2_result - std::sqrt(2.0)) << "\n";

    std::cout << "  Fixed-point iteration tests passed!\n\n";
}

void test_muller() {
    std::cout << "Testing Müller's method...\n";

    auto muller_root = muller(cubic_func<double>, 0.0, 1.0, 2.0);
    if (muller_root.has_value()) {
        std::cout << "  Cubic: root = " << *muller_root
                  << ", f(root) = " << cubic_func(*muller_root) << "\n";
        assert(std::abs(cubic_func(*muller_root)) < 1e-6);
    }

    // Müller can find complex roots when extended to complex numbers
    // For real fields, it finds real roots
    auto quad_root = muller(quadratic_func<double>, 1.5, 2.0, 2.5);
    assert(quad_root.has_value());
    assert(std::abs(*quad_root - 2.0) < 1e-6);
    std::cout << "  Quadratic: root = " << *quad_root << "\n";

    std::cout << "  Müller's method tests passed!\n\n";
}

void test_with_rational_numbers() {
    std::cout << "Testing root finding with rational numbers...\n";

    using Rat = rational<int>;

    // Simple rational function
    auto rat_func = [](Rat x) {
        return x * x - Rat(4, 1); // x^2 - 4
    };

    // Bisection works well with exact rational arithmetic
    Rat a(0, 1);  // 0
    Rat b(3, 1);  // 3

    // Manual bisection for demonstration
    for (int i = 0; i < 10; ++i) {
        Rat c = (a + b) / Rat(2, 1);
        Rat fc = rat_func(c);

        std::cout << "  Iteration " << i << ": c = "
                  << c.numerator() << "/" << c.denominator()
                  << ", f(c) = " << fc.numerator() << "/" << fc.denominator() << "\n";

        if (fc == Rat(0, 1)) {
            std::cout << "  Exact root found: " << c.numerator()
                      << "/" << c.denominator() << "\n";
            break;
        }

        if (rat_func(a) * fc < Rat(0, 1)) {
            b = c;
        } else {
            a = c;
        }
    }

    std::cout << "  Rational root finding tests completed!\n\n";
}

void test_polynomial_roots() {
    std::cout << "Testing root finding with polynomials...\n";

    // Create a polynomial: (x - 1)(x - 2)(x - 3) = x^3 - 6x^2 + 11x - 6
    // Coefficients: -6, 11, -6, 1
    auto poly_func = [](double x) {
        return x * x * x - 6 * x * x + 11 * x - 6;
    };

    // Find roots using different methods
    auto root1 = bisection(poly_func, 0.5, 1.5);
    auto root2 = bisection(poly_func, 1.5, 2.5);
    auto root3 = bisection(poly_func, 2.5, 3.5);

    assert(root1.has_value() && std::abs(*root1 - 1.0) < 1e-6);
    assert(root2.has_value() && std::abs(*root2 - 2.0) < 1e-6);
    assert(root3.has_value() && std::abs(*root3 - 3.0) < 1e-6);

    std::cout << "  Polynomial roots: " << *root1 << ", "
              << *root2 << ", " << *root3 << "\n";

    // Test with Brent for efficiency
    auto brent_poly_root = brent(poly_func, 0.0, 1.5);
    std::cout << "  Brent on polynomial: root = " << *brent_poly_root << "\n";

    std::cout << "  Polynomial root finding tests passed!\n\n";
}

int main() {
    std::cout << "=== Testing Root Finding Module ===\n\n";

    test_bisection();
    test_secant();
    test_false_position();
    test_brent();
    test_ridders();
    test_fixed_point();
    test_muller();
    test_with_rational_numbers();
    test_polynomial_roots();

    std::cout << "=== All root finding tests passed! ===\n";

    return 0;
}