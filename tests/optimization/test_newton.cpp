#include <iostream>
#include <cassert>
#include <cmath>
#include "../../include/stepanov/optimization.hpp"
#include "../../include/stepanov/rational.hpp"
#include "../../include/stepanov/polynomial.hpp"

using namespace stepanov;
using namespace stepanov::optimization;

// Test functions
template<typename T>
T quadratic(T x) {
    return x * x - T(4) * x + T(3); // Minimum at x = 2
}

template<typename T>
T cubic(T x) {
    return x * x * x - T(3) * x; // Roots at x = 0, ±√3
}

template<typename V>
typename V::value_type rosenbrock(const V& x) {
    using T = typename V::value_type;
    T a = T(1);
    T b = T(100);
    T term1 = a - x[0];
    T term2 = x[1] - x[0] * x[0];
    return term1 * term1 + b * term2 * term2;
}

template<typename V>
typename V::value_type sphere(const V& x) {
    return dot(x, x);
}

void test_newton_univariate() {
    std::cout << "Testing Newton's method for univariate functions...\n";

    // Test minimization of quadratic function
    auto result = newton_univariate(quadratic<double>, 0.0);
    assert(result.converged);
    assert(std::abs(result.solution - 2.0) < 1e-6);
    assert(std::abs(result.value - (-1.0)) < 1e-6);
    std::cout << "  Quadratic minimization: x* = " << result.solution
              << ", f(x*) = " << result.value << "\n";

    // Test with bounds
    auto bounded_result = newton_univariate(
        quadratic<double>, 0.0,
        tolerance_criterion<double>{},
        std::make_pair(1.5, 3.0)
    );
    assert(bounded_result.converged);
    assert(bounded_result.solution >= 1.5 && bounded_result.solution <= 3.0);
    std::cout << "  Bounded optimization: x* = " << bounded_result.solution << "\n";

    // Test root finding
    auto root = newton_root(cubic<double>, 1.5);
    assert(root.has_value());
    assert(std::abs(cubic(*root)) < 1e-6);
    std::cout << "  Root finding: x* = " << *root
              << ", f(x*) = " << cubic(*root) << "\n";

    std::cout << "  All univariate Newton tests passed!\n\n";
}

void test_newton_multivariate() {
    std::cout << "Testing Newton's method for multivariate functions...\n";

    // Test sphere function (minimum at origin)
    vector<double> x0{1.0, 1.0};
    auto result = newton_multivariate(sphere<vector<double>>, x0);
    assert(result.converged);
    assert(norm(result.solution) < 1e-6);
    std::cout << "  Sphere function: ||x*|| = " << norm(result.solution)
              << ", f(x*) = " << result.value << "\n";

    // Test Rosenbrock function (minimum at (1, 1))
    vector<double> x1{0.0, 0.0};
    auto rosenbrock_result = newton_multivariate(
        rosenbrock<vector<double>>, x1,
        tolerance_criterion<double>{1e-8, 1e-8, 1000}
    );
    if (rosenbrock_result.converged) {
        std::cout << "  Rosenbrock: x* = (" << rosenbrock_result.solution[0]
                  << ", " << rosenbrock_result.solution[1]
                  << "), f(x*) = " << rosenbrock_result.value << "\n";
    }

    std::cout << "  Multivariate Newton tests completed!\n\n";
}

void test_trust_region() {
    std::cout << "Testing trust region Newton method...\n";

    vector<double> x0{-1.0, 2.0};
    auto result = newton_trust_region(
        rosenbrock<vector<double>>, x0, 1.0,
        tolerance_criterion<double>{1e-8, 1e-8, 1000}
    );

    std::cout << "  Trust region result: x* = ("
              << result.solution[0] << ", " << result.solution[1]
              << "), f(x*) = " << result.value
              << ", converged = " << result.converged << "\n";

    std::cout << "  Trust region tests completed!\n\n";
}

void test_with_rational() {
    std::cout << "Testing Newton's method with rational numbers...\n";

    using Rat = rational<int>;

    // Define a simple rational function
    auto rational_func = [](Rat x) {
        return x * x - Rat(4, 1) * x + Rat(3, 1);
    };

    // Note: Newton with rationals requires careful handling
    // This is a simplified test
    Rat x0(0, 1);

    // Manual Newton iteration for demonstration
    for (int i = 0; i < 5; ++i) {
        Rat fx = rational_func(x0);
        Rat fpx = Rat(2, 1) * x0 - Rat(4, 1); // Derivative

        if (fpx.numerator() != 0) {
            x0 = x0 - fx / fpx;
            std::cout << "  Iteration " << i << ": x = "
                      << x0.numerator() << "/" << x0.denominator() << "\n";
        }
    }

    std::cout << "  Rational Newton tests completed!\n\n";
}

int main() {
    std::cout << "=== Testing Newton's Method Module ===\n\n";

    test_newton_univariate();
    test_newton_multivariate();
    test_trust_region();
    test_with_rational();

    std::cout << "=== All Newton's method tests passed! ===\n";

    return 0;
}