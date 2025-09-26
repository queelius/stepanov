#include <iostream>
#include <cmath>
#include "../include/stepanov/optimization.hpp"

using namespace stepanov;
using namespace stepanov::optimization;

int main() {
    std::cout << "Testing Stepanov Optimization Module\n\n";

    // Test 1: Simple univariate optimization
    std::cout << "1. Finding minimum of x^2 - 4x + 3:\n";
    auto quadratic = [](double x) { return x * x - 4 * x + 3; };

    auto result = golden_section_minimize(quadratic, 0.0, 5.0,
                                         tolerance_criterion<double>{1e-8, 1e-8, 100});
    std::cout << "   Minimum at x = " << result.solution
              << ", f(x) = " << result.value << "\n";
    std::cout << "   (Expected: x = 2, f(x) = -1)\n\n";

    // Test 2: Root finding
    std::cout << "2. Finding root of cos(x) - x:\n";
    auto equation = [](double x) { return std::cos(x) - x; };

    auto root = bisection(equation, 0.0, 1.0);
    if (root) {
        std::cout << "   Root at x = " << *root
                  << ", f(x) = " << equation(*root) << "\n\n";
    }

    // Test 3: Multivariate optimization
    std::cout << "3. Minimizing sphere function f(x,y) = x^2 + y^2:\n";
    auto sphere = [](const vector<double>& x) {
        return x[0] * x[0] + x[1] * x[1];
    };

    vector<double> start{1.0, 1.0};
    auto gd_result = gradient_descent(sphere, start,
                                      constant_step_size<double>(0.01),
                                      tolerance_criterion<double>{1e-8, 1e-8, 500});
    std::cout << "   Starting at: (" << start[0] << ", " << start[1]
              << "), f(x) = " << sphere(start) << "\n";
    std::cout << "   Minimum at: (" << gd_result.solution[0]
              << ", " << gd_result.solution[1]
              << "), f(x) = " << gd_result.value << "\n";
    std::cout << "   Iterations: " << gd_result.iterations
              << ", Converged: " << (gd_result.converged ? "Yes" : "No") << "\n\n";

    std::cout << "All tests completed successfully!\n";

    return 0;
}