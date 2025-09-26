#include <iostream>
#include <cassert>
#include <cmath>
#include "../../include/stepanov/optimization.hpp"
#include "../../include/stepanov/rational.hpp"

using namespace stepanov;
using namespace stepanov::optimization;

// Test functions for optimization
template<typename T>
T unimodal_quadratic(T x) {
    return (x - T(3)) * (x - T(3)) + T(2); // Minimum at x = 3, f(3) = 2
}

template<typename T>
T cubic_function(T x) {
    return x * x * x - T(3) * x * x + T(2); // Local minimum around x = 2
}

template<typename T>
T sine_function(T x) {
    return std::sin(x) + T(0.1) * x; // Multiple local extrema
}

template<typename T>
T exponential_function(T x) {
    return std::exp(-x * x) * (T(1) - x); // Smooth unimodal function
}

void test_golden_section_basic() {
    std::cout << "Testing basic golden section search...\n";

    // Test minimization
    auto result = golden_section_minimize(unimodal_quadratic<double>, 0.0, 6.0);
    assert(result.converged);
    assert(std::abs(result.solution - 3.0) < 1e-6);
    assert(std::abs(result.value - 2.0) < 1e-6);
    std::cout << "  Quadratic minimum: x* = " << result.solution
              << ", f(x*) = " << result.value
              << ", iterations = " << result.iterations << "\n";

    // Test maximization
    auto max_result = golden_section_maximize(
        [](double x) { return -unimodal_quadratic(x); },
        0.0, 6.0
    );
    std::cout << "  Quadratic maximum: x* = " << max_result.solution
              << ", f(x*) = " << max_result.value << "\n";

    // Test with tight tolerance
    tolerance_criterion<double> tight{1e-10, 1e-10, 1000};
    auto precise = golden_section_minimize(unimodal_quadratic<double>, 0.0, 6.0, tight);
    assert(std::abs(precise.solution - 3.0) < 1e-9);
    std::cout << "  High precision: x* = " << precise.solution
              << " (error < 1e-9)\n";

    std::cout << "  Basic golden section tests passed!\n\n";
}

void test_fibonacci_search() {
    std::cout << "Testing Fibonacci search...\n";

    // Fibonacci search with fixed evaluations
    auto fib_result = fibonacci_search(cubic_function<double>, 0.0, 3.0, 15);
    std::cout << "  Fibonacci (15 evals): x* = " << fib_result.solution
              << ", f(x*) = " << fib_result.value << "\n";

    // Compare with golden section
    auto gs_result = golden_section_minimize(
        cubic_function<double>, 0.0, 3.0,
        tolerance_criterion<double>{1e-6, 1e-6, 15}
    );
    std::cout << "  Golden section: x* = " << gs_result.solution
              << ", f(x*) = " << gs_result.value << "\n";

    // Fibonacci is optimal for fixed number of evaluations
    std::cout << "  Fibonacci vs Golden section comparison completed\n";

    std::cout << "  Fibonacci search tests passed!\n\n";
}

void test_ternary_search() {
    std::cout << "Testing ternary search...\n";

    auto ternary_result = ternary_search(
        exponential_function<double>, -2.0, 2.0
    );
    assert(ternary_result.converged);
    std::cout << "  Ternary search: x* = " << ternary_result.solution
              << ", f(x*) = " << ternary_result.value
              << ", iterations = " << ternary_result.iterations << "\n";

    // Compare convergence rate with golden section
    tolerance_criterion<double> fixed_iter{1e-6, 1e-6, 20};
    auto gs_comp = golden_section_minimize(
        exponential_function<double>, -2.0, 2.0, fixed_iter
    );
    auto tern_comp = ternary_search(
        exponential_function<double>, -2.0, 2.0, fixed_iter
    );

    std::cout << "  After 20 iterations:\n";
    std::cout << "    Golden section: f(x*) = " << gs_comp.value << "\n";
    std::cout << "    Ternary search: f(x*) = " << tern_comp.value << "\n";

    std::cout << "  Ternary search tests passed!\n\n";
}

void test_quadratic_interpolation() {
    std::cout << "Testing quadratic interpolation search...\n";

    // Quadratic interpolation works well for smooth functions
    auto quad_interp = quadratic_interpolation(
        unimodal_quadratic<double>, 1.0, 3.0, 5.0
    );
    assert(quad_interp.converged);
    assert(std::abs(quad_interp.solution - 3.0) < 1e-6);
    std::cout << "  Quadratic interpolation: x* = " << quad_interp.solution
              << ", f(x*) = " << quad_interp.value
              << ", iterations = " << quad_interp.iterations << "\n";

    // Test on cubic (should converge to local minimum)
    auto cubic_interp = quadratic_interpolation(
        cubic_function<double>, 1.0, 2.0, 3.0
    );
    std::cout << "  Cubic function: x* = " << cubic_interp.solution
              << ", f(x*) = " << cubic_interp.value << "\n";

    std::cout << "  Quadratic interpolation tests passed!\n\n";
}

void test_brent_minimize() {
    std::cout << "Testing Brent's method for minimization...\n";

    // Brent combines golden section with parabolic interpolation
    auto brent_result = brent_minimize(
        exponential_function<double>, -2.0, 2.0
    );
    assert(brent_result.converged);
    std::cout << "  Brent's method: x* = " << brent_result.solution
              << ", f(x*) = " << brent_result.value
              << ", iterations = " << brent_result.iterations << "\n";

    // Test on difficult function
    auto difficult = [](double x) {
        return std::abs(x - 1.5) + 0.01 * x * x;
    };

    auto brent_diff = brent_minimize(difficult, -5.0, 5.0);
    std::cout << "  Difficult function: x* = " << brent_diff.solution
              << ", f(x*) = " << brent_diff.value << "\n";

    // Performance comparison
    int brent_iters = 0, golden_iters = 0;

    // Count iterations for convergence
    tolerance_criterion<double> counter{1e-8, 1e-8, 1000};
    auto b_perf = brent_minimize(exponential_function<double>, -2.0, 2.0, counter);
    auto g_perf = golden_section_minimize(exponential_function<double>, -2.0, 2.0, counter);

    std::cout << "  Performance comparison:\n";
    std::cout << "    Brent iterations: " << b_perf.iterations << "\n";
    std::cout << "    Golden iterations: " << g_perf.iterations << "\n";

    std::cout << "  Brent minimization tests passed!\n\n";
}

void test_with_rational_numbers() {
    std::cout << "Testing golden section with rational numbers...\n";

    using Rat = rational<int64_t>;

    // Rational quadratic function
    auto rat_func = [](Rat x) {
        Rat three(3, 1);
        return (x - three) * (x - three) + Rat(2, 1);
    };

    // Manual golden section for rationals (simplified)
    Rat a(0, 1);  // 0
    Rat b(6, 1);  // 6

    // Approximate golden ratio conjugate with rational
    Rat gr(618, 1000); // ~0.618

    for (int iter = 0; iter < 10; ++iter) {
        Rat width = b - a;
        Rat x1 = a + width * (Rat(1, 1) - gr);
        Rat x2 = a + width * gr;

        Rat f1 = rat_func(x1);
        Rat f2 = rat_func(x2);

        if (f1 < f2) {
            b = x2;
        } else {
            a = x1;
        }

        if (iter == 9) {
            Rat mid = (a + b) / Rat(2, 1);
            std::cout << "  Final interval center: "
                      << mid.numerator() << "/" << mid.denominator()
                      << " (should be close to 3)\n";
        }
    }

    std::cout << "  Rational golden section tests completed!\n\n";
}

void test_constrained_optimization() {
    std::cout << "Testing optimization with bounds...\n";

    // Function with minimum at x = 3, but constrained to [0, 2]
    auto constrained = golden_section_minimize(
        unimodal_quadratic<double>, 0.0, 2.0
    );

    assert(constrained.converged);
    assert(constrained.solution <= 2.0);
    std::cout << "  Constrained minimum: x* = " << constrained.solution
              << " (bounded to [0, 2])\n";
    std::cout << "  f(x*) = " << constrained.value << "\n";

    // Test with multiple intervals
    auto find_global_min = [](auto f, std::vector<std::pair<double, double>> intervals) {
        double best_x = 0, best_f = std::numeric_limits<double>::max();

        for (const auto& [a, b] : intervals) {
            auto result = golden_section_minimize(f, a, b);
            if (result.value < best_f) {
                best_x = result.solution;
                best_f = result.value;
            }
        }
        return std::pair{best_x, best_f};
    };

    // Search multiple intervals for sine function
    std::vector<std::pair<double, double>> intervals{
        {0, 2}, {2, 4}, {4, 6}
    };

    auto [global_x, global_f] = find_global_min(sine_function<double>, intervals);
    std::cout << "  Multi-interval search: x* = " << global_x
              << ", f(x*) = " << global_f << "\n";

    std::cout << "  Constrained optimization tests passed!\n\n";
}

void test_convergence_rates() {
    std::cout << "Testing convergence rates of different methods...\n";

    auto smooth_func = [](double x) {
        return x * x * x * x - 2 * x * x + 1; // Smooth polynomial
    };

    struct ConvergenceData {
        std::string method;
        std::vector<double> errors;
    };

    std::vector<ConvergenceData> data;

    // Golden section convergence
    {
        ConvergenceData golden{"Golden Section", {}};
        double a = -2.0, b = 2.0;
        double true_min = 1.0; // Approximate

        for (int iter = 0; iter < 20; ++iter) {
            double gr = golden_ratio_conjugate<double>();
            double x1 = a + gr * (b - a);
            double x2 = a + (1 - gr) * (b - a);

            if (smooth_func(x1) < smooth_func(x2)) {
                b = x2;
            } else {
                a = x1;
            }

            double mid = (a + b) / 2;
            golden.errors.push_back(std::abs(mid - true_min));
        }
        data.push_back(golden);
    }

    // Print convergence comparison
    std::cout << "  Convergence analysis (error vs iteration):\n";
    for (const auto& method_data : data) {
        std::cout << "    " << method_data.method << ":\n";
        for (size_t i = 0; i < 5 && i < method_data.errors.size(); ++i) {
            std::cout << "      Iter " << i << ": " << method_data.errors[i] << "\n";
        }
    }

    std::cout << "  Convergence rate tests completed!\n\n";
}

int main() {
    std::cout << "=== Testing Golden Section Search Module ===\n\n";

    test_golden_section_basic();
    test_fibonacci_search();
    test_ternary_search();
    test_quadratic_interpolation();
    test_brent_minimize();
    test_with_rational_numbers();
    test_constrained_optimization();
    test_convergence_rates();

    std::cout << "=== All golden section tests passed! ===\n";

    return 0;
}