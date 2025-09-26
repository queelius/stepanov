/**
 * Demonstration of the Stepanov Library Optimization Module
 *
 * This example shows how to use various optimization algorithms
 * following generic programming principles.
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include "../include/stepanov/optimization.hpp"
#include "../include/stepanov/polynomial.hpp"
#include "../include/stepanov/rational.hpp"

using namespace stepanov;
using namespace stepanov::optimization;

// Example 1: Finding minimum of a polynomial
void polynomial_optimization_example() {
    std::cout << "=== Polynomial Optimization ===\n\n";

    // Define polynomial: p(x) = x^3 - 6x^2 + 9x + 1
    // Has local minimum around x ≈ 3
    polynomial<double> p;
    p.set_coefficient(0, 1.0);
    p.set_coefficient(1, 9.0);
    p.set_coefficient(2, -6.0);
    p.set_coefficient(3, 1.0);

    auto poly_func = [&p](double x) { return p.evaluate(x); };

    // Find minimum using Newton's method
    std::cout << "Using Newton's method:\n";
    auto newton_result = newton_univariate(poly_func, 2.5);
    std::cout << "  Minimum at x = " << newton_result.solution
              << ", p(x) = " << newton_result.value << "\n";
    std::cout << "  Converged in " << newton_result.iterations << " iterations\n\n";

    // Compare with golden section search
    std::cout << "Using golden section search:\n";
    auto golden_result = golden_section_minimize(poly_func, 0.0, 5.0);
    std::cout << "  Minimum at x = " << golden_result.solution
              << ", p(x) = " << golden_result.value << "\n";
    std::cout << "  Converged in " << golden_result.iterations << " iterations\n\n";
}

// Example 2: Multivariate optimization with gradient descent
void gradient_descent_example() {
    std::cout << "=== Gradient Descent Example ===\n\n";

    // Rosenbrock function - classic test problem
    auto rosenbrock = [](const vector<double>& x) {
        double a = 1.0, b = 100.0;
        double term1 = a - x[0];
        double term2 = x[1] - x[0] * x[0];
        return term1 * term1 + b * term2 * term2;
    };

    vector<double> start_point{-1.0, 2.0};
    std::cout << "Starting point: (" << start_point[0]
              << ", " << start_point[1] << ")\n";
    std::cout << "Initial value: " << rosenbrock(start_point) << "\n\n";

    // Try different optimizers
    std::cout << "1. Standard gradient descent:\n";
    auto gd_result = gradient_descent(rosenbrock, start_point,
                                      constant_step_size<double>(0.001));
    std::cout << "   Solution: (" << gd_result.solution[0]
              << ", " << gd_result.solution[1] << ")\n";
    std::cout << "   Value: " << gd_result.value
              << ", Iterations: " << gd_result.iterations << "\n\n";

    std::cout << "2. Adam optimizer:\n";
    auto adam_result = adam(rosenbrock, start_point);
    std::cout << "   Solution: (" << adam_result.solution[0]
              << ", " << adam_result.solution[1] << ")\n";
    std::cout << "   Value: " << adam_result.value
              << ", Iterations: " << adam_result.iterations << "\n\n";

    std::cout << "3. Conjugate gradient:\n";
    auto cg_result = conjugate_gradient(rosenbrock, start_point);
    std::cout << "   Solution: (" << cg_result.solution[0]
              << ", " << cg_result.solution[1] << ")\n";
    std::cout << "   Value: " << cg_result.value
              << ", Iterations: " << cg_result.iterations << "\n\n";
}

// Example 3: Root finding
void root_finding_example() {
    std::cout << "=== Root Finding Example ===\n\n";

    // Find where cos(x) = x
    auto equation = [](double x) { return std::cos(x) - x; };

    std::cout << "Solving cos(x) = x:\n\n";

    // Bisection method
    auto bisection_root = bisection(equation, 0.0, 1.0);
    if (bisection_root) {
        std::cout << "Bisection: x = " << *bisection_root
                  << ", f(x) = " << equation(*bisection_root) << "\n";
    }

    // Secant method
    auto secant_root = secant(equation, 0.0, 1.0);
    if (secant_root) {
        std::cout << "Secant:    x = " << *secant_root
                  << ", f(x) = " << equation(*secant_root) << "\n";
    }

    // Brent's method
    auto brent_root = brent(equation, 0.0, 1.0);
    if (brent_root) {
        std::cout << "Brent:     x = " << *brent_root
                  << ", f(x) = " << equation(*brent_root) << "\n";
    }

    // Fixed point iteration
    auto cos_func = [](double x) { return std::cos(x); };
    auto fixed_root = fixed_point(cos_func, 0.5);
    if (fixed_root) {
        std::cout << "Fixed pt:  x = " << *fixed_root
                  << ", cos(x) = " << std::cos(*fixed_root) << "\n";
    }

    std::cout << "\n";
}

// Example 4: Global optimization with simulated annealing
void simulated_annealing_example() {
    std::cout << "=== Simulated Annealing Example ===\n\n";

    // Ackley function - many local minima
    auto ackley = [](const std::vector<double>& x) {
        double a = 20.0, b = 0.2, c = 2 * M_PI;
        double sum1 = 0, sum2 = 0;
        for (double xi : x) {
            sum1 += xi * xi;
            sum2 += std::cos(c * xi);
        }
        double n = x.size();
        return -a * std::exp(-b * std::sqrt(sum1 / n)) -
               std::exp(sum2 / n) + a + std::exp(1.0);
    };

    // Neighbor generator
    auto neighbor = [](const std::vector<double>& x, std::mt19937& rng) {
        std::normal_distribution<double> dist(0, 0.5);
        std::vector<double> neighbor = x;
        for (double& xi : neighbor) {
            xi += dist(rng);
        }
        return neighbor;
    };

    std::vector<double> initial{5.0, 5.0};
    std::mt19937 rng(42);

    std::cout << "Optimizing Ackley function (global minimum at origin):\n";
    std::cout << "Starting from: (" << initial[0] << ", " << initial[1] << ")\n";
    std::cout << "Initial value: " << ackley(initial) << "\n\n";

    auto sa_result = simulated_annealing(
        ackley, initial, neighbor,
        exponential_cooling<double>(50.0, 0.95),
        3000, 0.001, rng
    );

    std::cout << "Best solution: (" << sa_result.best_solution[0]
              << ", " << sa_result.best_solution[1] << ")\n";
    std::cout << "Best value: " << sa_result.best_energy << "\n";
    std::cout << "Acceptance rate: "
              << (100.0 * sa_result.accepted_moves / sa_result.iterations)
              << "%\n\n";
}

// Example 5: Working with rational numbers
void rational_optimization_example() {
    std::cout << "=== Optimization with Rational Numbers ===\n\n";

    using Rat = rational<int64_t>;

    // Quadratic with rational coefficients
    auto rat_func = [](Rat x) {
        return x * x - Rat(4, 1) * x + Rat(3, 1); // x^2 - 4x + 3
    };

    std::cout << "Finding minimum of x^2 - 4x + 3 using rational arithmetic:\n\n";

    // Simple bisection for derivative root
    auto derivative = [](Rat x) {
        return Rat(2, 1) * x - Rat(4, 1); // 2x - 4
    };

    Rat a(-10, 1), b(10, 1);

    for (int iter = 0; iter < 20; ++iter) {
        Rat mid = (a + b) / Rat(2, 1);
        Rat f_mid = derivative(mid);

        if (f_mid == Rat(0, 1)) {
            std::cout << "Exact minimum at x = "
                      << mid.numerator() << "/" << mid.denominator()
                      << " = " << double(mid.numerator()) / mid.denominator() << "\n";
            Rat min_value = rat_func(mid);
            std::cout << "Minimum value = "
                      << min_value.numerator() << "/" << min_value.denominator()
                      << " = " << double(min_value.numerator()) / min_value.denominator() << "\n";
            break;
        }

        if (f_mid < Rat(0, 1)) {
            a = mid;
        } else {
            b = mid;
        }
    }

    std::cout << "\n";
}

int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "================================================\n";
    std::cout << "    Stepanov Optimization Module Examples      \n";
    std::cout << "================================================\n\n";

    polynomial_optimization_example();
    gradient_descent_example();
    root_finding_example();
    simulated_annealing_example();
    rational_optimization_example();

    std::cout << "================================================\n";
    std::cout << "         All examples completed!                \n";
    std::cout << "================================================\n";

    return 0;
}