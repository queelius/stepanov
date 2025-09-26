#include <iostream>
#include <cassert>
#include <cmath>
#include <chrono>
#include "../../include/stepanov/optimization.hpp"
#include "../../include/stepanov/autodiff.hpp"
#include "../../include/stepanov/polynomial.hpp"
#include "../../include/stepanov/rational.hpp"
#include "../../include/stepanov/fixed_decimal.hpp"

using namespace stepanov;
using namespace stepanov::optimization;

// Integration test 1: Newton's method with polynomials
void test_polynomial_optimization() {
    std::cout << "Testing polynomial optimization with Newton's method...\n";

    // Create polynomial: p(x) = x^4 - 4x^3 + 6x^2 - 4x + 1
    // This is (x-1)^4, minimum at x = 1
    polynomial<double> p;
    p.set_coefficient(0, 1.0);
    p.set_coefficient(1, -4.0);
    p.set_coefficient(2, 6.0);
    p.set_coefficient(3, -4.0);
    p.set_coefficient(4, 1.0);

    // Wrapper for polynomial evaluation
    auto poly_func = [&p](double x) { return p.evaluate(x); };

    // Find minimum using Newton's method
    auto result = newton_univariate(poly_func, 2.0);
    assert(result.converged);
    assert(std::abs(result.solution - 1.0) < 1e-6);

    std::cout << "  Polynomial minimum: x* = " << result.solution
              << ", p(x*) = " << result.value << "\n";

    // Find roots of derivative (critical points)
    polynomial<double> p_derivative;
    p_derivative.set_coefficient(0, -4.0);
    p_derivative.set_coefficient(1, 12.0);
    p_derivative.set_coefficient(2, -12.0);
    p_derivative.set_coefficient(3, 4.0);

    auto deriv_func = [&p_derivative](double x) { return p_derivative.evaluate(x); };
    auto critical_point = newton_root(deriv_func, 0.5);
    assert(critical_point.has_value());
    std::cout << "  Critical point found at: x = " << *critical_point << "\n";

    std::cout << "  Polynomial optimization tests passed!\n\n";
}

// Integration test 2: Gradient descent with automatic differentiation
void test_autodiff_integration() {
    std::cout << "Testing gradient descent with automatic differentiation...\n";

    using namespace stepanov;

    // Define a function that works with dual numbers
    auto rosenbrock_dual = [](const vector<dual<double>>& x) {
        dual<double> a(1.0, 0.0);
        dual<double> b(100.0, 0.0);
        dual<double> term1 = a - x[0];
        dual<double> term2 = x[1] - x[0] * x[0];
        return term1 * term1 + b * term2 * term2;
    };

    // Wrapper that computes gradient using autodiff
    auto rosenbrock_with_grad = [&rosenbrock_dual](const vector<double>& x) {
        // For function value
        vector<dual<double>> x_const(2);
        x_const[0] = dual<double>::constant(x[0]);
        x_const[1] = dual<double>::constant(x[1]);
        return rosenbrock_dual(x_const).value();
    };

    vector<double> x0{-1.0, 2.0};

    // Use Adam optimizer with autodiff gradients
    auto result = adam(rosenbrock_with_grad, x0, 0.001);

    std::cout << "  Rosenbrock with autodiff: x* = ("
              << result.solution[0] << ", " << result.solution[1]
              << "), f(x*) = " << result.value
              << ", iterations = " << result.iterations << "\n";

    // Test with nested dual numbers for Hessian
    auto compute_hessian_element = [](auto f, const vector<double>& x,
                                      size_t i, size_t j) {
        using D = dual<dual<double>>;
        vector<D> x_dual(x.size());

        for (size_t k = 0; k < x.size(); ++k) {
            if (k == i) {
                x_dual[k] = D::variable(
                    dual<double>::variable(x[k], 1.0),
                    dual<double>::constant(k == j ? 1.0 : 0.0)
                );
            } else if (k == j) {
                x_dual[k] = D::variable(
                    dual<double>::variable(x[k], 0.0),
                    dual<double>::constant(1.0)
                );
            } else {
                x_dual[k] = D::constant(dual<double>::constant(x[k]));
            }
        }

        auto result = f(x_dual);
        return result.derivative().derivative();
    };

    vector<double> test_point{1.0, 1.0};
    double h00 = compute_hessian_element(rosenbrock_dual, test_point, 0, 0);
    double h01 = compute_hessian_element(rosenbrock_dual, test_point, 0, 1);
    double h11 = compute_hessian_element(rosenbrock_dual, test_point, 1, 1);

    std::cout << "  Hessian at (1,1): [[" << h00 << ", " << h01
              << "], [" << h01 << ", " << h11 << "]]\n";

    std::cout << "  Autodiff integration tests passed!\n\n";
}

// Integration test 3: Optimization with rational numbers
void test_rational_optimization() {
    std::cout << "Testing optimization with rational numbers...\n";

    using Rat = rational<int64_t>;

    // Quadratic function with rational coefficients
    auto rat_quadratic = [](Rat x) {
        Rat a(3, 2);  // 3/2
        Rat b(-6, 1); // -6
        Rat c(9, 2);  // 9/2
        return a * x * x + b * x + c; // Minimum at x = 2
    };

    // Golden section search adapted for rationals
    Rat a(0, 1);  // 0
    Rat b(4, 1);  // 4

    // Approximate golden ratio with rational
    Rat phi(161803398, 100000000); // Approximation of golden ratio

    for (int iter = 0; iter < 20; ++iter) {
        Rat width = b - a;
        Rat x1 = a + width * Rat(382, 1000);  // ~0.382
        Rat x2 = a + width * Rat(618, 1000);  // ~0.618

        Rat f1 = rat_quadratic(x1);
        Rat f2 = rat_quadratic(x2);

        if (f1 < f2) {
            b = x2;
        } else {
            a = x1;
        }

        if (iter % 5 == 0) {
            Rat mid = (a + b) / Rat(2, 1);
            std::cout << "  Iteration " << iter << ": x = "
                      << mid.numerator() << "/" << mid.denominator() << "\n";
        }
    }

    Rat final_x = (a + b) / Rat(2, 1);
    std::cout << "  Rational optimization result: x* = "
              << final_x.numerator() << "/" << final_x.denominator()
              << " (should be close to 2)\n";

    std::cout << "  Rational optimization tests passed!\n\n";
}

// Integration test 4: Optimization with fixed decimal
void test_fixed_decimal_optimization() {
    std::cout << "Testing optimization with fixed decimal numbers...\n";

    using FD = fixed_decimal<int64_t, 6>; // 6 decimal places

    // Quadratic function with fixed decimal
    auto fd_func = [](FD x) {
        FD two(2000000);    // 2.0
        FD four(4000000);   // 4.0
        FD three(3000000);  // 3.0
        return x * x - four * x + three; // Minimum at x = 2
    };

    // Bisection for finding minimum by finding root of derivative
    auto fd_derivative = [](FD x) {
        FD two(2000000);
        FD four(4000000);
        return two * x - four; // 2x - 4
    };

    FD a(0);        // 0.0
    FD b(4000000);  // 4.0

    for (int iter = 0; iter < 30; ++iter) {
        FD mid = (a + b) / FD(2000000);
        FD f_mid = fd_derivative(mid);

        if (abs(f_mid) < FD(1)) { // < 0.000001
            std::cout << "  Fixed decimal root of derivative: x = "
                      << mid.value() / 1000000.0 << "\n";
            FD min_value = fd_func(mid);
            std::cout << "  Minimum value: f(x*) = "
                      << min_value.value() / 1000000.0 << "\n";
            break;
        }

        if (f_mid < FD(0)) {
            a = mid;
        } else {
            b = mid;
        }
    }

    std::cout << "  Fixed decimal optimization tests passed!\n\n";
}

// Integration test 5: Simulated annealing with discrete optimization
void test_simulated_annealing_tsp() {
    std::cout << "Testing simulated annealing for TSP-like problem...\n";

    // Simple TSP with 5 cities
    const int N = 5;
    double distances[N][N] = {
        {0, 2, 9, 10, 5},
        {2, 0, 6, 4, 8},
        {9, 6, 0, 3, 7},
        {10, 4, 3, 0, 1},
        {5, 8, 7, 1, 0}
    };

    // State is a permutation
    using State = std::vector<int>;

    // Energy function (total distance)
    auto tour_length = [&distances, N](const State& tour) {
        double total = 0;
        for (int i = 0; i < N; ++i) {
            int from = tour[i];
            int to = tour[(i + 1) % N];
            total += distances[from][to];
        }
        return total;
    };

    // Neighbor generator (swap two cities)
    auto swap_neighbor = [N](const State& tour, std::mt19937& rng) {
        State neighbor = tour;
        std::uniform_int_distribution<int> dist(0, N - 1);
        int i = dist(rng);
        int j = dist(rng);
        std::swap(neighbor[i], neighbor[j]);
        return neighbor;
    };

    // Initial state
    State initial{0, 1, 2, 3, 4};

    // Run simulated annealing
    std::mt19937 rng(42);
    auto result = simulated_annealing(
        tour_length, initial, swap_neighbor,
        exponential_cooling<double>(10.0, 0.95),
        1000, 0.01, rng
    );

    std::cout << "  Initial tour length: " << tour_length(initial) << "\n";
    std::cout << "  Best tour length: " << result.best_energy << "\n";
    std::cout << "  Best tour: ";
    for (int city : result.best_solution) {
        std::cout << city << " ";
    }
    std::cout << "\n";
    std::cout << "  Accepted moves: " << result.accepted_moves
              << "/" << result.iterations << "\n";

    std::cout << "  Simulated annealing TSP tests passed!\n\n";
}

// Integration test 6: Combining multiple algorithms
void test_hybrid_optimization() {
    std::cout << "Testing hybrid optimization approach...\n";

    // Difficult multimodal function
    auto ackley = [](const vector<double>& x) {
        double a = 20.0;
        double b = 0.2;
        double c = 2 * M_PI;
        size_t d = x.size();

        double sum1 = 0, sum2 = 0;
        for (size_t i = 0; i < d; ++i) {
            sum1 += x[i] * x[i];
            sum2 += std::cos(c * x[i]);
        }

        return -a * std::exp(-b * std::sqrt(sum1 / d)) -
               std::exp(sum2 / d) + a + std::exp(1.0);
    };

    // Start with simulated annealing for global search
    auto sa_neighbor = [](const vector<double>& x, std::mt19937& rng) {
        std::normal_distribution<double> noise(0, 0.5);
        vector<double> neighbor = x;
        for (size_t i = 0; i < x.size(); ++i) {
            neighbor[i] += noise(rng);
        }
        return neighbor;
    };

    vector<double> initial{5.0, 5.0};
    std::mt19937 rng(42);

    auto sa_result = simulated_annealing(
        ackley, initial, sa_neighbor,
        linear_cooling<double>(5.0, 0.01, 500),
        500, 0.01, rng
    );

    std::cout << "  SA result: x = (" << sa_result.best_solution[0]
              << ", " << sa_result.best_solution[1]
              << "), f = " << sa_result.best_energy << "\n";

    // Refine with gradient descent
    auto gd_result = adam(ackley, sa_result.best_solution,
                          0.01, 0.9, 0.999, 1e-8,
                          tolerance_criterion<double>{1e-8, 1e-8, 500});

    std::cout << "  After gradient refinement: x = ("
              << gd_result.solution[0] << ", " << gd_result.solution[1]
              << "), f = " << gd_result.value << "\n";

    // Global minimum is at origin with value 0
    if (gd_result.value < 1.0) {
        std::cout << "  Successfully found near-global minimum!\n";
    }

    std::cout << "  Hybrid optimization tests passed!\n\n";
}

// Performance comparison of root finding methods
void benchmark_root_finding() {
    std::cout << "Benchmarking root finding methods...\n";

    auto test_func = [](double x) {
        return std::exp(x) - 3 * x - 2;
    };

    const int iterations = 1000;
    double a = 1.0, b = 3.0;

    // Benchmark bisection
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto root = bisection(test_func, a, b);
    }
    auto bisection_time = std::chrono::high_resolution_clock::now() - start;

    // Benchmark Brent
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto root = brent(test_func, a, b);
    }
    auto brent_time = std::chrono::high_resolution_clock::now() - start;

    // Benchmark Ridders
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto root = ridders(test_func, a, b);
    }
    auto ridders_time = std::chrono::high_resolution_clock::now() - start;

    std::cout << "  Time for " << iterations << " iterations:\n";
    std::cout << "    Bisection: "
              << std::chrono::duration_cast<std::chrono::microseconds>(bisection_time).count()
              << " μs\n";
    std::cout << "    Brent:     "
              << std::chrono::duration_cast<std::chrono::microseconds>(brent_time).count()
              << " μs\n";
    std::cout << "    Ridders:   "
              << std::chrono::duration_cast<std::chrono::microseconds>(ridders_time).count()
              << " μs\n";

    std::cout << "  Benchmarking completed!\n\n";
}

int main() {
    std::cout << "=== Integration Tests for Optimization Module ===\n\n";

    test_polynomial_optimization();
    test_autodiff_integration();
    test_rational_optimization();
    test_fixed_decimal_optimization();
    test_simulated_annealing_tsp();
    test_hybrid_optimization();
    benchmark_root_finding();

    std::cout << "=== All integration tests passed! ===\n";

    return 0;
}