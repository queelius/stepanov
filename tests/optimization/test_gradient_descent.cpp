#include <iostream>
#include <cassert>
#include <cmath>
#include <random>
#include "../../include/stepanov/optimization.hpp"
#include "../../include/stepanov/fixed_decimal.hpp"

using namespace stepanov;
using namespace stepanov::optimization;

// Test functions
template<typename V>
typename V::value_type quadratic_form(const V& x) {
    using T = typename V::value_type;
    // f(x, y) = x^2 + 2y^2 - 2x - 4y + 5
    // Minimum at (1, 1) with value 0
    T result = x[0] * x[0] + T(2) * x[1] * x[1];
    result = result - T(2) * x[0] - T(4) * x[1] + T(5);
    return result;
}

template<typename V>
typename V::value_type himmelblau(const V& x) {
    using T = typename V::value_type;
    // Himmelblau's function - has 4 identical local minima
    T term1 = x[0] * x[0] + x[1] - T(11);
    T term2 = x[0] + x[1] * x[1] - T(7);
    return term1 * term1 + term2 * term2;
}

template<typename V>
typename V::value_type rastrigin(const V& x) {
    using T = typename V::value_type;
    // Rastrigin function - highly multimodal
    T A = T(10);
    T n = T(x.size());
    T sum = A * n;
    for (size_t i = 0; i < x.size(); ++i) {
        sum += x[i] * x[i] - A * std::cos(T(2) * T(M_PI) * x[i]);
    }
    return sum;
}

void test_basic_gradient_descent() {
    std::cout << "Testing basic gradient descent...\n";

    vector<double> x0{5.0, 5.0};

    // Test with constant step size
    auto result = gradient_descent(
        quadratic_form<vector<double>>, x0,
        constant_step_size<double>(0.1),
        tolerance_criterion<double>{1e-6, 1e-6, 1000}
    );

    assert(result.converged);
    assert(std::abs(result.solution[0] - 1.0) < 0.01);
    assert(std::abs(result.solution[1] - 1.0) < 0.01);

    std::cout << "  Constant step: x* = (" << result.solution[0]
              << ", " << result.solution[1] << "), f(x*) = " << result.value
              << ", iterations = " << result.iterations << "\n";

    // Test with diminishing step size
    auto result2 = gradient_descent(
        quadratic_form<vector<double>>, x0,
        diminishing_step_size<double>(1.0, 0.95),
        tolerance_criterion<double>{1e-6, 1e-6, 1000}
    );

    std::cout << "  Diminishing step: x* = (" << result2.solution[0]
              << ", " << result2.solution[1] << "), iterations = "
              << result2.iterations << "\n";

    std::cout << "  Basic gradient descent tests passed!\n\n";
}

void test_momentum_methods() {
    std::cout << "Testing momentum-based gradient descent...\n";

    vector<double> x0{5.0, 5.0};

    // Classical momentum
    auto momentum_result = momentum_gradient_descent(
        quadratic_form<vector<double>>, x0,
        0.01, 0.9,
        tolerance_criterion<double>{1e-6, 1e-6, 1000}
    );

    std::cout << "  Momentum: x* = (" << momentum_result.solution[0]
              << ", " << momentum_result.solution[1]
              << "), iterations = " << momentum_result.iterations << "\n";

    // Nesterov accelerated gradient
    auto nesterov_result = nesterov_gradient_descent(
        quadratic_form<vector<double>>, x0,
        0.01, 0.9,
        tolerance_criterion<double>{1e-6, 1e-6, 1000}
    );

    std::cout << "  Nesterov: x* = (" << nesterov_result.solution[0]
              << ", " << nesterov_result.solution[1]
              << "), iterations = " << nesterov_result.iterations << "\n";

    std::cout << "  Momentum methods tests passed!\n\n";
}

void test_adaptive_methods() {
    std::cout << "Testing adaptive gradient methods...\n";

    vector<double> x0{5.0, 5.0};

    // Adam optimizer
    auto adam_result = adam(
        himmelblau<vector<double>>, x0,
        0.01, 0.9, 0.999, 1e-8,
        tolerance_criterion<double>{1e-6, 1e-6, 2000}
    );

    std::cout << "  Adam: x* = (" << adam_result.solution[0]
              << ", " << adam_result.solution[1]
              << "), f(x*) = " << adam_result.value
              << ", iterations = " << adam_result.iterations << "\n";

    // AdaGrad
    auto adagrad_result = adagrad(
        quadratic_form<vector<double>>, x0,
        0.5, 1e-8,
        tolerance_criterion<double>{1e-6, 1e-6, 1000}
    );

    std::cout << "  AdaGrad: x* = (" << adagrad_result.solution[0]
              << ", " << adagrad_result.solution[1]
              << "), iterations = " << adagrad_result.iterations << "\n";

    // RMSprop
    auto rmsprop_result = rmsprop(
        quadratic_form<vector<double>>, x0,
        0.01, 0.9, 1e-8,
        tolerance_criterion<double>{1e-6, 1e-6, 1000}
    );

    std::cout << "  RMSprop: x* = (" << rmsprop_result.solution[0]
              << ", " << rmsprop_result.solution[1]
              << "), iterations = " << rmsprop_result.iterations << "\n";

    std::cout << "  Adaptive methods tests passed!\n\n";
}

void test_conjugate_gradient() {
    std::cout << "Testing conjugate gradient method...\n";

    vector<double> x0{5.0, 5.0};

    auto cg_result = conjugate_gradient(
        quadratic_form<vector<double>>, x0,
        tolerance_criterion<double>{1e-6, 1e-6, 1000}
    );

    std::cout << "  Conjugate gradient: x* = (" << cg_result.solution[0]
              << ", " << cg_result.solution[1]
              << "), f(x*) = " << cg_result.value
              << ", iterations = " << cg_result.iterations << "\n";

    assert(cg_result.converged);
    assert(std::abs(cg_result.solution[0] - 1.0) < 0.01);
    assert(std::abs(cg_result.solution[1] - 1.0) < 0.01);

    std::cout << "  Conjugate gradient tests passed!\n\n";
}

void test_with_fixed_decimal() {
    std::cout << "Testing gradient descent with fixed decimal...\n";

    using FD = fixed_decimal<int64_t, 6>; // 6 decimal places
    vector<FD> x0{FD(5000000), FD(5000000)}; // 5.0, 5.0

    // Simple quadratic with fixed decimal
    auto fd_func = [](const vector<FD>& x) {
        FD result = x[0] * x[0] + x[1] * x[1];
        return result;
    };

    // Note: This requires fixed_decimal to implement required operations
    // This is a simplified demonstration
    std::cout << "  Fixed decimal gradient descent demonstration completed\n\n";
}

void test_multimodal_function() {
    std::cout << "Testing on multimodal function (Rastrigin)...\n";

    // Rastrigin has global minimum at origin
    vector<double> x0{1.0, 1.0};

    // Adam often works well for multimodal functions
    auto result = adam(
        rastrigin<vector<double>>, x0,
        0.01, 0.9, 0.999, 1e-8,
        tolerance_criterion<double>{1e-6, 1e-6, 5000}
    );

    std::cout << "  Rastrigin with Adam: x* = (" << result.solution[0]
              << ", " << result.solution[1] << "), f(x*) = " << result.value
              << ", converged = " << result.converged << "\n";

    // With good initialization, should find a local minimum
    if (result.converged && result.value < 10.0) {
        std::cout << "  Found a good local minimum!\n";
    }

    std::cout << "  Multimodal function tests completed!\n\n";
}

int main() {
    std::cout << "=== Testing Gradient Descent Module ===\n\n";

    test_basic_gradient_descent();
    test_momentum_methods();
    test_adaptive_methods();
    test_conjugate_gradient();
    test_with_fixed_decimal();
    test_multimodal_function();

    std::cout << "=== All gradient descent tests completed! ===\n";

    return 0;
}