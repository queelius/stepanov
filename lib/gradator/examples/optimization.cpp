// optimization.cpp - Optimization Examples with gradator
//
// This example demonstrates using gradator for numerical optimization,
// including minimizing classic test functions like Rosenbrock.

#include <gradator.hpp>
#include <iostream>
#include <iomanip>

using namespace gradator;
using namespace elementa;

// Rosenbrock function: f(x,y) = (a-x)^2 + b*(y-x^2)^2
// Classic test function with a=1, b=100
// Minimum at (1, 1) with f(1,1) = 0
double rosenbrock(const matrix<double>& x, double a = 1.0, double b = 100.0) {
    double x1 = x(0, 0);
    double x2 = x(1, 0);
    return (a - x1) * (a - x1) + b * (x2 - x1 * x1) * (x2 - x1 * x1);
}

// Gradient of Rosenbrock
// df/dx1 = -2(a-x1) - 4*b*x1*(x2-x1^2)
// df/dx2 = 2*b*(x2-x1^2)
matrix<double> rosenbrock_grad(const matrix<double>& x, double a = 1.0, double b = 100.0) {
    double x1 = x(0, 0);
    double x2 = x(1, 0);
    matrix<double> grad(2, 1);
    grad(0, 0) = -2 * (a - x1) - 4 * b * x1 * (x2 - x1 * x1);
    grad(1, 0) = 2 * b * (x2 - x1 * x1);
    return grad;
}

// Quadratic function: f(x) = 0.5 * x^T * A * x - b^T * x
// Minimum at A^{-1} * b
double quadratic(const matrix<double>& x, const matrix<double>& A, const matrix<double>& b) {
    auto Ax = elementa::matmul(A, x);
    auto xAx = elementa::sum(elementa::hadamard(x, Ax));
    auto bx = elementa::sum(elementa::hadamard(b, x));
    return 0.5 * xAx - bx;
}

// Gradient: A*x - b
matrix<double> quadratic_grad(const matrix<double>& x, const matrix<double>& A, const matrix<double>& b) {
    return elementa::matmul(A, x) - b;
}

int main() {
    std::cout << std::fixed << std::setprecision(6);

    // =========================================================================
    // Example 1: Quadratic minimization with gradator
    // =========================================================================
    std::cout << "=== Example 1: Quadratic Function ===\n\n";

    // A is positive definite
    matrix<double> A{{4, 1}, {1, 3}};
    matrix<double> b{{1}, {2}};

    // Optimal solution: x* = A^{-1} * b
    auto x_opt = elementa::solve(A, b);
    std::cout << "Optimal solution: x* = [" << x_opt(0, 0) << ", " << x_opt(1, 0) << "]\n";
    std::cout << "Optimal value: f(x*) = " << quadratic(x_opt, A, b) << "\n\n";

    // Using gradator to compute gradient
    // For f(x) = 0.5 x^T A x - b^T x, gradient is Ax - b
    // (Simplified version using analytical gradient below)

    // Simpler: just use the analytical gradient
    auto grad_quad = [&A, &b](const matrix<double>& x) {
        return elementa::matmul(A, x) - b;
    };

    // Gradient descent
    matrix<double> x{{0}, {0}};  // Initial point
    double lr = 0.1;
    int iters = 50;

    std::cout << "Gradient descent (lr=" << lr << "):\n";
    for (int i = 0; i < iters; ++i) {
        auto g = grad_quad(x);
        x = x - g * lr;

        if (i < 5 || (i + 1) % 10 == 0) {
            std::cout << "  Iter " << std::setw(2) << (i + 1)
                      << ": x = [" << x(0, 0) << ", " << x(1, 0) << "]"
                      << ", f(x) = " << quadratic(x, A, b) << "\n";
        }
    }

    std::cout << "\nFinal: x = [" << x(0, 0) << ", " << x(1, 0) << "]\n";
    std::cout << "Error: ||x - x*|| = " << elementa::frobenius_norm(x - x_opt) << "\n\n";

    // =========================================================================
    // Example 2: Rosenbrock function (harder optimization)
    // =========================================================================
    std::cout << "=== Example 2: Rosenbrock Function ===\n\n";

    std::cout << "f(x,y) = (1-x)^2 + 100*(y-x^2)^2\n";
    std::cout << "Minimum at (1, 1) with f(1,1) = 0\n\n";

    // Verify gradient using finite differences
    std::cout << "Verifying gradient formula using finite differences:\n";
    matrix<double> test_point{{0.5}, {0.5}};
    auto rosenbrock_simple = [](const matrix<double>& x) { return rosenbrock(x); };
    auto numerical_grad = finite_diff_gradient(rosenbrock_simple, test_point, 1e-7);
    auto analytical_grad = rosenbrock_grad(test_point);
    std::cout << "  At point [0.5, 0.5]:\n";
    std::cout << "  Analytical: [" << analytical_grad(0, 0) << ", " << analytical_grad(1, 0) << "]\n";
    std::cout << "  Numerical:  [" << numerical_grad(0, 0) << ", " << numerical_grad(1, 0) << "]\n\n";

    // Gradient descent with smaller learning rate (Rosenbrock is ill-conditioned)
    matrix<double> r{{-1.5}, {2.0}};  // Starting point
    lr = 0.001;
    iters = 10000;

    std::cout << "Gradient descent (lr=" << lr << ", " << iters << " iterations):\n";
    std::cout << "  Initial: x = [" << r(0, 0) << ", " << r(1, 0) << "]"
              << ", f(x) = " << rosenbrock(r) << "\n";

    for (int i = 0; i < iters; ++i) {
        auto g = rosenbrock_grad(r);
        r = r - g * lr;

        if ((i + 1) == 100 || (i + 1) == 1000 || (i + 1) == 5000 || (i + 1) == iters) {
            std::cout << "  Iter " << std::setw(5) << (i + 1)
                      << ": x = [" << r(0, 0) << ", " << r(1, 0) << "]"
                      << ", f(x) = " << rosenbrock(r) << "\n";
        }
    }

    std::cout << "\nFinal: x = [" << r(0, 0) << ", " << r(1, 0) << "]\n";
    std::cout << "Distance from optimum: " << std::sqrt((r(0, 0) - 1) * (r(0, 0) - 1) +
                                                        (r(1, 0) - 1) * (r(1, 0) - 1)) << "\n\n";

    // =========================================================================
    // Example 3: Newton's method with Hessian
    // =========================================================================
    std::cout << "=== Example 3: Newton's Method ===\n\n";

    // For the quadratic, Newton converges in one step
    // Update: x_{k+1} = x_k - H^{-1} * grad

    matrix<double> x_newton{{0}, {0}};

    std::cout << "Newton's method on quadratic (should converge in 1 step):\n";
    std::cout << "  Initial: x = [" << x_newton(0, 0) << ", " << x_newton(1, 0) << "]\n";

    // For quadratic f(x) = 0.5*x^T*A*x - b^T*x:
    // grad = A*x - b
    // Hessian = A
    // Newton step: x_new = x - A^{-1}*(A*x - b) = A^{-1}*b

    auto g = quadratic_grad(x_newton, A, b);
    auto step = elementa::solve(A, g);
    x_newton = x_newton - step;

    std::cout << "  After 1 Newton step: x = [" << x_newton(0, 0) << ", " << x_newton(1, 0) << "]\n";
    std::cout << "  f(x) = " << quadratic(x_newton, A, b) << "\n";
    std::cout << "  (Compare to optimal: [" << x_opt(0, 0) << ", " << x_opt(1, 0) << "])\n\n";

    // =========================================================================
    // Example 4: Using gradator's sum of squares example
    // =========================================================================
    std::cout << "=== Example 4: Gradator sum of squares ===\n\n";

    // f(x) = sum(x^2) - simple quadratic
    auto sum_sq = [](const auto& x) {
        return sum(pow(x, 2.0));
    };

    auto grad_sum_sq = grad(sum_sq);

    matrix<double> v{{3}, {4}};
    auto computed_grad = grad_sum_sq(v);

    std::cout << "f(x) = sum(x^2)\n";
    std::cout << "At x = [3, 4]:\n";
    std::cout << "  Computed gradient: [" << computed_grad(0, 0) << ", " << computed_grad(1, 0) << "]\n";
    std::cout << "  Expected (2*x):    [6, 8]\n";

    return 0;
}
