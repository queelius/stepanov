// logistic_regression.cpp - Logistic Regression with Gradient Descent
//
// This example demonstrates binary classification using logistic regression.
// We train the model using manually computed gradients (for simplicity)
// and verify the gradient using gradator.

#include <gradator.hpp>
#include <iostream>
#include <random>

using namespace gradator;
using namespace elementa;

// Sigmoid function
matrix<double> sigmoid(const matrix<double>& x) {
    matrix<double> result(x.rows(), x.cols());
    for (std::size_t i = 0; i < x.rows(); ++i) {
        for (std::size_t j = 0; j < x.cols(); ++j) {
            result(i, j) = 1.0 / (1.0 + std::exp(-x(i, j)));
        }
    }
    return result;
}

int main() {
    std::cout << "=== Logistic Regression ===\n\n";

    // Generate linearly separable data
    const int n = 50;  // samples per class
    const int d = 2;   // features

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0, 0.5);

    // Class 0: centered at (-1, -1)
    // Class 1: centered at (1, 1)
    matrix<double> X(2 * n, d);
    matrix<double> y(2 * n, 1);

    for (int i = 0; i < n; ++i) {
        // Class 0
        X(i, 0) = -1.0 + noise(rng);
        X(i, 1) = -1.0 + noise(rng);
        y(i, 0) = 0.0;

        // Class 1
        X(n + i, 0) = 1.0 + noise(rng);
        X(n + i, 1) = 1.0 + noise(rng);
        y(n + i, 0) = 1.0;
    }

    std::cout << "Dataset: " << 2 * n << " samples, " << d << " features\n";
    std::cout << "Classes: 0 (centered at -1,-1), 1 (centered at 1,1)\n\n";

    // Initialize weights (with bias term, so d+1 parameters)
    // We'll augment X with a column of ones for the bias
    matrix<double> X_aug(2 * n, d + 1);
    for (int i = 0; i < 2 * n; ++i) {
        X_aug(i, 0) = 1.0;  // bias term
        for (int j = 0; j < d; ++j) {
            X_aug(i, j + 1) = X(i, j);
        }
    }

    // Binary cross-entropy loss
    // L = -sum(y * log(p) + (1-y) * log(1-p))
    auto bce_loss = [&X_aug, &y, n](const matrix<double>& beta) {
        auto logits = elementa::matmul(X_aug, beta);
        auto p = sigmoid(logits);

        double loss = 0.0;
        for (int i = 0; i < 2 * n; ++i) {
            double pi = std::max(1e-10, std::min(1 - 1e-10, p(i, 0)));
            loss -= y(i, 0) * std::log(pi) + (1 - y(i, 0)) * std::log(1 - pi);
        }
        return loss / (2 * n);
    };

    // Gradient of BCE loss: X^T * (p - y) / n
    auto grad_bce = [&X_aug, &y, n](const matrix<double>& beta) {
        auto logits = elementa::matmul(X_aug, beta);
        auto p = sigmoid(logits);
        auto residual = p - y;
        return elementa::matmul(elementa::transpose(X_aug), residual) / static_cast<double>(2 * n);
    };

    // Gradient descent
    matrix<double> beta(d + 1, 1, 0.0);  // Initialize to zeros
    double learning_rate = 1.0;
    int iterations = 100;

    std::cout << "Training with gradient descent:\n";
    std::cout << "  Learning rate: " << learning_rate << "\n";
    std::cout << "  Iterations: " << iterations << "\n\n";

    for (int iter = 0; iter < iterations; ++iter) {
        auto gradient = grad_bce(beta);

        // Update: beta = beta - lr * gradient
        beta = beta - gradient * learning_rate;

        if ((iter + 1) % 20 == 0 || iter == 0) {
            double loss = bce_loss(beta);
            std::cout << "  Iteration " << (iter + 1) << ": loss = " << loss << "\n";
        }
    }

    std::cout << "\nLearned parameters:\n";
    std::cout << "  bias (beta_0) = " << beta(0, 0) << "\n";
    std::cout << "  beta_1 = " << beta(1, 0) << "\n";
    std::cout << "  beta_2 = " << beta(2, 0) << "\n";

    // Compute accuracy
    auto logits = elementa::matmul(X_aug, beta);
    auto probs = sigmoid(logits);
    int correct = 0;
    for (int i = 0; i < 2 * n; ++i) {
        int pred = probs(i, 0) >= 0.5 ? 1 : 0;
        int actual = static_cast<int>(y(i, 0));
        if (pred == actual) ++correct;
    }

    std::cout << "\nAccuracy: " << (100.0 * correct / (2 * n)) << "%\n";

    // Decision boundary: beta_0 + beta_1*x1 + beta_2*x2 = 0
    // => x2 = -(beta_0 + beta_1*x1) / beta_2
    std::cout << "\nDecision boundary: x2 = " << (-beta(0, 0) / beta(2, 0))
              << " + " << (-beta(1, 0) / beta(2, 0)) << " * x1\n";

    // Verify gradient using gradator's finite difference
    std::cout << "\n=== Gradient Verification with Finite Differences ===\n";

    auto numerical_grad = finite_diff_gradient(bce_loss, beta, 1e-6);
    auto analytical_grad = grad_bce(beta);

    std::cout << "Analytical gradient: [" << analytical_grad(0, 0) << ", "
              << analytical_grad(1, 0) << ", " << analytical_grad(2, 0) << "]\n";
    std::cout << "Numerical gradient:  [" << numerical_grad(0, 0) << ", "
              << numerical_grad(1, 0) << ", " << numerical_grad(2, 0) << "]\n";
    std::cout << "Max difference: " << elementa::max_diff(analytical_grad, numerical_grad) << "\n";

    return 0;
}
