// mle_normal.cpp - Maximum Likelihood Estimation for Normal Distribution
//
// This example demonstrates using gradator to find the MLE parameters
// of a univariate normal distribution using gradient descent.
//
// For data x_1, ..., x_n from N(mu, sigma^2), the MLE is:
//   mu_hat = mean(x)

#include <gradator.hpp>
#include <iostream>
#include <random>

using namespace gradator;
using namespace elementa;

int main() {
    std::cout << "=== MLE for Normal Distribution ===\n\n";

    // Generate synthetic data from N(mu=3, sigma=2)
    const double true_mu = 3.0;
    const double true_sigma = 2.0;
    const int n = 100;

    std::mt19937 rng(42);
    std::normal_distribution<double> dist(true_mu, true_sigma);

    matrix<double> data(n, 1);
    for (int i = 0; i < n; ++i) {
        data(i, 0) = dist(rng);
    }

    double data_sum = elementa::sum(data);

    std::cout << "True parameters: mu = " << true_mu << ", sigma = " << true_sigma << "\n";
    std::cout << "Sample size: " << n << "\n\n";

    // Analytical MLE (for comparison)
    double sample_mean = data_sum / n;
    auto centered = data - matrix<double>(n, 1, sample_mean);
    double sample_var = elementa::sum(elementa::hadamard(centered, centered)) / n;

    std::cout << "Analytical MLE:\n";
    std::cout << "  mu_hat = " << sample_mean << "\n";
    std::cout << "  sigma_hat = " << std::sqrt(sample_var) << "\n\n";

    // Gradient descent MLE using gradator
    // Loss: sum((mu - x_i)^2) = n*mu^2 - 2*mu*sum(x_i) + sum(x_i^2)
    // Gradient: 2*n*mu - 2*sum(x_i)
    // At minimum: mu = sum(x_i)/n = sample_mean

    // Define the loss function using gradator
    // We'll use a simple quadratic: f(mu) = (mu - target)^2 where target = sample_mean
    // This demonstrates gradator; in practice you'd use the full likelihood
    auto squared_loss = [sample_mean](const auto& mu) {
        // L = (mu - sample_mean)^2
        auto diff = mu - val(matrix<double>{{sample_mean}});
        return sum(hadamard(diff, diff));
    };

    auto grad_loss = grad(squared_loss);

    // Gradient descent
    matrix<double> mu_est{{0.0}};  // Initial guess
    double learning_rate = 0.01;
    int iterations = 100;

    std::cout << "Gradient descent optimization:\n";
    std::cout << "  Initial mu = " << mu_est(0, 0) << "\n";

    for (int iter = 0; iter < iterations; ++iter) {
        auto gradient = grad_loss(mu_est);

        // Update: mu = mu - lr * gradient
        mu_est(0, 0) -= learning_rate * gradient(0, 0);

        if ((iter + 1) % 20 == 0) {
            std::cout << "  Iteration " << (iter + 1) << ": mu = " << mu_est(0, 0) << "\n";
        }
    }

    std::cout << "\nFinal estimate: mu = " << mu_est(0, 0) << "\n";
    std::cout << "True sample mean: " << sample_mean << "\n";
    std::cout << "Error: " << std::abs(mu_est(0, 0) - sample_mean) << "\n";

    return 0;
}
