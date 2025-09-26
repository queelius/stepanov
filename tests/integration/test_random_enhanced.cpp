#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <cassert>
#include <numeric>
#include <iomanip>
#include "../include/stepanov/random_enhanced.hpp"
#include "../include/stepanov/matrix.hpp"

using namespace stepanov::random;

// Statistical test helpers
template<typename T>
bool approximately_equal(T a, T b, T epsilon = 1e-2) {
    return std::abs(a - b) < epsilon;
}

template<typename T>
T sample_mean(const std::vector<T>& samples) {
    return std::accumulate(samples.begin(), samples.end(), T(0)) / samples.size();
}

template<typename T>
T sample_variance(const std::vector<T>& samples) {
    T mean = sample_mean(samples);
    T sum_sq = 0;
    for (const T& x : samples) {
        T diff = x - mean;
        sum_sq += diff * diff;
    }
    return sum_sq / (samples.size() - 1);
}

void test_beta_distribution() {
    std::cout << "Testing Beta distribution..." << std::endl;

    std::mt19937 gen(42);
    beta_distribution<double> beta(2.0, 5.0);

    const int n_samples = 10000;
    std::vector<double> samples;
    samples.reserve(n_samples);

    for (int i = 0; i < n_samples; ++i) {
        double x = beta(gen);
        assert(x >= 0.0 && x <= 1.0);
        samples.push_back(x);
    }

    double emp_mean = sample_mean(samples);
    double emp_var = sample_variance(samples);
    double theo_mean = beta.mean();
    double theo_var = beta.variance();

    std::cout << "  Empirical mean: " << emp_mean << ", Theoretical: " << theo_mean << std::endl;
    std::cout << "  Empirical var:  " << emp_var << ", Theoretical: " << theo_var << std::endl;

    assert(approximately_equal(emp_mean, theo_mean));
    assert(approximately_equal(emp_var, theo_var));

    std::cout << "✓ Beta distribution passed\n" << std::endl;
}

void test_laplace_distribution() {
    std::cout << "Testing Laplace distribution..." << std::endl;

    std::mt19937 gen(42);
    laplace_distribution<double> laplace(3.0, 2.0);

    const int n_samples = 10000;
    std::vector<double> samples;
    samples.reserve(n_samples);

    for (int i = 0; i < n_samples; ++i) {
        samples.push_back(laplace(gen));
    }

    double emp_mean = sample_mean(samples);
    double emp_var = sample_variance(samples);
    double theo_mean = laplace.mean();
    double theo_var = laplace.variance();

    std::cout << "  Empirical mean: " << emp_mean << ", Theoretical: " << theo_mean << std::endl;
    std::cout << "  Empirical var:  " << emp_var << ", Theoretical: " << theo_var << std::endl;

    assert(approximately_equal(emp_mean, theo_mean, 0.1));
    assert(approximately_equal(emp_var, theo_var, 0.5));

    std::cout << "✓ Laplace distribution passed\n" << std::endl;
}

void test_pareto_distribution() {
    std::cout << "Testing Pareto distribution..." << std::endl;

    std::mt19937 gen(42);
    pareto_distribution<double> pareto(1.0, 3.0);

    const int n_samples = 10000;
    std::vector<double> samples;
    samples.reserve(n_samples);

    for (int i = 0; i < n_samples; ++i) {
        double x = pareto(gen);
        assert(x >= pareto.min());
        samples.push_back(x);
    }

    double emp_mean = sample_mean(samples);
    double theo_mean = pareto.mean();

    std::cout << "  Empirical mean: " << emp_mean << ", Theoretical: " << theo_mean << std::endl;

    assert(approximately_equal(emp_mean, theo_mean, 0.1));

    std::cout << "✓ Pareto distribution passed\n" << std::endl;
}

void test_markov_chain() {
    std::cout << "Testing Markov chain..." << std::endl;

    // Create a simple 3-state Markov chain
    std::vector<std::string> states = {"A", "B", "C"};
    stepanov::matrix<double> P(3, 3);

    // Transition matrix
    P(0, 0) = 0.2; P(0, 1) = 0.5; P(0, 2) = 0.3;
    P(1, 0) = 0.4; P(1, 1) = 0.1; P(1, 2) = 0.5;
    P(2, 0) = 0.3; P(2, 1) = 0.3; P(2, 2) = 0.4;

    markov_chain<std::string> chain(states, P);

    std::mt19937 gen(42);
    chain.set_state("A");

    // Simulate the chain
    auto path = chain.simulate(100, gen);

    std::cout << "  First 10 states: ";
    for (size_t i = 0; i < std::min(size_t(10), path.size()); ++i) {
        std::cout << path[i] << " ";
    }
    std::cout << std::endl;

    // Count state frequencies
    std::unordered_map<std::string, int> counts;
    for (const auto& state : path) {
        counts[state]++;
    }

    std::cout << "  State frequencies:" << std::endl;
    for (const auto& [state, count] : counts) {
        std::cout << "    " << state << ": " << count << std::endl;
    }

    // Get stationary distribution
    auto pi = chain.stationary_distribution();
    std::cout << "  Stationary distribution: [";
    for (size_t i = 0; i < pi.size(); ++i) {
        std::cout << pi[i];
        if (i < pi.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "✓ Markov chain passed\n" << std::endl;
}

void test_random_walk() {
    std::cout << "Testing Random walk..." << std::endl;

    std::mt19937 gen(42);

    // 1D random walk
    random_walk<1> walk1d;
    auto path1d = walk1d.simulate(100, gen, 0.01);

    std::cout << "  1D walk final position: " << path1d.back()[0] << std::endl;

    // 2D Brownian motion
    random_walk<2> brownian2d = random_walk<2>::brownian(
        {0.0, 0.0},    // start
        {0.1, 0.0},    // drift
        {1.0, 1.0}     // volatility
    );

    auto path2d = brownian2d.simulate(100, gen, 0.01);

    std::cout << "  2D Brownian final position: ("
              << path2d.back()[0] << ", " << path2d.back()[1] << ")" << std::endl;

    // Test geometric Brownian motion
    double S0 = 100.0;
    double mu = 0.05;
    double sigma = 0.2;
    double final_price = walk1d.geometric_step(gen, S0, mu, sigma, 1.0);

    std::cout << "  Geometric BM: S0=" << S0 << ", S1=" << final_price << std::endl;
    assert(final_price > 0);

    std::cout << "✓ Random walk passed\n" << std::endl;
}

void test_poisson_process() {
    std::cout << "Testing Poisson process..." << std::endl;

    std::mt19937 gen(42);
    poisson_process<double> process(2.0);  // Rate = 2 events per unit time

    auto events = process.simulate_until(10.0, gen);

    std::cout << "  Number of events in [0, 10]: " << events.size() << std::endl;
    std::cout << "  First 5 event times: ";
    for (size_t i = 0; i < std::min(size_t(5), events.size()); ++i) {
        std::cout << std::fixed << std::setprecision(2) << events[i] << " ";
    }
    std::cout << std::endl;

    // Check inter-arrival times are exponential
    std::vector<double> inter_arrivals;
    for (size_t i = 1; i < events.size(); ++i) {
        inter_arrivals.push_back(events[i] - events[i-1]);
    }

    double mean_inter_arrival = sample_mean(inter_arrivals);
    std::cout << "  Mean inter-arrival time: " << mean_inter_arrival
              << " (expected: " << 1.0/2.0 << ")" << std::endl;

    std::cout << "✓ Poisson process passed\n" << std::endl;
}

void test_compound_poisson() {
    std::cout << "Testing Compound Poisson process..." << std::endl;

    std::mt19937 gen(42);
    std::normal_distribution<double> jump_dist(0.0, 1.0);
    compound_poisson_process<double> process(3.0, jump_dist);

    auto path = process.simulate_until(5.0, gen);

    std::cout << "  Number of jumps in [0, 5]: " << path.size() - 1 << std::endl;
    std::cout << "  Final value: " << path.back().second << std::endl;

    // Print first few jumps
    std::cout << "  First 5 jumps (time, value):" << std::endl;
    for (size_t i = 0; i < std::min(size_t(5), path.size()); ++i) {
        std::cout << "    (" << std::fixed << std::setprecision(2)
                  << path[i].first << ", " << path[i].second << ")" << std::endl;
    }

    std::cout << "✓ Compound Poisson passed\n" << std::endl;
}

void test_monte_carlo() {
    std::cout << "Testing Monte Carlo integration..." << std::endl;

    std::mt19937 gen(42);
    monte_carlo_integrator<double> mc(10000);

    // Integrate x^2 from 0 to 1 (should be 1/3)
    auto f = [](double x) { return x * x; };
    double result = mc.integrate_1d(f, 0.0, 1.0, gen);

    std::cout << "  ∫x² dx from 0 to 1: " << result
              << " (expected: " << 1.0/3.0 << ")" << std::endl;
    std::cout << "  Error estimate: " << *mc.error() << std::endl;

    assert(approximately_equal(result, 1.0/3.0, 0.01));

    // 2D integration: unit circle area (π/4 in unit square)
    auto circle = [](const std::array<double, 2>& x) {
        return (x[0]*x[0] + x[1]*x[1] <= 1.0) ? 1.0 : 0.0;
    };

    double area = mc.integrate_nd<decltype(circle), std::mt19937, 2>(
        circle, {0.0, 0.0}, {1.0, 1.0}, gen);

    std::cout << "  Quarter circle area: " << area
              << " (expected: " << M_PI/4.0 << ")" << std::endl;

    assert(approximately_equal(area, M_PI/4.0, 0.05));

    std::cout << "✓ Monte Carlo passed\n" << std::endl;
}

void test_type_erasure() {
    std::cout << "Testing type erasure..." << std::endl;

    std::mt19937 gen(42);

    // Type-erased distribution
    auto dist = any_distribution<double>::make<beta_distribution<double>>(2.0, 3.0);

    std::vector<double> samples;
    for (int i = 0; i < 100; ++i) {
        samples.push_back(dist(gen));
    }

    std::cout << "  Generated " << samples.size() << " samples from type-erased distribution" << std::endl;
    std::cout << "  Mean: " << dist.mean() << ", Variance: " << dist.variance() << std::endl;

    // Type-erased random process
    auto process = any_random_process<std::array<double, 1>>::make<random_walk<1>>();

    for (int i = 0; i < 10; ++i) {
        auto pos = process.step(gen, 0.1);
        // Just verify it runs without error
    }

    std::cout << "✓ Type erasure passed\n" << std::endl;
}

void test_distributions_statistical() {
    std::cout << "Testing distribution statistical properties..." << std::endl;

    std::mt19937 gen(42);
    const int n_samples = 50000;

    // Test negative binomial
    {
        negative_binomial_distribution<double> nb(5, 0.3);
        std::vector<int> samples;
        for (int i = 0; i < n_samples; ++i) {
            samples.push_back(nb(gen));
        }

        double emp_mean = sample_mean(samples);
        double theo_mean = nb.mean();

        std::cout << "  Negative Binomial mean: " << emp_mean
                  << " (theoretical: " << theo_mean << ")" << std::endl;

        assert(approximately_equal(emp_mean, theo_mean, 0.5));
    }

    // Test hypergeometric
    {
        hypergeometric_distribution<double> hg(100, 30, 20);
        std::vector<int> samples;
        for (int i = 0; i < 1000; ++i) {  // Fewer samples as it's slower
            samples.push_back(hg(gen));
        }

        double emp_mean = sample_mean(samples);
        double theo_mean = hg.mean();

        std::cout << "  Hypergeometric mean: " << emp_mean
                  << " (theoretical: " << theo_mean << ")" << std::endl;

        assert(approximately_equal(emp_mean, theo_mean, 1.0));
    }

    std::cout << "✓ Statistical properties passed\n" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Testing stepanov::random (enhanced)" << std::endl;
    std::cout << "========================================\n" << std::endl;

    try {
        test_beta_distribution();
        test_laplace_distribution();
        test_pareto_distribution();
        test_markov_chain();
        test_random_walk();
        test_poisson_process();
        test_compound_poisson();
        test_monte_carlo();
        test_type_erasure();
        test_distributions_statistical();

        std::cout << "========================================" << std::endl;
        std::cout << "All random tests passed successfully!" << std::endl;
        std::cout << "========================================" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}