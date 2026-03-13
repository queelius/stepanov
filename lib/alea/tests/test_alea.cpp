// test_alea.cpp - Tests for alea probability distribution library
//
// Tests verify distributional properties using statistical methods:
// - Empirical moments converge to theoretical values
// - PDF/PMF integrate/sum correctly
// - Algebraic composition preserves structure
// - Multivariate normal works with elementa matrices

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <alea.hpp>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

using Catch::Approx;
using namespace alea;

// =============================================================================
// Helpers
// =============================================================================

constexpr int N_SAMPLES = 50000;
constexpr double STAT_TOL = 0.05;  // 5% relative tolerance for statistical tests
constexpr double PDF_TOL = 1e-6;

template <typename Dist, typename RNG>
auto sample_moments(const Dist& dist, RNG& rng, int n = N_SAMPLES)
    -> std::pair<double, double>
{
    double sum = 0, sum_sq = 0;
    for (int i = 0; i < n; ++i) {
        double x = dist.sample(rng);
        sum += x;
        sum_sq += x * x;
    }
    double mean = sum / n;
    double var = sum_sq / n - mean * mean;
    return {mean, var};
}

// =============================================================================
// Normal Distribution Tests
// =============================================================================

TEST_CASE("Normal distribution", "[alea][normal]") {
    std::mt19937 rng(42);

    SECTION("Standard normal: mean=0, variance=1") {
        normal<double> dist(0.0, 1.0);
        REQUIRE(dist.mean() == Approx(0.0));
        REQUIRE(dist.variance() == Approx(1.0));

        auto [emp_mean, emp_var] = sample_moments(dist, rng);
        REQUIRE(emp_mean == Approx(0.0).margin(STAT_TOL));
        REQUIRE(emp_var == Approx(1.0).epsilon(STAT_TOL));
    }

    SECTION("Non-standard normal: mean=5, sigma=2") {
        normal<double> dist(5.0, 2.0);
        REQUIRE(dist.mean() == Approx(5.0));
        REQUIRE(dist.variance() == Approx(4.0));

        auto [emp_mean, emp_var] = sample_moments(dist, rng);
        REQUIRE(emp_mean == Approx(5.0).epsilon(STAT_TOL));
        REQUIRE(emp_var == Approx(4.0).epsilon(STAT_TOL));
    }

    SECTION("PDF at mean") {
        normal<double> dist(0.0, 1.0);
        // PDF at 0 = 1/sqrt(2*pi) ≈ 0.3989
        REQUIRE(dist.pdf(0.0) == Approx(1.0 / std::sqrt(2.0 * M_PI)).epsilon(PDF_TOL));
    }

    SECTION("CDF symmetry") {
        normal<double> dist(0.0, 1.0);
        REQUIRE(dist.cdf(0.0) == Approx(0.5).epsilon(PDF_TOL));
        // CDF(-x) + CDF(x) = 1
        REQUIRE(dist.cdf(-1.0) + dist.cdf(1.0) == Approx(1.0).epsilon(PDF_TOL));
    }
}

// =============================================================================
// Uniform Distribution Tests
// =============================================================================

TEST_CASE("Uniform distribution", "[alea][uniform]") {
    std::mt19937 rng(123);

    SECTION("Standard uniform [0,1]") {
        uniform<double> dist(0.0, 1.0);
        REQUIRE(dist.mean() == Approx(0.5));
        REQUIRE(dist.variance() == Approx(1.0 / 12.0));

        auto [emp_mean, emp_var] = sample_moments(dist, rng);
        REQUIRE(emp_mean == Approx(0.5).epsilon(STAT_TOL));
        REQUIRE(emp_var == Approx(1.0 / 12.0).epsilon(STAT_TOL));
    }

    SECTION("PDF is constant") {
        uniform<double> dist(2.0, 5.0);
        REQUIRE(dist.pdf(3.0) == Approx(1.0 / 3.0));
        REQUIRE(dist.pdf(1.0) == Approx(0.0));  // outside support
        REQUIRE(dist.pdf(6.0) == Approx(0.0));
    }

    SECTION("Samples within bounds") {
        uniform<double> dist(-1.0, 1.0);
        for (int i = 0; i < 1000; ++i) {
            double x = dist.sample(rng);
            REQUIRE(x >= -1.0);
            REQUIRE(x <= 1.0);
        }
    }
}

// =============================================================================
// Exponential Distribution Tests
// =============================================================================

TEST_CASE("Exponential distribution", "[alea][exponential]") {
    std::mt19937 rng(456);

    SECTION("Moments") {
        exponential<double> dist(2.0);  // lambda = 2
        REQUIRE(dist.mean() == Approx(0.5));       // 1/lambda
        REQUIRE(dist.variance() == Approx(0.25));   // 1/lambda^2

        auto [emp_mean, emp_var] = sample_moments(dist, rng);
        REQUIRE(emp_mean == Approx(0.5).epsilon(STAT_TOL));
        REQUIRE(emp_var == Approx(0.25).epsilon(STAT_TOL));
    }

    SECTION("PDF and CDF") {
        exponential<double> dist(1.0);
        REQUIRE(dist.pdf(0.0) == Approx(1.0));
        REQUIRE(dist.cdf(0.0) == Approx(0.0));
        // CDF(1) = 1 - e^(-1)
        REQUIRE(dist.cdf(1.0) == Approx(1.0 - std::exp(-1.0)).epsilon(PDF_TOL));
    }

    SECTION("All samples non-negative") {
        exponential<double> dist(3.0);
        for (int i = 0; i < 1000; ++i) {
            REQUIRE(dist.sample(rng) >= 0.0);
        }
    }
}

// =============================================================================
// Bernoulli Distribution Tests
// =============================================================================

TEST_CASE("Bernoulli distribution", "[alea][bernoulli]") {
    std::mt19937 rng(789);

    SECTION("Fair coin") {
        bernoulli<double> dist(0.5);
        REQUIRE(dist.mean() == Approx(0.5));
        REQUIRE(dist.variance() == Approx(0.25));

        auto [emp_mean, emp_var] = sample_moments(dist, rng);
        REQUIRE(emp_mean == Approx(0.5).epsilon(STAT_TOL));
    }

    SECTION("PMF") {
        bernoulli<double> dist(0.3);
        REQUIRE(dist.pmf(0.0) == Approx(0.7));
        REQUIRE(dist.pmf(1.0) == Approx(0.3));
    }

    SECTION("Samples are 0 or 1") {
        bernoulli<double> dist(0.7);
        for (int i = 0; i < 1000; ++i) {
            double x = dist.sample(rng);
            REQUIRE((x == 0.0 || x == 1.0));
        }
    }
}

// =============================================================================
// Poisson Distribution Tests
// =============================================================================

TEST_CASE("Poisson distribution", "[alea][poisson]") {
    std::mt19937 rng(101);

    SECTION("Moments") {
        poisson<double> dist(5.0);  // lambda = 5
        REQUIRE(dist.mean() == Approx(5.0));
        REQUIRE(dist.variance() == Approx(5.0));  // mean == variance for Poisson

        auto [emp_mean, emp_var] = sample_moments(dist, rng);
        REQUIRE(emp_mean == Approx(5.0).epsilon(STAT_TOL));
        REQUIRE(emp_var == Approx(5.0).epsilon(STAT_TOL));
    }

    SECTION("PMF sums to ~1") {
        poisson<double> dist(3.0);
        double total = 0;
        for (int k = 0; k < 30; ++k) {
            total += dist.pmf(static_cast<double>(k));
        }
        REQUIRE(total == Approx(1.0).epsilon(1e-6));
    }

    SECTION("Samples are non-negative integers") {
        poisson<double> dist(2.0);
        for (int i = 0; i < 1000; ++i) {
            double x = dist.sample(rng);
            REQUIRE(x >= 0.0);
            REQUIRE(x == std::floor(x));  // integer-valued
        }
    }
}

// =============================================================================
// Beta Distribution Tests
// =============================================================================

TEST_CASE("Beta distribution", "[alea][beta]") {
    std::mt19937 rng(202);

    SECTION("Symmetric beta(2,2)") {
        beta<double> dist(2.0, 2.0);
        REQUIRE(dist.mean() == Approx(0.5));
        // Variance = alpha*beta / ((a+b)^2 * (a+b+1))
        REQUIRE(dist.variance() == Approx(1.0 / 20.0));

        auto [emp_mean, emp_var] = sample_moments(dist, rng);
        REQUIRE(emp_mean == Approx(0.5).epsilon(STAT_TOL));
        REQUIRE(emp_var == Approx(0.05).epsilon(STAT_TOL));
    }

    SECTION("Samples in [0,1]") {
        beta<double> dist(0.5, 0.5);
        for (int i = 0; i < 1000; ++i) {
            double x = dist.sample(rng);
            REQUIRE(x >= 0.0);
            REQUIRE(x <= 1.0);
        }
    }
}

// =============================================================================
// Algebraic Composition Tests
// =============================================================================

TEST_CASE("Random element composition", "[alea][composition]") {
    std::mt19937 rng(303);

    SECTION("Sum of two normals is normal") {
        // N(mu1, sigma1^2) + N(mu2, sigma2^2) = N(mu1+mu2, sigma1^2+sigma2^2)
        normal<double> d1(1.0, 1.0);
        normal<double> d2(2.0, 1.0);

        auto re1 = make_random_element<double>(d1);
        auto re2 = make_random_element<double>(d2);
        auto re_sum = re1 + re2;

        // Expected: mean = 3, variance = 2
        double sum_total = 0, sum_sq = 0;
        for (int i = 0; i < N_SAMPLES; ++i) {
            double x = re_sum.sample(rng);
            sum_total += x;
            sum_sq += x * x;
        }
        double emp_mean = sum_total / N_SAMPLES;
        double emp_var = sum_sq / N_SAMPLES - emp_mean * emp_mean;

        REQUIRE(emp_mean == Approx(3.0).epsilon(STAT_TOL));
        REQUIRE(emp_var == Approx(2.0).epsilon(STAT_TOL));
    }

    SECTION("Transform preserves sampling") {
        normal<double> dist(0.0, 1.0);
        auto squared = transform([](double x) { return x * x; }, dist);

        // E[X^2] = Var[X] + E[X]^2 = 1 + 0 = 1 for standard normal
        double sum_total = 0;
        for (int i = 0; i < N_SAMPLES; ++i) {
            sum_total += squared.sample(rng);
        }
        REQUIRE(sum_total / N_SAMPLES == Approx(1.0).epsilon(STAT_TOL));
    }

    SECTION("Scalar multiplication") {
        normal<double> dist(0.0, 1.0);
        auto re = make_random_element<double>(dist);
        auto scaled = 3.0 * re;

        // E[3X] = 0, Var[3X] = 9
        double sum_total = 0, sum_sq = 0;
        for (int i = 0; i < N_SAMPLES; ++i) {
            double x = scaled.sample(rng);
            sum_total += x;
            sum_sq += x * x;
        }
        double emp_mean = sum_total / N_SAMPLES;
        double emp_var = sum_sq / N_SAMPLES - emp_mean * emp_mean;

        REQUIRE(emp_mean == Approx(0.0).margin(STAT_TOL));
        REQUIRE(emp_var == Approx(9.0).epsilon(STAT_TOL));
    }
}

// =============================================================================
// Multivariate Normal Tests
// =============================================================================

TEST_CASE("Multivariate normal", "[alea][mvn]") {
    std::mt19937 rng(404);
    using mat = elementa::matrix<double>;

    SECTION("2D standard normal") {
        mat mu{{0}, {0}};
        mat sigma{{1, 0}, {0, 1}};

        multivariate_normal<double> dist(mu, sigma);

        // Sample and check empirical mean
        mat sum_samples(2, 1, 0.0);
        int n = 10000;
        for (int i = 0; i < n; ++i) {
            auto x = dist.sample(rng);
            for (std::size_t r = 0; r < 2; ++r) {
                sum_samples(r, 0) += x(r, 0);
            }
        }

        REQUIRE(sum_samples(0, 0) / n == Approx(0.0).margin(STAT_TOL));
        REQUIRE(sum_samples(1, 0) / n == Approx(0.0).margin(STAT_TOL));
    }

    SECTION("Correlated 2D normal") {
        mat mu{{1}, {2}};
        mat sigma{{4, 1}, {1, 2}};  // positive definite

        multivariate_normal<double> dist(mu, sigma);

        mat sum_samples(2, 1, 0.0);
        double sum_cov = 0;
        int n = 20000;
        for (int i = 0; i < n; ++i) {
            auto x = dist.sample(rng);
            sum_samples(0, 0) += x(0, 0);
            sum_samples(1, 0) += x(1, 0);
            sum_cov += (x(0, 0) - 1.0) * (x(1, 0) - 2.0);
        }

        double emp_mean0 = sum_samples(0, 0) / n;
        double emp_mean1 = sum_samples(1, 0) / n;
        double emp_cov = sum_cov / n;

        REQUIRE(emp_mean0 == Approx(1.0).epsilon(STAT_TOL));
        REQUIRE(emp_mean1 == Approx(2.0).epsilon(STAT_TOL));
        REQUIRE(emp_cov == Approx(1.0).epsilon(0.1));  // looser tolerance for covariance
    }
}

// =============================================================================
// Cholesky Integration Test
// =============================================================================

TEST_CASE("Elementa Cholesky decomposition", "[alea][cholesky]") {
    using mat = elementa::matrix<double>;

    SECTION("Simple positive definite matrix") {
        mat A{{4, 2}, {2, 3}};
        auto L = elementa::cholesky(A);

        // Verify L * L^T = A
        auto LLT = elementa::matmul(L, elementa::transpose(L));
        REQUIRE(elementa::approx_equal(LLT, A));
    }

    SECTION("3x3 positive definite") {
        mat A{{4, 2, 1}, {2, 5, 3}, {1, 3, 6}};
        auto L = elementa::cholesky(A);

        auto LLT = elementa::matmul(L, elementa::transpose(L));
        REQUIRE(elementa::approx_equal(LLT, A));
    }

    SECTION("L is lower triangular") {
        mat A{{4, 2}, {2, 3}};
        auto L = elementa::cholesky(A);

        // Upper triangle should be zero
        REQUIRE(L(0, 1) == Approx(0.0));
        // Diagonal should be positive
        REQUIRE(L(0, 0) > 0);
        REQUIRE(L(1, 1) > 0);
    }
}

// =============================================================================
// Utility Function Tests
// =============================================================================

TEST_CASE("Sample statistics utilities", "[alea][utility]") {
    std::mt19937 rng(505);

    SECTION("Sample mean and variance") {
        normal<double> dist(3.0, 2.0);
        std::vector<double> samples;
        for (int i = 0; i < N_SAMPLES; ++i) {
            samples.push_back(dist.sample(rng));
        }

        double mean = sample_mean(samples);
        double var = sample_variance(samples);

        REQUIRE(mean == Approx(3.0).epsilon(STAT_TOL));
        REQUIRE(var == Approx(4.0).epsilon(STAT_TOL));
    }
}

// =============================================================================
// Concept Satisfaction Tests
// =============================================================================

TEST_CASE("Concept satisfaction", "[alea][concepts]") {
    // Verify at compile time that distributions satisfy the shared concepts
    static_assert(stepanov::Distribution<normal<double>>);
    static_assert(stepanov::Distribution<uniform<double>>);
    static_assert(stepanov::Distribution<exponential<double>>);
    static_assert(stepanov::Distribution<beta<double>>);
    static_assert(stepanov::Distribution<bernoulli<double>>);
    static_assert(stepanov::Distribution<poisson<double>>);

    static_assert(stepanov::ContinuousDistribution<normal<double>>);
    static_assert(stepanov::ContinuousDistribution<uniform<double>>);
    static_assert(stepanov::ContinuousDistribution<exponential<double>>);

    static_assert(stepanov::DiscreteDistribution<bernoulli<double>>);
    static_assert(stepanov::DiscreteDistribution<poisson<double>>);

    REQUIRE(true);  // If we got here, all static_asserts passed
}
