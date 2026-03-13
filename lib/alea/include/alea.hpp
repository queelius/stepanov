// alea.hpp - Pedagogical Probability Distribution Library
//
// A teaching-oriented implementation of probability distributions and
// random element composition. Demonstrates that distributions are
// algebraic types: composition preserves distributional structure.
//
// Key design principles:
// - Concept-based: every distribution satisfies stepanov::Distribution
// - No Eigen: uses elementa's matrix<T> for multivariate operations
// - Value semantics: distributions are copyable, movable, cheap to pass
// - Algebraic composition: sum, product, transform of random elements
//
// Dependencies: concepts/random.hpp, concepts/matrix.hpp, elementa.hpp

#pragma once

#include <concepts/random.hpp>
#include <concepts/matrix.hpp>
#include <elementa.hpp>

#include <cmath>
#include <functional>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace alea {

// Import shared concepts
using stepanov::Distribution;
using stepanov::ContinuousDistribution;
using stepanov::DiscreteDistribution;
using stepanov::Matrix;

// =============================================================================
// Mathematical helpers
// =============================================================================

namespace detail {

/// Log-gamma function using Stirling's approximation (Lanczos method).
/// Needed for beta distribution PDF computation.
template <typename T>
[[nodiscard]] auto log_gamma(T x) -> T {
    // Lanczos approximation coefficients (g=7)
    static constexpr double c[] = {
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    };

    if (x < T{0.5}) {
        // Reflection formula: Gamma(1-x)*Gamma(x) = pi/sin(pi*x)
        return std::log(std::numbers::pi_v<T> / std::sin(std::numbers::pi_v<T> * x))
             - log_gamma(T{1} - x);
    }

    x -= T{1};
    T a = static_cast<T>(c[0]);
    for (int i = 1; i < 9; ++i) {
        a += static_cast<T>(c[i]) / (x + static_cast<T>(i));
    }

    T t = x + T{7.5};
    return T{0.5} * std::log(T{2} * std::numbers::pi_v<T>)
         + (x + T{0.5}) * std::log(t)
         - t + std::log(a);
}

/// Log of the beta function: B(a,b) = Gamma(a)*Gamma(b)/Gamma(a+b)
template <typename T>
[[nodiscard]] auto log_beta(T a, T b) -> T {
    return log_gamma(a) + log_gamma(b) - log_gamma(a + b);
}

/// Gamma variate sampling using Marsaglia & Tsang's method.
/// For shape alpha >= 1. For alpha < 1, use the relation:
///   X ~ Gamma(alpha) => X * U^(1/alpha) ~ Gamma(alpha) for U ~ Uniform(0,1)
template <typename T, typename RNG>
auto sample_gamma(T alpha, RNG& rng) -> T {
    if (alpha < T{1}) {
        // Boost: Gamma(alpha) = Gamma(alpha+1) * U^(1/alpha)
        std::uniform_real_distribution<T> u01(T{0}, T{1});
        T x = sample_gamma(alpha + T{1}, rng);
        return x * std::pow(u01(rng), T{1} / alpha);
    }

    // Marsaglia & Tsang's method for alpha >= 1
    T d = alpha - T{1} / T{3};
    T c = T{1} / std::sqrt(T{9} * d);

    std::normal_distribution<T> norm(T{0}, T{1});
    std::uniform_real_distribution<T> u01(T{0}, T{1});

    while (true) {
        T x, v;
        do {
            x = norm(rng);
            v = T{1} + c * x;
        } while (v <= T{0});

        v = v * v * v;
        T u = u01(rng);

        // Squeeze test
        if (u < T{1} - T{0.0331} * x * x * x * x) {
            return d * v;
        }

        // Full acceptance test
        if (std::log(u) < T{0.5} * x * x + d * (T{1} - v + std::log(v))) {
            return d * v;
        }
    }
}

} // namespace detail

// =============================================================================
// Normal Distribution
// =============================================================================

/// Gaussian distribution N(mu, sigma^2).
///
/// The normal distribution is the cornerstone of probability theory.
/// By the Central Limit Theorem, sums of independent random variables
/// converge to normal regardless of their individual distributions.
///
/// Sampling uses the Box-Muller transform:
///   Z = sqrt(-2*ln(U1)) * cos(2*pi*U2)
/// where U1, U2 ~ Uniform(0,1).
template <typename T = double>
class normal {
    T mu_;
    T sigma_;

public:
    using value_type = T;
    using scalar_type = T;

    explicit normal(T mu = T{0}, T sigma = T{1})
        : mu_(mu), sigma_(sigma)
    {
        if (sigma <= T{0}) {
            throw std::invalid_argument("Normal: sigma must be positive");
        }
    }

    [[nodiscard]] auto mean() const -> T { return mu_; }
    [[nodiscard]] auto variance() const -> T { return sigma_ * sigma_; }

    /// PDF: f(x) = (1/(sigma*sqrt(2*pi))) * exp(-(x-mu)^2 / (2*sigma^2))
    [[nodiscard]] auto pdf(T x) const -> T {
        T z = (x - mu_) / sigma_;
        return std::exp(-T{0.5} * z * z) / (sigma_ * std::sqrt(T{2} * std::numbers::pi_v<T>));
    }

    /// CDF via the error function: Phi(x) = 0.5 * (1 + erf((x-mu)/(sigma*sqrt(2))))
    [[nodiscard]] auto cdf(T x) const -> T {
        return T{0.5} * (T{1} + std::erf((x - mu_) / (sigma_ * std::sqrt(T{2}))));
    }

    /// Sample using the standard library normal distribution
    template <typename RNG>
    auto sample(RNG& rng) const -> T {
        std::normal_distribution<T> dist(mu_, sigma_);
        return dist(rng);
    }
};

// =============================================================================
// Uniform Distribution
// =============================================================================

/// Continuous uniform distribution on [a, b].
///
/// The simplest continuous distribution. PDF is constant:
///   f(x) = 1/(b-a) for a <= x <= b
///
/// Every point in the interval is equally likely.
template <typename T = double>
class uniform {
    T a_;
    T b_;

public:
    using value_type = T;
    using scalar_type = T;

    explicit uniform(T a = T{0}, T b = T{1})
        : a_(a), b_(b)
    {
        if (a >= b) {
            throw std::invalid_argument("Uniform: a must be less than b");
        }
    }

    [[nodiscard]] auto mean() const -> T { return (a_ + b_) / T{2}; }
    [[nodiscard]] auto variance() const -> T {
        T range = b_ - a_;
        return range * range / T{12};
    }

    /// PDF: constant 1/(b-a) on [a,b], zero elsewhere
    [[nodiscard]] auto pdf(T x) const -> T {
        if (x < a_ || x > b_) return T{0};
        return T{1} / (b_ - a_);
    }

    /// CDF: linear ramp from 0 to 1
    [[nodiscard]] auto cdf(T x) const -> T {
        if (x < a_) return T{0};
        if (x > b_) return T{1};
        return (x - a_) / (b_ - a_);
    }

    template <typename RNG>
    auto sample(RNG& rng) const -> T {
        std::uniform_real_distribution<T> dist(a_, b_);
        return dist(rng);
    }
};

// =============================================================================
// Exponential Distribution
// =============================================================================

/// Exponential distribution with rate parameter lambda.
///
/// The exponential is the only continuous distribution with the
/// memoryless property: P(X > s+t | X > s) = P(X > t).
/// This makes it the natural model for waiting times.
///
/// Sampling via inverse CDF: X = -ln(U)/lambda where U ~ Uniform(0,1).
template <typename T = double>
class exponential {
    T lambda_;

public:
    using value_type = T;
    using scalar_type = T;

    explicit exponential(T lambda = T{1})
        : lambda_(lambda)
    {
        if (lambda <= T{0}) {
            throw std::invalid_argument("Exponential: lambda must be positive");
        }
    }

    [[nodiscard]] auto mean() const -> T { return T{1} / lambda_; }
    [[nodiscard]] auto variance() const -> T { return T{1} / (lambda_ * lambda_); }

    /// PDF: f(x) = lambda * exp(-lambda * x) for x >= 0
    [[nodiscard]] auto pdf(T x) const -> T {
        if (x < T{0}) return T{0};
        return lambda_ * std::exp(-lambda_ * x);
    }

    /// CDF: F(x) = 1 - exp(-lambda * x)
    [[nodiscard]] auto cdf(T x) const -> T {
        if (x < T{0}) return T{0};
        return T{1} - std::exp(-lambda_ * x);
    }

    template <typename RNG>
    auto sample(RNG& rng) const -> T {
        std::exponential_distribution<T> dist(lambda_);
        return dist(rng);
    }
};

// =============================================================================
// Beta Distribution
// =============================================================================

/// Beta distribution with shape parameters alpha and beta.
///
/// Supported on [0, 1], making it ideal for modeling probabilities.
/// It is the conjugate prior of the Bernoulli distribution.
///
/// PDF: f(x) = x^(a-1) * (1-x)^(b-1) / B(a,b)
///
/// Sampling: Generate X ~ Gamma(alpha), Y ~ Gamma(beta),
/// then X/(X+Y) ~ Beta(alpha, beta).
template <typename T = double>
class beta {
    T alpha_;
    T beta_;

public:
    using value_type = T;
    using scalar_type = T;

    explicit beta(T alpha, T beta)
        : alpha_(alpha), beta_(beta)
    {
        if (alpha <= T{0} || beta <= T{0}) {
            throw std::invalid_argument("Beta: parameters must be positive");
        }
    }

    [[nodiscard]] auto mean() const -> T {
        return alpha_ / (alpha_ + beta_);
    }

    [[nodiscard]] auto variance() const -> T {
        T ab = alpha_ + beta_;
        return (alpha_ * beta_) / (ab * ab * (ab + T{1}));
    }

    /// PDF: f(x) = x^(a-1) * (1-x)^(b-1) / B(a,b)
    [[nodiscard]] auto pdf(T x) const -> T {
        if (x < T{0} || x > T{1}) return T{0};
        if (x == T{0} && alpha_ < T{1}) return std::numeric_limits<T>::infinity();
        if (x == T{1} && beta_ < T{1}) return std::numeric_limits<T>::infinity();

        T log_pdf = (alpha_ - T{1}) * std::log(x)
                  + (beta_ - T{1}) * std::log(T{1} - x)
                  - detail::log_beta(alpha_, beta_);
        return std::exp(log_pdf);
    }

    /// CDF: computed via numerical integration (regularized incomplete beta)
    /// For pedagogical simplicity, we use a basic series expansion.
    [[nodiscard]] auto cdf(T x) const -> T {
        if (x <= T{0}) return T{0};
        if (x >= T{1}) return T{1};

        // Use the continued fraction / series representation
        // For pedagogical clarity, numerical integration via Simpson's rule
        constexpr int n = 1000;
        T h = x / static_cast<T>(n);
        T sum = pdf(T{0}) + pdf(x);

        for (int i = 1; i < n; i += 2) {
            sum += T{4} * pdf(static_cast<T>(i) * h);
        }
        for (int i = 2; i < n; i += 2) {
            sum += T{2} * pdf(static_cast<T>(i) * h);
        }

        return sum * h / T{3};
    }

    /// Sample via the gamma ratio method
    template <typename RNG>
    auto sample(RNG& rng) const -> T {
        T x = detail::sample_gamma(alpha_, rng);
        T y = detail::sample_gamma(beta_, rng);
        return x / (x + y);
    }
};

// =============================================================================
// Bernoulli Distribution
// =============================================================================

/// Bernoulli distribution: P(X=1) = p, P(X=0) = 1-p.
///
/// The simplest discrete distribution. A single trial with
/// two outcomes (success/failure, heads/tails, etc.).
template <typename T = double>
class bernoulli {
    T p_;

public:
    using value_type = T;
    using scalar_type = T;

    explicit bernoulli(T p = T{0.5})
        : p_(p)
    {
        if (p < T{0} || p > T{1}) {
            throw std::invalid_argument("Bernoulli: p must be in [0, 1]");
        }
    }

    [[nodiscard]] auto mean() const -> T { return p_; }
    [[nodiscard]] auto variance() const -> T { return p_ * (T{1} - p_); }

    /// PMF: P(X=k) = p^k * (1-p)^(1-k) for k in {0, 1}
    [[nodiscard]] auto pmf(T x) const -> T {
        if (x == T{0}) return T{1} - p_;
        if (x == T{1}) return p_;
        return T{0};
    }

    /// CDF
    [[nodiscard]] auto cdf(T x) const -> T {
        if (x < T{0}) return T{0};
        if (x < T{1}) return T{1} - p_;
        return T{1};
    }

    template <typename RNG>
    auto sample(RNG& rng) const -> T {
        std::uniform_real_distribution<T> u(T{0}, T{1});
        return u(rng) < p_ ? T{1} : T{0};
    }
};

// =============================================================================
// Poisson Distribution
// =============================================================================

/// Poisson distribution with rate parameter lambda.
///
/// Models the number of events in a fixed interval when events
/// occur independently at a constant rate. Mean == variance == lambda.
///
/// Sum-stable: Poisson(a) + Poisson(b) = Poisson(a+b).
///
/// Sampling uses Knuth's algorithm for small lambda.
template <typename T = double>
class poisson {
    T lambda_;

public:
    using value_type = T;
    using scalar_type = T;

    explicit poisson(T lambda = T{1})
        : lambda_(lambda)
    {
        if (lambda <= T{0}) {
            throw std::invalid_argument("Poisson: lambda must be positive");
        }
    }

    [[nodiscard]] auto mean() const -> T { return lambda_; }
    [[nodiscard]] auto variance() const -> T { return lambda_; }

    /// PMF: P(X=k) = lambda^k * exp(-lambda) / k!
    [[nodiscard]] auto pmf(T x) const -> T {
        if (x < T{0} || x != std::floor(x)) return T{0};
        auto k = static_cast<int>(x);

        // Use log to avoid overflow: log(PMF) = k*log(lambda) - lambda - log(k!)
        T log_pmf = static_cast<T>(k) * std::log(lambda_) - lambda_;
        for (int i = 1; i <= k; ++i) {
            log_pmf -= std::log(static_cast<T>(i));
        }
        return std::exp(log_pmf);
    }

    /// CDF: sum of PMF from 0 to floor(x)
    [[nodiscard]] auto cdf(T x) const -> T {
        if (x < T{0}) return T{0};
        auto k = static_cast<int>(std::floor(x));
        T total = T{0};
        for (int i = 0; i <= k; ++i) {
            total += pmf(static_cast<T>(i));
        }
        return total;
    }

    /// Knuth's algorithm for Poisson sampling
    template <typename RNG>
    auto sample(RNG& rng) const -> T {
        std::uniform_real_distribution<T> u(T{0}, T{1});
        T L = std::exp(-lambda_);
        int k = 0;
        T p = T{1};

        do {
            ++k;
            p *= u(rng);
        } while (p > L);

        return static_cast<T>(k - 1);
    }
};

// =============================================================================
// Random Element — Type-Erased Samplable Value
// =============================================================================

/// A random_element<T> is a type-erased wrapper around any distribution
/// that produces values of type T. This enables algebraic composition:
/// you can add, multiply, and transform random elements without knowing
/// their concrete distribution type.
///
/// This is the key algebraic abstraction: random elements form a
/// module over the scalar ring under addition and scalar multiplication.
template <typename T>
class random_element {
    std::function<T(std::mt19937&)> sampler_;

public:
    using value_type = T;

    /// Construct from any callable that takes an RNG and returns T
    template <typename F>
        requires std::invocable<F, std::mt19937&>
    explicit random_element(F&& f) : sampler_(std::forward<F>(f)) {}

    /// Sample a value
    auto sample(std::mt19937& rng) const -> T {
        return sampler_(rng);
    }
};

/// Wrap any distribution into a random_element
template <typename T, typename Dist>
auto make_random_element(const Dist& dist) -> random_element<T> {
    return random_element<T>([dist](std::mt19937& rng) { return dist.sample(rng); });
}

/// Sum of random elements: (X + Y)(omega) = X(omega) + Y(omega)
/// The sum of independent normals is normal — algebraic closure.
template <typename T>
auto operator+(const random_element<T>& lhs, const random_element<T>& rhs) -> random_element<T> {
    return random_element<T>([lhs, rhs](std::mt19937& rng) {
        return lhs.sample(rng) + rhs.sample(rng);
    });
}

/// Scalar multiplication: (s * X)(omega) = s * X(omega)
template <typename T>
auto operator*(T scalar, const random_element<T>& re) -> random_element<T> {
    return random_element<T>([scalar, re](std::mt19937& rng) {
        return scalar * re.sample(rng);
    });
}

template <typename T>
auto operator*(const random_element<T>& re, T scalar) -> random_element<T> {
    return scalar * re;
}

/// Transform (pushforward): apply a function to the sampled value.
/// If X ~ Dist, then transform(f, X) samples from the pushforward measure f_*(Dist).
template <typename F, typename Dist>
auto transform(F&& f, const Dist& dist) -> random_element<
    std::invoke_result_t<F, typename Dist::value_type>>
{
    using R = std::invoke_result_t<F, typename Dist::value_type>;
    return random_element<R>([f = std::forward<F>(f), dist](std::mt19937& rng) {
        return f(dist.sample(rng));
    });
}

/// Joint distribution: sample from both distributions independently.
/// Returns a pair of values.
template <typename Dist1, typename Dist2>
auto joint(const Dist1& d1, const Dist2& d2)
    -> random_element<std::pair<typename Dist1::value_type, typename Dist2::value_type>>
{
    using P = std::pair<typename Dist1::value_type, typename Dist2::value_type>;
    return random_element<P>([d1, d2](std::mt19937& rng) -> P {
        return {d1.sample(rng), d2.sample(rng)};
    });
}

// =============================================================================
// Multivariate Normal Distribution
// =============================================================================

/// Multivariate normal distribution N(mu, Sigma).
///
/// Parameterized by:
///   mu    — mean vector (N x 1 matrix)
///   Sigma — covariance matrix (N x N, must be positive-definite)
///
/// Sampling algorithm:
///   1. Compute L = cholesky(Sigma) where Sigma = L * L^T
///   2. Generate z ~ N(0, I) (vector of independent standard normals)
///   3. Return x = mu + L * z
///
/// This works because: Cov(L*z) = L * Cov(z) * L^T = L * I * L^T = Sigma
///
/// Uses elementa for all matrix operations — no Eigen dependency.
template <typename T = double>
class multivariate_normal {
    using mat = elementa::matrix<T>;

    mat mu_;       // Mean vector (n x 1)
    mat sigma_;    // Covariance matrix (n x n)
    mat chol_L_;   // Lower Cholesky factor: Sigma = L * L^T
    std::size_t n_;

public:
    using value_type = mat;
    using scalar_type = T;

    multivariate_normal(const mat& mu, const mat& sigma)
        : mu_(mu), sigma_(sigma), chol_L_(elementa::cholesky(sigma)), n_(mu.rows())
    {
        if (mu.cols() != 1) {
            throw std::invalid_argument("Multivariate normal: mu must be a column vector");
        }
        if (sigma.rows() != n_ || sigma.cols() != n_) {
            throw std::invalid_argument("Multivariate normal: sigma dimensions must match mu");
        }
    }

    [[nodiscard]] auto mean() const -> mat { return mu_; }

    [[nodiscard]] auto variance() const -> mat { return sigma_; }

    [[nodiscard]] auto dimension() const -> std::size_t { return n_; }

    /// Log PDF: -0.5 * (k*log(2*pi) + log|Sigma| + (x-mu)^T * Sigma^{-1} * (x-mu))
    [[nodiscard]] auto log_pdf(const mat& x) const -> T {
        auto diff = x - mu_;
        auto [sign, log_det] = elementa::logdet(sigma_);

        // Solve Sigma * y = diff for y, then compute diff^T * y
        auto y = elementa::solve(sigma_, diff);
        T quad = T{0};
        for (std::size_t i = 0; i < n_; ++i) {
            quad += diff(i, 0) * y(i, 0);
        }

        return -T{0.5} * (static_cast<T>(n_) * std::log(T{2} * std::numbers::pi_v<T>)
                         + log_det + quad);
    }

    [[nodiscard]] auto pdf(const mat& x) const -> T {
        return std::exp(log_pdf(x));
    }

    /// Sample: x = mu + L * z where z ~ N(0, I)
    template <typename RNG>
    auto sample(RNG& rng) const -> mat {
        std::normal_distribution<T> std_norm(T{0}, T{1});

        // Generate z ~ N(0, I)
        mat z(n_, 1);
        for (std::size_t i = 0; i < n_; ++i) {
            z(i, 0) = std_norm(rng);
        }

        // x = mu + L * z
        return mu_ + elementa::matmul(chol_L_, z);
    }
};

// =============================================================================
// Utility Functions
// =============================================================================

/// Compute sample mean from a vector of values
template <typename T>
[[nodiscard]] auto sample_mean(const std::vector<T>& samples) -> T {
    T sum{0};
    for (const auto& x : samples) {
        sum += x;
    }
    return sum / static_cast<T>(samples.size());
}

/// Compute sample variance (unbiased, using Bessel's correction)
template <typename T>
[[nodiscard]] auto sample_variance(const std::vector<T>& samples) -> T {
    T mean = sample_mean(samples);
    T sum_sq{0};
    for (const auto& x : samples) {
        T d = x - mean;
        sum_sq += d * d;
    }
    return sum_sq / static_cast<T>(samples.size() - 1);
}

/// Kolmogorov-Smirnov test statistic.
/// Compares the empirical CDF of samples against a theoretical CDF.
/// Returns the maximum absolute difference (D statistic).
template <typename Dist>
    requires ContinuousDistribution<Dist>
[[nodiscard]] auto ks_statistic(const Dist& dist, std::vector<typename Dist::value_type> samples)
    -> typename Dist::scalar_type
{
    using T = typename Dist::scalar_type;
    std::sort(samples.begin(), samples.end());
    auto n = samples.size();

    T max_diff{0};
    for (std::size_t i = 0; i < n; ++i) {
        T empirical = static_cast<T>(i + 1) / static_cast<T>(n);
        T theoretical = dist.cdf(samples[i]);
        max_diff = std::max(max_diff, std::abs(empirical - theoretical));
    }

    return max_diff;
}

/// Numerical PDF from CDF via finite differences (for testing).
template <typename Dist>
    requires ContinuousDistribution<Dist>
[[nodiscard]] auto finite_diff_pdf(const Dist& dist, typename Dist::value_type x,
                                    typename Dist::scalar_type eps = 1e-7)
    -> typename Dist::scalar_type
{
    return (dist.cdf(x + eps) - dist.cdf(x - eps)) / (2 * eps);
}

} // namespace alea
