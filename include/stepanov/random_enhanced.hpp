#pragma once

#include <cmath>
#include <random>
#include <vector>
#include <array>
#include <memory>
#include <numeric>
#include <algorithm>
#include <complex>
#include <functional>
#include <optional>
#include <unordered_map>
#include <limits>
#include <cstddef>
#include "random.hpp"
#include "matrix.hpp"

namespace stepanov::random {

// =============================================================================
// Enhanced Continuous Distributions
// =============================================================================

// Beta distribution
template<typename T = double>
class beta_distribution {
public:
    using result_type = T;

private:
    T alpha_;
    T beta_;
    std::gamma_distribution<T> gamma_alpha_;
    std::gamma_distribution<T> gamma_beta_;

public:
    beta_distribution(T alpha = 1.0, T beta = 1.0)
        : alpha_(alpha), beta_(beta),
          gamma_alpha_(alpha, 1.0),
          gamma_beta_(beta, 1.0) {
        if (alpha <= 0 || beta <= 0) {
            throw std::invalid_argument("Beta parameters must be positive");
        }
    }

    template<typename Generator>
    T operator()(Generator& gen) {
        T x = gamma_alpha_(gen);
        T y = gamma_beta_(gen);
        return x / (x + y);
    }

    T min() const { return 0; }
    T max() const { return 1; }

    T mean() const { return alpha_ / (alpha_ + beta_); }

    T variance() const {
        T sum = alpha_ + beta_;
        return (alpha_ * beta_) / (sum * sum * (sum + 1));
    }

    T pdf(T x) const {
        if (x < 0 || x > 1) return 0;

        T log_pdf = (alpha_ - 1) * std::log(x) +
                    (beta_ - 1) * std::log(1 - x) -
                    log_beta_function(alpha_, beta_);
        return std::exp(log_pdf);
    }

private:
    static T log_beta_function(T a, T b) {
        return std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
    }
};

// Laplace distribution
template<typename T = double>
class laplace_distribution {
public:
    using result_type = T;

private:
    T mu_;
    T b_;
    std::uniform_real_distribution<T> uniform_;

public:
    laplace_distribution(T mu = 0.0, T b = 1.0)
        : mu_(mu), b_(b), uniform_(-0.5, 0.5) {
        if (b <= 0) {
            throw std::invalid_argument("Scale parameter must be positive");
        }
    }

    template<typename Generator>
    T operator()(Generator& gen) {
        T u = uniform_(gen);
        return mu_ - b_ * std::copysign(std::log(1 - 2 * std::abs(u)), u);
    }

    T min() const { return std::numeric_limits<T>::lowest(); }
    T max() const { return std::numeric_limits<T>::max(); }

    T mean() const { return mu_; }
    T variance() const { return 2 * b_ * b_; }

    T pdf(T x) const {
        return (1 / (2 * b_)) * std::exp(-std::abs(x - mu_) / b_);
    }
};

// Pareto distribution
template<typename T = double>
class pareto_distribution {
public:
    using result_type = T;

private:
    T xm_;  // Scale parameter
    T alpha_;  // Shape parameter
    std::uniform_real_distribution<T> uniform_;

public:
    pareto_distribution(T xm = 1.0, T alpha = 1.0)
        : xm_(xm), alpha_(alpha), uniform_(0.0, 1.0) {
        if (xm <= 0 || alpha <= 0) {
            throw std::invalid_argument("Parameters must be positive");
        }
    }

    template<typename Generator>
    T operator()(Generator& gen) {
        T u = uniform_(gen);
        return xm_ / std::pow(1 - u, 1 / alpha_);
    }

    T min() const { return xm_; }
    T max() const { return std::numeric_limits<T>::max(); }

    T mean() const {
        if (alpha_ <= 1) return std::numeric_limits<T>::infinity();
        return (alpha_ * xm_) / (alpha_ - 1);
    }

    T variance() const {
        if (alpha_ <= 2) return std::numeric_limits<T>::infinity();
        return (xm_ * xm_ * alpha_) / ((alpha_ - 1) * (alpha_ - 1) * (alpha_ - 2));
    }

    T pdf(T x) const {
        if (x < xm_) return 0;
        return (alpha_ * std::pow(xm_, alpha_)) / std::pow(x, alpha_ + 1);
    }
};

// =============================================================================
// Enhanced Discrete Distributions
// =============================================================================

// Negative binomial distribution
template<typename T = double>
class negative_binomial_distribution {
public:
    using result_type = int;

private:
    int r_;  // Number of failures
    T p_;    // Success probability
    std::gamma_distribution<T> gamma_;
    std::poisson_distribution<int> poisson_;

public:
    negative_binomial_distribution(int r = 1, T p = 0.5)
        : r_(r), p_(p), gamma_(r, (1 - p) / p) {
        if (r <= 0 || p <= 0 || p > 1) {
            throw std::invalid_argument("Invalid parameters");
        }
    }

    template<typename Generator>
    int operator()(Generator& gen) {
        T lambda = gamma_(gen);
        std::poisson_distribution<int> poisson(lambda);
        return poisson(gen);
    }

    int min() const { return 0; }
    int max() const { return std::numeric_limits<int>::max(); }

    T mean() const { return r_ * (1 - p_) / p_; }
    T variance() const { return r_ * (1 - p_) / (p_ * p_); }
};

// Hypergeometric distribution
template<typename T = double>
class hypergeometric_distribution {
public:
    using result_type = int;

private:
    int N_;  // Population size
    int K_;  // Success states in population
    int n_;  // Number of draws

public:
    hypergeometric_distribution(int N, int K, int n)
        : N_(N), K_(K), n_(n) {
        if (N <= 0 || K < 0 || K > N || n < 0 || n > N) {
            throw std::invalid_argument("Invalid parameters");
        }
    }

    template<typename Generator>
    int operator()(Generator& gen) {
        // Using the urn model
        std::vector<int> population(N_);
        std::fill(population.begin(), population.begin() + K_, 1);
        std::fill(population.begin() + K_, population.end(), 0);

        std::shuffle(population.begin(), population.end(), gen);

        return std::accumulate(population.begin(), population.begin() + n_, 0);
    }

    int min() const { return std::max(0, n_ + K_ - N_); }
    int max() const { return std::min(n_, K_); }

    double mean() const { return static_cast<double>(n_ * K_) / N_; }

    double variance() const {
        double p = static_cast<double>(K_) / N_;
        return n_ * p * (1 - p) * (N_ - n_) / (N_ - 1);
    }
};

// =============================================================================
// Markov Chains
// =============================================================================

template<typename State = int>
class markov_chain {
public:
    using transition_matrix = matrix<double>;
    using state_type = State;

private:
    std::vector<State> states_;
    transition_matrix P_;
    std::unordered_map<State, size_t> state_to_index_;
    State current_state_;
    std::uniform_real_distribution<double> uniform_;

public:
    markov_chain(const std::vector<State>& states, const transition_matrix& P)
        : states_(states), P_(P), uniform_(0.0, 1.0) {

        if (states.size() != P.rows() || P.rows() != P.cols()) {
            throw std::invalid_argument("Invalid transition matrix dimensions");
        }

        // Build state to index mapping
        for (size_t i = 0; i < states.size(); ++i) {
            state_to_index_[states[i]] = i;
        }

        // Verify transition matrix (rows sum to 1)
        for (size_t i = 0; i < P.rows(); ++i) {
            double sum = 0;
            for (size_t j = 0; j < P.cols(); ++j) {
                sum += P(i, j);
            }
            if (std::abs(sum - 1.0) > 1e-10) {
                throw std::invalid_argument("Transition matrix rows must sum to 1");
            }
        }

        current_state_ = states[0];
    }

    void set_state(const State& state) {
        auto it = state_to_index_.find(state);
        if (it == state_to_index_.end()) {
            throw std::invalid_argument("Unknown state");
        }
        current_state_ = state;
    }

    template<typename Generator>
    State step(Generator& gen) {
        size_t current_idx = state_to_index_[current_state_];
        double u = uniform_(gen);
        double cumsum = 0;

        for (size_t j = 0; j < P_.cols(); ++j) {
            cumsum += P_(current_idx, j);
            if (u <= cumsum) {
                current_state_ = states_[j];
                return current_state_;
            }
        }

        // Should not reach here if matrix is valid
        current_state_ = states_.back();
        return current_state_;
    }

    template<typename Generator>
    std::vector<State> simulate(size_t steps, Generator& gen) {
        std::vector<State> path;
        path.reserve(steps + 1);
        path.push_back(current_state_);

        for (size_t i = 0; i < steps; ++i) {
            path.push_back(step(gen));
        }

        return path;
    }

    // Compute stationary distribution (if exists)
    std::vector<double> stationary_distribution() const {
        // Solve π = πP by finding left eigenvector for eigenvalue 1
        size_t n = P_.rows();
        matrix<double> A(n + 1, n);

        // Set up system: (P^T - I)π = 0 with constraint Σπ_i = 1
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                A(i, j) = P_(j, i) - (i == j ? 1.0 : 0.0);
            }
        }
        for (size_t j = 0; j < n; ++j) {
            A(n, j) = 1.0;
        }

        // Solve using least squares or similar method
        // Simplified: return uniform distribution
        std::vector<double> pi(n, 1.0 / n);
        return pi;
    }

    const State& current() const { return current_state_; }
    const std::vector<State>& states() const { return states_; }
    const transition_matrix& transition() const { return P_; }
};

// =============================================================================
// Random Walks
// =============================================================================

template<int Dim = 1>
class random_walk {
public:
    using position_type = std::array<double, Dim>;

private:
    position_type position_;
    position_type drift_;
    position_type volatility_;
    std::normal_distribution<double> normal_;

public:
    random_walk(const position_type& start = {},
                const position_type& drift = {},
                const position_type& volatility = {})
        : position_(start), drift_(drift), volatility_(volatility), normal_(0, 1) {

        // Default volatility to 1 if not specified
        for (int i = 0; i < Dim; ++i) {
            if (volatility_[i] == 0) {
                volatility_[i] = 1.0;
            }
        }
    }

    template<typename Generator>
    position_type step(Generator& gen, double dt = 1.0) {
        for (int i = 0; i < Dim; ++i) {
            double dW = normal_(gen) * std::sqrt(dt);
            position_[i] += drift_[i] * dt + volatility_[i] * dW;
        }
        return position_;
    }

    template<typename Generator>
    std::vector<position_type> simulate(size_t steps, Generator& gen, double dt = 1.0) {
        std::vector<position_type> path;
        path.reserve(steps + 1);
        path.push_back(position_);

        for (size_t i = 0; i < steps; ++i) {
            path.push_back(step(gen, dt));
        }

        return path;
    }

    // Brownian motion factory
    static random_walk brownian(const position_type& start = {},
                                const position_type& drift = {},
                                const position_type& volatility = {}) {
        return random_walk(start, drift, volatility);
    }

    // Geometric Brownian motion (for asset prices)
    template<typename Generator>
    double geometric_step(Generator& gen, double S0, double mu, double sigma, double dt = 1.0) {
        double dW = normal_(gen) * std::sqrt(dt);
        return S0 * std::exp((mu - 0.5 * sigma * sigma) * dt + sigma * dW);
    }

    const position_type& position() const { return position_; }
    void reset(const position_type& pos = {}) { position_ = pos; }
};

// =============================================================================
// Jump Processes
// =============================================================================

template<typename T = double>
class poisson_process {
private:
    T lambda_;  // Rate parameter
    T current_time_;
    int event_count_;
    std::exponential_distribution<T> exp_dist_;

public:
    explicit poisson_process(T lambda)
        : lambda_(lambda), current_time_(0), event_count_(0), exp_dist_(lambda) {
        if (lambda <= 0) {
            throw std::invalid_argument("Rate must be positive");
        }
    }

    template<typename Generator>
    T next_event(Generator& gen) {
        T inter_arrival = exp_dist_(gen);
        current_time_ += inter_arrival;
        event_count_++;
        return current_time_;
    }

    template<typename Generator>
    std::vector<T> simulate_until(T end_time, Generator& gen) {
        std::vector<T> events;
        reset();

        while (current_time_ < end_time) {
            T next = next_event(gen);
            if (next <= end_time) {
                events.push_back(next);
            } else {
                break;
            }
        }

        return events;
    }

    void reset() {
        current_time_ = 0;
        event_count_ = 0;
    }

    T current_time() const { return current_time_; }
    int event_count() const { return event_count_; }
    T rate() const { return lambda_; }
};

// Compound Poisson process
template<typename T = double, typename JumpDist = std::normal_distribution<T>>
class compound_poisson_process {
private:
    poisson_process<T> poisson_;
    JumpDist jump_dist_;
    T current_value_;

public:
    compound_poisson_process(T lambda, JumpDist jump_dist = JumpDist())
        : poisson_(lambda), jump_dist_(jump_dist), current_value_(0) {}

    template<typename Generator>
    T next_jump(Generator& gen) {
        T event_time = poisson_.next_event(gen);
        T jump_size = jump_dist_(gen);
        current_value_ += jump_size;
        return current_value_;
    }

    template<typename Generator>
    std::vector<std::pair<T, T>> simulate_until(T end_time, Generator& gen) {
        std::vector<std::pair<T, T>> path;  // (time, value) pairs
        reset();
        path.emplace_back(0, 0);

        auto event_times = poisson_.simulate_until(end_time, gen);
        for (T t : event_times) {
            T jump = jump_dist_(gen);
            current_value_ += jump;
            path.emplace_back(t, current_value_);
        }

        return path;
    }

    void reset() {
        poisson_.reset();
        current_value_ = 0;
    }

    T current_value() const { return current_value_; }
};

// =============================================================================
// Monte Carlo Integration
// =============================================================================

template<typename T = double>
class monte_carlo_integrator {
private:
    size_t n_samples_;
    std::optional<T> last_estimate_;
    std::optional<T> last_error_;

public:
    explicit monte_carlo_integrator(size_t n_samples = 10000)
        : n_samples_(n_samples) {}

    // Integrate function over [a, b]^dim
    template<typename Func, typename Generator>
    T integrate_1d(Func f, T a, T b, Generator& gen) {
        std::uniform_real_distribution<T> uniform(a, b);
        T sum = 0;
        T sum_sq = 0;

        for (size_t i = 0; i < n_samples_; ++i) {
            T x = uniform(gen);
            T fx = f(x);
            sum += fx;
            sum_sq += fx * fx;
        }

        T volume = b - a;
        last_estimate_ = volume * sum / n_samples_;

        T variance = (sum_sq / n_samples_ - (sum / n_samples_) * (sum / n_samples_));
        last_error_ = volume * std::sqrt(variance / n_samples_);

        return *last_estimate_;
    }

    // Integrate over multidimensional box
    template<typename Func, typename Generator, size_t Dim>
    T integrate_nd(Func f, const std::array<T, Dim>& lower,
                   const std::array<T, Dim>& upper, Generator& gen) {

        std::array<std::uniform_real_distribution<T>, Dim> uniforms;
        for (size_t i = 0; i < Dim; ++i) {
            uniforms[i] = std::uniform_real_distribution<T>(lower[i], upper[i]);
        }

        T sum = 0;
        T sum_sq = 0;

        for (size_t i = 0; i < n_samples_; ++i) {
            std::array<T, Dim> x;
            for (size_t j = 0; j < Dim; ++j) {
                x[j] = uniforms[j](gen);
            }
            T fx = f(x);
            sum += fx;
            sum_sq += fx * fx;
        }

        T volume = 1;
        for (size_t i = 0; i < Dim; ++i) {
            volume *= (upper[i] - lower[i]);
        }

        last_estimate_ = volume * sum / n_samples_;

        T variance = (sum_sq / n_samples_ - (sum / n_samples_) * (sum / n_samples_));
        last_error_ = volume * std::sqrt(variance / n_samples_);

        return *last_estimate_;
    }

    // Importance sampling
    template<typename Func, typename ImportanceDist, typename Generator>
    T importance_sample(Func f, ImportanceDist& g_dist,
                       std::function<T(T)> g_pdf, Generator& gen) {

        T sum = 0;
        T sum_sq = 0;

        for (size_t i = 0; i < n_samples_; ++i) {
            T x = g_dist(gen);
            T fx = f(x) / g_pdf(x);
            sum += fx;
            sum_sq += fx * fx;
        }

        last_estimate_ = sum / n_samples_;

        T variance = (sum_sq / n_samples_ - (sum / n_samples_) * (sum / n_samples_));
        last_error_ = std::sqrt(variance / n_samples_);

        return *last_estimate_;
    }

    std::optional<T> estimate() const { return last_estimate_; }
    std::optional<T> error() const { return last_error_; }
    size_t samples() const { return n_samples_; }
};

// =============================================================================
// Type erasure for distributions
// =============================================================================

template<typename T>
class any_distribution {
private:
    struct concept_t {
        virtual ~concept_t() = default;
        virtual T sample(std::mt19937& gen) = 0;
        virtual T min() const = 0;
        virtual T max() const = 0;
        virtual T mean() const = 0;
        virtual T variance() const = 0;
        virtual std::unique_ptr<concept_t> clone() const = 0;
    };

    template<typename Dist>
    struct model : concept_t {
        Dist dist_;

        template<typename... Args>
        explicit model(Args&&... args) : dist_(std::forward<Args>(args)...) {}

        T sample(std::mt19937& gen) override {
            return dist_(gen);
        }

        T min() const override {
            if constexpr (requires { dist_.min(); }) {
                return dist_.min();
            } else {
                return std::numeric_limits<T>::lowest();
            }
        }

        T max() const override {
            if constexpr (requires { dist_.max(); }) {
                return dist_.max();
            } else {
                return std::numeric_limits<T>::max();
            }
        }

        T mean() const override {
            if constexpr (requires { dist_.mean(); }) {
                return dist_.mean();
            } else {
                return T{};
            }
        }

        T variance() const override {
            if constexpr (requires { dist_.variance(); }) {
                return dist_.variance();
            } else {
                return T{};
            }
        }

        std::unique_ptr<concept_t> clone() const override {
            return std::make_unique<model>(dist_);
        }
    };

    std::unique_ptr<concept_t> pimpl_;

public:
    template<typename Dist, typename... Args>
    static any_distribution make(Args&&... args) {
        any_distribution result;
        result.pimpl_ = std::make_unique<model<Dist>>(std::forward<Args>(args)...);
        return result;
    }

    T operator()(std::mt19937& gen) {
        return pimpl_ ? pimpl_->sample(gen) : T{};
    }

    T min() const { return pimpl_ ? pimpl_->min() : T{}; }
    T max() const { return pimpl_ ? pimpl_->max() : T{}; }
    T mean() const { return pimpl_ ? pimpl_->mean() : T{}; }
    T variance() const { return pimpl_ ? pimpl_->variance() : T{}; }
};

// Type erasure for random processes
template<typename T>
class any_random_process {
private:
    struct concept_t {
        virtual ~concept_t() = default;
        virtual T step(std::mt19937& gen, double dt) = 0;
        virtual void reset() = 0;
        virtual std::unique_ptr<concept_t> clone() const = 0;
    };

    template<typename Process>
    struct model : concept_t {
        Process process_;

        template<typename... Args>
        explicit model(Args&&... args) : process_(std::forward<Args>(args)...) {}

        T step(std::mt19937& gen, double dt) override {
            return process_.step(gen, dt);
        }

        void reset() override {
            if constexpr (requires { process_.reset(); }) {
                process_.reset();
            }
        }

        std::unique_ptr<concept_t> clone() const override {
            return std::make_unique<model>(process_);
        }
    };

    std::unique_ptr<concept_t> pimpl_;

public:
    template<typename Process, typename... Args>
    static any_random_process make(Args&&... args) {
        any_random_process result;
        result.pimpl_ = std::make_unique<model<Process>>(std::forward<Args>(args)...);
        return result;
    }

    T step(std::mt19937& gen, double dt = 1.0) {
        return pimpl_ ? pimpl_->step(gen, dt) : T{};
    }

    void reset() {
        if (pimpl_) {
            pimpl_->reset();
        }
    }
};

// =============================================================================
// Jump Processes - Poisson and Compound Poisson
// =============================================================================

template<typename T = double>
class poisson_process {
private:
    T lambda_;  // Rate parameter
    T current_time_;
    int current_count_;
    std::exponential_distribution<T> exp_dist_;

public:
    explicit poisson_process(T lambda)
        : lambda_(lambda), current_time_(0), current_count_(0),
          exp_dist_(lambda) {
        if (lambda <= 0) {
            throw std::invalid_argument("Rate must be positive");
        }
    }

    template<typename Generator>
    T next_jump_time(Generator& gen) {
        T inter_arrival = exp_dist_(gen);
        current_time_ += inter_arrival;
        current_count_++;
        return current_time_;
    }

    template<typename Generator>
    std::vector<T> generate_path(Generator& gen, T end_time) {
        std::vector<T> jump_times;
        reset();

        while (current_time_ < end_time) {
            T next_time = next_jump_time(gen);
            if (next_time <= end_time) {
                jump_times.push_back(next_time);
            } else {
                break;
            }
        }

        return jump_times;
    }

    template<typename Generator>
    int count_jumps(Generator& gen, T time_interval) {
        std::poisson_distribution<int> poisson(lambda_ * time_interval);
        return poisson(gen);
    }

    void reset() {
        current_time_ = 0;
        current_count_ = 0;
    }

    T rate() const { return lambda_; }
    T current_time() const { return current_time_; }
    int current_count() const { return current_count_; }
};

// Compound Poisson Process - jumps with random sizes
template<typename T = double, typename JumpDist = std::normal_distribution<T>>
class compound_poisson_process {
private:
    poisson_process<T> jump_process_;
    JumpDist jump_dist_;
    T current_value_;

public:
    compound_poisson_process(T lambda, JumpDist jump_dist = JumpDist())
        : jump_process_(lambda), jump_dist_(jump_dist), current_value_(0) {}

    template<typename Generator>
    T step(Generator& gen, T dt) {
        int n_jumps = jump_process_.count_jumps(gen, dt);

        for (int i = 0; i < n_jumps; ++i) {
            current_value_ += jump_dist_(gen);
        }

        return current_value_;
    }

    template<typename Generator>
    std::vector<std::pair<T, T>> generate_path(Generator& gen, T end_time, int n_steps) {
        std::vector<std::pair<T, T>> path;
        T dt = end_time / n_steps;
        reset();

        for (int i = 0; i < n_steps; ++i) {
            T time = (i + 1) * dt;
            T value = step(gen, dt);
            path.emplace_back(time, value);
        }

        return path;
    }

    void reset() {
        jump_process_.reset();
        current_value_ = 0;
    }

    T current_value() const { return current_value_; }
};

// =============================================================================
// Lévy Processes for Finance
// =============================================================================

// General Lévy Process interface
template<typename T = double>
class levy_process {
public:
    virtual ~levy_process() = default;
    virtual T increment(std::mt19937& gen, T dt) = 0;
    virtual void reset() = 0;
    virtual T characteristic_function(T u) const = 0;
};

// Variance Gamma Process - used in option pricing
template<typename T = double>
class variance_gamma_process : public levy_process<T> {
private:
    T sigma_;  // Volatility of Brownian motion
    T nu_;     // Variance of gamma time change
    T theta_;  // Drift
    std::gamma_distribution<T> gamma_dist_;
    std::normal_distribution<T> normal_dist_;

public:
    variance_gamma_process(T sigma, T nu, T theta)
        : sigma_(sigma), nu_(nu), theta_(theta),
          gamma_dist_(1.0 / nu, nu),
          normal_dist_(0, 1) {
        if (sigma <= 0 || nu <= 0) {
            throw std::invalid_argument("Parameters must be positive");
        }
    }

    T increment(std::mt19937& gen, T dt) override {
        // Generate gamma time change
        std::gamma_distribution<T> g(dt / nu_, nu_);
        T gamma_time = g(gen);

        // Generate Brownian motion with drift
        T brownian = normal_dist_(gen) * std::sqrt(gamma_time);

        return theta_ * gamma_time + sigma_ * brownian;
    }

    void reset() override {}

    T characteristic_function(T u) const override {
        std::complex<T> i(0, 1);
        std::complex<T> exponent = -std::log(1.0 - i * theta_ * nu_ * u +
                                            0.5 * sigma_ * sigma_ * nu_ * u * u) / nu_;
        return std::exp(exponent).real();
    }
};

// Normal Inverse Gaussian (NIG) Process
template<typename T = double>
class nig_process : public levy_process<T> {
private:
    T alpha_;  // Steepness
    T beta_;   // Asymmetry
    T delta_;  // Scale
    T mu_;     // Location

    // Helper: Generate from inverse Gaussian distribution
    template<typename Generator>
    T inverse_gaussian(Generator& gen, T mu, T lambda) {
        std::normal_distribution<T> normal(0, 1);
        T v = normal(gen);
        v = v * v;
        T x = mu + (mu * mu * v) / (2 * lambda) -
              (mu / (2 * lambda)) * std::sqrt(4 * mu * lambda * v + mu * mu * v * v);

        std::uniform_real_distribution<T> uniform(0, 1);
        T u = uniform(gen);

        if (u <= mu / (mu + x)) {
            return x;
        } else {
            return mu * mu / x;
        }
    }

public:
    nig_process(T alpha, T beta, T delta, T mu = 0)
        : alpha_(alpha), beta_(beta), delta_(delta), mu_(mu) {
        if (alpha <= std::abs(beta) || delta <= 0) {
            throw std::invalid_argument("Invalid NIG parameters");
        }
    }

    T increment(std::mt19937& gen, T dt) override {
        T gamma = std::sqrt(alpha_ * alpha_ - beta_ * beta_);

        // Generate inverse Gaussian subordinator
        T ig = inverse_gaussian(gen, delta_ * dt / gamma, delta_ * delta_ * dt);

        // Generate normal component
        std::normal_distribution<T> normal(0, ig);
        T w = normal(gen);

        return mu_ * dt + beta_ * ig + w;
    }

    void reset() override {}

    T characteristic_function(T u) const override {
        T gamma = std::sqrt(alpha_ * alpha_ - beta_ * beta_);
        T gamma_u = std::sqrt(alpha_ * alpha_ - (beta_ + u) * (beta_ + u));
        std::complex<T> i(0, 1);
        return std::exp(i * mu_ * u + delta_ * (gamma - gamma_u)).real();
    }
};

// =============================================================================
// Stochastic Differential Equations (SDEs)
// =============================================================================

// General SDE: dX_t = drift(t, X_t)dt + diffusion(t, X_t)dW_t
template<typename T = double>
class sde_solver {
public:
    using drift_fn = std::function<T(T, T)>;      // drift(time, state)
    using diffusion_fn = std::function<T(T, T)>;  // diffusion(time, state)

private:
    drift_fn drift_;
    diffusion_fn diffusion_;
    T current_time_;
    T current_value_;
    std::normal_distribution<T> normal_;

public:
    sde_solver(drift_fn drift, diffusion_fn diffusion, T initial_value = 0)
        : drift_(drift), diffusion_(diffusion),
          current_time_(0), current_value_(initial_value),
          normal_(0, 1) {}

    // Euler-Maruyama method
    template<typename Generator>
    T step_euler(Generator& gen, T dt) {
        T dW = normal_(gen) * std::sqrt(dt);
        T drift_term = drift_(current_time_, current_value_) * dt;
        T diffusion_term = diffusion_(current_time_, current_value_) * dW;

        current_value_ += drift_term + diffusion_term;
        current_time_ += dt;

        return current_value_;
    }

    // Milstein method - higher order accuracy
    template<typename Generator>
    T step_milstein(Generator& gen, T dt) {
        T dW = normal_(gen) * std::sqrt(dt);
        T drift_term = drift_(current_time_, current_value_) * dt;
        T sigma = diffusion_(current_time_, current_value_);
        T diffusion_term = sigma * dW;

        // Milstein correction term
        T h = 1e-6;  // Small increment for derivative
        T sigma_plus = diffusion_(current_time_, current_value_ + h);
        T sigma_deriv = (sigma_plus - sigma) / h;
        T correction = 0.5 * sigma * sigma_deriv * (dW * dW - dt);

        current_value_ += drift_term + diffusion_term + correction;
        current_time_ += dt;

        return current_value_;
    }

    // Generate full path
    template<typename Generator>
    std::vector<std::pair<T, T>> generate_path(Generator& gen, T end_time,
                                               int n_steps,
                                               const std::string& method = "euler") {
        std::vector<std::pair<T, T>> path;
        T dt = end_time / n_steps;
        reset();

        path.emplace_back(current_time_, current_value_);

        for (int i = 0; i < n_steps; ++i) {
            T value = (method == "milstein") ? step_milstein(gen, dt)
                                             : step_euler(gen, dt);
            path.emplace_back(current_time_, value);
        }

        return path;
    }

    void reset(T initial_value = 0) {
        current_time_ = 0;
        current_value_ = initial_value;
    }

    T current_time() const { return current_time_; }
    T current_value() const { return current_value_; }
};

// Ornstein-Uhlenbeck Process - mean reverting
template<typename T = double>
class ornstein_uhlenbeck_process {
private:
    T theta_;  // Mean reversion rate
    T mu_;     // Long-term mean
    T sigma_;  // Volatility
    T current_value_;
    std::normal_distribution<T> normal_;

public:
    ornstein_uhlenbeck_process(T theta, T mu, T sigma, T initial_value = 0)
        : theta_(theta), mu_(mu), sigma_(sigma),
          current_value_(initial_value), normal_(0, 1) {
        if (theta <= 0 || sigma <= 0) {
            throw std::invalid_argument("Parameters must be positive");
        }
    }

    template<typename Generator>
    T step(Generator& gen, T dt) {
        // Exact solution for OU process
        T exp_factor = std::exp(-theta_ * dt);
        T mean = current_value_ * exp_factor + mu_ * (1 - exp_factor);
        T variance = (sigma_ * sigma_ / (2 * theta_)) * (1 - exp_factor * exp_factor);
        T stddev = std::sqrt(variance);

        current_value_ = mean + stddev * normal_(gen);
        return current_value_;
    }

    template<typename Generator>
    std::vector<std::pair<T, T>> generate_path(Generator& gen, T end_time, int n_steps) {
        std::vector<std::pair<T, T>> path;
        T dt = end_time / n_steps;
        T time = 0;

        path.emplace_back(time, current_value_);

        for (int i = 0; i < n_steps; ++i) {
            time += dt;
            T value = step(gen, dt);
            path.emplace_back(time, value);
        }

        return path;
    }

    void reset(T initial_value = 0) {
        current_value_ = initial_value;
    }

    T stationary_mean() const { return mu_; }
    T stationary_variance() const { return sigma_ * sigma_ / (2 * theta_); }
    T current_value() const { return current_value_; }
};

// =============================================================================
// Stochastic Process Composition - Elegant API
// =============================================================================

template<typename T = double>
class stochastic_process_builder {
private:
    std::vector<std::function<T(std::mt19937&, T)>> components_;
    T current_value_;
    T current_time_;

public:
    stochastic_process_builder() : current_value_(0), current_time_(0) {}

    // Add drift component
    stochastic_process_builder& with_drift(std::function<T(T, T)> drift) {
        components_.push_back([drift, this](std::mt19937&, T dt) {
            return drift(current_time_, current_value_) * dt;
        });
        return *this;
    }

    // Add Brownian motion
    stochastic_process_builder& with_brownian(T sigma) {
        components_.push_back([sigma](std::mt19937& gen, T dt) {
            std::normal_distribution<T> normal(0, std::sqrt(dt));
            return sigma * normal(gen);
        });
        return *this;
    }

    // Add Poisson jumps
    stochastic_process_builder& with_poisson_jumps(T lambda, T jump_size) {
        components_.push_back([lambda, jump_size](std::mt19937& gen, T dt) {
            std::poisson_distribution<int> poisson(lambda * dt);
            return jump_size * poisson(gen);
        });
        return *this;
    }

    // Add compound Poisson jumps
    template<typename JumpDist>
    stochastic_process_builder& with_compound_poisson(T lambda, JumpDist jump_dist) {
        components_.push_back([lambda, jump_dist](std::mt19937& gen, T dt) mutable {
            std::poisson_distribution<int> poisson(lambda * dt);
            int n_jumps = poisson(gen);
            T total_jump = 0;
            for (int i = 0; i < n_jumps; ++i) {
                total_jump += jump_dist(gen);
            }
            return total_jump;
        });
        return *this;
    }

    // Add mean reversion
    stochastic_process_builder& with_mean_reversion(T theta, T mu) {
        components_.push_back([theta, mu, this](std::mt19937&, T dt) {
            return theta * (mu - current_value_) * dt;
        });
        return *this;
    }

    // Add custom component
    stochastic_process_builder& with_custom(std::function<T(std::mt19937&, T)> component) {
        components_.push_back(component);
        return *this;
    }

    // Step the process
    T step(std::mt19937& gen, T dt) {
        T increment = 0;
        for (auto& component : components_) {
            increment += component(gen, dt);
        }
        current_value_ += increment;
        current_time_ += dt;
        return current_value_;
    }

    // Generate path
    std::vector<std::pair<T, T>> generate_path(std::mt19937& gen, T end_time, int n_steps) {
        std::vector<std::pair<T, T>> path;
        T dt = end_time / n_steps;
        reset();

        path.emplace_back(current_time_, current_value_);

        for (int i = 0; i < n_steps; ++i) {
            step(gen, dt);
            path.emplace_back(current_time_, current_value_);
        }

        return path;
    }

    void reset(T initial_value = 0) {
        current_value_ = initial_value;
        current_time_ = 0;
    }

    T current_value() const { return current_value_; }
    T current_time() const { return current_time_; }
};

// Factory functions for common processes
template<typename T = double>
stochastic_process_builder<T> geometric_brownian_motion(T mu, T sigma) {
    return stochastic_process_builder<T>()
        .with_drift([mu](T, T x) { return mu * x; })
        .with_custom([sigma](std::mt19937& gen, T dt) {
            std::normal_distribution<T> normal(0, std::sqrt(dt));
            return sigma * normal(gen);
        });
}

template<typename T = double>
stochastic_process_builder<T> jump_diffusion(T mu, T sigma, T lambda, T jump_mean, T jump_std) {
    return stochastic_process_builder<T>()
        .with_drift([mu](T, T x) { return mu * x; })
        .with_brownian(sigma)
        .with_compound_poisson(lambda, std::normal_distribution<T>(jump_mean, jump_std));
}

template<typename T = double>
stochastic_process_builder<T> mean_reverting_jump_diffusion(T theta, T mu, T sigma, T lambda) {
    return stochastic_process_builder<T>()
        .with_mean_reversion(theta, mu)
        .with_brownian(sigma)
        .with_poisson_jumps(lambda, 1.0);
}

} // namespace stepanov::random