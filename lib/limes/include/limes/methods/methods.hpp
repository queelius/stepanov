#pragma once

/**
 * @file methods.hpp
 * @brief Integration method objects for composable numerical integration.
 *
 * Methods are first-class objects passed to `.eval()` on integrals to control
 * how integration is performed.
 *
 * @code{.cpp}
 * using namespace limes::expr;
 * using namespace limes::methods;
 *
 * auto x = arg<0>;
 * auto I = integral(sin(x)).over<0>(0.0, 3.14159);
 *
 * auto r1 = I.eval(gauss<7>());
 * auto r2 = I.eval(adaptive_method().with_tolerance(1e-12));
 * auto r3 = I.eval(monte_carlo_method(100000).with_seed(42));
 * auto r4 = I.eval(simpson_method<100>());
 * @endcode
 */

#include <cstddef>
#include <optional>
#include <random>
#include <functional>
#include "concepts.hpp"
#include "../algorithms/integrators/integrators.hpp"
#include "../algorithms/quadrature/quadrature.hpp"

namespace limes::methods {

// =============================================================================
// Gauss-Legendre Quadrature Method
// =============================================================================

/// N-point Gauss-Legendre quadrature, achieving degree 2N-1 polynomial exactness.
template<std::size_t N, typename T = double>
struct gauss_legendre {
    static constexpr std::size_t order = N;
    using value_type = T;

    constexpr gauss_legendre() noexcept = default;

    [[nodiscard]] algorithms::integration_result<T>
    operator()(std::function<T(T)> const& f, T a, T b) const {
        algorithms::quadrature_integrator<T, algorithms::quadrature::gauss_legendre<T, N>> integrator;
        return integrator(f, a, b);
    }
};

template<std::size_t N, typename T>
struct is_integration_method<gauss_legendre<N, T>> : std::true_type {};

/// Factory for Gauss-Legendre quadrature.
template<std::size_t N, typename T = double>
[[nodiscard]] constexpr auto gauss() {
    return gauss_legendre<N, T>{};
}

// =============================================================================
// Adaptive Integration Method
// =============================================================================

/// Adaptive integration with recursive interval subdivision until convergence.
template<typename T = double>
struct adaptive {
    using value_type = T;

    T tolerance;
    std::size_t max_subdivisions;

    constexpr adaptive(T tol = T(1e-10), std::size_t max_sub = 1000) noexcept
        : tolerance{tol}, max_subdivisions{max_sub} {}

    [[nodiscard]] algorithms::integration_result<T>
    operator()(std::function<T(T)> const& f, T a, T b) const {
        algorithms::adaptive_integrator<T> integrator;
        return integrator(f, a, b, tolerance);
    }

    [[nodiscard]] constexpr adaptive with_tolerance(T tol) const noexcept {
        return adaptive{tol, max_subdivisions};
    }

    [[nodiscard]] constexpr adaptive with_max_subdivisions(std::size_t max_sub) const noexcept {
        return adaptive{tolerance, max_sub};
    }
};

template<typename T>
struct is_integration_method<adaptive<T>> : std::true_type {};

/// Factory for adaptive integration.
template<typename T = double>
[[nodiscard]] constexpr auto adaptive_method(T tol = T(1e-10)) {
    return adaptive<T>{tol};
}

// =============================================================================
// Monte Carlo Integration Method
// =============================================================================

/// Monte Carlo integration using random sampling. Error decreases as O(1/sqrt(n)).
template<typename T = double>
struct monte_carlo {
    using value_type = T;

    std::size_t samples;
    std::optional<std::size_t> seed;

    constexpr monte_carlo(std::size_t n, std::optional<std::size_t> s = std::nullopt) noexcept
        : samples{n}, seed{s} {}

    [[nodiscard]] algorithms::integration_result<T>
    operator()(std::function<T(T)> const& f, T a, T b) const {
        std::mt19937_64 rng;
        if (seed) {
            rng.seed(*seed);
        } else {
            std::random_device rd;
            rng.seed(rd());
        }

        std::uniform_real_distribution<T> dist(a, b);

        T sum = T(0);
        T sum_sq = T(0);

        for (std::size_t i = 0; i < samples; ++i) {
            T y = f(dist(rng));
            sum += y;
            sum_sq += y * y;
        }

        T interval_length = b - a;
        T mean = sum / T(samples);
        T variance = (sum_sq / T(samples) - mean * mean) / T(samples);
        T value = mean * interval_length;
        T error = std::sqrt(variance) * interval_length;

        algorithms::integration_result<T> result{value, error, samples, samples};
        result.variance_ = variance;
        return result;
    }

    [[nodiscard]] constexpr monte_carlo with_seed(std::size_t s) const noexcept {
        return monte_carlo{samples, s};
    }

    [[nodiscard]] constexpr monte_carlo with_samples(std::size_t n) const noexcept {
        return monte_carlo{n, seed};
    }
};

template<typename T>
struct is_integration_method<monte_carlo<T>> : std::true_type {};

/// Factory for Monte Carlo integration.
template<typename T = double>
[[nodiscard]] constexpr auto monte_carlo_method(std::size_t n) {
    return monte_carlo<T>{n};
}

// =============================================================================
// Simpson's Rule Method
// =============================================================================

/// Simpson's 1/3 rule with N subdivisions (N must be even). Achieves O(h^4) convergence.
template<std::size_t N, typename T = double>
struct simpson {
    static_assert(N % 2 == 0, "Simpson's rule requires even number of subdivisions");

    static constexpr std::size_t subdivisions = N;
    using value_type = T;

    constexpr simpson() noexcept = default;

    [[nodiscard]] algorithms::integration_result<T>
    operator()(std::function<T(T)> const& f, T a, T b) const {
        T h = (b - a) / T(N);
        T sum = f(a) + f(b);

        for (std::size_t i = 1; i < N; i += 2) {
            sum += T(4) * f(a + T(i) * h);
        }
        for (std::size_t i = 2; i < N; i += 2) {
            sum += T(2) * f(a + T(i) * h);
        }

        T value = sum * h / T(3);
        T error = std::abs(value) * std::pow(h, 4);

        return algorithms::integration_result<T>{value, error, N + 1, N + 1};
    }
};

template<std::size_t N, typename T>
struct is_integration_method<simpson<N, T>> : std::true_type {};

/// Factory for Simpson's rule.
template<std::size_t N, typename T = double>
[[nodiscard]] constexpr auto simpson_method() {
    return simpson<N, T>{};
}

// =============================================================================
// Trapezoidal Rule Method
// =============================================================================

/// Trapezoidal rule with N subdivisions. Achieves O(h^2) convergence.
template<std::size_t N, typename T = double>
struct trapezoidal {
    static constexpr std::size_t subdivisions = N;
    using value_type = T;

    constexpr trapezoidal() noexcept = default;

    [[nodiscard]] algorithms::integration_result<T>
    operator()(std::function<T(T)> const& f, T a, T b) const {
        T h = (b - a) / T(N);
        T sum = (f(a) + f(b)) / T(2);

        for (std::size_t i = 1; i < N; ++i) {
            sum += f(a + T(i) * h);
        }

        T value = sum * h;
        T error = std::abs(value) * h * h;

        return algorithms::integration_result<T>{value, error, N + 1, N + 1};
    }
};

template<std::size_t N, typename T>
struct is_integration_method<trapezoidal<N, T>> : std::true_type {};

/// Factory for trapezoidal rule.
template<std::size_t N, typename T = double>
[[nodiscard]] constexpr auto trapezoidal_method() {
    return trapezoidal<N, T>{};
}

// =============================================================================
// Composed Adaptive Method (wraps another method for adaptive refinement)
// =============================================================================

/// Wraps any base method with adaptive interval subdivision until convergence.
template<typename BaseMethod, typename T = double>
struct adaptive_composed {
    using value_type = T;

    BaseMethod base;
    T tolerance;
    std::size_t max_depth;

    constexpr adaptive_composed(BaseMethod m, T tol = T(1e-10), std::size_t depth = 20) noexcept
        : base{m}, tolerance{tol}, max_depth{depth} {}

    [[nodiscard]] algorithms::integration_result<T>
    operator()(std::function<T(T)> const& f, T a, T b) const {
        return integrate_adaptive(f, a, b, 0);
    }

    [[nodiscard]] constexpr adaptive_composed with_tolerance(T tol) const noexcept {
        return adaptive_composed{base, tol, max_depth};
    }

private:
    [[nodiscard]] algorithms::integration_result<T>
    integrate_adaptive(std::function<T(T)> const& f, T a, T b, std::size_t depth) const {
        auto full = base(f, a, b);

        if (depth >= max_depth) {
            full.converged_ = false;
            return full;
        }

        T mid = (a + b) / T(2);
        auto left = base(f, a, mid);
        auto right = base(f, mid, b);
        auto refined = left + right;

        T error = std::abs(full.value() - refined.value());

        if (error < tolerance) {
            refined.error_ = error;
            return refined;
        }

        auto left_refined = integrate_adaptive(f, a, mid, depth + 1);
        auto right_refined = integrate_adaptive(f, mid, b, depth + 1);

        return left_refined + right_refined;
    }
};

template<typename M, typename T>
struct is_integration_method<adaptive_composed<M, T>> : std::true_type {};

/// Factory for adaptive composition of any base method.
template<typename M, typename T = double>
[[nodiscard]] constexpr auto make_adaptive(M base_method, T tol = T(1e-10)) {
    return adaptive_composed<M, T>{base_method, tol};
}

// =============================================================================
// Default method
// =============================================================================

/// The default integration method (adaptive Gauss-Legendre)
template<typename T = double>
using default_method = adaptive<T>;

} // namespace limes::methods
