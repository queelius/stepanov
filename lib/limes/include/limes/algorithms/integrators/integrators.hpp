#pragma once

#include <cmath>
#include <limits>
#include <functional>
#include "../concepts/concepts.hpp"
#include "../core/result.hpp"
#include "../accumulators/accumulators.hpp"
#include "../quadrature/quadrature.hpp"

namespace limes::algorithms {

// Basic quadrature integrator using any quadrature rule and accumulator
template<
    concepts::Field T,
    concepts::QuadratureRule<T> Rule,
    concepts::Accumulator<T> Acc = accumulators::simple_accumulator<T>
>
class quadrature_integrator {
public:
    using value_type = T;
    using result_type = integration_result<T>;
    using rule_type = Rule;
    using accumulator_type = Acc;

    constexpr quadrature_integrator() noexcept = default;

    constexpr explicit quadrature_integrator(Rule rule, Acc acc = {}) noexcept
        : rule_{std::move(rule)}, accumulator_template_{std::move(acc)} {}

    template<concepts::UnivariateFunction<T> F>
    constexpr result_type operator()(F&& f, T a, T b) const {
        return integrate_interval(std::forward<F>(f), a, b);
    }

    template<concepts::UnivariateFunction<T> F>
    constexpr result_type operator()(F&& f, T a, T b, T tol) const {
        return adaptive_integrate(std::forward<F>(f), a, b, tol);
    }

private:
    Rule rule_;
    Acc accumulator_template_;

    template<typename F>
    constexpr result_type integrate_interval(F&& f, T a, T b) const {
        Acc acc = accumulator_template_;
        const T half_width = (b - a) / T(2);
        const T center = (a + b) / T(2);

        for (std::size_t i = 0; i < rule_.size(); ++i) {
            T x = center + half_width * rule_.abscissa(i);
            acc += rule_.weight(i) * f(x);
        }

        T integral = half_width * acc();
        return {integral, std::numeric_limits<T>::epsilon() * std::abs(integral), rule_.size()};
    }

    template<typename F>
    constexpr result_type adaptive_integrate(F&& f, T a, T b, T tol, std::size_t depth = 0) const {
        constexpr std::size_t max_depth = 30;

        result_type whole = integrate_interval(f, a, b);

        if (depth >= max_depth) {
            return whole;
        }

        T mid = (a + b) / T(2);
        result_type left = integrate_interval(f, a, mid);
        result_type right = integrate_interval(f, mid, b);
        result_type combined = left + right;

        T error = std::abs(whole.value() - combined.value());

        if (error <= tol) {
            // Richardson extrapolation for better accuracy
            T refined = combined.value() + (combined.value() - whole.value()) / T(15);
            return {refined, error, combined.evaluations()};
        }

        // Recursive subdivision
        result_type left_refined = adaptive_integrate(f, a, mid, tol / T(2), depth + 1);
        result_type right_refined = adaptive_integrate(f, mid, b, tol / T(2), depth + 1);

        return left_refined + right_refined;
    }
};

// Specialized integrator for infinite intervals using tanh-sinh transform
template<
    concepts::Field T,
    concepts::Accumulator<T> Acc = accumulators::neumaier_accumulator<T>
>
class tanh_sinh_integrator {
public:
    using value_type = T;
    using result_type = integration_result<T>;
    using accumulator_type = Acc;

    constexpr tanh_sinh_integrator() noexcept = default;

    template<concepts::UnivariateFunction<T> F>
    constexpr result_type operator()(F&& f, T a, T b, T tol = default_tolerance()) const {
        if (std::isinf(a) && std::isinf(b)) {
            return integrate_infinite(std::forward<F>(f), tol);
        } else if (std::isinf(a)) {
            return integrate_left_infinite(std::forward<F>(f), b, tol);
        } else if (std::isinf(b)) {
            return integrate_right_infinite(std::forward<F>(f), a, tol);
        } else {
            return integrate_finite(std::forward<F>(f), a, b, tol);
        }
    }

private:
    static constexpr T default_tolerance() { return std::sqrt(std::numeric_limits<T>::epsilon()); }

    template<typename F>
    constexpr result_type integrate_finite(F&& f, T a, T b, T tol) const {
        const T center = (a + b) / T(2);
        const T half_width = (b - a) / T(2);

        auto transformed = [&](T t) -> T {
            return f(center + half_width * t);
        };

        return integrate_symmetric(transformed, half_width, tol);
    }

    template<typename F>
    constexpr result_type integrate_infinite(F&& f, T tol) const {
        auto transformed = [&](T t) -> T {
            T scale = T(1) / (T(1) - t * t);
            return f(t * scale) * scale * scale;
        };

        return integrate_symmetric(transformed, T(1), tol);
    }

    template<typename F>
    constexpr result_type integrate_left_infinite(F&& f, T b, T tol) const {
        auto transformed = [&](T t) -> T {
            T scale = T(1) / (T(1) + t);
            T x = b - scale * (T(1) - t);
            return f(x) * scale * scale;
        };

        return integrate_symmetric(transformed, T(1), tol);
    }

    template<typename F>
    constexpr result_type integrate_right_infinite(F&& f, T a, T tol) const {
        auto transformed = [&](T t) -> T {
            T scale = T(1) / (T(1) - t);
            T x = a + scale * (T(1) + t);
            return f(x) * scale * scale;
        };

        return integrate_symmetric(transformed, T(1), tol);
    }

    template<typename F>
    constexpr result_type integrate_symmetric(F&& f, T scale, T tol) const {
        quadrature::tanh_sinh_nodes<T> nodes;
        Acc acc{};
        T h = T(1);
        T prev_sum = T(0);
        std::size_t total_evals = 0;

        for (std::size_t level = 0; level < nodes.max_level; ++level) {
            Acc level_acc{};
            std::size_t level_evals = 0;

            if (level == 0) {
                T w = nodes.weight(0, 0);
                level_acc += w * f(T(0));
                level_evals = 1;
            } else {
                std::size_t n = 1 << (level - 1);
                for (std::size_t i = 1; i <= n; ++i) {
                    std::size_t index = 2 * i - 1;
                    T x = nodes.abscissa(level, index);
                    T w = nodes.weight(level, index);

                    T val = f(x) + f(-x);
                    if (!std::isfinite(val)) break;

                    level_acc += w * val;
                    level_evals += 2;
                }
            }

            acc += level_acc();
            total_evals += level_evals;

            T sum = scale * h * acc();
            T error = std::abs(sum - prev_sum);

            if (level > 3 && error < tol * std::abs(sum)) {
                return {sum, error, level + 1, total_evals};
            }

            prev_sum = sum;
            h /= T(2);
        }

        return {scale * h * acc(), tol, nodes.max_level, total_evals};
    }
};

// Romberg integrator with Richardson extrapolation
template<
    concepts::Field T,
    concepts::Accumulator<T> Acc = accumulators::kahan_accumulator<T>
>
class romberg_integrator {
public:
    using value_type = T;
    using result_type = integration_result<T>;
    using accumulator_type = Acc;

    static constexpr std::size_t max_iterations = 20;

    constexpr romberg_integrator() noexcept = default;

    template<concepts::UnivariateFunction<T> F>
    constexpr result_type operator()(F&& f, T a, T b, T tol = default_tolerance()) const {
        T R[max_iterations][max_iterations] = {};

        // Initial trapezoidal estimate
        R[0][0] = (b - a) * (f(a) + f(b)) / T(2);
        std::size_t total_evals = 2;

        for (std::size_t i = 1; i < max_iterations; ++i) {
            // Refine trapezoidal rule
            std::size_t n = 1 << (i - 1);
            T h = (b - a) / T(n << 1);
            Acc sum{};

            for (std::size_t k = 0; k < n; ++k) {
                sum += f(a + h * (T(2) * k + T(1)));
            }

            R[i][0] = R[i-1][0] / T(2) + h * sum();
            total_evals += n;

            // Richardson extrapolation
            T power = T(4);
            for (std::size_t j = 1; j <= i; ++j) {
                R[i][j] = R[i][j-1] + (R[i][j-1] - R[i-1][j-1]) / (power - T(1));
                power *= T(4);
            }

            // Check convergence
            if (i > 3) {
                T error = std::abs(R[i][i] - R[i-1][i-1]);
                if (error < tol * std::abs(R[i][i])) {
                    return {R[i][i], error, i + 1, total_evals};
                }
            }
        }

        return {R[max_iterations-1][max_iterations-1], tol, max_iterations, total_evals};
    }

private:
    static constexpr T default_tolerance() {
        return T(1000) * std::numeric_limits<T>::epsilon();
    }
};

// Recommended adaptive integrator: Gauss-Kronrod 15 with Neumaier accumulation
template<concepts::Field T>
using adaptive_integrator = quadrature_integrator<
    T,
    quadrature::gauss_kronrod_15<T>,
    accumulators::neumaier_accumulator<T>
>;

} // namespace limes::algorithms
