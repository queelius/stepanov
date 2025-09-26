#pragma once

#include "../concepts.hpp"
#include <concepts>
#include <cmath>
#include <functional>
#include <limits>
#include <utility>
#include <array>

namespace stepanov {
namespace integration {

// Integration result type
template<typename T>
struct integration_result {
    T value;
    T error_estimate;
    std::size_t evaluations;

    operator T() const { return value; }

    integration_result(T v = T{}, T e = T{}, std::size_t n = 0)
        : value(v), error_estimate(e), evaluations(n) {}
};

// Adaptive integration using Simpson's rule
template<field T>
class simpson_integrator {
private:
    static constexpr std::size_t max_depth = 20;
    static constexpr T default_tolerance = std::numeric_limits<T>::epsilon() * T(100);

    template<typename F>
    static T simpson_rule(F&& f, T a, T b) {
        T h = (b - a) / T(2);
        T mid = (a + b) / T(2);
        return h / T(3) * (f(a) + T(4) * f(mid) + f(b));
    }

    template<typename F>
    integration_result<T> adaptive_simpson_impl(
        F&& f, T a, T b, T tolerance,
        T whole, std::size_t depth, std::size_t& evaluations
    ) const {
        if (depth >= max_depth) {
            return {whole, std::numeric_limits<T>::infinity(), evaluations};
        }

        T mid = (a + b) / T(2);
        T left = simpson_rule(f, a, mid);
        T right = simpson_rule(f, mid, b);
        evaluations += 4;

        T delta = left + right - whole;
        T error = std::abs(delta) / T(15);

        if (error < tolerance || depth >= max_depth - 1) {
            return {left + right + delta / T(15), error, evaluations};
        }

        auto left_result = adaptive_simpson_impl(
            f, a, mid, tolerance / T(2), left, depth + 1, evaluations
        );
        auto right_result = adaptive_simpson_impl(
            f, mid, b, tolerance / T(2), right, depth + 1, evaluations
        );

        return {
            left_result.value + right_result.value,
            left_result.error_estimate + right_result.error_estimate,
            evaluations
        };
    }

public:
    template<typename F>
    integration_result<T> operator()(F&& f, T a, T b, T tolerance = default_tolerance) const {
        std::size_t evaluations = 3;
        T whole = simpson_rule(f, a, b);
        return adaptive_simpson_impl(f, a, b, tolerance, whole, 0, evaluations);
    }
};

// Gauss-Legendre quadrature
template<field T, std::size_t N>
class gauss_legendre {
private:
    struct quadrature_point {
        T weight;
        T abscissa;
    };

    std::array<quadrature_point, N> points;

    // Initialize Gauss-Legendre nodes and weights
    void initialize() {
        // For demonstration, implementing 2, 3, 4, 5 point rules
        if constexpr (N == 2) {
            T sqrt3 = std::sqrt(T(3));
            points[0] = {T(1), -T(1)/sqrt3};
            points[1] = {T(1), T(1)/sqrt3};
        } else if constexpr (N == 3) {
            T sqrt35 = std::sqrt(T(3)/T(5));
            points[0] = {T(5)/T(9), -sqrt35};
            points[1] = {T(8)/T(9), T(0)};
            points[2] = {T(5)/T(9), sqrt35};
        } else if constexpr (N == 4) {
            T a = std::sqrt(T(3)/T(7) - T(2)/T(7)*std::sqrt(T(6)/T(5)));
            T b = std::sqrt(T(3)/T(7) + T(2)/T(7)*std::sqrt(T(6)/T(5)));
            T wa = (T(18) + std::sqrt(T(30)))/T(36);
            T wb = (T(18) - std::sqrt(T(30)))/T(36);
            points[0] = {wb, -b};
            points[1] = {wa, -a};
            points[2] = {wa, a};
            points[3] = {wb, b};
        } else if constexpr (N == 5) {
            T sqrt5 = std::sqrt(T(5));
            T sqrt10_7 = std::sqrt(T(10)/T(7));
            T x1 = T(1)/T(3) * std::sqrt(T(5) - T(2)*sqrt10_7);
            T x2 = T(1)/T(3) * std::sqrt(T(5) + T(2)*sqrt10_7);
            T w0 = T(128)/T(225);
            T w1 = (T(322) + T(13)*sqrt(T(70)))/T(900);
            T w2 = (T(322) - T(13)*sqrt(T(70)))/T(900);
            points[0] = {w2, -x2};
            points[1] = {w1, -x1};
            points[2] = {w0, T(0)};
            points[3] = {w1, x1};
            points[4] = {w2, x2};
        }
    }

public:
    gauss_legendre() { initialize(); }

    template<typename F>
    T operator()(F&& f, T a, T b) const {
        T sum = T(0);
        T scale = (b - a) / T(2);
        T shift = (a + b) / T(2);

        for (const auto& point : points) {
            T x = scale * point.abscissa + shift;
            sum += point.weight * f(x);
        }

        return scale * sum;
    }

    template<typename F>
    integration_result<T> integrate(F&& f, T a, T b) const {
        T value = operator()(f, a, b);
        return {value, T(0), N};  // No error estimate for fixed quadrature
    }
};

// Romberg integration
template<field T>
class romberg_integrator {
private:
    static constexpr std::size_t max_iterations = 20;
    static constexpr T default_tolerance = std::numeric_limits<T>::epsilon() * T(1000);

    template<typename F>
    T trapezoid_rule(F&& f, T a, T b, std::size_t n) const {
        T h = (b - a) / T(n);
        T sum = (f(a) + f(b)) / T(2);

        for (std::size_t i = 1; i < n; ++i) {
            sum += f(a + T(i) * h);
        }

        return h * sum;
    }

public:
    template<typename F>
    integration_result<T> operator()(F&& f, T a, T b, T tolerance = default_tolerance) const {
        std::vector<std::vector<T>> rom(max_iterations, std::vector<T>(max_iterations));
        std::size_t evaluations = 0;

        // First trapezoid estimate
        rom[0][0] = trapezoid_rule(f, a, b, 1);
        evaluations += 2;

        for (std::size_t i = 1; i < max_iterations; ++i) {
            // Richardson extrapolation
            std::size_t n = 1 << i;  // 2^i
            rom[i][0] = trapezoid_rule(f, a, b, n);
            evaluations += n - 1;  // We already evaluated endpoints

            T power_of_4 = T(4);
            for (std::size_t j = 1; j <= i; ++j) {
                rom[i][j] = (power_of_4 * rom[i][j-1] - rom[i-1][j-1]) / (power_of_4 - T(1));
                power_of_4 *= T(4);
            }

            // Check convergence
            if (i > 0) {
                T error = std::abs(rom[i][i] - rom[i-1][i-1]);
                if (error < tolerance) {
                    return {rom[i][i], error, evaluations};
                }
            }
        }

        // Return best estimate if max iterations reached
        std::size_t last = max_iterations - 1;
        T error = std::abs(rom[last][last] - rom[last-1][last-1]);
        return {rom[last][last], error, evaluations};
    }
};

// Double exponential (tanh-sinh) integration for improper integrals
template<field T>
class double_exponential_integrator {
private:
    static constexpr std::size_t max_levels = 10;
    static constexpr T default_tolerance = std::numeric_limits<T>::epsilon() * T(100);

    struct node {
        T weight;
        T abscissa;
    };

    std::vector<std::vector<node>> levels;

    void initialize() {
        // Precompute nodes for efficiency
        T h = T(1);

        for (std::size_t level = 0; level < max_levels; ++level) {
            std::vector<node> nodes;
            T t = h / T(2);

            while (t < T(4)) {
                T exp_t = std::exp(t);
                T exp_neg_t = T(1) / exp_t;
                T sinh_t = (exp_t - exp_neg_t) / T(2);
                T cosh_t = (exp_t + exp_neg_t) / T(2);

                T exp_sinh = std::exp(T(M_PI) / T(2) * sinh_t);
                T x = (exp_sinh - T(1) / exp_sinh) / (exp_sinh + T(1) / exp_sinh);
                T w = T(M_PI) / T(2) * cosh_t / (exp_sinh + T(1) / exp_sinh) /
                      (exp_sinh + T(1) / exp_sinh) * h;

                if (std::abs(x) < T(1) - std::numeric_limits<T>::epsilon()) {
                    nodes.push_back({w, x});
                }

                t += h;
            }

            levels.push_back(std::move(nodes));
            h /= T(2);
        }
    }

public:
    double_exponential_integrator() { initialize(); }

    template<typename F>
    integration_result<T> operator()(F&& f, T a, T b, T tolerance = default_tolerance) const {
        bool infinite_a = std::isinf(a);
        bool infinite_b = std::isinf(b);

        if (infinite_a && infinite_b) {
            // Transform (-∞, ∞) to (-1, 1)
            auto g = [&f](T t) {
                T scale = T(1) / (T(1) - t * t);
                T x = t * scale;
                return f(x) * scale * scale * (T(1) + t * t);
            };
            return integrate_finite(g, T(-1), T(1), tolerance);
        } else if (infinite_a) {
            // Transform (-∞, b] to [-1, 1]
            auto g = [&f, b](T t) {
                T scale = T(1) / (T(1) + t);
                T x = b - (T(1) - t) * scale;
                return f(x) * scale * scale * T(2);
            };
            return integrate_finite(g, T(-1), T(1), tolerance);
        } else if (infinite_b) {
            // Transform [a, ∞) to [-1, 1]
            auto g = [&f, a](T t) {
                T scale = T(1) / (T(1) - t);
                T x = a + (T(1) + t) * scale;
                return f(x) * scale * scale * T(2);
            };
            return integrate_finite(g, T(-1), T(1), tolerance);
        } else {
            // Finite interval: transform [a, b] to [-1, 1]
            T scale = (b - a) / T(2);
            T shift = (a + b) / T(2);
            auto g = [&f, scale, shift](T t) {
                return f(scale * t + shift);
            };
            auto result = integrate_finite(g, T(-1), T(1), tolerance);
            result.value *= scale;
            result.error_estimate *= scale;
            return result;
        }
    }

private:
    template<typename F>
    integration_result<T> integrate_finite(F&& f, T a, T b, T tolerance) const {
        T integral = T(0);
        T prev_integral = T(0);
        std::size_t evaluations = 0;

        // Midpoint
        T mid = (a + b) / T(2);
        T f_mid = f(mid);
        evaluations++;

        for (std::size_t level = 0; level < levels.size(); ++level) {
            T level_sum = f_mid;

            for (const auto& node : levels[level]) {
                if (node.abscissa > T(0)) {
                    T x1 = mid + (b - mid) * node.abscissa;
                    T x2 = mid - (mid - a) * node.abscissa;
                    level_sum += node.weight * (f(x1) + f(x2));
                    evaluations += 2;
                }
            }

            integral = level_sum * (b - a) / T(2);

            if (level > 0) {
                T error = std::abs(integral - prev_integral);
                if (error < tolerance) {
                    return {integral, error, evaluations};
                }
            }

            prev_integral = integral;
        }

        return {integral, std::abs(integral - prev_integral), evaluations};
    }
};

// Convenience function for adaptive integration
template<field T, typename F>
integration_result<T> integrate(F&& f, T a, T b, T tolerance = std::numeric_limits<T>::epsilon() * T(100)) {
    if (std::isinf(a) || std::isinf(b)) {
        double_exponential_integrator<T> integrator;
        return integrator(std::forward<F>(f), a, b, tolerance);
    } else {
        simpson_integrator<T> integrator;
        return integrator(std::forward<F>(f), a, b, tolerance);
    }
}

// Line integral for parametric curves
template<field T>
class line_integral {
public:
    // Scalar field line integral: ∫_C f(r(t)) |r'(t)| dt
    template<typename ScalarField, typename Curve, typename CurveDerivative>
    integration_result<T> scalar_field(
        ScalarField&& f,
        Curve&& r,
        CurveDerivative&& r_prime,
        T t0, T t1,
        T tolerance = std::numeric_limits<T>::epsilon() * T(100)
    ) {
        auto integrand = [&](T t) {
            auto position = r(t);
            auto velocity = r_prime(t);
            T speed = T(0);

            // Calculate |r'(t)|
            if constexpr (std::is_arithmetic_v<decltype(velocity)>) {
                speed = std::abs(velocity);
            } else {
                // Assume vector with norm() method or array-like
                for (const auto& component : velocity) {
                    speed += component * component;
                }
                speed = std::sqrt(speed);
            }

            return f(position) * speed;
        };

        simpson_integrator<T> integrator;
        return integrator(integrand, t0, t1, tolerance);
    }

    // Vector field line integral: ∫_C F(r(t)) · r'(t) dt
    template<typename VectorField, typename Curve, typename CurveDerivative>
    integration_result<T> vector_field(
        VectorField&& F,
        Curve&& r,
        CurveDerivative&& r_prime,
        T t0, T t1,
        T tolerance = std::numeric_limits<T>::epsilon() * T(100)
    ) {
        auto integrand = [&](T t) {
            auto position = r(t);
            auto velocity = r_prime(t);
            auto field = F(position);

            // Calculate dot product F · r'
            T dot = T(0);
            if constexpr (std::is_arithmetic_v<decltype(field)>) {
                dot = field * velocity;
            } else {
                // Assume both are vectors/arrays of same dimension
                auto it_f = std::begin(field);
                auto it_v = std::begin(velocity);
                while (it_f != std::end(field)) {
                    dot += (*it_f) * (*it_v);
                    ++it_f;
                    ++it_v;
                }
            }

            return dot;
        };

        simpson_integrator<T> integrator;
        return integrator(integrand, t0, t1, tolerance);
    }
};

} // namespace integration
} // namespace stepanov