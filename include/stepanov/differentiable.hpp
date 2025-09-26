// differentiable.hpp - Differentiable Programming
// Makes discrete algorithms continuous and differentiable
// Enables gradient-based optimization of traditionally non-differentiable operations

#ifndef STEPANOV_DIFFERENTIABLE_HPP
#define STEPANOV_DIFFERENTIABLE_HPP

#include <concepts>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>
#include <optional>
#include <memory>
#include <unordered_map>
#include <array>
#include <limits>

namespace stepanov::differentiable {

// ============================================================================
// Dual Numbers - Forward Mode Automatic Differentiation
// ============================================================================

template<typename T = double>
struct dual {
    T value;
    T derivative;

    dual(T v = 0, T d = 0) : value(v), derivative(d) {}

    // Arithmetic operations
    dual operator+(const dual& other) const {
        return dual(value + other.value, derivative + other.derivative);
    }

    dual operator-(const dual& other) const {
        return dual(value - other.value, derivative - other.derivative);
    }

    dual operator*(const dual& other) const {
        return dual(value * other.value,
                   derivative * other.value + value * other.derivative);
    }

    dual operator/(const dual& other) const {
        T denom = other.value * other.value;
        return dual(value / other.value,
                   (derivative * other.value - value * other.derivative) / denom);
    }

    // Transcendental functions
    friend dual sin(const dual& x) {
        return dual(std::sin(x.value), x.derivative * std::cos(x.value));
    }

    friend dual cos(const dual& x) {
        return dual(std::cos(x.value), -x.derivative * std::sin(x.value));
    }

    friend dual exp(const dual& x) {
        T e = std::exp(x.value);
        return dual(e, x.derivative * e);
    }

    friend dual log(const dual& x) {
        return dual(std::log(x.value), x.derivative / x.value);
    }

    friend dual sqrt(const dual& x) {
        T s = std::sqrt(x.value);
        return dual(s, x.derivative / (2 * s));
    }

    friend dual pow(const dual& x, const dual& y) {
        T p = std::pow(x.value, y.value);
        return dual(p, p * (y.derivative * std::log(x.value) +
                           y.value * x.derivative / x.value));
    }

    // Comparison (based on value only)
    bool operator<(const dual& other) const { return value < other.value; }
    bool operator>(const dual& other) const { return value > other.value; }
    bool operator<=(const dual& other) const { return value <= other.value; }
    bool operator>=(const dual& other) const { return value >= other.value; }
};

// ============================================================================
// Differentiable Sorting - Smooth Approximations of Discrete Operations
// ============================================================================

template<typename T = double>
class soft_sort {
    T temperature_;  // Controls smoothness (lower = closer to hard sort)

public:
    explicit soft_sort(T temp = 1.0) : temperature_(temp) {}

    // Soft maximum using log-sum-exp
    dual<T> soft_max(const std::vector<dual<T>>& values) const {
        // Numerically stable computation
        dual<T> max_val = *std::max_element(values.begin(), values.end());

        dual<T> sum_exp(0, 0);
        for (const auto& v : values) {
            dual<T> scaled = (v - max_val) / temperature_;
            sum_exp = sum_exp + exp(scaled);
        }

        return max_val + temperature_ * log(sum_exp);
    }

    // Soft minimum
    dual<T> soft_min(const std::vector<dual<T>>& values) const {
        std::vector<dual<T>> negated;
        negated.reserve(values.size());

        for (const auto& v : values) {
            negated.push_back(dual<T>(-v.value, -v.derivative));
        }

        auto result = soft_max(negated);
        return dual<T>(-result.value, -result.derivative);
    }

    // Soft ranking - returns smooth approximation of ranks
    std::vector<dual<T>> soft_ranks(const std::vector<dual<T>>& values) const {
        size_t n = values.size();
        std::vector<dual<T>> ranks(n);

        for (size_t i = 0; i < n; ++i) {
            dual<T> rank(0, 0);

            for (size_t j = 0; j < n; ++j) {
                if (i != j) {
                    // Sigmoid comparison
                    dual<T> diff = (values[j] - values[i]) / temperature_;
                    dual<T> sigmoid = dual<T>(1, 0) / (dual<T>(1, 0) + exp(-diff));
                    rank = rank + sigmoid;
                }
            }

            ranks[i] = rank + dual<T>(1, 0);  // 1-indexed ranks
        }

        return ranks;
    }

    // Soft sorting network - differentiable approximation
    std::vector<dual<T>> soft_sort_values(std::vector<dual<T>> values) const {
        size_t n = values.size();

        // Bitonic sort network with soft comparators
        for (size_t k = 2; k <= n; k *= 2) {
            for (size_t j = k / 2; j > 0; j /= 2) {
                for (size_t i = 0; i < n; ++i) {
                    size_t ixj = i ^ j;
                    if (ixj > i) {
                        if ((i & k) == 0 && values[i] > values[ixj]) {
                            soft_swap(values[i], values[ixj]);
                        }
                        if ((i & k) != 0 && values[i] < values[ixj]) {
                            soft_swap(values[i], values[ixj]);
                        }
                    }
                }
            }
        }

        return values;
    }

    // Top-k selection with soft approximation
    std::vector<dual<T>> soft_top_k(const std::vector<dual<T>>& values, size_t k) const {
        auto ranks = soft_ranks(values);
        std::vector<dual<T>> result;

        for (size_t i = 0; i < values.size(); ++i) {
            // Soft threshold based on rank
            dual<T> threshold = dual<T>(k + 0.5, 0);
            dual<T> diff = (threshold - ranks[i]) / temperature_;
            dual<T> weight = dual<T>(1, 0) / (dual<T>(1, 0) + exp(-diff));

            result.push_back(values[i] * weight);
        }

        return result;
    }

private:
    void soft_swap(dual<T>& a, dual<T>& b) const {
        // Soft swap using sigmoid blending
        dual<T> diff = (a - b) / temperature_;
        dual<T> weight = dual<T>(1, 0) / (dual<T>(1, 0) + exp(-diff));

        dual<T> new_a = weight * b + (dual<T>(1, 0) - weight) * a;
        dual<T> new_b = weight * a + (dual<T>(1, 0) - weight) * b;

        a = new_a;
        b = new_b;
    }
};

// ============================================================================
// Differentiable Search - Smooth Approximations of Discrete Search
// ============================================================================

template<typename T = double>
class soft_search {
    T temperature_;

public:
    explicit soft_search(T temp = 1.0) : temperature_(temp) {}

    // Soft binary search - returns weighted average position
    dual<T> soft_binary_search(const std::vector<dual<T>>& sorted_array,
                               const dual<T>& target) const {
        size_t n = sorted_array.size();
        dual<T> position(0, 0);
        dual<T> total_weight(0, 0);

        for (size_t i = 0; i < n; ++i) {
            // Gaussian-like weight based on distance to target
            dual<T> diff = sorted_array[i] - target;
            dual<T> dist_sq = diff * diff;
            dual<T> weight = exp(-dist_sq / (temperature_ * temperature_));

            position = position + dual<T>(i, 0) * weight;
            total_weight = total_weight + weight;
        }

        return position / total_weight;
    }

    // Soft argmax - returns weighted index
    dual<T> soft_argmax(const std::vector<dual<T>>& values) const {
        dual<T> weighted_index(0, 0);
        dual<T> total_weight(0, 0);

        // Find max for numerical stability
        dual<T> max_val = *std::max_element(values.begin(), values.end());

        for (size_t i = 0; i < values.size(); ++i) {
            dual<T> weight = exp((values[i] - max_val) / temperature_);
            weighted_index = weighted_index + dual<T>(i, 0) * weight;
            total_weight = total_weight + weight;
        }

        return weighted_index / total_weight;
    }

    // Soft k-nearest neighbors
    std::vector<dual<T>> soft_knn(const std::vector<dual<T>>& points,
                                  const dual<T>& query,
                                  size_t k) const {
        std::vector<dual<T>> distances;
        for (const auto& p : points) {
            dual<T> diff = p - query;
            distances.push_back(sqrt(diff * diff));
        }

        // Soft selection of k nearest
        soft_sort<T> sorter(temperature_);
        return sorter.soft_top_k(distances, k);
    }
};

// ============================================================================
// Differentiable Dynamic Programming
// ============================================================================

template<typename T = double>
class soft_dynamic_programming {
    T temperature_;

public:
    explicit soft_dynamic_programming(T temp = 1.0) : temperature_(temp) {}

    // Soft longest common subsequence
    dual<T> soft_lcs(const std::vector<dual<T>>& seq1,
                     const std::vector<dual<T>>& seq2) const {
        size_t m = seq1.size();
        size_t n = seq2.size();

        std::vector<std::vector<dual<T>>> dp(m + 1,
            std::vector<dual<T>>(n + 1, dual<T>(0, 0)));

        for (size_t i = 1; i <= m; ++i) {
            for (size_t j = 1; j <= n; ++j) {
                // Soft equality check
                dual<T> diff = seq1[i-1] - seq2[j-1];
                dual<T> similarity = exp(-(diff * diff) / (temperature_ * temperature_));

                // Soft maximum of three choices
                std::vector<dual<T>> choices = {
                    dp[i-1][j],      // Delete from seq1
                    dp[i][j-1],      // Delete from seq2
                    dp[i-1][j-1] + similarity  // Match/mismatch
                };

                soft_sort<T> sorter(temperature_);
                dp[i][j] = sorter.soft_max(choices);
            }
        }

        return dp[m][n];
    }

    // Soft edit distance (Levenshtein)
    dual<T> soft_edit_distance(const std::vector<dual<T>>& seq1,
                               const std::vector<dual<T>>& seq2) const {
        size_t m = seq1.size();
        size_t n = seq2.size();

        std::vector<std::vector<dual<T>>> dp(m + 1,
            std::vector<dual<T>>(n + 1));

        // Initialize base cases
        for (size_t i = 0; i <= m; ++i) {
            dp[i][0] = dual<T>(i, 0);
        }
        for (size_t j = 0; j <= n; ++j) {
            dp[0][j] = dual<T>(j, 0);
        }

        for (size_t i = 1; i <= m; ++i) {
            for (size_t j = 1; j <= n; ++j) {
                // Soft equality check
                dual<T> diff = seq1[i-1] - seq2[j-1];
                dual<T> mismatch_cost = dual<T>(1, 0) -
                    exp(-(diff * diff) / (temperature_ * temperature_));

                // Soft minimum of three choices
                std::vector<dual<T>> choices = {
                    dp[i-1][j] + dual<T>(1, 0),     // Deletion
                    dp[i][j-1] + dual<T>(1, 0),     // Insertion
                    dp[i-1][j-1] + mismatch_cost    // Substitution
                };

                soft_sort<T> sorter(temperature_);
                dp[i][j] = sorter.soft_min(choices);
            }
        }

        return dp[m][n];
    }

    // Soft knapsack problem
    dual<T> soft_knapsack(const std::vector<dual<T>>& values,
                          const std::vector<dual<T>>& weights,
                          const dual<T>& capacity) const {
        size_t n = values.size();
        size_t max_cap = static_cast<size_t>(capacity.value);

        std::vector<std::vector<dual<T>>> dp(n + 1,
            std::vector<dual<T>>(max_cap + 1, dual<T>(0, 0)));

        for (size_t i = 1; i <= n; ++i) {
            for (size_t w = 0; w <= max_cap; ++w) {
                dual<T> current_weight = dual<T>(w, 0);

                // Soft constraint: can we include item i?
                dual<T> weight_diff = current_weight - weights[i-1];
                dual<T> include_weight = dual<T>(1, 0) /
                    (dual<T>(1, 0) + exp(-weight_diff * 10 / temperature_));

                // Value if we include item i
                size_t prev_w = w >= weights[i-1].value ?
                    static_cast<size_t>(w - weights[i-1].value) : 0;

                dual<T> include_value = dp[i-1][prev_w] + values[i-1] * include_weight;
                dual<T> exclude_value = dp[i-1][w];

                // Soft maximum
                soft_sort<T> sorter(temperature_);
                dp[i][w] = sorter.soft_max({include_value, exclude_value});
            }
        }

        return dp[n][max_cap];
    }
};

// ============================================================================
// Neural ODEs - Continuous-Depth Models
// ============================================================================

template<typename T = double>
class neural_ode {
    using State = std::vector<dual<T>>;
    using TimeDerivative = std::function<State(T, const State&)>;

    TimeDerivative dynamics_;
    T dt_;  // Integration step size

public:
    neural_ode(TimeDerivative f, T step = 0.01)
        : dynamics_(std::move(f)), dt_(step) {}

    // Forward Euler integration
    State integrate(const State& initial, T t0, T t1) const {
        State state = initial;
        T t = t0;

        while (t < t1) {
            State derivative = dynamics_(t, state);

            for (size_t i = 0; i < state.size(); ++i) {
                state[i] = state[i] + derivative[i] * dual<T>(dt_, 0);
            }

            t += dt_;
        }

        return state;
    }

    // Runge-Kutta 4 integration for better accuracy
    State integrate_rk4(const State& initial, T t0, T t1) const {
        State state = initial;
        T t = t0;

        while (t < t1) {
            State k1 = dynamics_(t, state);

            State state2 = state;
            for (size_t i = 0; i < state.size(); ++i) {
                state2[i] = state[i] + k1[i] * dual<T>(dt_ / 2, 0);
            }
            State k2 = dynamics_(t + dt_ / 2, state2);

            State state3 = state;
            for (size_t i = 0; i < state.size(); ++i) {
                state3[i] = state[i] + k2[i] * dual<T>(dt_ / 2, 0);
            }
            State k3 = dynamics_(t + dt_ / 2, state3);

            State state4 = state;
            for (size_t i = 0; i < state.size(); ++i) {
                state4[i] = state[i] + k3[i] * dual<T>(dt_, 0);
            }
            State k4 = dynamics_(t + dt_, state4);

            // Combine
            for (size_t i = 0; i < state.size(); ++i) {
                state[i] = state[i] + (k1[i] + k2[i] * dual<T>(2, 0) +
                                       k3[i] * dual<T>(2, 0) + k4[i]) *
                                      dual<T>(dt_ / 6, 0);
            }

            t += dt_;
        }

        return state;
    }

    // Adjoint method for efficient gradient computation
    struct adjoint_state {
        State forward;
        State adjoint;
    };

    adjoint_state adjoint_solve(const State& initial, T t0, T t1,
                                const State& terminal_gradient) const {
        // Forward solve
        State forward = integrate_rk4(initial, t0, t1);

        // Backward solve for adjoint
        State adjoint = terminal_gradient;
        T t = t1;

        while (t > t0) {
            // Negative time step for backward integration
            State derivative = dynamics_(t, forward);

            // Adjoint dynamics (linearized around forward trajectory)
            for (size_t i = 0; i < adjoint.size(); ++i) {
                dual<T> eps(0, 1);  // Infinitesimal perturbation

                State perturbed = forward;
                perturbed[i] = perturbed[i] + eps;

                State perturbed_derivative = dynamics_(t, perturbed);

                // Extract gradient
                for (size_t j = 0; j < adjoint.size(); ++j) {
                    adjoint[j] = adjoint[j] -
                        (perturbed_derivative[j].derivative * adjoint[i]) *
                        dual<T>(dt_, 0);
                }
            }

            t -= dt_;
        }

        return {forward, adjoint};
    }
};

// ============================================================================
// Differentiable Optimal Control
// ============================================================================

template<typename T = double>
class differential_dynamic_programming {
    using State = std::vector<dual<T>>;
    using Control = std::vector<dual<T>>;
    using Dynamics = std::function<State(const State&, const Control&)>;
    using Cost = std::function<dual<T>(const State&, const Control&)>;

    Dynamics dynamics_;
    Cost running_cost_;
    Cost terminal_cost_;
    size_t horizon_;

public:
    differential_dynamic_programming(Dynamics f, Cost l, Cost lf, size_t T)
        : dynamics_(std::move(f)), running_cost_(std::move(l)),
          terminal_cost_(std::move(lf)), horizon_(T) {}

    struct trajectory {
        std::vector<State> states;
        std::vector<Control> controls;
        dual<T> cost;
    };

    // iLQR (iterative Linear Quadratic Regulator)
    trajectory solve(const State& initial_state,
                    const std::vector<Control>& initial_controls) const {
        trajectory traj;
        traj.states.push_back(initial_state);
        traj.controls = initial_controls;

        // Forward pass - simulate trajectory
        for (size_t t = 0; t < horizon_; ++t) {
            State next = dynamics_(traj.states[t], traj.controls[t]);
            traj.states.push_back(next);
            traj.cost = traj.cost + running_cost_(traj.states[t], traj.controls[t]);
        }
        traj.cost = traj.cost + terminal_cost_(traj.states.back());

        // Backward pass - compute optimal control adjustments
        const size_t max_iterations = 100;
        for (size_t iter = 0; iter < max_iterations; ++iter) {
            auto gains = backward_pass(traj);
            auto new_traj = forward_pass(traj, gains);

            if ((new_traj.cost.value - traj.cost.value) < 1e-6) {
                break;  // Converged
            }

            traj = new_traj;
        }

        return traj;
    }

private:
    struct control_gains {
        std::vector<std::vector<dual<T>>> K;  // Feedback gain
        std::vector<Control> k;                // Feedforward term
    };

    control_gains backward_pass(const trajectory& traj) const {
        control_gains gains;
        gains.K.resize(horizon_);
        gains.k.resize(horizon_);

        // Value function gradients
        State Vx = compute_gradient(terminal_cost_, traj.states.back());

        for (int t = horizon_ - 1; t >= 0; --t) {
            // Compute Q-function gradients
            State Qx = compute_gradient(running_cost_, traj.states[t]);
            Control Qu = compute_control_gradient(running_cost_,
                                                  traj.states[t],
                                                  traj.controls[t]);

            // Linearize dynamics
            auto [Fx, Fu] = linearize_dynamics(traj.states[t], traj.controls[t]);

            // Compute gains
            gains.k[t] = solve_linear(Fu, Qu);  // Simplified - should use Quu inverse
            gains.K[t] = compute_feedback_gain(Fx, Fu);

            // Update value function gradient
            Vx = Qx;  // Simplified update
        }

        return gains;
    }

    trajectory forward_pass(const trajectory& nominal,
                           const control_gains& gains) const {
        trajectory new_traj;
        new_traj.states.push_back(nominal.states[0]);

        for (size_t t = 0; t < horizon_; ++t) {
            // Compute control adjustment
            State state_diff = subtract(new_traj.states[t], nominal.states[t]);
            Control control_adjustment = gains.k[t];

            // Add feedback term (simplified)
            Control new_control = add(nominal.controls[t], control_adjustment);
            new_traj.controls.push_back(new_control);

            // Simulate forward
            State next = dynamics_(new_traj.states[t], new_control);
            new_traj.states.push_back(next);

            new_traj.cost = new_traj.cost +
                           running_cost_(new_traj.states[t], new_control);
        }

        new_traj.cost = new_traj.cost + terminal_cost_(new_traj.states.back());
        return new_traj;
    }

    // Helper functions for linearization and linear algebra
    std::pair<std::vector<std::vector<dual<T>>>, std::vector<std::vector<dual<T>>>>
    linearize_dynamics(const State& x, const Control& u) const {
        // Compute Jacobians using dual numbers
        size_t n = x.size();
        size_t m = u.size();

        std::vector<std::vector<dual<T>>> Fx(n, std::vector<dual<T>>(n));
        std::vector<std::vector<dual<T>>> Fu(n, std::vector<dual<T>>(m));

        // Compute Fx
        for (size_t i = 0; i < n; ++i) {
            State x_perturbed = x;
            x_perturbed[i] = dual<T>(x[i].value, 1);  // Unit perturbation

            State f_perturbed = dynamics_(x_perturbed, u);

            for (size_t j = 0; j < n; ++j) {
                Fx[j][i] = dual<T>(f_perturbed[j].derivative, 0);
            }
        }

        // Compute Fu
        for (size_t i = 0; i < m; ++i) {
            Control u_perturbed = u;
            u_perturbed[i] = dual<T>(u[i].value, 1);

            State f_perturbed = dynamics_(x, u_perturbed);

            for (size_t j = 0; j < n; ++j) {
                Fu[j][i] = dual<T>(f_perturbed[j].derivative, 0);
            }
        }

        return {Fx, Fu};
    }

    State compute_gradient(const Cost& cost_fn, const State& x) const {
        State gradient;

        for (size_t i = 0; i < x.size(); ++i) {
            State x_perturbed = x;
            x_perturbed[i] = dual<T>(x[i].value, 1);

            Control dummy_control(1, dual<T>(0, 0));
            dual<T> cost_perturbed = cost_fn(x_perturbed, dummy_control);

            gradient.push_back(dual<T>(cost_perturbed.derivative, 0));
        }

        return gradient;
    }

    Control compute_control_gradient(const Cost& cost_fn,
                                    const State& x,
                                    const Control& u) const {
        Control gradient;

        for (size_t i = 0; i < u.size(); ++i) {
            Control u_perturbed = u;
            u_perturbed[i] = dual<T>(u[i].value, 1);

            dual<T> cost_perturbed = cost_fn(x, u_perturbed);
            gradient.push_back(dual<T>(cost_perturbed.derivative, 0));
        }

        return gradient;
    }

    // Simplified linear algebra operations
    Control solve_linear(const std::vector<std::vector<dual<T>>>& A,
                        const Control& b) const {
        // Simplified - would need proper linear solver
        return b;
    }

    std::vector<std::vector<dual<T>>> compute_feedback_gain(
        const std::vector<std::vector<dual<T>>>& Fx,
        const std::vector<std::vector<dual<T>>>& Fu) const {
        // Simplified - would compute K = -inv(Quu) * Qux
        return Fx;
    }

    State subtract(const State& a, const State& b) const {
        State result;
        for (size_t i = 0; i < a.size(); ++i) {
            result.push_back(a[i] - b[i]);
        }
        return result;
    }

    Control add(const Control& a, const Control& b) const {
        Control result;
        for (size_t i = 0; i < a.size(); ++i) {
            result.push_back(a[i] + b[i]);
        }
        return result;
    }
};

} // namespace stepanov::differentiable

#endif // STEPANOV_DIFFERENTIABLE_HPP