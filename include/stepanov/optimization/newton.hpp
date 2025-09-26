#pragma once

#include "concepts.hpp"
#include "../autodiff.hpp"
#include <optional>

namespace stepanov::optimization {

/**
 * Newton's method for optimization following Stepanov's generic programming principles
 *
 * Finds local minima using second-order Taylor approximation:
 * x_{k+1} = x_k - H^{-1}(x_k) * g(x_k)
 * where g is gradient and H is Hessian
 */

// Newton's method for univariate functions
template<typename T, typename F>
    requires field<T> && evaluable<F, T>
optimization_result<T> newton_univariate(
    F f,
    T initial_guess,
    tolerance_criterion<T> stopping = {},
    std::optional<std::pair<T, T>> bounds = std::nullopt)
{
    using namespace stepanov;

    // Use automatic differentiation for derivatives
    auto eval_with_derivatives = [&f](T x) {
        dual<dual<T>> xx = dual<dual<T>>::variable(
            dual<T>::variable(x, T(1)),
            dual<T>::constant(T(1))
        );
        auto result = f(xx);
        return std::tuple{
            result.value().value(),           // f(x)
            result.value().derivative(),      // f'(x)
            result.derivative().value()       // f''(x)
        };
    };

    T x = initial_guess;
    T prev_x = x;
    size_t iter = 0;

    for (; iter < stopping.max_iterations; ++iter) {
        auto [fx, grad, hess] = eval_with_derivatives(x);

        // Check if Hessian is too small (near critical point)
        if (abs(hess) < std::numeric_limits<T>::epsilon()) {
            // Fall back to gradient descent step
            x = x - stopping.absolute_tolerance * grad;
        } else {
            // Newton step
            T newton_step = grad / hess;
            T alpha = T(1); // Step size

            // Line search with backtracking
            T current_fx = fx;
            while (alpha > T(1e-10)) {
                T x_new = x - alpha * newton_step;

                // Check bounds if provided
                if (bounds) {
                    x_new = std::max(bounds->first, std::min(x_new, bounds->second));
                }

                T new_fx = f(x_new);

                // Armijo condition for sufficient decrease
                if (new_fx < current_fx - alpha * T(0.5) * grad * newton_step) {
                    x = x_new;
                    break;
                }
                alpha = alpha * T(0.5);
            }
        }

        // Check convergence
        if (stopping(x, prev_x, iter)) {
            return optimization_result<T>{x, fx, iter, true};
        }

        prev_x = x;
    }

    auto [final_fx, _, __] = eval_with_derivatives(x);
    return optimization_result<T>{x, final_fx, iter, false};
}

// Newton's method for multivariate functions
template<typename V, typename F, typename C = std::function<bool(const V&)>>
    requires vector_space<V> && evaluable<F, V>
optimization_result<typename V::value_type, V> newton_multivariate(
    F f,
    V initial_guess,
    tolerance_criterion<typename V::value_type> stopping = {},
    std::optional<C> feasible_region = std::nullopt)
{
    using T = typename V::value_type;
    using namespace stepanov;

    V x = initial_guess;
    V prev_x = x;
    size_t iter = 0;

    for (; iter < stopping.max_iterations; ++iter) {
        // Compute gradient and Hessian using autodiff
        auto grad = compute_gradient(f, x);
        auto hess = compute_hessian(f, x);

        // Solve H * p = -g for Newton direction p
        V newton_direction = solve_linear_system(hess, -grad);

        // Line search with Armijo backtracking
        T alpha = T(1);
        T current_fx = f(x);
        T grad_dot_dir = dot(grad, newton_direction);

        while (alpha > T(1e-10)) {
            V x_new = x + alpha * newton_direction;

            // Check feasibility constraint if provided
            if (feasible_region && !(*feasible_region)(x_new)) {
                alpha = alpha * T(0.5);
                continue;
            }

            T new_fx = f(x_new);

            // Armijo condition
            if (new_fx < current_fx + alpha * T(0.1) * grad_dot_dir) {
                x = x_new;
                break;
            }
            alpha = alpha * T(0.5);
        }

        // Check convergence
        T norm_diff = norm(x - prev_x);
        if (stopping(norm_diff, norm(prev_x), iter)) {
            return optimization_result<T, V>{x, f(x), iter, true};
        }

        prev_x = x;
    }

    return optimization_result<T, V>{x, f(x), iter, false};
}

// Newton's method for root finding (solving f(x) = 0)
template<typename T, typename F>
    requires field<T> && evaluable<F, T>
std::optional<T> newton_root(
    F f,
    T initial_guess,
    tolerance_criterion<T> stopping = {})
{
    using namespace stepanov;

    T x = initial_guess;
    T prev_x = x;

    for (size_t iter = 0; iter < stopping.max_iterations; ++iter) {
        // Use autodiff for derivative
        dual<T> x_dual = dual<T>::variable(x, T(1));
        dual<T> fx_dual = f(x_dual);

        T fx = fx_dual.value();
        T fpx = fx_dual.derivative();

        // Check if derivative is too small
        if (abs(fpx) < std::numeric_limits<T>::epsilon()) {
            return std::nullopt; // Cannot continue
        }

        // Newton step for root finding
        x = x - fx / fpx;

        // Check convergence
        if (abs(fx) < stopping.absolute_tolerance ||
            stopping(x, prev_x, iter)) {
            return x;
        }

        prev_x = x;
    }

    return std::nullopt;
}

// Damped Newton method with trust region
template<typename V, typename F>
    requires vector_space<V> && evaluable<F, V>
optimization_result<typename V::value_type, V> newton_trust_region(
    F f,
    V initial_guess,
    typename V::value_type trust_radius = 1.0,
    tolerance_criterion<typename V::value_type> stopping = {})
{
    using T = typename V::value_type;

    V x = initial_guess;
    T radius = trust_radius;
    size_t iter = 0;

    for (; iter < stopping.max_iterations; ++iter) {
        auto grad = compute_gradient(f, x);
        auto hess = compute_hessian(f, x);

        // Solve trust region subproblem
        V step = solve_trust_region_subproblem(grad, hess, radius);

        // Evaluate actual vs predicted reduction
        T fx = f(x);
        T fx_new = f(x + step);
        T actual_reduction = fx - fx_new;
        T predicted_reduction = -dot(grad, step) - T(0.5) * quadratic_form(step, hess, step);

        T ratio = actual_reduction / predicted_reduction;

        // Update trust region radius
        if (ratio < T(0.25)) {
            radius = radius * T(0.25);
        } else if (ratio > T(0.75) && norm(step) == radius) {
            radius = std::min(T(2) * radius, T(10));
        }

        // Accept or reject step
        if (ratio > T(0)) {
            x = x + step;

            // Check convergence
            if (norm(grad) < stopping.absolute_tolerance) {
                return optimization_result<T, V>{x, fx_new, iter, true};
            }
        }

        if (radius < stopping.absolute_tolerance) {
            return optimization_result<T, V>{x, f(x), iter, false};
        }
    }

    return optimization_result<T, V>{x, f(x), iter, false};
}

} // namespace stepanov::optimization