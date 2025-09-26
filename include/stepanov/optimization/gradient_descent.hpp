#pragma once

#include "concepts.hpp"
#include "../autodiff.hpp"
#include <cmath>
#include <optional>

namespace stepanov::optimization {

/**
 * Gradient descent algorithms following generic programming principles
 *
 * Provides various gradient-based optimization methods:
 * - Standard gradient descent
 * - Momentum-based methods (classical, Nesterov)
 * - Adaptive learning rate methods (AdaGrad, RMSprop, Adam)
 * - Line search strategies
 */

// Step size strategies
template<typename T>
    requires field<T>
struct constant_step_size {
    T alpha;

    explicit constant_step_size(T a = T(0.01)) : alpha(a) {}

    T operator()(size_t iteration) const { return alpha; }
};

template<typename T>
    requires field<T>
struct diminishing_step_size {
    T initial_alpha;
    T decay_rate;

    diminishing_step_size(T init = T(1), T decay = T(0.99))
        : initial_alpha(init), decay_rate(decay) {}

    T operator()(size_t iteration) const {
        return initial_alpha * std::pow(decay_rate, T(iteration));
    }
};

template<typename T>
    requires field<T>
struct armijo_line_search {
    T c1;  // Sufficient decrease parameter
    T tau; // Backtracking factor

    armijo_line_search(T c = T(0.1), T t = T(0.5))
        : c1(c), tau(t) {}

    template<typename F, typename V>
    T operator()(F f, const V& x, const V& gradient, const V& direction, T initial_alpha = T(1)) {
        T alpha = initial_alpha;
        T fx = f(x);
        T grad_dot_dir = dot(gradient, direction);

        while (alpha > T(1e-10)) {
            T fx_new = f(x + alpha * direction);
            if (fx_new <= fx + alpha * c1 * grad_dot_dir) {
                return alpha;
            }
            alpha = alpha * tau;
        }
        return alpha;
    }
};

// Basic gradient descent
template<typename V, typename F, typename StepSize = constant_step_size<typename V::value_type>>
    requires vector_space<V> && evaluable<F, V>
optimization_result<typename V::value_type, V> gradient_descent(
    F f,
    V initial_guess,
    StepSize step_size = {},
    tolerance_criterion<typename V::value_type> stopping = {})
{
    using T = typename V::value_type;

    V x = initial_guess;
    V prev_x = x;

    for (size_t iter = 0; iter < stopping.max_iterations; ++iter) {
        // Compute gradient using finite differences
        V grad = compute_gradient_finite_diff(f, x);

        // Check for convergence
        if (norm(grad) < stopping.absolute_tolerance) {
            return optimization_result<T, V>{x, f(x), iter, true};
        }

        // Update with gradient descent step
        T alpha = step_size(iter);
        x = x - alpha * grad;

        // Check stopping criterion
        if (stopping(norm(x - prev_x), norm(prev_x), iter)) {
            return optimization_result<T, V>{x, f(x), iter, true};
        }

        prev_x = x;
    }

    return optimization_result<T, V>{x, f(x), stopping.max_iterations, false};
}

// Gradient descent with momentum
template<typename V, typename F>
    requires vector_space<V> && evaluable<F, V>
optimization_result<typename V::value_type, V> momentum_gradient_descent(
    F f,
    V initial_guess,
    typename V::value_type learning_rate = 0.01,
    typename V::value_type momentum = 0.9,
    tolerance_criterion<typename V::value_type> stopping = {})
{
    using T = typename V::value_type;

    V x = initial_guess;
    V velocity = V{}; // Zero-initialized velocity

    for (size_t iter = 0; iter < stopping.max_iterations; ++iter) {
        V grad = compute_gradient(f, x);

        // Check for convergence
        if (norm(grad) < stopping.absolute_tolerance) {
            return optimization_result<T, V>{x, f(x), iter, true};
        }

        // Update velocity with momentum
        velocity = momentum * velocity - learning_rate * grad;

        // Update position
        V prev_x = x;
        x = x + velocity;

        // Check stopping criterion
        if (stopping(norm(x - prev_x), norm(prev_x), iter)) {
            return optimization_result<T, V>{x, f(x), iter, true};
        }
    }

    return optimization_result<T, V>{x, f(x), stopping.max_iterations, false};
}

// Nesterov accelerated gradient
template<typename V, typename F>
    requires vector_space<V> && evaluable<F, V>
optimization_result<typename V::value_type, V> nesterov_gradient_descent(
    F f,
    V initial_guess,
    typename V::value_type learning_rate = 0.01,
    typename V::value_type momentum = 0.9,
    tolerance_criterion<typename V::value_type> stopping = {})
{
    using T = typename V::value_type;

    V x = initial_guess;
    V velocity = V{}; // Zero-initialized velocity

    for (size_t iter = 0; iter < stopping.max_iterations; ++iter) {
        // Look ahead
        V x_ahead = x + momentum * velocity;

        // Compute gradient at look-ahead position
        V grad = compute_gradient(f, x_ahead);

        // Check for convergence
        if (norm(grad) < stopping.absolute_tolerance) {
            return optimization_result<T, V>{x, f(x), iter, true};
        }

        // Update velocity
        velocity = momentum * velocity - learning_rate * grad;

        // Update position
        V prev_x = x;
        x = x + velocity;

        // Check stopping criterion
        if (stopping(norm(x - prev_x), norm(prev_x), iter)) {
            return optimization_result<T, V>{x, f(x), iter, true};
        }
    }

    return optimization_result<T, V>{x, f(x), stopping.max_iterations, false};
}

// Adam optimizer (Adaptive Moment Estimation)
template<typename V, typename F>
    requires vector_space<V> && evaluable<F, V>
optimization_result<typename V::value_type, V> adam(
    F f,
    V initial_guess,
    typename V::value_type learning_rate = 0.001,
    typename V::value_type beta1 = 0.9,
    typename V::value_type beta2 = 0.999,
    typename V::value_type epsilon = 1e-8,
    tolerance_criterion<typename V::value_type> stopping = {})
{
    using T = typename V::value_type;

    V x = initial_guess;
    V m = V{}; // First moment estimate
    V v = V{}; // Second moment estimate

    for (size_t iter = 0; iter < stopping.max_iterations; ++iter) {
        V grad = compute_gradient(f, x);

        // Check for convergence
        if (norm(grad) < stopping.absolute_tolerance) {
            return optimization_result<T, V>{x, f(x), iter, true};
        }

        // Update biased first moment estimate
        m = beta1 * m + (T(1) - beta1) * grad;

        // Update biased second raw moment estimate
        v = beta2 * v + (T(1) - beta2) * element_wise_multiply(grad, grad);

        // Compute bias-corrected first moment estimate
        T bias_correction1 = T(1) - std::pow(beta1, T(iter + 1));
        V m_hat = m / bias_correction1;

        // Compute bias-corrected second raw moment estimate
        T bias_correction2 = T(1) - std::pow(beta2, T(iter + 1));
        V v_hat = v / bias_correction2;

        // Update parameters
        V prev_x = x;
        x = x - learning_rate * element_wise_divide(m_hat, element_wise_sqrt(v_hat) + epsilon);

        // Check stopping criterion
        if (stopping(norm(x - prev_x), norm(prev_x), iter)) {
            return optimization_result<T, V>{x, f(x), iter, true};
        }
    }

    return optimization_result<T, V>{x, f(x), stopping.max_iterations, false};
}

// AdaGrad (Adaptive Gradient)
template<typename V, typename F>
    requires vector_space<V> && evaluable<F, V>
optimization_result<typename V::value_type, V> adagrad(
    F f,
    V initial_guess,
    typename V::value_type learning_rate = 0.01,
    typename V::value_type epsilon = 1e-8,
    tolerance_criterion<typename V::value_type> stopping = {})
{
    using T = typename V::value_type;

    V x = initial_guess;
    V accumulated_grad = V{}; // Accumulated squared gradients

    for (size_t iter = 0; iter < stopping.max_iterations; ++iter) {
        V grad = compute_gradient(f, x);

        // Check for convergence
        if (norm(grad) < stopping.absolute_tolerance) {
            return optimization_result<T, V>{x, f(x), iter, true};
        }

        // Accumulate squared gradients
        accumulated_grad = accumulated_grad + element_wise_multiply(grad, grad);

        // Update with adaptive learning rate
        V prev_x = x;
        x = x - learning_rate * element_wise_divide(grad,
            element_wise_sqrt(accumulated_grad) + epsilon);

        // Check stopping criterion
        if (stopping(norm(x - prev_x), norm(prev_x), iter)) {
            return optimization_result<T, V>{x, f(x), iter, true};
        }
    }

    return optimization_result<T, V>{x, f(x), stopping.max_iterations, false};
}

// RMSprop (Root Mean Square Propagation)
template<typename V, typename F>
    requires vector_space<V> && evaluable<F, V>
optimization_result<typename V::value_type, V> rmsprop(
    F f,
    V initial_guess,
    typename V::value_type learning_rate = 0.001,
    typename V::value_type decay_rate = 0.9,
    typename V::value_type epsilon = 1e-8,
    tolerance_criterion<typename V::value_type> stopping = {})
{
    using T = typename V::value_type;

    V x = initial_guess;
    V moving_avg = V{}; // Moving average of squared gradients

    for (size_t iter = 0; iter < stopping.max_iterations; ++iter) {
        V grad = compute_gradient(f, x);

        // Check for convergence
        if (norm(grad) < stopping.absolute_tolerance) {
            return optimization_result<T, V>{x, f(x), iter, true};
        }

        // Update moving average of squared gradients
        moving_avg = decay_rate * moving_avg +
                    (T(1) - decay_rate) * element_wise_multiply(grad, grad);

        // Update with adaptive learning rate
        V prev_x = x;
        x = x - learning_rate * element_wise_divide(grad,
            element_wise_sqrt(moving_avg) + epsilon);

        // Check stopping criterion
        if (stopping(norm(x - prev_x), norm(prev_x), iter)) {
            return optimization_result<T, V>{x, f(x), iter, true};
        }
    }

    return optimization_result<T, V>{x, f(x), stopping.max_iterations, false};
}

// Conjugate gradient method
template<typename V, typename F>
    requires vector_space<V> && evaluable<F, V>
optimization_result<typename V::value_type, V> conjugate_gradient(
    F f,
    V initial_guess,
    tolerance_criterion<typename V::value_type> stopping = {})
{
    using T = typename V::value_type;

    V x = initial_guess;
    V grad = compute_gradient(f, x);
    V direction = -grad;
    V prev_grad = grad;

    for (size_t iter = 0; iter < stopping.max_iterations; ++iter) {
        // Check for convergence
        if (norm(grad) < stopping.absolute_tolerance) {
            return optimization_result<T, V>{x, f(x), iter, true};
        }

        // Line search along direction
        armijo_line_search<T> line_search;
        T alpha = line_search(f, x, grad, direction);

        // Update position
        V prev_x = x;
        x = x + alpha * direction;

        // Compute new gradient
        prev_grad = grad;
        grad = compute_gradient(f, x);

        // Compute beta (Polak-Ribiere formula)
        T beta = dot(grad, grad - prev_grad) / dot(prev_grad, prev_grad);
        beta = std::max(T(0), beta); // Restart if beta < 0

        // Update search direction
        direction = -grad + beta * direction;

        // Check stopping criterion
        if (stopping(norm(x - prev_x), norm(prev_x), iter)) {
            return optimization_result<T, V>{x, f(x), iter, true};
        }
    }

    return optimization_result<T, V>{x, f(x), stopping.max_iterations, false};
}

} // namespace stepanov::optimization