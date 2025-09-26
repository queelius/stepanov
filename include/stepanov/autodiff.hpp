#pragma once

#include <cmath>
#include <unordered_map>
#include <memory>
#include <vector>
#include <functional>
#include <initializer_list>
#include <numeric>
#include <algorithm>
#include <span>
#include <array>
#include <cstring>
#include <random>
#include "concepts.hpp"
#include "type_erasure.hpp"

namespace stepanov {

/**
 * Enhanced automatic differentiation using dual numbers, tape-based AD, and tensors
 *
 * Provides both forward-mode (dual numbers) and reverse-mode (tape-based)
 * automatic differentiation following generic programming principles.
 *
 * Key features:
 * - Forward-mode AD via dual numbers for efficient gradient computation
 * - Reverse-mode AD via computational graph for many inputs
 * - Tensor support for multi-dimensional arrays
 * - Expression templates for zero-overhead composition
 * - Higher-order derivatives via nesting
 * - Integration with type erasure for runtime polymorphism
 */

// =============================================================================
// Forward-mode automatic differentiation (dual numbers)
// =============================================================================

template<typename T>
    requires field<T>
class dual {
private:
    T value_;       // Function value
    T derivative_;  // Derivative value

public:
    using value_type = T;

    // Constructors
    dual() : value_(T(0)), derivative_(T(0)) {}

    explicit dual(T val, T deriv = T(0)) : value_(val), derivative_(deriv) {}

    // Create variable with seed derivative
    static dual variable(T val, T seed = T(1)) {
        return dual(val, seed);
    }

    // Create constant (zero derivative)
    static dual constant(T val) {
        return dual(val, T(0));
    }

    // Accessors
    T value() const { return value_; }
    T derivative() const { return derivative_; }

    // Arithmetic operations with chain rule
    dual operator-() const {
        return dual(-value_, -derivative_);
    }

    dual operator+(const dual& other) const {
        return dual(value_ + other.value_, derivative_ + other.derivative_);
    }

    dual operator-(const dual& other) const {
        return dual(value_ - other.value_, derivative_ - other.derivative_);
    }

    dual operator*(const dual& other) const {
        // Product rule: (fg)' = f'g + fg'
        return dual(value_ * other.value_,
                   derivative_ * other.value_ + value_ * other.derivative_);
    }

    dual operator/(const dual& other) const {
        // Quotient rule: (f/g)' = (f'g - fg')/g²
        T g_squared = other.value_ * other.value_;
        return dual(value_ / other.value_,
                   (derivative_ * other.value_ - value_ * other.derivative_) / g_squared);
    }

    // Compound assignments
    dual& operator+=(const dual& other) {
        value_ += other.value_;
        derivative_ += other.derivative_;
        return *this;
    }

    dual& operator-=(const dual& other) {
        value_ -= other.value_;
        derivative_ -= other.derivative_;
        return *this;
    }

    dual& operator*=(const dual& other) {
        derivative_ = derivative_ * other.value_ + value_ * other.derivative_;
        value_ *= other.value_;
        return *this;
    }

    dual& operator/=(const dual& other) {
        T g_squared = other.value_ * other.value_;
        derivative_ = (derivative_ * other.value_ - value_ * other.derivative_) / g_squared;
        value_ /= other.value_;
        return *this;
    }

    // Comparison (based on value only)
    bool operator==(const dual& other) const {
        return value_ == other.value_;
    }

    bool operator!=(const dual& other) const {
        return value_ != other.value_;
    }

    bool operator<(const dual& other) const {
        return value_ < other.value_;
    }

    bool operator<=(const dual& other) const {
        return value_ <= other.value_;
    }

    bool operator>(const dual& other) const {
        return value_ > other.value_;
    }

    bool operator>=(const dual& other) const {
        return value_ >= other.value_;
    }

    // Mixed operations with scalars
    dual operator+(T scalar) const {
        return dual(value_ + scalar, derivative_);
    }

    dual operator-(T scalar) const {
        return dual(value_ - scalar, derivative_);
    }

    dual operator*(T scalar) const {
        return dual(value_ * scalar, derivative_ * scalar);
    }

    dual operator/(T scalar) const {
        return dual(value_ / scalar, derivative_ / scalar);
    }

    friend dual operator+(T scalar, const dual& d) {
        return d + scalar;
    }

    friend dual operator-(T scalar, const dual& d) {
        return dual(scalar - d.value_, -d.derivative_);
    }

    friend dual operator*(T scalar, const dual& d) {
        return d * scalar;
    }

    friend dual operator/(T scalar, const dual& d) {
        T d_squared = d.value_ * d.value_;
        return dual(scalar / d.value_, -scalar * d.derivative_ / d_squared);
    }
};

// Mathematical functions for dual numbers

template<typename T>
dual<T> sqrt(const dual<T>& x) {
    T sqrt_val = std::sqrt(x.value());
    return dual<T>(sqrt_val, x.derivative() / (T(2) * sqrt_val));
}

template<typename T>
dual<T> exp(const dual<T>& x) {
    T exp_val = std::exp(x.value());
    return dual<T>(exp_val, x.derivative() * exp_val);
}

template<typename T>
dual<T> log(const dual<T>& x) {
    return dual<T>(std::log(x.value()), x.derivative() / x.value());
}

template<typename T>
dual<T> sin(const dual<T>& x) {
    return dual<T>(std::sin(x.value()), x.derivative() * std::cos(x.value()));
}

template<typename T>
dual<T> cos(const dual<T>& x) {
    return dual<T>(std::cos(x.value()), -x.derivative() * std::sin(x.value()));
}

template<typename T>
dual<T> tan(const dual<T>& x) {
    T tan_val = std::tan(x.value());
    T sec_squared = T(1) + tan_val * tan_val;
    return dual<T>(tan_val, x.derivative() * sec_squared);
}

template<typename T>
dual<T> asin(const dual<T>& x) {
    return dual<T>(std::asin(x.value()),
                  x.derivative() / std::sqrt(T(1) - x.value() * x.value()));
}

template<typename T>
dual<T> acos(const dual<T>& x) {
    return dual<T>(std::acos(x.value()),
                  -x.derivative() / std::sqrt(T(1) - x.value() * x.value()));
}

template<typename T>
dual<T> atan(const dual<T>& x) {
    return dual<T>(std::atan(x.value()),
                  x.derivative() / (T(1) + x.value() * x.value()));
}

template<typename T>
dual<T> sinh(const dual<T>& x) {
    return dual<T>(std::sinh(x.value()), x.derivative() * std::cosh(x.value()));
}

template<typename T>
dual<T> cosh(const dual<T>& x) {
    return dual<T>(std::cosh(x.value()), x.derivative() * std::sinh(x.value()));
}

template<typename T>
dual<T> tanh(const dual<T>& x) {
    T tanh_val = std::tanh(x.value());
    return dual<T>(tanh_val, x.derivative() * (T(1) - tanh_val * tanh_val));
}

template<typename T>
dual<T> pow(const dual<T>& base, const dual<T>& exp) {
    T val = std::pow(base.value(), exp.value());
    T deriv = val * (exp.derivative() * std::log(base.value()) +
                     exp.value() * base.derivative() / base.value());
    return dual<T>(val, deriv);
}

template<typename T>
dual<T> pow(const dual<T>& base, T exp) {
    T val = std::pow(base.value(), exp);
    return dual<T>(val, base.derivative() * exp * std::pow(base.value(), exp - T(1)));
}

template<typename T>
dual<T> abs(const dual<T>& x) {
    if (x.value() > T(0)) {
        return x;
    } else if (x.value() < T(0)) {
        return -x;
    } else {
        // Non-differentiable at 0
        return dual<T>(T(0), T(0));
    }
}

// =============================================================================
// Reverse-mode automatic differentiation (tape-based)
// =============================================================================

template<typename T>
    requires field<T>
class tape_var {
private:
    static inline size_t next_id = 0;
    static inline std::vector<T> values;
    static inline std::vector<T> adjoints;
    static inline std::vector<std::function<void()>> tape;

    size_t id_;

public:
    tape_var(T val) : id_(next_id++) {
        if (values.size() <= id_) {
            values.resize(id_ + 1);
            adjoints.resize(id_ + 1);
        }
        values[id_] = val;
        adjoints[id_] = T(0);
    }

    // Reset tape for new computation
    static void reset_tape() {
        next_id = 0;
        values.clear();
        adjoints.clear();
        tape.clear();
    }

    // Get value
    T value() const { return values[id_]; }

    // Get gradient after backward pass
    T gradient() const { return adjoints[id_]; }

    // Seed gradient for backward pass
    void seed_gradient(T seed = T(1)) {
        adjoints[id_] = seed;
    }

    // Arithmetic operations that build the tape
    tape_var operator+(const tape_var& other) const {
        tape_var result(values[id_] + values[other.id_]);

        size_t result_id = result.id_;
        size_t this_id = id_;
        size_t other_id = other.id_;

        tape.push_back([result_id, this_id, other_id]() {
            adjoints[this_id] += adjoints[result_id];
            adjoints[other_id] += adjoints[result_id];
        });

        return result;
    }

    tape_var operator-(const tape_var& other) const {
        tape_var result(values[id_] - values[other.id_]);

        size_t result_id = result.id_;
        size_t this_id = id_;
        size_t other_id = other.id_;

        tape.push_back([result_id, this_id, other_id]() {
            adjoints[this_id] += adjoints[result_id];
            adjoints[other_id] -= adjoints[result_id];
        });

        return result;
    }

    tape_var operator*(const tape_var& other) const {
        T this_val = values[id_];
        T other_val = values[other.id_];
        tape_var result(this_val * other_val);

        size_t result_id = result.id_;
        size_t this_id = id_;
        size_t other_id = other.id_;

        tape.push_back([result_id, this_id, other_id, this_val, other_val]() {
            adjoints[this_id] += adjoints[result_id] * other_val;
            adjoints[other_id] += adjoints[result_id] * this_val;
        });

        return result;
    }

    tape_var operator/(const tape_var& other) const {
        T this_val = values[id_];
        T other_val = values[other.id_];
        tape_var result(this_val / other_val);

        size_t result_id = result.id_;
        size_t this_id = id_;
        size_t other_id = other.id_;

        tape.push_back([result_id, this_id, other_id, this_val, other_val]() {
            adjoints[this_id] += adjoints[result_id] / other_val;
            adjoints[other_id] -= adjoints[result_id] * this_val / (other_val * other_val);
        });

        return result;
    }

    // Backward pass - compute all gradients
    static void backward() {
        // Execute tape in reverse order
        for (auto it = tape.rbegin(); it != tape.rend(); ++it) {
            (*it)();
        }
    }
};

// Mathematical functions for tape variables

template<typename T>
tape_var<T> exp(const tape_var<T>& x) {
    T val = std::exp(x.value());
    tape_var<T> result(val);

    auto& tape = tape_var<T>::tape;
    size_t x_id = x.id_;
    size_t result_id = result.id_;

    tape.push_back([x_id, result_id, val]() {
        tape_var<T>::adjoints[x_id] += tape_var<T>::adjoints[result_id] * val;
    });

    return result;
}

template<typename T>
tape_var<T> log(const tape_var<T>& x) {
    T x_val = x.value();
    tape_var<T> result(std::log(x_val));

    auto& tape = tape_var<T>::tape;
    size_t x_id = x.id_;
    size_t result_id = result.id_;

    tape.push_back([x_id, result_id, x_val]() {
        tape_var<T>::adjoints[x_id] += tape_var<T>::adjoints[result_id] / x_val;
    });

    return result;
}

// =============================================================================
// Higher-order derivatives via nesting
// =============================================================================

template<typename T>
using dual2 = dual<dual<T>>;  // Second-order dual numbers

template<typename T>
dual2<T> make_dual2(T val, T first_deriv = T(0), T second_deriv = T(0)) {
    return dual2<T>(dual<T>(val, first_deriv), dual<T>(first_deriv, second_deriv));
}

// Extract derivatives
template<typename T>
struct derivatives {
    T value;
    T first;
    T second;

    derivatives(const dual2<T>& d)
        : value(d.value().value())
        , first(d.value().derivative())
        , second(d.derivative().derivative()) {}
};

// =============================================================================
// Gradient computation utilities
// =============================================================================

/**
 * Compute gradient of a function at a point
 * Uses forward-mode AD with multiple passes
 */
template<typename F, typename T>
std::vector<T> gradient(F f, const std::vector<T>& x) {
    std::vector<T> grad(x.size());

    for (size_t i = 0; i < x.size(); ++i) {
        // Create dual numbers with i-th unit vector as seed
        std::vector<dual<T>> x_dual(x.size());
        for (size_t j = 0; j < x.size(); ++j) {
            x_dual[j] = (i == j) ? dual<T>::variable(x[j], T(1))
                                 : dual<T>::constant(x[j]);
        }

        // Evaluate function and extract i-th partial derivative
        auto result = f(x_dual);
        grad[i] = result.derivative();
    }

    return grad;
}

/**
 * Compute Jacobian matrix of a vector function
 */
template<typename F, typename T>
std::vector<std::vector<T>> jacobian(F f, const std::vector<T>& x) {
    // Evaluate f to get output dimension
    auto y0 = f(x);
    size_t m = y0.size();
    size_t n = x.size();

    std::vector<std::vector<T>> jac(m, std::vector<T>(n));

    for (size_t j = 0; j < n; ++j) {
        // Create dual numbers with j-th unit vector as seed
        std::vector<dual<T>> x_dual(n);
        for (size_t k = 0; k < n; ++k) {
            x_dual[k] = (j == k) ? dual<T>::variable(x[k], T(1))
                                 : dual<T>::constant(x[k]);
        }

        // Evaluate function
        auto y_dual = f(x_dual);

        // Extract j-th column of Jacobian
        for (size_t i = 0; i < m; ++i) {
            jac[i][j] = y_dual[i].derivative();
        }
    }

    return jac;
}

/**
 * Compute Hessian matrix (matrix of second partial derivatives)
 */
template<typename F, typename T>
std::vector<std::vector<T>> hessian(F f, const std::vector<T>& x) {
    size_t n = x.size();
    std::vector<std::vector<T>> hess(n, std::vector<T>(n));

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            // Use second-order dual numbers
            std::vector<dual2<T>> x_dual2(n);
            for (size_t k = 0; k < n; ++k) {
                if (k == i && k == j) {
                    // Diagonal element
                    x_dual2[k] = make_dual2(x[k], T(1), T(1));
                } else if (k == i) {
                    x_dual2[k] = dual2<T>(dual<T>::variable(x[k], T(1)),
                                         dual<T>::constant(T(1)));
                } else if (k == j) {
                    x_dual2[k] = dual2<T>(dual<T>::constant(x[k]),
                                         dual<T>::variable(T(0), T(1)));
                } else {
                    x_dual2[k] = dual2<T>(dual<T>::constant(x[k]),
                                         dual<T>::constant(T(0)));
                }
            }

            auto result = f(x_dual2);
            derivatives<T> d(result);

            hess[i][j] = d.second;
            if (i != j) {
                hess[j][i] = d.second;  // Hessian is symmetric
            }
        }
    }

    return hess;
}

// =============================================================================
// Optimization utilities using automatic differentiation
// =============================================================================

/**
 * Gradient descent with automatic differentiation
 */
template<typename T>
struct optimization_result {
    std::vector<T> x;
    T value;
    int iterations;
    bool converged;
};

template<typename F, typename T>
optimization_result<T> gradient_descent(
    F f,
    std::vector<T> x0,
    T learning_rate = T(0.01),
    T tolerance = T(1e-6),
    int max_iter = 1000)
{
    std::vector<T> x = x0;
    int iter = 0;
    bool converged = false;

    for (; iter < max_iter; ++iter) {
        auto grad = gradient(f, x);

        // Check convergence
        T grad_norm = T(0);
        for (const auto& g : grad) {
            grad_norm += g * g;
        }
        grad_norm = std::sqrt(grad_norm);

        if (grad_norm < tolerance) {
            converged = true;
            break;
        }

        // Update parameters
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] -= learning_rate * grad[i];
        }
    }

    // Evaluate final value
    std::vector<dual<T>> x_dual(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        x_dual[i] = dual<T>::constant(x[i]);
    }
    T final_value = f(x_dual).value();

    return {x, final_value, iter, converged};
}

/**
 * Newton's method using Hessian from automatic differentiation
 */
template<typename F, typename T>
optimization_result<T> newton_optimize(
    F f,
    std::vector<T> x0,
    T tolerance = T(1e-9),
    int max_iter = 100)
{
    std::vector<T> x = x0;
    int iter = 0;
    bool converged = false;

    for (; iter < max_iter; ++iter) {
        auto grad = gradient(f, x);
        auto hess = hessian(f, x);

        // Check convergence
        T grad_norm = T(0);
        for (const auto& g : grad) {
            grad_norm += g * g;
        }
        grad_norm = std::sqrt(grad_norm);

        if (grad_norm < tolerance) {
            converged = true;
            break;
        }

        // Solve Hessian * delta = -gradient
        // For simplicity, using basic Gaussian elimination
        // In production, use a proper linear solver

        size_t n = x.size();
        std::vector<std::vector<T>> A = hess;
        std::vector<T> b(n);
        for (size_t i = 0; i < n; ++i) {
            b[i] = -grad[i];
        }

        // Gaussian elimination with partial pivoting
        for (size_t k = 0; k < n; ++k) {
            // Find pivot
            size_t pivot = k;
            for (size_t i = k + 1; i < n; ++i) {
                if (std::abs(A[i][k]) > std::abs(A[pivot][k])) {
                    pivot = i;
                }
            }

            // Swap rows
            if (pivot != k) {
                std::swap(A[k], A[pivot]);
                std::swap(b[k], b[pivot]);
            }

            // Eliminate
            for (size_t i = k + 1; i < n; ++i) {
                T factor = A[i][k] / A[k][k];
                for (size_t j = k + 1; j < n; ++j) {
                    A[i][j] -= factor * A[k][j];
                }
                b[i] -= factor * b[k];
            }
        }

        // Back substitution
        std::vector<T> delta(n);
        for (int i = n - 1; i >= 0; --i) {
            delta[i] = b[i];
            for (size_t j = i + 1; j < n; ++j) {
                delta[i] -= A[i][j] * delta[j];
            }
            delta[i] /= A[i][i];
        }

        // Update x
        for (size_t i = 0; i < n; ++i) {
            x[i] += delta[i];
        }
    }

    // Evaluate final value
    std::vector<dual<T>> x_dual(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        x_dual[i] = dual<T>::constant(x[i]);
    }
    T final_value = f(x_dual).value();

    return {x, final_value, iter, converged};
}

// =============================================================================
// Tensor support for automatic differentiation
// =============================================================================

/**
 * Tensor class with automatic differentiation support
 * Inspired by autograd-cpp but following Stepanov principles
 */
template<typename T = float>
    requires field<T>
class tensor {
private:
    std::shared_ptr<std::vector<T>> data_;
    std::shared_ptr<std::vector<T>> grad_;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::size_t size_;
    bool requires_grad_;

    // Computation graph for backpropagation
    std::vector<std::shared_ptr<tensor>> children_;
    std::function<void()> backward_fn_;

    void compute_strides() {
        strides_.resize(shape_.size());
        std::size_t stride = 1;
        for (int i = shape_.size() - 1; i >= 0; --i) {
            strides_[i] = stride;
            stride *= shape_[i];
        }
    }

    void compute_size() {
        size_ = 1;
        for (auto dim : shape_) {
            size_ *= dim;
        }
    }

public:
    // Constructors
    tensor(std::initializer_list<std::size_t> shape, bool requires_grad = true)
        : shape_(shape), requires_grad_(requires_grad) {
        compute_size();
        compute_strides();
        data_ = std::make_shared<std::vector<T>>(size_, T{0});
        if (requires_grad) {
            grad_ = std::make_shared<std::vector<T>>(size_, T{0});
        }
    }

    tensor(std::vector<std::size_t> shape, bool requires_grad = true)
        : shape_(std::move(shape)), requires_grad_(requires_grad) {
        compute_size();
        compute_strides();
        data_ = std::make_shared<std::vector<T>>(size_, T{0});
        if (requires_grad) {
            grad_ = std::make_shared<std::vector<T>>(size_, T{0});
        }
    }

    // Factory methods
    static tensor zeros(std::vector<std::size_t> shape, bool requires_grad = true) {
        return tensor(std::move(shape), requires_grad);
    }

    static tensor ones(std::vector<std::size_t> shape, bool requires_grad = true) {
        tensor t(std::move(shape), requires_grad);
        std::fill(t.data_->begin(), t.data_->end(), T{1});
        return t;
    }

    static tensor randn(std::vector<std::size_t> shape, bool requires_grad = true) {
        tensor t(std::move(shape), requires_grad);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dist(0.0, 1.0);
        for (auto& val : *t.data_) {
            val = static_cast<T>(dist(gen));
        }
        return t;
    }

    static tensor uniform(std::vector<std::size_t> shape, T min = T{0}, T max = T{1},
                          bool requires_grad = true) {
        tensor t(std::move(shape), requires_grad);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(static_cast<double>(min),
                                                    static_cast<double>(max));
        for (auto& val : *t.data_) {
            val = static_cast<T>(dist(gen));
        }
        return t;
    }

    // Shape and size accessors
    const std::vector<std::size_t>& shape() const { return shape_; }
    const std::vector<std::size_t>& strides() const { return strides_; }
    std::size_t size() const { return size_; }
    std::size_t ndim() const { return shape_.size(); }
    bool requires_grad() const { return requires_grad_; }

    // Data access
    T* data() { return data_->data(); }
    const T* data() const { return data_->data(); }
    T* grad() { return grad_ ? grad_->data() : nullptr; }
    const T* grad() const { return grad_ ? grad_->data() : nullptr; }

    // Element access
    T& operator[](std::size_t index) {
        return (*data_)[index];
    }

    const T& operator[](std::size_t index) const {
        return (*data_)[index];
    }

    // Multi-dimensional indexing
    T& at(std::initializer_list<std::size_t> indices) {
        std::size_t flat_index = 0;
        auto it = indices.begin();
        for (std::size_t i = 0; i < shape_.size() && it != indices.end(); ++i, ++it) {
            flat_index += (*it) * strides_[i];
        }
        return (*data_)[flat_index];
    }

    const T& at(std::initializer_list<std::size_t> indices) const {
        std::size_t flat_index = 0;
        auto it = indices.begin();
        for (std::size_t i = 0; i < shape_.size() && it != indices.end(); ++i, ++it) {
            flat_index += (*it) * strides_[i];
        }
        return (*data_)[flat_index];
    }

    // Gradient operations
    void zero_grad() {
        if (grad_) {
            std::fill(grad_->begin(), grad_->end(), T{0});
        }
    }

    void backward(T gradient = T{1}) {
        if (!requires_grad_) return;

        // Seed gradient
        if (grad_) {
            std::fill(grad_->begin(), grad_->end(), gradient);
        }

        // Execute backward function if it exists
        if (backward_fn_) {
            backward_fn_();
        }

        // Propagate to children
        for (auto& child : children_) {
            if (child && child->requires_grad()) {
                child->backward();
            }
        }
    }

    // Arithmetic operations
    tensor operator+(const tensor& other) const {
        if (shape_ != other.shape_) {
            throw std::runtime_error("Tensor shapes must match for addition");
        }

        tensor result(shape_, requires_grad_ || other.requires_grad_);
        for (std::size_t i = 0; i < size_; ++i) {
            result[i] = (*data_)[i] + (*other.data_)[i];
        }

        // Set up backward pass
        if (result.requires_grad_) {
            auto self_ptr = std::make_shared<tensor>(*this);
            auto other_ptr = std::make_shared<tensor>(other);
            result.children_ = {self_ptr, other_ptr};

            result.backward_fn_ = [self_ptr, other_ptr, &result]() {
                if (self_ptr->grad_) {
                    for (std::size_t i = 0; i < self_ptr->size_; ++i) {
                        (*self_ptr->grad_)[i] += (*result.grad_)[i];
                    }
                }
                if (other_ptr->grad_) {
                    for (std::size_t i = 0; i < other_ptr->size_; ++i) {
                        (*other_ptr->grad_)[i] += (*result.grad_)[i];
                    }
                }
            };
        }

        return result;
    }

    tensor operator-(const tensor& other) const {
        if (shape_ != other.shape_) {
            throw std::runtime_error("Tensor shapes must match for subtraction");
        }

        tensor result(shape_, requires_grad_ || other.requires_grad_);
        for (std::size_t i = 0; i < size_; ++i) {
            result[i] = (*data_)[i] - (*other.data_)[i];
        }

        // Set up backward pass
        if (result.requires_grad_) {
            auto self_ptr = std::make_shared<tensor>(*this);
            auto other_ptr = std::make_shared<tensor>(other);
            result.children_ = {self_ptr, other_ptr};

            result.backward_fn_ = [self_ptr, other_ptr, &result]() {
                if (self_ptr->grad_) {
                    for (std::size_t i = 0; i < self_ptr->size_; ++i) {
                        (*self_ptr->grad_)[i] += (*result.grad_)[i];
                    }
                }
                if (other_ptr->grad_) {
                    for (std::size_t i = 0; i < other_ptr->size_; ++i) {
                        (*other_ptr->grad_)[i] -= (*result.grad_)[i];
                    }
                }
            };
        }

        return result;
    }

    // Element-wise multiplication
    tensor operator*(const tensor& other) const {
        if (shape_ != other.shape_) {
            throw std::runtime_error("Tensor shapes must match for element-wise multiplication");
        }

        tensor result(shape_, requires_grad_ || other.requires_grad_);
        for (std::size_t i = 0; i < size_; ++i) {
            result[i] = (*data_)[i] * (*other.data_)[i];
        }

        // Set up backward pass
        if (result.requires_grad_) {
            auto self_ptr = std::make_shared<tensor>(*this);
            auto other_ptr = std::make_shared<tensor>(other);
            result.children_ = {self_ptr, other_ptr};

            result.backward_fn_ = [self_ptr, other_ptr, &result]() {
                if (self_ptr->grad_) {
                    for (std::size_t i = 0; i < self_ptr->size_; ++i) {
                        (*self_ptr->grad_)[i] += (*result.grad_)[i] * (*other_ptr->data_)[i];
                    }
                }
                if (other_ptr->grad_) {
                    for (std::size_t i = 0; i < other_ptr->size_; ++i) {
                        (*other_ptr->grad_)[i] += (*result.grad_)[i] * (*self_ptr->data_)[i];
                    }
                }
            };
        }

        return result;
    }

    // Scalar operations
    tensor operator*(T scalar) const {
        tensor result(shape_, requires_grad_);
        for (std::size_t i = 0; i < size_; ++i) {
            result[i] = (*data_)[i] * scalar;
        }

        if (result.requires_grad_) {
            auto self_ptr = std::make_shared<tensor>(*this);
            result.children_ = {self_ptr};

            result.backward_fn_ = [self_ptr, scalar, &result]() {
                if (self_ptr->grad_) {
                    for (std::size_t i = 0; i < self_ptr->size_; ++i) {
                        (*self_ptr->grad_)[i] += (*result.grad_)[i] * scalar;
                    }
                }
            };
        }

        return result;
    }

    friend tensor operator*(T scalar, const tensor& t) {
        return t * scalar;
    }

    tensor operator+(T scalar) const {
        tensor result(shape_, requires_grad_);
        for (std::size_t i = 0; i < size_; ++i) {
            result[i] = (*data_)[i] + scalar;
        }

        if (result.requires_grad_) {
            auto self_ptr = std::make_shared<tensor>(*this);
            result.children_ = {self_ptr};

            result.backward_fn_ = [self_ptr, &result]() {
                if (self_ptr->grad_) {
                    for (std::size_t i = 0; i < self_ptr->size_; ++i) {
                        (*self_ptr->grad_)[i] += (*result.grad_)[i];
                    }
                }
            };
        }

        return result;
    }

    // Matrix multiplication (2D tensors only)
    tensor matmul(const tensor& other) const {
        if (shape_.size() != 2 || other.shape_.size() != 2) {
            throw std::runtime_error("matmul requires 2D tensors");
        }
        if (shape_[1] != other.shape_[0]) {
            throw std::runtime_error("Inner dimensions must match for matmul");
        }

        std::vector<std::size_t> result_shape = {shape_[0], other.shape_[1]};
        tensor result(result_shape, requires_grad_ || other.requires_grad_);

        // Perform matrix multiplication
        for (std::size_t i = 0; i < shape_[0]; ++i) {
            for (std::size_t j = 0; j < other.shape_[1]; ++j) {
                T sum = T{0};
                for (std::size_t k = 0; k < shape_[1]; ++k) {
                    sum += (*data_)[i * shape_[1] + k] *
                           (*other.data_)[k * other.shape_[1] + j];
                }
                result[i * other.shape_[1] + j] = sum;
            }
        }

        // Set up backward pass
        if (result.requires_grad_) {
            auto self_ptr = std::make_shared<tensor>(*this);
            auto other_ptr = std::make_shared<tensor>(other);
            result.children_ = {self_ptr, other_ptr};

            result.backward_fn_ = [self_ptr, other_ptr, &result]() {
                // Gradient for self: result.grad @ other.T
                if (self_ptr->grad_) {
                    for (std::size_t i = 0; i < self_ptr->shape_[0]; ++i) {
                        for (std::size_t j = 0; j < self_ptr->shape_[1]; ++j) {
                            T sum = T{0};
                            for (std::size_t k = 0; k < other_ptr->shape_[1]; ++k) {
                                sum += (*result.grad_)[i * other_ptr->shape_[1] + k] *
                                       (*other_ptr->data_)[j * other_ptr->shape_[1] + k];
                            }
                            (*self_ptr->grad_)[i * self_ptr->shape_[1] + j] += sum;
                        }
                    }
                }

                // Gradient for other: self.T @ result.grad
                if (other_ptr->grad_) {
                    for (std::size_t i = 0; i < other_ptr->shape_[0]; ++i) {
                        for (std::size_t j = 0; j < other_ptr->shape_[1]; ++j) {
                            T sum = T{0};
                            for (std::size_t k = 0; k < self_ptr->shape_[0]; ++k) {
                                sum += (*self_ptr->data_)[k * self_ptr->shape_[1] + i] *
                                       (*result.grad_)[k * other_ptr->shape_[1] + j];
                            }
                            (*other_ptr->grad_)[i * other_ptr->shape_[1] + j] += sum;
                        }
                    }
                }
            };
        }

        return result;
    }

    // Reshape
    tensor reshape(std::vector<std::size_t> new_shape) const {
        std::size_t new_size = 1;
        for (auto dim : new_shape) {
            new_size *= dim;
        }
        if (new_size != size_) {
            throw std::runtime_error("Cannot reshape tensor: size mismatch");
        }

        tensor result(new_shape, requires_grad_);
        std::copy(data_->begin(), data_->end(), result.data_->begin());

        if (requires_grad_ && grad_) {
            std::copy(grad_->begin(), grad_->end(), result.grad_->begin());
        }

        return result;
    }

    // Transpose (2D tensors only)
    tensor transpose() const {
        if (shape_.size() != 2) {
            throw std::runtime_error("transpose requires a 2D tensor");
        }

        std::vector<std::size_t> new_shape = {shape_[1], shape_[0]};
        tensor result(new_shape, requires_grad_);

        for (std::size_t i = 0; i < shape_[0]; ++i) {
            for (std::size_t j = 0; j < shape_[1]; ++j) {
                result[j * shape_[0] + i] = (*data_)[i * shape_[1] + j];
            }
        }

        return result;
    }

    // Reduction operations
    T sum() const {
        return std::accumulate(data_->begin(), data_->end(), T{0});
    }

    T mean() const {
        return sum() / static_cast<T>(size_);
    }

    T max() const {
        return *std::max_element(data_->begin(), data_->end());
    }

    T min() const {
        return *std::min_element(data_->begin(), data_->end());
    }

    // Activation functions
    tensor relu() const {
        tensor result(shape_, requires_grad_);
        for (std::size_t i = 0; i < size_; ++i) {
            result[i] = std::max(T{0}, (*data_)[i]);
        }

        if (result.requires_grad_) {
            auto self_ptr = std::make_shared<tensor>(*this);
            result.children_ = {self_ptr};

            result.backward_fn_ = [self_ptr, &result]() {
                if (self_ptr->grad_) {
                    for (std::size_t i = 0; i < self_ptr->size_; ++i) {
                        (*self_ptr->grad_)[i] += (*result.grad_)[i] *
                                                 ((*self_ptr->data_)[i] > T{0} ? T{1} : T{0});
                    }
                }
            };
        }

        return result;
    }

    tensor sigmoid() const {
        tensor result(shape_, requires_grad_);
        for (std::size_t i = 0; i < size_; ++i) {
            result[i] = T{1} / (T{1} + std::exp(-(*data_)[i]));
        }

        if (result.requires_grad_) {
            auto self_ptr = std::make_shared<tensor>(*this);
            result.children_ = {self_ptr};

            result.backward_fn_ = [self_ptr, &result]() {
                if (self_ptr->grad_) {
                    for (std::size_t i = 0; i < self_ptr->size_; ++i) {
                        T sig_val = (*result.data_)[i];
                        (*self_ptr->grad_)[i] += (*result.grad_)[i] * sig_val * (T{1} - sig_val);
                    }
                }
            };
        }

        return result;
    }

    tensor tanh() const {
        tensor result(shape_, requires_grad_);
        for (std::size_t i = 0; i < size_; ++i) {
            result[i] = std::tanh((*data_)[i]);
        }

        if (result.requires_grad_) {
            auto self_ptr = std::make_shared<tensor>(*this);
            result.children_ = {self_ptr};

            result.backward_fn_ = [self_ptr, &result]() {
                if (self_ptr->grad_) {
                    for (std::size_t i = 0; i < self_ptr->size_; ++i) {
                        T tanh_val = (*result.data_)[i];
                        (*self_ptr->grad_)[i] += (*result.grad_)[i] * (T{1} - tanh_val * tanh_val);
                    }
                }
            };
        }

        return result;
    }
};

// =============================================================================
// Optimizers for gradient descent
// =============================================================================

/**
 * Stochastic Gradient Descent optimizer
 */
template<typename T = float>
class sgd_optimizer {
private:
    T learning_rate_;
    T momentum_;
    std::unordered_map<void*, std::vector<T>> velocities_;

public:
    sgd_optimizer(T learning_rate, T momentum = T{0})
        : learning_rate_(learning_rate), momentum_(momentum) {}

    void step(std::vector<tensor<T>*>& parameters) {
        for (auto* param : parameters) {
            if (!param->requires_grad() || !param->grad()) continue;

            auto* data = param->data();
            auto* grad = param->grad();
            auto size = param->size();

            if (momentum_ > T{0}) {
                // Get or create velocity buffer
                void* key = static_cast<void*>(param);
                if (velocities_.find(key) == velocities_.end()) {
                    velocities_[key] = std::vector<T>(size, T{0});
                }
                auto& velocity = velocities_[key];

                // Update with momentum
                for (std::size_t i = 0; i < size; ++i) {
                    velocity[i] = momentum_ * velocity[i] - learning_rate_ * grad[i];
                    data[i] += velocity[i];
                }
            } else {
                // Simple gradient descent
                for (std::size_t i = 0; i < size; ++i) {
                    data[i] -= learning_rate_ * grad[i];
                }
            }
        }
    }

    void zero_grad(std::vector<tensor<T>*>& parameters) {
        for (auto* param : parameters) {
            param->zero_grad();
        }
    }
};

/**
 * Adam optimizer
 */
template<typename T = float>
class adam_optimizer {
private:
    T learning_rate_;
    T beta1_;
    T beta2_;
    T epsilon_;
    std::size_t step_count_;
    std::unordered_map<void*, std::vector<T>> m_buffers_;
    std::unordered_map<void*, std::vector<T>> v_buffers_;

public:
    adam_optimizer(T learning_rate = T{0.001},
                  T beta1 = T{0.9},
                  T beta2 = T{0.999},
                  T epsilon = T{1e-8})
        : learning_rate_(learning_rate),
          beta1_(beta1),
          beta2_(beta2),
          epsilon_(epsilon),
          step_count_(0) {}

    void step(std::vector<tensor<T>*>& parameters) {
        ++step_count_;

        for (auto* param : parameters) {
            if (!param->requires_grad() || !param->grad()) continue;

            auto* data = param->data();
            auto* grad = param->grad();
            auto size = param->size();

            // Get or create moment buffers
            void* key = static_cast<void*>(param);
            if (m_buffers_.find(key) == m_buffers_.end()) {
                m_buffers_[key] = std::vector<T>(size, T{0});
                v_buffers_[key] = std::vector<T>(size, T{0});
            }
            auto& m = m_buffers_[key];
            auto& v = v_buffers_[key];

            // Bias correction
            T bias_correction1 = T{1} - std::pow(beta1_, step_count_);
            T bias_correction2 = T{1} - std::pow(beta2_, step_count_);

            // Update parameters
            for (std::size_t i = 0; i < size; ++i) {
                m[i] = beta1_ * m[i] + (T{1} - beta1_) * grad[i];
                v[i] = beta2_ * v[i] + (T{1} - beta2_) * grad[i] * grad[i];

                T m_hat = m[i] / bias_correction1;
                T v_hat = v[i] / bias_correction2;

                data[i] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
            }
        }
    }

    void zero_grad(std::vector<tensor<T>*>& parameters) {
        for (auto* param : parameters) {
            param->zero_grad();
        }
    }
};

// =============================================================================
// Type-erased differentiable wrapper
// =============================================================================

// Use the any_differentiable from type_erasure.hpp
template<typename T>
using differentiable = any_differentiable<T>;

// Factory function for creating type-erased differentiable values
template<typename T>
differentiable<T> make_differentiable(dual<T> value) {
    return differentiable<T>(value);
}

template<typename T>
differentiable<T> make_differentiable(tensor<T> value) {
    return differentiable<T>(value);
}

} // namespace stepanov