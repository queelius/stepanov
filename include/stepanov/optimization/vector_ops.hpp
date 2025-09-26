#pragma once

#include "../concepts.hpp"
#include "../autodiff.hpp"
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace stepanov::optimization {

/**
 * Vector operations for optimization algorithms
 *
 * Provides generic implementations of common vector operations needed
 * for optimization, following Stepanov's generic programming principles.
 */

// Basic vector type using std::vector
template<typename T>
    requires field<T>
class vector {
private:
    std::vector<T> data_;

public:
    using value_type = T;
    using size_type = std::size_t;

    // Constructors
    vector() = default;
    explicit vector(size_type n, T val = T(0)) : data_(n, val) {}
    vector(std::initializer_list<T> init) : data_(init) {}
    vector(const std::vector<T>& v) : data_(v) {}

    // Size operations
    size_type size() const { return data_.size(); }
    bool empty() const { return data_.empty(); }
    void resize(size_type n, T val = T(0)) { data_.resize(n, val); }

    // Element access
    T& operator[](size_type i) { return data_[i]; }
    const T& operator[](size_type i) const { return data_[i]; }
    T& at(size_type i) { return data_.at(i); }
    const T& at(size_type i) const { return data_.at(i); }

    // Iterators
    auto begin() { return data_.begin(); }
    auto end() { return data_.end(); }
    auto begin() const { return data_.begin(); }
    auto end() const { return data_.end(); }

    // Vector arithmetic
    vector operator-() const {
        vector result(size());
        for (size_type i = 0; i < size(); ++i) {
            result[i] = -data_[i];
        }
        return result;
    }

    vector operator+(const vector& other) const {
        vector result(size());
        for (size_type i = 0; i < size(); ++i) {
            result[i] = data_[i] + other[i];
        }
        return result;
    }

    vector operator-(const vector& other) const {
        vector result(size());
        for (size_type i = 0; i < size(); ++i) {
            result[i] = data_[i] - other[i];
        }
        return result;
    }

    vector operator*(T scalar) const {
        vector result(size());
        for (size_type i = 0; i < size(); ++i) {
            result[i] = data_[i] * scalar;
        }
        return result;
    }

    friend vector operator*(T scalar, const vector& v) {
        return v * scalar;
    }

    vector operator/(T scalar) const {
        vector result(size());
        for (size_type i = 0; i < size(); ++i) {
            result[i] = data_[i] / scalar;
        }
        return result;
    }

    // Compound assignments
    vector& operator+=(const vector& other) {
        for (size_type i = 0; i < size(); ++i) {
            data_[i] += other[i];
        }
        return *this;
    }

    vector& operator-=(const vector& other) {
        for (size_type i = 0; i < size(); ++i) {
            data_[i] -= other[i];
        }
        return *this;
    }

    vector& operator*=(T scalar) {
        for (size_type i = 0; i < size(); ++i) {
            data_[i] *= scalar;
        }
        return *this;
    }

    vector& operator/=(T scalar) {
        for (size_type i = 0; i < size(); ++i) {
            data_[i] /= scalar;
        }
        return *this;
    }

    // Underlying data access
    const std::vector<T>& data() const { return data_; }
    std::vector<T>& data() { return data_; }
};

// Dot product
template<typename V>
    requires vector_space<V>
typename V::value_type dot(const V& a, const V& b) {
    using T = typename V::value_type;
    T result = T(0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// Euclidean norm
template<typename V>
    requires vector_space<V>
typename V::value_type norm(const V& v) {
    return std::sqrt(dot(v, v));
}

// L1 norm
template<typename V>
    requires vector_space<V>
typename V::value_type norm_l1(const V& v) {
    using T = typename V::value_type;
    T result = T(0);
    for (std::size_t i = 0; i < v.size(); ++i) {
        result += abs(v[i]);
    }
    return result;
}

// L-infinity norm
template<typename V>
    requires vector_space<V>
typename V::value_type norm_inf(const V& v) {
    using T = typename V::value_type;
    T result = T(0);
    for (std::size_t i = 0; i < v.size(); ++i) {
        result = std::max(result, abs(v[i]));
    }
    return result;
}

// Normalize vector
template<typename V>
    requires vector_space<V>
V normalize(const V& v) {
    auto n = norm(v);
    if (n == typename V::value_type(0)) return v;
    return v / n;
}

// Element-wise operations
template<typename V>
    requires vector_space<V>
V element_wise_multiply(const V& a, const V& b) {
    V result(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] * b[i];
    }
    return result;
}

template<typename V>
    requires vector_space<V>
V element_wise_divide(const V& a, const V& b) {
    V result(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] / b[i];
    }
    return result;
}

template<typename V>
    requires vector_space<V>
V element_wise_sqrt(const V& v) {
    V result(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        result[i] = std::sqrt(v[i]);
    }
    return result;
}

// Compute gradient using finite differences
template<typename F, typename V>
    requires evaluable<F, V> && vector_space<V>
V compute_gradient_finite_diff(F f, const V& x,
                               typename V::value_type epsilon = 1e-8) {
    using T = typename V::value_type;
    V grad(x.size());
    T fx = f(x);

    for (std::size_t i = 0; i < x.size(); ++i) {
        V x_plus = x;
        x_plus[i] += epsilon;
        T fx_plus = f(x_plus);
        grad[i] = (fx_plus - fx) / epsilon;
    }

    return grad;
}

// Compute gradient using automatic differentiation
template<typename F, typename V>
    requires vector_space<V>
V compute_gradient(F f, const V& x) {
    using T = typename V::value_type;
    using namespace stepanov;

    V grad(x.size());

    // For each component, compute partial derivative
    for (std::size_t i = 0; i < x.size(); ++i) {
        // Create dual number vector with seed in i-th component
        vector<dual<T>> x_dual(x.size());
        for (std::size_t j = 0; j < x.size(); ++j) {
            x_dual[j] = (i == j) ? dual<T>::variable(x[j], T(1))
                                 : dual<T>::constant(x[j]);
        }

        // Evaluate function with dual numbers
        dual<T> result = f(x_dual);
        grad[i] = result.derivative();
    }

    return grad;
}

// Simple matrix class for Hessian
template<typename T>
    requires field<T>
class matrix_type {
private:
    std::vector<std::vector<T>> data_;
    std::size_t rows_, cols_;

public:
    using value_type = T;

    matrix_type(std::size_t r, std::size_t c, T val = T(0))
        : data_(r, std::vector<T>(c, val)), rows_(r), cols_(c) {}

    T& operator()(std::size_t i, std::size_t j) { return data_[i][j]; }
    const T& operator()(std::size_t i, std::size_t j) const { return data_[i][j]; }

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }

    matrix_type operator+(const matrix_type& other) const {
        matrix_type result(rows_, cols_);
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < cols_; ++j) {
                result(i, j) = data_[i][j] + other(i, j);
            }
        }
        return result;
    }

    matrix_type operator*(const matrix_type& other) const {
        matrix_type result(rows_, other.cols_);
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < other.cols_; ++j) {
                T sum = T(0);
                for (std::size_t k = 0; k < cols_; ++k) {
                    sum += data_[i][k] * other(k, j);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }

    vector<T> operator*(const vector<T>& v) const {
        vector<T> result(rows_);
        for (std::size_t i = 0; i < rows_; ++i) {
            T sum = T(0);
            for (std::size_t j = 0; j < cols_; ++j) {
                sum += data_[i][j] * v[j];
            }
            result[i] = sum;
        }
        return result;
    }

    matrix_type operator*(T scalar) const {
        matrix_type result(rows_, cols_);
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < cols_; ++j) {
                result(i, j) = data_[i][j] * scalar;
            }
        }
        return result;
    }
};

// Compute Hessian using finite differences
template<typename F, typename V>
    requires evaluable<F, V> && vector_space<V>
matrix_type<typename V::value_type> compute_hessian_finite_diff(
    F f, const V& x, typename V::value_type epsilon = 1e-5)
{
    using T = typename V::value_type;
    std::size_t n = x.size();
    matrix_type<T> hess(n, n);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            V x_pp = x, x_pm = x, x_mp = x, x_mm = x;

            x_pp[i] += epsilon; x_pp[j] += epsilon;
            x_pm[i] += epsilon; x_pm[j] -= epsilon;
            x_mp[i] -= epsilon; x_mp[j] += epsilon;
            x_mm[i] -= epsilon; x_mm[j] -= epsilon;

            T f_pp = f(x_pp);
            T f_pm = f(x_pm);
            T f_mp = f(x_mp);
            T f_mm = f(x_mm);

            hess(i, j) = (f_pp - f_pm - f_mp + f_mm) / (T(4) * epsilon * epsilon);
            hess(j, i) = hess(i, j); // Symmetry
        }
    }

    return hess;
}

// Compute Hessian using automatic differentiation
template<typename F, typename V>
    requires vector_space<V>
matrix_type<typename V::value_type> compute_hessian(F f, const V& x) {
    using T = typename V::value_type;
    using namespace stepanov;

    std::size_t n = x.size();
    matrix_type<T> hess(n, n);

    // Use nested dual numbers for second derivatives
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            vector<dual<dual<T>>> x_dual(n);

            for (std::size_t k = 0; k < n; ++k) {
                if (k == i) {
                    x_dual[k] = dual<dual<T>>::variable(
                        dual<T>::variable(x[k], T(1)),
                        dual<T>::constant(T(k == j ? 1 : 0))
                    );
                } else if (k == j) {
                    x_dual[k] = dual<dual<T>>::variable(
                        dual<T>::variable(x[k], T(0)),
                        dual<T>::constant(T(1))
                    );
                } else {
                    x_dual[k] = dual<dual<T>>::constant(
                        dual<T>::constant(x[k])
                    );
                }
            }

            auto result = f(x_dual);
            hess(i, j) = result.derivative().derivative();
            hess(j, i) = hess(i, j); // Symmetry
        }
    }

    return hess;
}

// Solve linear system Ax = b using Gaussian elimination
template<typename T>
    requires field<T>
vector<T> solve_linear_system(matrix_type<T> A, vector<T> b) {
    std::size_t n = b.size();
    vector<T> x = b;

    // Forward elimination
    for (std::size_t k = 0; k < n; ++k) {
        // Find pivot
        std::size_t pivot = k;
        for (std::size_t i = k + 1; i < n; ++i) {
            if (abs(A(i, k)) > abs(A(pivot, k))) {
                pivot = i;
            }
        }

        // Swap rows
        if (pivot != k) {
            for (std::size_t j = k; j < n; ++j) {
                std::swap(A(k, j), A(pivot, j));
            }
            std::swap(x[k], x[pivot]);
        }

        // Eliminate column
        for (std::size_t i = k + 1; i < n; ++i) {
            T factor = A(i, k) / A(k, k);
            for (std::size_t j = k + 1; j < n; ++j) {
                A(i, j) -= factor * A(k, j);
            }
            x[i] -= factor * x[k];
        }
    }

    // Back substitution
    for (std::size_t i = n; i > 0; --i) {
        std::size_t k = i - 1;
        for (std::size_t j = k + 1; j < n; ++j) {
            x[k] -= A(k, j) * x[j];
        }
        x[k] /= A(k, k);
    }

    return x;
}

// Matrix inverse using Gauss-Jordan elimination
template<typename T>
    requires field<T>
matrix_type<T> inverse(const matrix_type<T>& A) {
    std::size_t n = A.rows();
    matrix_type<T> aug(n, 2 * n); // Augmented matrix [A | I]

    // Initialize augmented matrix
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            aug(i, j) = A(i, j);
            aug(i, j + n) = (i == j) ? T(1) : T(0);
        }
    }

    // Gauss-Jordan elimination
    for (std::size_t k = 0; k < n; ++k) {
        // Find pivot
        std::size_t pivot = k;
        for (std::size_t i = k + 1; i < n; ++i) {
            if (abs(aug(i, k)) > abs(aug(pivot, k))) {
                pivot = i;
            }
        }

        // Swap rows
        if (pivot != k) {
            for (std::size_t j = 0; j < 2 * n; ++j) {
                std::swap(aug(k, j), aug(pivot, j));
            }
        }

        // Scale pivot row
        T pivot_val = aug(k, k);
        for (std::size_t j = 0; j < 2 * n; ++j) {
            aug(k, j) /= pivot_val;
        }

        // Eliminate column
        for (std::size_t i = 0; i < n; ++i) {
            if (i != k) {
                T factor = aug(i, k);
                for (std::size_t j = 0; j < 2 * n; ++j) {
                    aug(i, j) -= factor * aug(k, j);
                }
            }
        }
    }

    // Extract inverse from augmented matrix
    matrix_type<T> inv(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            inv(i, j) = aug(i, j + n);
        }
    }

    return inv;
}

// Quadratic form: x^T * A * x
template<typename T>
    requires field<T>
T quadratic_form(const vector<T>& x, const matrix_type<T>& A, const vector<T>& y) {
    T result = T(0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        T row_sum = T(0);
        for (std::size_t j = 0; j < y.size(); ++j) {
            row_sum += A(i, j) * y[j];
        }
        result += x[i] * row_sum;
    }
    return result;
}

// Trust region subproblem solver (simplified)
template<typename T>
    requires field<T>
vector<T> solve_trust_region_subproblem(const vector<T>& grad,
                                        const matrix_type<T>& hess,
                                        T radius) {
    // Simplified: just use Newton direction scaled to fit trust region
    vector<T> newton_dir = solve_linear_system(hess, -grad);
    T dir_norm = norm(newton_dir);

    if (dir_norm <= radius) {
        return newton_dir;
    } else {
        return newton_dir * (radius / dir_norm);
    }
}

// Absolute value for optimization
template<typename T>
    requires ordered_field<T>
T abs(T x) {
    return x < T(0) ? -x : x;
}

// Half operation for bisection
template<typename T>
    requires field<T>
T half(T x) {
    return x / T(2);
}

// Square root wrapper
template<typename T>
    requires field<T>
T sqrt(T x) {
    return std::sqrt(x);
}

} // namespace stepanov::optimization