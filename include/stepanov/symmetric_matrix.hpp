#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <optional>
#include "concepts.hpp"
#include "matrix.hpp"
#include "matrix_expressions.hpp"  // Need this for property tags

namespace stepanov {

/**
 * Symmetric matrix storage - stores only upper triangle
 * For an n×n symmetric matrix, we store only n(n+1)/2 elements
 *
 * This is particularly useful for:
 * - Adjacency matrices of undirected graphs
 * - Covariance/correlation matrices
 * - Distance matrices
 * - Gram matrices
 */
template<typename T>
class symmetric_storage {
public:
    using value_type = T;

private:
    std::vector<T> data_;  // Stores upper triangle in row-major order
    size_t n_;  // Matrix dimension

    // Convert (i,j) to linear index in upper triangle storage
    size_t index(size_t i, size_t j) const {
        if (i > j) std::swap(i, j);
        // Upper triangle indexing: i <= j
        // Index = i*n - i*(i-1)/2 + (j-i)
        return i * n_ - (i * (i - 1)) / 2 + (j - i);
    }

public:
    symmetric_storage() : n_(0) {}

    void allocate(size_t n) {
        n_ = n;
        data_.resize((n * (n + 1)) / 2);
    }

    void allocate(size_t n, const T& val) {
        n_ = n;
        data_.resize((n * (n + 1)) / 2, val);
    }

    size_t rows() const { return n_; }
    size_t cols() const { return n_; }
    size_t size() const { return n_; }

    T& get(size_t i, size_t j) {
        return data_[index(i, j)];
    }

    const T& get(size_t i, size_t j) const {
        return data_[index(i, j)];
    }

    void set(size_t i, size_t j, const T& val) {
        data_[index(i, j)] = val;
    }

    // Memory usage: n(n+1)/2 vs n^2 for dense storage
    size_t memory_usage() const {
        return data_.size() * sizeof(T);
    }

    // Efficient iteration over unique elements
    template<typename F>
    void for_each_unique(F&& f) {
        for (size_t i = 0; i < n_; ++i) {
            for (size_t j = i; j < n_; ++j) {
                f(i, j, get(i, j));
            }
        }
    }

    // Direct data access for optimized algorithms
    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }
};

/**
 * Symmetric matrix class - guarantees symmetry by construction
 */
template<typename T>
    requires ring<T>
class symmetric_matrix {
public:
    using value_type = T;
    using storage_type = symmetric_storage<T>;
    using property_tag = matrix_expr::symmetric_tag;  // Add property tag for compile-time dispatch

private:
    storage_type storage_;

public:
    // Constructors
    symmetric_matrix() = default;

    explicit symmetric_matrix(size_t n) {
        storage_.allocate(n, T(0));
    }

    symmetric_matrix(size_t n, const T& val) {
        storage_.allocate(n, val);
    }

    // Factory methods
    static symmetric_matrix identity(size_t n) {
        symmetric_matrix result(n, T(0));
        for (size_t i = 0; i < n; ++i) {
            result(i, i) = T(1);
        }
        return result;
    }

    static symmetric_matrix from_function(size_t n, auto f) {
        symmetric_matrix result(n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i; j < n; ++j) {
                result(i, j) = f(i, j);
            }
        }
        return result;
    }

    // Size
    size_t rows() const { return storage_.rows(); }
    size_t cols() const { return storage_.cols(); }
    size_t size() const { return storage_.size(); }

    // Element access
    T operator()(size_t i, size_t j) const {
        return storage_.get(i, j);
    }

    T& operator()(size_t i, size_t j) {
        return storage_.get(i, j);
    }

    // Matrix-vector multiplication (optimized for symmetry)
    std::vector<T> operator*(const std::vector<T>& v) const {
        size_t n = size();
        if (v.size() != n) {
            throw std::invalid_argument("Vector size mismatch");
        }

        std::vector<T> result(n, T(0));

        // Exploit symmetry: when processing element (i,j),
        // update both result[i] and result[j]
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i; j < n; ++j) {
                T val = storage_.get(i, j);
                result[i] += val * v[j];
                if (i != j) {
                    result[j] += val * v[i];
                }
            }
        }

        return result;
    }

    // Efficient quadratic form: x^T * A * x
    T quadratic_form(const std::vector<T>& x) const {
        size_t n = size();
        if (x.size() != n) {
            throw std::invalid_argument("Vector size mismatch");
        }

        T result = T(0);

        // Only iterate over upper triangle
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i; j < n; ++j) {
                T val = storage_.get(i, j) * x[i] * x[j];
                result += (i == j) ? val : T(2) * val;  // Double off-diagonal elements
            }
        }

        return result;
    }

    // Eigenvalue algorithms for symmetric matrices
    // Power method for largest eigenvalue
    std::pair<T, std::vector<T>> power_method(size_t max_iters = 1000,
                                               T tolerance = T(1e-10)) const {
        size_t n = size();
        std::vector<T> v(n, T(1) / sqrt(T(n)));  // Initial guess
        std::vector<T> Av(n);
        T eigenvalue = T(0);

        for (size_t iter = 0; iter < max_iters; ++iter) {
            // Compute Av
            Av = (*this) * v;

            // Compute Rayleigh quotient
            T vAv = T(0), vv = T(0);
            for (size_t i = 0; i < n; ++i) {
                vAv += v[i] * Av[i];
                vv += v[i] * v[i];
            }
            T new_eigenvalue = vAv / vv;

            // Check convergence
            if (abs(new_eigenvalue - eigenvalue) < tolerance) {
                return {new_eigenvalue, v};
            }
            eigenvalue = new_eigenvalue;

            // Normalize Av
            T norm = T(0);
            for (size_t i = 0; i < n; ++i) {
                norm += Av[i] * Av[i];
            }
            norm = sqrt(norm);
            for (size_t i = 0; i < n; ++i) {
                v[i] = Av[i] / norm;
            }
        }

        return {eigenvalue, v};
    }

    // Check if matrix is positive definite (all eigenvalues > 0)
    // Using Cholesky decomposition attempt
    bool is_positive_definite() const {
        size_t n = size();
        symmetric_matrix L(n, T(0));  // Lower triangular

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                T sum = storage_.get(i, j);

                for (size_t k = 0; k < j; ++k) {
                    sum -= L(i, k) * L(j, k);
                }

                if (i == j) {
                    if (sum <= T(0)) return false;
                    L(i, j) = sqrt(sum);
                } else {
                    L(i, j) = sum / L(j, j);
                }
            }
        }
        return true;
    }

    // Add scalar to diagonal (useful for regularization)
    symmetric_matrix& add_to_diagonal(const T& val) {
        size_t n = size();
        for (size_t i = 0; i < n; ++i) {
            storage_.get(i, i) += val;
        }
        return *this;
    }

    // Trace (sum of diagonal elements)
    T trace() const {
        T result = T(0);
        size_t n = size();
        for (size_t i = 0; i < n; ++i) {
            result += storage_.get(i, i);
        }
        return result;
    }
};

/**
 * Specialized algorithms for graph adjacency matrices
 */
template<typename T>
class graph_matrix : public symmetric_matrix<T> {
public:
    using symmetric_matrix<T>::symmetric_matrix;

    // Degree of vertex i (sum of row i)
    T degree(size_t i) const {
        T deg = T(0);
        size_t n = this->size();
        for (size_t j = 0; j < n; ++j) {
            deg += (*this)(i, j);
        }
        return deg;
    }

    // Degree vector
    std::vector<T> degree_vector() const {
        size_t n = this->size();
        std::vector<T> degrees(n);
        for (size_t i = 0; i < n; ++i) {
            degrees[i] = degree(i);
        }
        return degrees;
    }

    // Laplacian matrix: L = D - A
    symmetric_matrix<T> laplacian() const {
        size_t n = this->size();
        symmetric_matrix<T> L(n);

        for (size_t i = 0; i < n; ++i) {
            T deg = degree(i);
            L(i, i) = deg;
            for (size_t j = i + 1; j < n; ++j) {
                L(i, j) = -(*this)(i, j);
            }
        }

        return L;
    }

    // Number of triangles in the graph (tr(A^3) / 6)
    T triangle_count() const {
        size_t n = this->size();
        T count = T(0);

        // For each triple (i,j,k) with i < j < k
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                if ((*this)(i, j) != T(0)) {
                    for (size_t k = j + 1; k < n; ++k) {
                        if ((*this)(i, k) != T(0) && (*this)(j, k) != T(0)) {
                            count += T(1);
                        }
                    }
                }
            }
        }

        return count;
    }

    // Check if graph is connected using BFS
    bool is_connected() const {
        size_t n = this->size();
        if (n == 0) return true;

        std::vector<bool> visited(n, false);
        std::vector<size_t> queue;
        queue.push_back(0);
        visited[0] = true;
        size_t visited_count = 1;
        size_t front = 0;

        while (front < queue.size()) {
            size_t u = queue[front++];
            for (size_t v = 0; v < n; ++v) {
                if (!visited[v] && (*this)(u, v) != T(0)) {
                    visited[v] = true;
                    visited_count++;
                    queue.push_back(v);
                }
            }
        }

        return visited_count == n;
    }
};

} // namespace stepanov