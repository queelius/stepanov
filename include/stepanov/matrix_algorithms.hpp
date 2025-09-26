#pragma once

#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <functional>
#include <concepts>
#include <utility>
#include <span>
#include <ranges>
#include <cmath>
#include <optional>
#include <variant>
#include <memory>
#include <execution>
#include <tuple>
#include <type_traits>
#include <complex>
#include "concepts.hpp"
#include "math.hpp"
#include "matrix.hpp"
#include "symmetric_matrix.hpp"
#include "matrix_expressions.hpp"
#include "matrix_type_erasure.hpp"

namespace stepanov::matrix_algorithms {

// =============================================================================
// Matrix Property Traits - Compile-time property detection and propagation
// =============================================================================

/**
 * Traits for detecting and propagating matrix properties through operations
 * Following Stepanov's principle: "Use the type system to enforce invariants"
 */
template<typename M>
struct matrix_traits {
    using value_type = typename M::value_type;
    using property_tag = typename M::property_tag;

    static constexpr bool is_symmetric = matrix_expr::is_symmetric<property_tag>;
    static constexpr bool is_diagonal = matrix_expr::is_diagonal<property_tag>;
    static constexpr bool is_triangular = matrix_expr::is_triangular<property_tag>;

    // Closure properties under operations
    static constexpr bool closed_under_addition = is_diagonal || is_symmetric;
    static constexpr bool closed_under_multiplication = is_diagonal;
    static constexpr bool closed_under_inversion = is_diagonal || is_triangular;
};

/**
 * Result type deduction for matrix operations
 * Encodes mathematical laws about closure properties
 */
template<typename M1, typename M2, typename Op>
struct operation_result {
    using type = matrix<typename M1::value_type>;  // Default to general matrix
};

// Specializations for preserving structure
template<typename T>
struct operation_result<matrix_expr::diagonal_matrix<T>,
                       matrix_expr::diagonal_matrix<T>,
                       std::multiplies<>> {
    using type = matrix_expr::diagonal_matrix<T>;  // Diagonal × Diagonal = Diagonal
};

template<typename T>
struct operation_result<symmetric_matrix<T>, symmetric_matrix<T>, std::plus<>> {
    using type = symmetric_matrix<T>;  // Symmetric + Symmetric = Symmetric
};

// =============================================================================
// Generic Inverse Algorithms with Compile-Time Dispatch
// =============================================================================

/**
 * Generic matrix inverse with optimal algorithm selection based on matrix type
 * Follows the principle: "Algorithm = Logic + Control"
 */
template<typename Matrix>
    requires field<typename Matrix::value_type>
class inverse_algorithm {
public:
    using value_type = typename Matrix::value_type;
    using result_type = Matrix;

    /**
     * Compute inverse with compile-time dispatch to optimal algorithm
     */
    static result_type compute(const Matrix& m) {
        if constexpr (matrix_traits<Matrix>::is_diagonal) {
            return inverse_diagonal(m);
        } else if constexpr (matrix_traits<Matrix>::is_triangular) {
            return inverse_triangular(m);
        } else if constexpr (matrix_traits<Matrix>::is_symmetric) {
            return inverse_symmetric(m);
        } else {
            return inverse_general(m);
        }
    }

private:
    // O(n) inverse for diagonal matrices
    static result_type inverse_diagonal(const Matrix& m) {
        result_type result(m.rows());
        for (size_t i = 0; i < m.rows(); ++i) {
            result.diagonal(i) = value_type(1) / m.diagonal(i);
        }
        return result;
    }

    // O(n²) inverse for triangular matrices using back substitution
    static result_type inverse_triangular(const Matrix& m) {
        size_t n = m.rows();
        result_type result(n);

        // Solve LX = I or UX = I column by column
        for (size_t j = 0; j < n; ++j) {
            std::vector<value_type> e(n, value_type(0));
            e[j] = value_type(1);
            auto col = m.solve(e);  // Uses efficient triangular solve
            for (size_t i = 0; i < n; ++i) {
                result.at(i, j) = col[i];
            }
        }
        return result;
    }

    // Inverse for symmetric positive definite using Cholesky
    static result_type inverse_symmetric(const Matrix& m) {
        // Try Cholesky decomposition first (most efficient for SPD)
        try {
            auto L = cholesky_decomposition(m);
            return cholesky_inverse(L);
        } catch (...) {
            // Fall back to LDL^T for indefinite symmetric
            return ldlt_inverse(m);
        }
    }

    // General inverse using LU decomposition
    static result_type inverse_general(const Matrix& m) {
        return m.inverse();  // Delegate to existing implementation
    }

    // Efficient inverse using Cholesky factor
    static result_type cholesky_inverse(const auto& L) {
        // A = LL^T => A^(-1) = (L^T)^(-1) L^(-1)
        auto L_inv = inverse_triangular(L);
        return L_inv.transpose() * L_inv;
    }

    // LDL^T inverse for symmetric indefinite matrices
    static result_type ldlt_inverse(const Matrix& m) {
        auto [L, D] = ldlt_decomposition(m);
        auto L_inv = inverse_triangular(L);
        auto D_inv = inverse_diagonal(D);
        return L_inv.transpose() * D_inv * L_inv;
    }
};

// Convenience function
template<typename Matrix>
auto inverse(const Matrix& m) {
    return inverse_algorithm<Matrix>::compute(m);
}

// =============================================================================
// Generic Linear System Solvers with Property Exploitation
// =============================================================================

/**
 * Solve Ax = b with optimal algorithm based on matrix structure
 * Embodies the principle of "zero-cost abstractions"
 */
template<typename Matrix, typename Vector>
    requires field<typename Matrix::value_type>
class linear_solver {
public:
    using value_type = typename Matrix::value_type;
    using vector_type = Vector;

    static vector_type solve(const Matrix& A, const vector_type& b) {
        if constexpr (matrix_traits<Matrix>::is_diagonal) {
            return solve_diagonal(A, b);
        } else if constexpr (matrix_traits<Matrix>::is_triangular) {
            return solve_triangular(A, b);
        } else if constexpr (matrix_traits<Matrix>::is_symmetric) {
            return solve_symmetric(A, b);
        } else if (is_sparse(A)) {
            return solve_sparse_iterative(A, b);
        } else {
            return solve_general(A, b);
        }
    }

private:
    // O(n) solve for diagonal systems
    static vector_type solve_diagonal(const Matrix& A, const vector_type& b) {
        size_t n = A.rows();
        vector_type x(n);
        for (size_t i = 0; i < n; ++i) {
            x[i] = b[i] / A.diagonal(i);
        }
        return x;
    }

    // O(n²) solve for triangular systems
    static vector_type solve_triangular(const Matrix& A, const vector_type& b) {
        return A.solve(b);  // Use specialized triangular solve
    }

    // Cholesky/LDLT for symmetric systems
    static vector_type solve_symmetric(const Matrix& A, const vector_type& b) {
        if (A.is_positive_definite()) {
            auto L = cholesky_decomposition(A);
            return cholesky_solve(L, b);
        } else {
            auto [L, D] = ldlt_decomposition(A);
            return ldlt_solve(L, D, b);
        }
    }

    // Iterative solver for sparse systems
    static vector_type solve_sparse_iterative(const Matrix& A, const vector_type& b) {
        // Use conjugate gradient for symmetric positive definite
        if constexpr (matrix_traits<Matrix>::is_symmetric) {
            return conjugate_gradient(A, b);
        } else {
            // BiCGSTAB for general sparse systems
            return bicgstab(A, b);
        }
    }

    // LU decomposition for general dense systems
    static vector_type solve_general(const Matrix& A, const vector_type& b) {
        return A.solve_gaussian(matrix<value_type>(b.size(), 1, b)).col(0);
    }

    // Helper: Cholesky solve
    static vector_type cholesky_solve(const auto& L, const vector_type& b) {
        // Solve Ly = b (forward substitution)
        auto y = L.solve(b);
        // Solve L^T x = y (back substitution)
        return L.transpose().solve(y);
    }

    // Helper: LDLT solve
    static vector_type ldlt_solve(const auto& L, const auto& D, const vector_type& b) {
        auto y = L.solve(b);
        for (size_t i = 0; i < y.size(); ++i) {
            y[i] /= D.diagonal(i);
        }
        return L.transpose().solve(y);
    }
};

// =============================================================================
// Advanced Decomposition Algorithms
// =============================================================================

/**
 * LDLT decomposition for symmetric matrices
 * A = LDL^T where L is unit lower triangular, D is diagonal
 */
template<typename Matrix>
    requires matrix_traits<Matrix>::is_symmetric && field<typename Matrix::value_type>
auto ldlt_decomposition(const Matrix& A) {
    using T = typename Matrix::value_type;
    size_t n = A.rows();

    matrix_expr::lower_triangular<T> L(n);
    matrix_expr::diagonal_matrix<T> D(n);

    for (size_t j = 0; j < n; ++j) {
        T sum = A(j, j);

        for (size_t k = 0; k < j; ++k) {
            sum -= L.at(j, k) * L.at(j, k) * D.diagonal(k);
        }
        D.diagonal(j) = sum;
        L.at(j, j) = T(1);  // Unit diagonal

        for (size_t i = j + 1; i < n; ++i) {
            sum = A(i, j);
            for (size_t k = 0; k < j; ++k) {
                sum -= L.at(i, k) * L.at(j, k) * D.diagonal(k);
            }
            L.at(i, j) = sum / D.diagonal(j);
        }
    }

    return std::make_pair(L, D);
}

/**
 * Householder QR decomposition with column pivoting
 * More numerically stable than Gram-Schmidt
 */
template<typename Matrix>
    requires field<typename Matrix::value_type>
auto qr_householder(const Matrix& A) {
    using T = typename Matrix::value_type;
    size_t m = A.rows(), n = A.cols();

    matrix<T> Q = matrix<T>::identity(m);
    matrix<T> R = A;
    std::vector<size_t> pivot(n);
    std::iota(pivot.begin(), pivot.end(), 0);

    for (size_t k = 0; k < std::min(m, n); ++k) {
        // Find pivot column (maximum norm)
        size_t max_col = k;
        T max_norm = T(0);
        for (size_t j = k; j < n; ++j) {
            T norm = T(0);
            for (size_t i = k; i < m; ++i) {
                norm += R(i, j) * R(i, j);
            }
            if (norm > max_norm) {
                max_norm = norm;
                max_col = j;
            }
        }

        // Swap columns
        if (max_col != k) {
            R.swap_cols(k, max_col);
            std::swap(pivot[k], pivot[max_col]);
        }

        // Compute Householder reflection
        std::vector<T> x(m - k);
        for (size_t i = 0; i < m - k; ++i) {
            x[i] = R(k + i, k);
        }

        T norm = std::sqrt(std::inner_product(x.begin(), x.end(), x.begin(), T(0)));
        if (norm == T(0)) continue;

        x[0] += (x[0] >= T(0) ? norm : -norm);
        T h_norm = std::sqrt(std::inner_product(x.begin(), x.end(), x.begin(), T(0)));

        for (auto& xi : x) xi /= h_norm;

        // Apply Householder reflection to R
        for (size_t j = k; j < n; ++j) {
            T dot = T(0);
            for (size_t i = 0; i < m - k; ++i) {
                dot += x[i] * R(k + i, j);
            }
            for (size_t i = 0; i < m - k; ++i) {
                R(k + i, j) -= T(2) * dot * x[i];
            }
        }

        // Apply Householder reflection to Q
        for (size_t j = 0; j < m; ++j) {
            T dot = T(0);
            for (size_t i = 0; i < m - k; ++i) {
                dot += x[i] * Q(k + i, j);
            }
            for (size_t i = 0; i < m - k; ++i) {
                Q(k + i, j) -= T(2) * dot * x[i];
            }
        }
    }

    return std::make_tuple(Q.transpose(), R, pivot);
}

// Forward declaration for SVD
template<typename Matrix>
    requires field<typename Matrix::value_type> && has_sqrt<typename Matrix::value_type>
auto svd(const Matrix& A, size_t max_iter = 1000);

// =============================================================================
// Eigenvalue Algorithms with Property Exploitation
// =============================================================================

// Forward declaration
template<typename Matrix>
    requires field<typename Matrix::value_type>
auto symmetric_eigendecomposition(const Matrix& A, size_t max_iter);

/**
 * Symmetric eigendecomposition using Jacobi method
 * Guaranteed convergence for symmetric matrices
 */
template<typename Matrix>
    requires field<typename Matrix::value_type>
auto symmetric_eigendecomposition(const Matrix& A, size_t max_iter) {
    using T = typename Matrix::value_type;
    size_t n = A.rows();

    matrix<T> V = matrix<T>::identity(n);  // Eigenvectors
    matrix<T> D(n, n);  // Will become diagonal

    // Copy A into D
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            D(i, j) = A(i, j);
        }
    }

    // Compute frobenius norm
    T fnorm = T(0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            fnorm += D(i, j) * D(i, j);
        }
    }
    T tolerance = T(1e-10) * std::sqrt(fnorm);

    for (size_t iter = 0; iter < max_iter; ++iter) {
        // Find largest off-diagonal element
        size_t p = 0, q = 1;
        T max_off = std::abs(D(0, 1));

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                T val = std::abs(D(i, j));
                if (val > max_off) {
                    max_off = val;
                    p = i;
                    q = j;
                }
            }
        }

        if (max_off < tolerance) break;

        // Compute Jacobi rotation
        T theta = (D(q, q) - D(p, p)) / (T(2) * D(p, q));
        T t = T(1) / (std::abs(theta) + std::sqrt(T(1) + theta * theta));
        if (theta < T(0)) t = -t;

        T c = T(1) / std::sqrt(T(1) + t * t);
        T s = t * c;

        // Apply rotation to D
        T app = D(p, p);
        T aqq = D(q, q);
        T apq = D(p, q);

        D(p, p) = c * c * app - T(2) * c * s * apq + s * s * aqq;
        D(q, q) = s * s * app + T(2) * c * s * apq + c * c * aqq;
        D(p, q) = D(q, p) = T(0);

        for (size_t i = 0; i < n; ++i) {
            if (i != p && i != q) {
                T aip = D(i, p);
                T aiq = D(i, q);
                D(i, p) = D(p, i) = c * aip - s * aiq;
                D(i, q) = D(q, i) = s * aip + c * aiq;
            }
        }

        // Update eigenvectors
        for (size_t i = 0; i < n; ++i) {
            T vip = V(i, p);
            T viq = V(i, q);
            V(i, p) = c * vip - s * viq;
            V(i, q) = s * vip + c * viq;
        }
    }

    // Extract eigenvalues
    std::vector<T> eigenvalues(n);
    for (size_t i = 0; i < n; ++i) {
        eigenvalues[i] = D(i, i);
    }

    return std::make_pair(V, eigenvalues);
}

/**
 * Arnoldi iteration for computing k largest eigenvalues of general matrices
 */
template<typename Matrix>
    requires field<typename Matrix::value_type>
auto arnoldi_eigenvalues(const Matrix& A, size_t k, size_t max_iter = 100) {
    using T = typename Matrix::value_type;
    size_t n = A.rows();

    if (k > n) k = n;

    // Initialize with random vector
    std::vector<T> v(n);
    for (auto& vi : v) vi = T(rand()) / T(RAND_MAX);
    T norm = std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), T(0)));
    for (auto& vi : v) vi /= norm;

    std::vector<std::vector<T>> V(k + 1, std::vector<T>(n));
    matrix<T> H(k + 1, k, T(0));

    V[0] = v;

    for (size_t j = 0; j < k; ++j) {
        // Compute w = A * v_j
        auto w = A * V[j];

        // Orthogonalize against previous vectors
        for (size_t i = 0; i <= j; ++i) {
            H(i, j) = std::inner_product(w.begin(), w.end(), V[i].begin(), T(0));
            for (size_t l = 0; l < n; ++l) {
                w[l] -= H(i, j) * V[i][l];
            }
        }

        // Compute norm
        H(j + 1, j) = std::sqrt(std::inner_product(w.begin(), w.end(), w.begin(), T(0)));

        if (H(j + 1, j) == T(0)) break;

        // Normalize
        for (size_t l = 0; l < n; ++l) {
            V[j + 1][l] = w[l] / H(j + 1, j);
        }
    }

    // Extract upper Hessenberg matrix and compute its eigenvalues
    matrix<T> H_square(k, k);
    for (size_t i = 0; i < k; ++i) {
        for (size_t j = 0; j < k; ++j) {
            H_square(i, j) = H(i, j);
        }
    }

    return H_square.qr_eigenvalues(max_iter);
}

/**
 * Singular Value Decomposition using QR algorithm
 * A = UΣV^T where U and V are orthogonal, Σ is diagonal
 */
template<typename Matrix>
    requires field<typename Matrix::value_type> && has_sqrt<typename Matrix::value_type>
auto svd(const Matrix& A, size_t max_iter) {
    using T = typename Matrix::value_type;
    size_t m = A.rows(), n = A.cols();

    // Compute A^T A and its eigenvalue decomposition
    auto ATA = A.transpose() * A;
    auto [V, singular_values_sq] = symmetric_eigendecomposition(ATA, max_iter);

    // Compute singular values
    matrix_expr::diagonal_matrix<T> Sigma(std::min(m, n));
    for (size_t i = 0; i < Sigma.rows(); ++i) {
        Sigma.diagonal(i) = std::sqrt(singular_values_sq[i]);
    }

    // Compute U = AV Σ^(-1)
    matrix<T> U(m, m);
    for (size_t i = 0; i < std::min(m, n); ++i) {
        if (Sigma.diagonal(i) != T(0)) {
            // Extract column i from V
            std::vector<T> v_col(n);
            for (size_t j = 0; j < n; ++j) {
                v_col[j] = V(j, i);
            }
            auto u_col = A * v_col;
            T norm = T(1) / Sigma.diagonal(i);
            for (size_t j = 0; j < m; ++j) {
                U(j, i) = u_col[j] * norm;
            }
        }
    }

    // Complete U with orthonormal basis if m > n
    if (m > n) {
        // Use Gram-Schmidt to complete the basis
        for (size_t i = n; i < m; ++i) {
            std::vector<T> v(m, T(0));
            v[i] = T(1);

            // Orthogonalize against existing columns
            for (size_t j = 0; j < i; ++j) {
                T dot = T(0);
                for (size_t k = 0; k < m; ++k) {
                    dot += U(k, j) * v[k];
                }
                for (size_t k = 0; k < m; ++k) {
                    v[k] -= dot * U(k, j);
                }
            }

            // Normalize
            T norm = std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), T(0)));
            for (size_t k = 0; k < m; ++k) {
                U(k, i) = v[k] / norm;
            }
        }
    }

    return std::make_tuple(U, Sigma, V);
}

// =============================================================================
// Forward Declarations for Helper Functions
// =============================================================================

template<typename Matrix>
Matrix pade_exp_approximation(const Matrix& X, size_t order);

template<typename Matrix>
Matrix pade_log_approximation(const Matrix& X, size_t order);

// =============================================================================
// Matrix Functions and Special Operations
// =============================================================================

/**
 * Matrix square root using Denman-Beavers iteration
 * Quadratically convergent for matrices with positive eigenvalues
 */
template<typename Matrix>
    requires field<typename Matrix::value_type>
Matrix matrix_sqrt_denman_beavers(const Matrix& A, size_t max_iter = 50) {
    using T = typename Matrix::value_type;

    Matrix Y = A;
    Matrix Z = Matrix::identity(A.rows());

    for (size_t iter = 0; iter < max_iter; ++iter) {
        Matrix Y_new = T(0.5) * (Y + inverse(Z));
        Matrix Z_new = T(0.5) * (Z + inverse(Y));

        // Check convergence
        T diff = (Y_new - Y).frobenius_norm();
        if (diff < T(1e-10)) {
            return Y_new;
        }

        Y = Y_new;
        Z = Z_new;
    }

    return Y;
}

/**
 * Matrix logarithm using inverse scaling and squaring with Padé approximation
 * For matrices close to identity
 */
template<typename Matrix>
    requires field<typename Matrix::value_type>
Matrix matrix_log_pade(const Matrix& A, size_t order = 7) {
    using T = typename Matrix::value_type;

    // Scale matrix to be close to identity
    T norm = (A - Matrix::identity(A.rows())).frobenius_norm();
    size_t s = 0;
    while (norm > T(0.5)) {
        norm /= T(2);
        s++;
    }

    Matrix B = A;
    for (size_t i = 0; i < s; ++i) {
        B = matrix_sqrt_denman_beavers(B);
    }

    // Compute log(B) using Padé approximation
    Matrix X = B - Matrix::identity(A.rows());
    Matrix log_B = pade_log_approximation(X, order);

    // Scale back
    return T(1 << s) * log_B;
}

/**
 * Matrix exponential using scaling and squaring with Padé approximation
 * Industry standard algorithm used in MATLAB and scipy
 */
template<typename Matrix>
    requires field<typename Matrix::value_type>
Matrix matrix_exp_pade(const Matrix& A, size_t order = 13) {
    using T = typename Matrix::value_type;

    // Determine scaling factor
    T norm = A.frobenius_norm();
    size_t s = std::max(0, static_cast<int>(std::ceil(std::log2(norm))));

    // Scale matrix
    Matrix B = A / T(1 << s);

    // Compute Padé approximant
    Matrix exp_B = pade_exp_approximation(B, order);

    // Square s times
    for (size_t i = 0; i < s; ++i) {
        exp_B = exp_B * exp_B;
    }

    return exp_B;
}

/**
 * Polar decomposition: A = UP where U is orthogonal, P is positive semidefinite
 * Applications: closest orthogonal matrix, matrix sign function
 */
template<typename Matrix>
    requires field<typename Matrix::value_type>
auto polar_decomposition(const Matrix& A, size_t max_iter = 100) {
    using T = typename Matrix::value_type;

    // Newton iteration: U_{k+1} = (U_k + (U_k^T)^{-1}) / 2
    Matrix U = A;

    for (size_t iter = 0; iter < max_iter; ++iter) {
        Matrix U_new = T(0.5) * (U + inverse(U.transpose()));

        if ((U_new - U).frobenius_norm() < T(1e-10)) {
            // Compute P = U^T A
            Matrix P = U.transpose() * A;
            return std::make_pair(U_new, P);
        }

        U = U_new;
    }

    Matrix P = U.transpose() * A;
    return std::make_pair(U, P);
}

// =============================================================================
// Iterative Solvers for Large Sparse Systems
// =============================================================================

/**
 * Conjugate Gradient method for symmetric positive definite systems
 * Krylov subspace method with optimal convergence properties
 */
template<typename Matrix, typename Vector>
    requires field<typename Matrix::value_type>
Vector conjugate_gradient(const Matrix& A, const Vector& b,
                          typename Matrix::value_type tolerance = 1e-10,
                          size_t max_iter = 0) {
    using T = typename Matrix::value_type;
    size_t n = A.rows();
    if (max_iter == 0) max_iter = n;

    Vector x(n, T(0));  // Initial guess
    Vector r = b;       // Initial residual
    Vector p = r;       // Initial search direction
    T rsold = std::inner_product(r.begin(), r.end(), r.begin(), T(0));

    for (size_t iter = 0; iter < max_iter; ++iter) {
        Vector Ap = A * p;
        T alpha = rsold / std::inner_product(p.begin(), p.end(), Ap.begin(), T(0));

        // Update solution and residual
        for (size_t i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        T rsnew = std::inner_product(r.begin(), r.end(), r.begin(), T(0));

        if (std::sqrt(rsnew) < tolerance) break;

        // Update search direction
        T beta = rsnew / rsold;
        for (size_t i = 0; i < n; ++i) {
            p[i] = r[i] + beta * p[i];
        }

        rsold = rsnew;
    }

    return x;
}

/**
 * BiCGSTAB (Bi-Conjugate Gradient Stabilized) for general non-symmetric systems
 * Combines BiCG with GMRES(1) for improved stability
 */
template<typename Matrix, typename Vector>
    requires field<typename Matrix::value_type>
Vector bicgstab(const Matrix& A, const Vector& b,
                typename Matrix::value_type tolerance = 1e-10,
                size_t max_iter = 0) {
    using T = typename Matrix::value_type;
    size_t n = A.rows();
    if (max_iter == 0) max_iter = n;

    Vector x(n, T(0));     // Initial guess
    Vector r = b;          // Initial residual
    Vector r_tilde = r;    // Shadow residual

    T rho = T(1), alpha = T(1), omega = T(1);
    Vector v(n, T(0)), p(n, T(0)), s(n), t(n);

    for (size_t iter = 0; iter < max_iter; ++iter) {
        T rho_new = std::inner_product(r_tilde.begin(), r_tilde.end(), r.begin(), T(0));

        if (std::abs(rho_new) < tolerance * tolerance) break;

        T beta = (rho_new / rho) * (alpha / omega);

        // Update search direction
        for (size_t i = 0; i < n; ++i) {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }

        v = A * p;
        alpha = rho_new / std::inner_product(r_tilde.begin(), r_tilde.end(), v.begin(), T(0));

        // Compute intermediate residual
        for (size_t i = 0; i < n; ++i) {
            s[i] = r[i] - alpha * v[i];
        }

        // Check convergence
        T s_norm = std::sqrt(std::inner_product(s.begin(), s.end(), s.begin(), T(0)));
        if (s_norm < tolerance) {
            for (size_t i = 0; i < n; ++i) {
                x[i] += alpha * p[i];
            }
            break;
        }

        t = A * s;
        omega = std::inner_product(t.begin(), t.end(), s.begin(), T(0)) /
                std::inner_product(t.begin(), t.end(), t.begin(), T(0));

        // Update solution and residual
        for (size_t i = 0; i < n; ++i) {
            x[i] += alpha * p[i] + omega * s[i];
            r[i] = s[i] - omega * t[i];
        }

        // Check convergence
        T r_norm = std::sqrt(std::inner_product(r.begin(), r.end(), r.begin(), T(0)));
        if (r_norm < tolerance) break;

        rho = rho_new;
    }

    return x;
}

/**
 * GMRES (Generalized Minimal Residual) for general non-symmetric systems
 * Minimizes residual over Krylov subspace
 */
template<typename Matrix, typename Vector>
    requires field<typename Matrix::value_type>
Vector gmres(const Matrix& A, const Vector& b, size_t restart = 30,
             typename Matrix::value_type tolerance = 1e-10,
             size_t max_iter = 0) {
    using T = typename Matrix::value_type;
    size_t n = A.rows();
    if (max_iter == 0) max_iter = n;

    Vector x(n, T(0));  // Initial guess

    for (size_t outer = 0; outer < max_iter / restart; ++outer) {
        Vector r = b - A * x;
        T r_norm = std::sqrt(std::inner_product(r.begin(), r.end(), r.begin(), T(0)));

        if (r_norm < tolerance) break;

        std::vector<Vector> V(restart + 1, Vector(n));
        matrix<T> H(restart + 1, restart, T(0));
        std::vector<T> g(restart + 1, T(0));
        g[0] = r_norm;

        // Normalize first basis vector
        for (size_t i = 0; i < n; ++i) {
            V[0][i] = r[i] / r_norm;
        }

        // Arnoldi process
        size_t j;
        for (j = 0; j < restart; ++j) {
            Vector w = A * V[j];

            // Orthogonalize
            for (size_t i = 0; i <= j; ++i) {
                H(i, j) = std::inner_product(w.begin(), w.end(), V[i].begin(), T(0));
                for (size_t k = 0; k < n; ++k) {
                    w[k] -= H(i, j) * V[i][k];
                }
            }

            H(j + 1, j) = std::sqrt(std::inner_product(w.begin(), w.end(), w.begin(), T(0)));

            if (H(j + 1, j) < tolerance) {
                j++;
                break;
            }

            for (size_t k = 0; k < n; ++k) {
                V[j + 1][k] = w[k] / H(j + 1, j);
            }

            // Apply Givens rotations to maintain QR of H
            for (size_t i = 0; i < j; ++i) {
                T temp = H(i, j);
                H(i, j) = H(i, j) * cos(i) + H(i + 1, j) * sin(i);
                H(i + 1, j) = -temp * sin(i) + H(i + 1, j) * cos(i);
            }

            // Compute new Givens rotation
            T h_jj = H(j, j);
            T h_jp1j = H(j + 1, j);
            T r = std::sqrt(h_jj * h_jj + h_jp1j * h_jp1j);
            T c = h_jj / r;
            T s = h_jp1j / r;

            H(j, j) = r;
            H(j + 1, j) = T(0);

            // Apply to g
            T temp = g[j];
            g[j] = c * g[j] + s * g[j + 1];
            g[j + 1] = -s * temp + c * g[j + 1];

            if (std::abs(g[j + 1]) < tolerance) {
                j++;
                break;
            }
        }

        // Solve upper triangular system
        std::vector<T> y(j);
        for (int i = j - 1; i >= 0; --i) {
            T sum = g[i];
            for (size_t k = i + 1; k < j; ++k) {
                sum -= H(i, k) * y[k];
            }
            y[i] = sum / H(i, i);
        }

        // Update solution
        for (size_t i = 0; i < j; ++i) {
            for (size_t k = 0; k < n; ++k) {
                x[k] += y[i] * V[i][k];
            }
        }
    }

    return x;
}

// =============================================================================
// FFT Helper Functions (placeholders)
// =============================================================================

// Forward declarations
template<typename T>
std::vector<std::complex<T>> fft_impl(const std::vector<std::complex<T>>& x) {
    // Placeholder - would implement Cooley-Tukey FFT
    return x;
}

template<typename T>
std::vector<std::complex<T>> ifft_impl(const std::vector<std::complex<T>>& x) {
    // Placeholder - would implement inverse FFT
    return x;
}

// =============================================================================
// Specialized Algorithms for Structured Matrices
// =============================================================================

/**
 * Fast multiplication for Toeplitz matrices using FFT
 * O(n log n) instead of O(n²)
 */
template<typename T>
    requires field<T>
std::vector<T> toeplitz_multiply(const std::vector<T>& c, const std::vector<T>& r,
                                 const std::vector<T>& x) {
    size_t n = x.size();

    // Embed Toeplitz matrix in circulant
    std::vector<std::complex<double>> circ(2 * n);
    circ[0] = std::complex<double>(c[0]);
    for (size_t i = 1; i < n; ++i) {
        circ[i] = std::complex<double>(c[i]);
        circ[2 * n - i] = std::complex<double>(r[i]);
    }

    // FFT of first column
    auto circ_fft = fft_impl(circ);

    // Pad x with zeros
    std::vector<std::complex<double>> x_padded(2 * n);
    for (size_t i = 0; i < n; ++i) {
        x_padded[i] = std::complex<double>(x[i]);
    }

    // FFT of x
    auto x_fft = fft_impl(x_padded);

    // Pointwise multiply
    std::vector<std::complex<double>> y_fft(2 * n);
    for (size_t i = 0; i < 2 * n; ++i) {
        y_fft[i] = circ_fft[i] * x_fft[i];
    }

    // Inverse FFT
    auto y_padded = ifft_impl(y_fft);

    // Extract result
    std::vector<T> y(n);
    for (size_t i = 0; i < n; ++i) {
        y[i] = T(y_padded[i].real());
    }

    return y;
}

/**
 * Fast solver for tridiagonal systems using Thomas algorithm
 * O(n) time and O(1) extra space
 */
template<typename T>
    requires field<T>
std::vector<T> tridiagonal_solve(const std::vector<T>& a,  // Lower diagonal
                                 const std::vector<T>& b,  // Main diagonal
                                 const std::vector<T>& c,  // Upper diagonal
                                 const std::vector<T>& d) { // RHS
    size_t n = d.size();
    std::vector<T> x(n);
    std::vector<T> c_prime(n);
    std::vector<T> d_prime(n);

    // Forward sweep
    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];

    for (size_t i = 1; i < n; ++i) {
        T m = b[i] - a[i - 1] * c_prime[i - 1];
        c_prime[i] = (i < n - 1) ? c[i] / m : T(0);
        d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / m;
    }

    // Back substitution
    x[n - 1] = d_prime[n - 1];
    for (int i = n - 2; i >= 0; --i) {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    return x;
}

/**
 * Fast multiplication for banded matrices
 * O(n * bandwidth) instead of O(n²)
 */
template<typename Matrix, typename Vector>
    requires field<typename Matrix::value_type>
Vector banded_multiply(const Matrix& A, const Vector& x, size_t lower_bandwidth,
                      size_t upper_bandwidth) {
    using T = typename Matrix::value_type;
    size_t n = A.rows();
    Vector y(n, T(0));

    for (size_t i = 0; i < n; ++i) {
        size_t j_min = (i > lower_bandwidth) ? i - lower_bandwidth : 0;
        size_t j_max = std::min(i + upper_bandwidth + 1, n);

        for (size_t j = j_min; j < j_max; ++j) {
            y[i] += A(i, j) * x[j];
        }
    }

    return y;
}

// =============================================================================
// Matrix Norms and Condition Numbers
// =============================================================================

/**
 * Compute various matrix norms with optimal algorithms
 */
template<typename Matrix>
class matrix_norms {
public:
    using T = typename Matrix::value_type;

    // Frobenius norm (already implemented in base matrix class)
    static T frobenius(const Matrix& A) {
        return A.frobenius_norm();
    }

    // 1-norm (maximum column sum)
    static T norm_1(const Matrix& A) {
        T max_sum = T(0);
        for (size_t j = 0; j < A.cols(); ++j) {
            T col_sum = T(0);
            for (size_t i = 0; i < A.rows(); ++i) {
                col_sum += std::abs(A(i, j));
            }
            max_sum = std::max(max_sum, col_sum);
        }
        return max_sum;
    }

    // Infinity norm (maximum row sum)
    static T norm_inf(const Matrix& A) {
        T max_sum = T(0);
        for (size_t i = 0; i < A.rows(); ++i) {
            T row_sum = T(0);
            for (size_t j = 0; j < A.cols(); ++j) {
                row_sum += std::abs(A(i, j));
            }
            max_sum = std::max(max_sum, row_sum);
        }
        return max_sum;
    }

    // 2-norm (spectral norm) - largest singular value
    static T norm_2(const Matrix& A) {
        if constexpr (matrix_traits<Matrix>::is_symmetric) {
            // For symmetric matrices, spectral norm is largest eigenvalue magnitude
            auto eigenvalues = A.qr_eigenvalues();
            T max_eigen = T(0);
            for (const auto& lambda : eigenvalues) {
                max_eigen = std::max(max_eigen, std::abs(lambda));
            }
            return max_eigen;
        } else {
            // Use power iteration on A^T A
            auto ATA = A.transpose() * A;
            auto [lambda, v] = ATA.power_iteration();
            return std::sqrt(lambda);
        }
    }

    // Nuclear norm (sum of singular values)
    static T nuclear(const Matrix& A) {
        auto [U, Sigma, V] = svd(A);
        T sum = T(0);
        for (size_t i = 0; i < Sigma.rows(); ++i) {
            sum += Sigma.diagonal(i);
        }
        return sum;
    }
};

/**
 * Condition number estimation
 * Measures sensitivity to perturbations
 */
template<typename Matrix>
    requires field<typename Matrix::value_type>
auto condition_number(const Matrix& A, const std::string& norm_type = "2") {
    using T = typename Matrix::value_type;

    T norm_A, norm_A_inv;

    if (norm_type == "1") {
        norm_A = matrix_norms<Matrix>::norm_1(A);
        norm_A_inv = matrix_norms<Matrix>::norm_1(inverse(A));
    } else if (norm_type == "inf") {
        norm_A = matrix_norms<Matrix>::norm_inf(A);
        norm_A_inv = matrix_norms<Matrix>::norm_inf(inverse(A));
    } else if (norm_type == "2") {
        // For 2-norm, use singular values
        auto [U, Sigma, V] = svd(A);
        T max_sv = T(0), min_sv = std::numeric_limits<T>::max();
        for (size_t i = 0; i < Sigma.rows(); ++i) {
            max_sv = std::max(max_sv, Sigma.diagonal(i));
            if (Sigma.diagonal(i) > T(0)) {
                min_sv = std::min(min_sv, Sigma.diagonal(i));
            }
        }
        return max_sv / min_sv;
    } else {
        norm_A = matrix_norms<Matrix>::frobenius(A);
        norm_A_inv = matrix_norms<Matrix>::frobenius(inverse(A));
    }

    return norm_A * norm_A_inv;
}

// =============================================================================
// Matrix Completion and Low-Rank Approximation
// =============================================================================

/**
 * Low-rank approximation using truncated SVD
 * Optimal in Frobenius and spectral norms (Eckart-Young theorem)
 */
template<typename Matrix>
    requires field<typename Matrix::value_type>
Matrix low_rank_approximation(const Matrix& A, size_t rank) {
    using T = typename Matrix::value_type;

    auto [U, Sigma, V] = svd(A);

    // Truncate to specified rank
    size_t k = std::min(rank, Sigma.rows());

    // Reconstruct: A_k = U_k Σ_k V_k^T
    Matrix A_approx(A.rows(), A.cols(), T(0));

    for (size_t r = 0; r < k; ++r) {
        T sigma = Sigma.diagonal(r);
        for (size_t i = 0; i < A.rows(); ++i) {
            for (size_t j = 0; j < A.cols(); ++j) {
                A_approx(i, j) += sigma * U(i, r) * V(j, r);
            }
        }
    }

    return A_approx;
}

/**
 * Nuclear norm minimization for matrix completion
 * Solves: min ||X||_* subject to X_ij = M_ij for (i,j) in Omega
 * Using soft-impute algorithm
 */
template<typename Matrix>
    requires field<typename Matrix::value_type>
Matrix matrix_completion(const Matrix& M_partial,
                        const std::vector<std::pair<size_t, size_t>>& observed,
                        typename Matrix::value_type lambda = 1.0,
                        size_t max_iter = 1000) {
    using T = typename Matrix::value_type;

    Matrix X = M_partial;  // Initialize with observed entries

    for (size_t iter = 0; iter < max_iter; ++iter) {
        // SVD of current iterate
        auto [U, Sigma, V] = svd(X);

        // Soft thresholding of singular values
        for (size_t i = 0; i < Sigma.rows(); ++i) {
            T sigma = Sigma.diagonal(i);
            Sigma.diagonal(i) = std::max(T(0), sigma - lambda);
        }

        // Reconstruct
        Matrix X_new(X.rows(), X.cols(), T(0));
        for (size_t r = 0; r < Sigma.rows(); ++r) {
            T sigma = Sigma.diagonal(r);
            if (sigma > T(0)) {
                for (size_t i = 0; i < X.rows(); ++i) {
                    for (size_t j = 0; j < X.cols(); ++j) {
                        X_new(i, j) += sigma * U(i, r) * V(j, r);
                    }
                }
            }
        }

        // Project back observed entries
        for (const auto& [i, j] : observed) {
            X_new(i, j) = M_partial(i, j);
        }

        // Check convergence
        if ((X_new - X).frobenius_norm() < T(1e-6)) {
            return X_new;
        }

        X = X_new;
    }

    return X;
}

// =============================================================================
// Utility Functions and Helpers
// =============================================================================

/**
 * Check if a matrix is sparse (heuristic based on non-zero ratio)
 */
template<typename Matrix>
bool is_sparse(const Matrix& A, double sparsity_threshold = 0.3) {
    size_t nnz = 0;
    size_t total = A.rows() * A.cols();

    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            if (A(i, j) != typename Matrix::value_type(0)) {
                nnz++;
            }
        }
    }

    return (static_cast<double>(nnz) / total) < sparsity_threshold;
}

/**
 * Padé approximation coefficients for exp and log
 */
template<typename Matrix>
Matrix pade_exp_approximation(const Matrix& X, size_t order) {
    using T = typename Matrix::value_type;

    // Compute Padé coefficients
    std::vector<T> p_coef(order + 1), q_coef(order + 1);
    T factorial = T(1);

    for (size_t k = 0; k <= order; ++k) {
        if (k > 0) factorial *= T(k);
        T binom = T(1);
        for (size_t j = 1; j <= k; ++j) {
            binom *= T(order - j + 1) / T(j);
        }
        p_coef[k] = binom / factorial;
        q_coef[k] = (k % 2 == 0) ? p_coef[k] : -p_coef[k];
    }

    // Evaluate polynomials using Horner's method
    Matrix P = p_coef[order] * Matrix::identity(X.rows());
    Matrix Q = q_coef[order] * Matrix::identity(X.rows());

    for (int k = order - 1; k >= 0; --k) {
        P = P * X + p_coef[k] * Matrix::identity(X.rows());
        Q = Q * X + q_coef[k] * Matrix::identity(X.rows());
    }

    return inverse(Q) * P;
}

template<typename Matrix>
Matrix pade_log_approximation(const Matrix& X, size_t order) {
    using T = typename Matrix::value_type;

    // For log(I + X) where ||X|| < 1
    // Use series: log(I + X) = X - X²/2 + X³/3 - ...

    Matrix result = Matrix::zeros(X.rows(), X.cols());
    Matrix X_power = X;

    for (size_t k = 1; k <= order; ++k) {
        T coef = (k % 2 == 1) ? T(1) / T(k) : -T(1) / T(k);
        result = result + coef * X_power;
        if (k < order) {
            X_power = X_power * X;
        }
    }

    return result;
}


// =============================================================================
// Matrix Calculus and Differentiation
// =============================================================================

/**
 * Fréchet derivative of matrix functions
 * Computes directional derivative of f(A) in direction E
 */
template<typename Matrix, typename Function>
    requires field<typename Matrix::value_type>
Matrix frechet_derivative(const Matrix& A, const Matrix& E, Function f,
                          typename Matrix::value_type h = 1e-8) {
    using T = typename Matrix::value_type;

    // Use finite differences: Df(A)[E] ≈ (f(A + hE) - f(A)) / h
    Matrix A_plus = A + h * E;
    Matrix f_A = f(A);
    Matrix f_A_plus = f(A_plus);

    return (f_A_plus - f_A) / h;
}

/**
 * Matrix derivative rules for common functions
 */
template<typename Matrix>
class matrix_derivatives {
public:
    using T = typename Matrix::value_type;

    // d/dX (X^{-1}) = -X^{-1} dX X^{-1}
    static Matrix inverse_derivative(const Matrix& X, const Matrix& dX) {
        Matrix X_inv = inverse(X);
        return -X_inv * dX * X_inv;
    }

    // d/dX det(X) = det(X) tr(X^{-1} dX)
    static T determinant_derivative(const Matrix& X, const Matrix& dX) {
        return X.determinant() * (inverse(X) * dX).trace();
    }

    // d/dX tr(X) = tr(dX)
    static T trace_derivative(const Matrix& X, const Matrix& dX) {
        return dX.trace();
    }

    // d/dX ||X||_F^2 = 2 tr(X^T dX)
    static T frobenius_squared_derivative(const Matrix& X, const Matrix& dX) {
        return T(2) * (X.transpose() * dX).trace();
    }
};

} // namespace stepanov::matrix_algorithms