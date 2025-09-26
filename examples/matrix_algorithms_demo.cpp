#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>
#include <stepanov/matrix.hpp>
#include <stepanov/symmetric_matrix.hpp>
#include <stepanov/matrix_expressions.hpp>
#include <stepanov/matrix_algorithms.hpp>

using namespace stepanov;
using namespace stepanov::matrix_expr;
using namespace std;
using namespace std::chrono;

template<typename F>
double time_ms(const string& name, F&& f) {
    auto start = high_resolution_clock::now();
    auto result = f();
    auto end = high_resolution_clock::now();
    auto ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    cout << name << ": " << ms << " ms";
    return result;
}

int main() {
    cout << "=== Matrix Algorithms Demo ===" << endl;
    cout << "Demonstrating how algorithms exploit matrix structure\n" << endl;

    // 1. Diagonal Matrix Inverse - O(n) instead of O(n³)
    cout << "1. DIAGONAL MATRIX INVERSE" << endl;
    cout << "-------------------------" << endl;

    diagonal_matrix<double> D(100);
    for (size_t i = 0; i < 100; ++i) {
        D.diagonal(i) = 2.0 + i;
    }

    auto D_inv = time_ms("  Diagonal inverse (100x100)", [&]() {
        return matrix_algorithms<double>::inverse(D);
    });
    cout << " - O(n) complexity!" << endl;

    // Verify: D * D_inv = I
    auto should_be_identity = D * D_inv;
    double max_error = 0.0;
    for (size_t i = 0; i < 100; ++i) {
        for (size_t j = 0; j < 100; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            max_error = max(max_error, abs(should_be_identity(i, j) - expected));
        }
    }
    cout << "  Verification error: " << max_error << " (should be ~0)" << endl << endl;

    // 2. Triangular System Solve - O(n²) instead of O(n³)
    cout << "2. TRIANGULAR SYSTEM SOLVE" << endl;
    cout << "-------------------------" << endl;

    lower_triangular<double> L(50);
    vector<double> b(50);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.1, 2.0);

    // Fill lower triangular matrix and vector
    for (size_t i = 0; i < 50; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            L.at(i, j) = dis(gen);
        }
        b[i] = dis(gen);
    }

    auto x = time_ms("  Lower triangular solve (50x50)", [&]() {
        return matrix_algorithms<double>::solve(L, b);
    });
    cout << " - O(n²) forward substitution!" << endl;

    // Verify: L * x = b
    auto Lx = L * x;
    double solve_error = 0.0;
    for (size_t i = 0; i < 50; ++i) {
        solve_error += abs(Lx[i] - b[i]);
    }
    cout << "  Verification error: " << solve_error << " (should be ~0)" << endl << endl;

    // 3. Symmetric Matrix Eigenvalues - Specialized algorithm
    cout << "3. SYMMETRIC MATRIX EIGENVALUES" << endl;
    cout << "-------------------------------" << endl;

    symmetric_matrix<double> S(5);
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = i; j < 5; ++j) {
            S(i, j) = dis(gen);
        }
    }

    cout << "  Symmetric matrix (5x5):" << endl;
    auto [eigenvalues, eigenvectors] = matrix_algorithms<double>::eigen_symmetric(S);

    cout << "  Eigenvalues: ";
    for (auto& lambda : eigenvalues) {
        cout << fixed << setprecision(3) << lambda << " ";
    }
    cout << endl << endl;

    // 4. Property Preservation in Operations
    cout << "4. PROPERTY PRESERVATION" << endl;
    cout << "------------------------" << endl;

    diagonal_matrix<double> D1(3);
    diagonal_matrix<double> D2(3);
    D1.diagonal(0) = 1; D1.diagonal(1) = 2; D1.diagonal(2) = 3;
    D2.diagonal(0) = 4; D2.diagonal(1) = 5; D2.diagonal(2) = 6;

    // Diagonal × Diagonal = Diagonal (tracked at compile time!)
    auto D3 = D1 * D2;  // Result type knows it's diagonal

    cout << "  D1 × D2 (both diagonal):" << endl;
    cout << "  Result is also diagonal: ";
    for (size_t i = 0; i < 3; ++i) {
        cout << D3(i, i) << " ";
    }
    cout << "(only diagonal stored)" << endl << endl;

    // 5. Matrix Decompositions with Structure Exploitation
    cout << "5. MATRIX DECOMPOSITIONS" << endl;
    cout << "-----------------------" << endl;

    // Cholesky for symmetric positive definite
    symmetric_matrix<double> SPD(4);
    // Create positive definite matrix: A^T * A + I
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = i; j < 4; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < 4; ++k) {
                sum += dis(gen) * dis(gen);
            }
            SPD(i, j) = sum;
        }
        SPD(i, i) += 2.0;  // Ensure positive definite
    }

    auto L_chol = time_ms("  Cholesky decomposition (symmetric)", [&]() {
        return matrix_algorithms<double>::cholesky(SPD);
    });
    cout << " - Exploits symmetry!" << endl;

    // LU for general matrix
    matrix<double> A(4, 4);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            A(i, j) = dis(gen);
        }
    }

    auto [L_lu, U_lu, P_lu] = time_ms("  LU decomposition (general)", [&]() {
        return matrix_algorithms<double>::lu_decompose(A);
    });
    cout << " - Standard algorithm" << endl << endl;

    // 6. Iterative Solvers for Large Sparse Systems
    cout << "6. ITERATIVE SOLVERS" << endl;
    cout << "-------------------" << endl;

    // Create a large sparse symmetric positive definite system
    const size_t n = 100;
    symmetric_matrix<double> A_sparse(n);
    vector<double> b_sparse(n);

    // Tridiagonal system (common in PDEs)
    for (size_t i = 0; i < n; ++i) {
        A_sparse(i, i) = 4.0;  // Diagonal
        if (i > 0) A_sparse(i-1, i) = -1.0;  // Off-diagonal
        b_sparse[i] = dis(gen);
    }

    vector<double> x0(n, 0.0);  // Initial guess
    auto x_cg = time_ms("  Conjugate Gradient (100x100 sparse)", [&]() {
        return matrix_algorithms<double>::conjugate_gradient(A_sparse, b_sparse, x0, 1e-10, 100);
    });
    cout << " - Efficient for sparse!" << endl;

    // Verify convergence
    auto Ax = A_sparse * x_cg;
    double cg_error = 0.0;
    for (size_t i = 0; i < n; ++i) {
        cg_error += abs(Ax[i] - b_sparse[i]);
    }
    cout << "  Residual: " << cg_error << endl << endl;

    // 7. Matrix Functions
    cout << "7. MATRIX FUNCTIONS" << endl;
    cout << "------------------" << endl;

    diagonal_matrix<double> D_small(3);
    D_small.diagonal(0) = 1.0;
    D_small.diagonal(1) = 2.0;
    D_small.diagonal(2) = 3.0;

    auto D_exp = matrix_algorithms<double>::matrix_exp(D_small);
    cout << "  exp(diagonal) = diagonal with exp(d_ii):" << endl;
    cout << "  ";
    for (size_t i = 0; i < 3; ++i) {
        cout << "e^" << D_small(i,i) << "=" << fixed << setprecision(3) << D_exp(i,i) << " ";
    }
    cout << endl << endl;

    cout << "=== SUMMARY ===" << endl;
    cout << "The algorithms automatically exploit matrix structure:" << endl;
    cout << "  • Diagonal operations: O(n) instead of O(n³)" << endl;
    cout << "  • Triangular systems: O(n²) instead of O(n³)" << endl;
    cout << "  • Symmetric matrices: Half the storage and computation" << endl;
    cout << "  • Sparse matrices: Only process non-zero elements" << endl;
    cout << "  • Properties tracked at compile-time for zero overhead" << endl;
    cout << "\nThis is generic programming at its finest!" << endl;

    return 0;
}