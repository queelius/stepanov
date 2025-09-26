#include <iostream>
#include <iomanip>
#include <chrono>
#include <complex>
#include "../include/stepanov/matrix.hpp"

using namespace stepanov;
using namespace std;
using namespace chrono;

// Helper function to print a matrix
template<typename T>
void print_matrix(const matrix<T>& m, const string& name) {
    cout << name << " (" << m.rows() << "x" << m.cols() << "):\n";
    for (size_t i = 0; i < m.rows(); ++i) {
        cout << "  [";
        for (size_t j = 0; j < m.cols(); ++j) {
            cout << setw(8) << fixed << setprecision(3) << m(i, j);
            if (j < m.cols() - 1) cout << ", ";
        }
        cout << "]\n";
    }
    cout << "\n";
}

int main() {
    cout << "========================================\n";
    cout << "Stepanov Generic Matrix Library Demo\n";
    cout << "========================================\n\n";

    // 1. Basic matrix operations
    cout << "1. Basic Matrix Operations\n";
    cout << "--------------------------\n";

    matrix<double> A = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    matrix<double> B = {
        {9, 8, 7},
        {6, 5, 4},
        {3, 2, 1}
    };

    print_matrix(A, "Matrix A");
    print_matrix(B, "Matrix B");

    auto C = A + B;
    print_matrix(C, "A + B");

    auto D = A * B;
    print_matrix(D, "A * B");

    auto At = A.transpose();
    print_matrix(At, "A^T");

    cout << "Trace of A: " << A.trace() << "\n\n";

    // 2. Special matrices
    cout << "2. Special Matrices\n";
    cout << "-------------------\n";

    auto I = matrix<double>::identity(4);
    print_matrix(I, "4x4 Identity Matrix");

    vector<double> diag_vals = {1, 2, 3, 4};
    auto D_diag = matrix<double>::diagonal(diag_vals);
    print_matrix(D_diag, "Diagonal Matrix");

    // 3. Matrix decompositions
    cout << "3. Matrix Decompositions\n";
    cout << "------------------------\n";

    matrix<double> M = {
        {4, 3, 2},
        {3, 5, 1},
        {2, 1, 6}
    };

    print_matrix(M, "Matrix M (symmetric positive definite)");

    // LU Decomposition
    auto [L, U, pivot] = M.lu_decomposition();
    print_matrix(L, "L (from LU decomposition)");
    print_matrix(U, "U (from LU decomposition)");

    // QR Decomposition
    auto [Q, R] = M.qr_decomposition();
    print_matrix(Q, "Q (from QR decomposition)");
    print_matrix(R, "R (from QR decomposition)");

    // Verify QR = M
    auto QR = Q * R;
    print_matrix(QR, "Q * R (should equal M)");

    // 4. Linear system solving
    cout << "4. Linear System Solving\n";
    cout << "------------------------\n";

    matrix<double> A_sys = {
        {2, -1, 0},
        {-1, 2, -1},
        {0, -1, 2}
    };

    matrix<double> b = {
        {1},
        {0},
        {1}
    };

    print_matrix(A_sys, "System matrix A");
    print_matrix(b, "Right-hand side b");

    auto x = A_sys.solve_gaussian(b);
    print_matrix(x, "Solution x (from Gaussian elimination)");

    auto Ax = A_sys * x;
    print_matrix(Ax, "A * x (verification)");

    // 5. Eigenvalues
    cout << "5. Eigenvalues\n";
    cout << "--------------\n";

    matrix<double> E = {
        {4, -2, 1},
        {-2, 5, -1},
        {1, -1, 3}
    };

    print_matrix(E, "Matrix E");

    auto [lambda, v] = E.power_iteration(1000);
    cout << "Dominant eigenvalue: " << lambda << "\n";
    cout << "Corresponding eigenvector: [";
    for (size_t i = 0; i < v.size(); ++i) {
        cout << v[i];
        if (i < v.size() - 1) cout << ", ";
    }
    cout << "]\n\n";

    // 6. Matrix functions
    cout << "6. Matrix Functions\n";
    cout << "-------------------\n";

    matrix<double> F = {
        {0, -1},
        {1, 0}
    };

    print_matrix(F, "Matrix F (rotation by π/2)");

    auto expF = F.exp(20);
    print_matrix(expF, "exp(F)");

    cout << "Note: exp(F) represents rotation by 1 radian\n\n";

    // 7. Performance comparison
    cout << "7. Performance Comparison\n";
    cout << "-------------------------\n";

    size_t n = 256;
    auto rand_matrix = []() -> double { return (rand() % 100) / 10.0; };

    auto X = matrix<double>::random(n, n, rand_matrix);
    auto Y = matrix<double>::random(n, n, rand_matrix);

    cout << "Multiplying two " << n << "x" << n << " matrices:\n";

    // Standard multiplication
    auto start = high_resolution_clock::now();
    auto Z1 = X.multiply_standard(Y);
    auto end = high_resolution_clock::now();
    auto standard_time = duration_cast<milliseconds>(end - start).count();
    cout << "  Standard algorithm: " << standard_time << " ms\n";

    // Strassen multiplication
    start = high_resolution_clock::now();
    auto Z2 = X.multiply_strassen(Y);
    end = high_resolution_clock::now();
    auto strassen_time = duration_cast<milliseconds>(end - start).count();
    cout << "  Strassen algorithm: " << strassen_time << " ms\n";

    // Cache-oblivious multiplication
    start = high_resolution_clock::now();
    auto Z3 = X.multiply_cache_oblivious(Y);
    end = high_resolution_clock::now();
    auto cache_time = duration_cast<milliseconds>(end - start).count();
    cout << "  Cache-oblivious algorithm: " << cache_time << " ms\n";

    // 8. Complex matrices
    cout << "\n8. Complex Matrices\n";
    cout << "-------------------\n";

    using Complex = complex<double>;

    // Pauli matrices
    matrix<Complex> sigma_x = {
        {Complex(0, 0), Complex(1, 0)},
        {Complex(1, 0), Complex(0, 0)}
    };

    matrix<Complex> sigma_y = {
        {Complex(0, 0), Complex(0, -1)},
        {Complex(0, 1), Complex(0, 0)}
    };

    matrix<Complex> sigma_z = {
        {Complex(1, 0), Complex(0, 0)},
        {Complex(0, 0), Complex(-1, 0)}
    };

    cout << "Pauli matrices:\n";
    print_matrix(sigma_x, "σ_x");
    print_matrix(sigma_y, "σ_y");
    print_matrix(sigma_z, "σ_z");

    // Verify anticommutation relation: {σ_x, σ_y} = 0
    auto anticomm = sigma_x * sigma_y + sigma_y * sigma_x;
    print_matrix(anticomm, "{σ_x, σ_y} (should be zero)");

    // 9. Kronecker and Hadamard products
    cout << "9. Special Products\n";
    cout << "-------------------\n";

    matrix<double> P = {{1, 2}, {3, 4}};
    matrix<double> Q_small = {{5, 6}, {7, 8}};

    print_matrix(P, "Matrix P");
    print_matrix(Q_small, "Matrix Q");

    auto H = P.hadamard(Q_small);
    print_matrix(H, "P ⊙ Q (Hadamard product)");

    auto K = P.kronecker(Q_small);
    print_matrix(K, "P ⊗ Q (Kronecker product)");

    cout << "========================================\n";
    cout << "Demo Complete!\n";
    cout << "========================================\n";

    return 0;
}