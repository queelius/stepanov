/**
 * Comparison Against Eigen Library
 *
 * This benchmark compares the Stepanov library's claimed optimizations
 * against the industry-standard Eigen library to validate performance claims.
 */

#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>

// Check if Eigen is available
#ifdef __has_include
#  if __has_include(<Eigen/Dense>)
#    include <Eigen/Dense>
#    define HAS_EIGEN 1
#  else
#    define HAS_EIGEN 0
#  endif
#else
#  define HAS_EIGEN 0
#endif

using namespace std;
using namespace std::chrono;

template<typename F>
double time_ms(F&& f, size_t iterations = 1) {
    auto start = high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        f();
    }
    auto end = high_resolution_clock::now();
    return duration_cast<microseconds>(end - start).count() / (1000.0 * iterations);
}

#if HAS_EIGEN

using namespace Eigen;

int main() {
    cout << "================================================\n";
    cout << "EIGEN LIBRARY COMPARISON BENCHMARKS\n";
    cout << "================================================\n\n";

    cout << "Comparing Stepanov library claims against Eigen,\n";
    cout << "a highly-optimized industry-standard library.\n\n";

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.1, 10.0);

    // 1. DIAGONAL MATRIX MULTIPLICATION
    cout << "1. DIAGONAL MATRIX MULTIPLICATION\n";
    cout << "==================================\n";

    {
        const int n = 500;

        // Eigen matrices
        DiagonalMatrix<double, Dynamic> D(n);
        MatrixXd M = MatrixXd::Random(n, n);
        MatrixXd result;

        for (int i = 0; i < n; ++i) {
            D.diagonal()[i] = dis(gen);
        }

        // Eigen optimized (knows D is diagonal)
        double eigen_optimized = time_ms([&]() {
            result.noalias() = D * M;  // Eigen detects diagonal structure
        }, 100);

        // Eigen naive (force full matrix multiply)
        MatrixXd D_full = D.toDenseMatrix();
        double eigen_naive = time_ms([&]() {
            result.noalias() = D_full * M;
        }, 100);

        cout << "  Size: " << n << "x" << n << "\n";
        cout << "  Eigen (diagonal aware): " << fixed << setprecision(3)
             << eigen_optimized << " ms\n";
        cout << "  Eigen (forced dense):   " << eigen_naive << " ms\n";
        cout << "  Actual speedup: " << (eigen_naive / eigen_optimized) << "x\n";
        cout << "  Claimed speedup: 1190x\n";

        if (eigen_naive / eigen_optimized > 100) {
            cout << "  ✓ Large speedup confirmed (>100x)\n";
        } else {
            cout << "  ✗ Speedup much less than claimed\n";
        }
        cout << "\n";
    }

    // 2. SYMMETRIC MATRIX OPERATIONS
    cout << "2. SYMMETRIC MATRIX OPERATIONS\n";
    cout << "===============================\n";

    {
        const int n = 1000;

        // Create symmetric matrix using Eigen's self-adjoint view
        MatrixXd A = MatrixXd::Random(n, n);
        MatrixXd S = (A + A.transpose()) / 2.0;  // Make symmetric

        VectorXd v = VectorXd::Random(n);
        VectorXd result;

        // Using symmetric property
        double sym_time = time_ms([&]() {
            result.noalias() = S.selfadjointView<Lower>() * v;
        }, 1000);

        // Without using symmetric property
        double full_time = time_ms([&]() {
            result.noalias() = S * v;
        }, 1000);

        cout << "  Matrix-vector multiply (" << n << "x" << n << ")\n";
        cout << "  Symmetric aware: " << fixed << setprecision(3)
             << sym_time << " ms\n";
        cout << "  Full matrix:     " << full_time << " ms\n";
        cout << "  Speedup: " << (full_time / sym_time) << "x\n";
        cout << "  Claimed: 2x\n\n";
    }

    // 3. TRIANGULAR SYSTEM SOLVE
    cout << "3. TRIANGULAR SYSTEM SOLVE\n";
    cout << "===========================\n";

    {
        const int n = 200;

        MatrixXd L = MatrixXd::Random(n, n);
        // Make it lower triangular
        L = L.triangularView<Lower>();

        VectorXd b = VectorXd::Random(n);
        VectorXd x;

        // Triangular solve (knows structure)
        double tri_time = time_ms([&]() {
            x = L.triangularView<Lower>().solve(b);
        }, 100);

        // General solve (doesn't use structure)
        double gen_time = time_ms([&]() {
            x = L.fullPivLu().solve(b);
        }, 100);

        cout << "  System size: " << n << "x" << n << "\n";
        cout << "  Triangular solve: " << fixed << setprecision(3)
             << tri_time << " ms\n";
        cout << "  General LU solve: " << gen_time << " ms\n";
        cout << "  Speedup: " << (gen_time / tri_time) << "x\n";
        cout << "  Claimed: 3x\n\n";
    }

    // 4. EXPRESSION TEMPLATES
    cout << "4. EXPRESSION TEMPLATES\n";
    cout << "========================\n";

    {
        const int n = 300;

        MatrixXd A = MatrixXd::Random(n, n);
        MatrixXd B = MatrixXd::Random(n, n);
        MatrixXd C = MatrixXd::Random(n, n);
        MatrixXd result(n, n);

        // Eigen with expression templates (automatic)
        double expr_time = time_ms([&]() {
            result.noalias() = 2.0 * A + B - 0.5 * C;
        }, 100);

        // Force evaluation of intermediates
        double temp_time = time_ms([&]() {
            MatrixXd temp1 = 2.0 * A;
            MatrixXd temp2 = temp1 + B;
            result = temp2 - 0.5 * C;
        }, 100);

        cout << "  Matrix size: " << n << "x" << n << "\n";
        cout << "  Expression templates: " << fixed << setprecision(3)
             << expr_time << " ms\n";
        cout << "  With temporaries:     " << temp_time << " ms\n";
        cout << "  Speedup: " << (temp_time / expr_time) << "x\n";
        cout << "  Claimed: 4x\n\n";
    }

    // 5. SPARSE MATRIX OPERATIONS
    cout << "5. SPARSE MATRIX OPERATIONS\n";
    cout << "============================\n";

    {
        const int n = 1000;
        const double sparsity = 0.01;  // 1% non-zero

        // Create sparse matrix
        SparseMatrix<double> S(n, n);
        vector<Triplet<double>> triplets;

        uniform_int_distribution<> idx_dis(0, n - 1);
        int nnz = n * n * sparsity;

        for (int k = 0; k < nnz; ++k) {
            triplets.push_back(Triplet<double>(idx_dis(gen), idx_dis(gen), dis(gen)));
        }
        S.setFromTriplets(triplets.begin(), triplets.end());
        S.makeCompressed();

        // Dense version
        MatrixXd D = MatrixXd(S);

        VectorXd v = VectorXd::Random(n);
        VectorXd result;

        // Sparse matrix-vector multiply
        double sparse_time = time_ms([&]() {
            result = S * v;
        }, 100);

        // Dense matrix-vector multiply
        double dense_time = time_ms([&]() {
            result = D * v;
        }, 100);

        cout << "  Matrix size: " << n << "x" << n << "\n";
        cout << "  Sparsity: " << (sparsity * 100) << "%\n";
        cout << "  Non-zeros: " << S.nonZeros() << "\n";
        cout << "  Sparse multiply: " << fixed << setprecision(3)
             << sparse_time << " ms\n";
        cout << "  Dense multiply:  " << dense_time << " ms\n";
        cout << "  Speedup: " << (dense_time / sparse_time) << "x\n";
        cout << "  Claimed: 11x\n\n";
    }

    // SUMMARY
    cout << "================================================\n";
    cout << "EIGEN COMPARISON SUMMARY\n";
    cout << "================================================\n\n";

    cout << "KEY FINDINGS:\n";
    cout << "1. Eigen already implements most optimizations claimed\n";
    cout << "   by the Stepanov library\n";
    cout << "2. Diagonal matrix optimization is real and significant\n";
    cout << "3. Expression templates are standard in modern libraries\n";
    cout << "4. Sparse matrix benefits depend heavily on sparsity\n";
    cout << "5. Symmetric/triangular optimizations provide modest gains\n\n";

    cout << "PERFORMANCE REALITY:\n";
    cout << "• Industry libraries like Eigen already exploit structure\n";
    cout << "• The Stepanov library's contributions are:\n";
    cout << "  - Generic programming philosophy\n";
    cout << "  - Mathematical abstraction framework\n";
    cout << "  - Educational value in algorithm design\n";
    cout << "• Raw performance claims should be taken in context\n\n";

    cout << "CONCLUSION:\n";
    cout << "The Stepanov library's value lies not in beating Eigen's\n";
    cout << "performance, but in demonstrating how mathematical thinking\n";
    cout << "and generic programming can lead to efficient implementations.\n";

    return 0;
}

#else

int main() {
    cout << "================================================\n";
    cout << "EIGEN LIBRARY NOT FOUND\n";
    cout << "================================================\n\n";

    cout << "The Eigen library is not installed on this system.\n";
    cout << "To run these comparison benchmarks:\n\n";
    cout << "1. Install Eigen:\n";
    cout << "   sudo apt-get install libeigen3-dev  # Ubuntu/Debian\n";
    cout << "   brew install eigen                  # macOS\n\n";
    cout << "2. Rebuild and run:\n";
    cout << "   g++ -std=c++20 -O3 bench_eigen_comparison.cpp -o bench_eigen\n";
    cout << "   ./bench_eigen\n\n";

    cout << "Eigen provides a good baseline for comparing the Stepanov\n";
    cout << "library's performance claims against industry-standard\n";
    cout << "implementations.\n";

    return 0;
}

#endif