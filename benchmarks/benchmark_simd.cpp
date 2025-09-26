// SIMD and OpenMP Optimized Benchmark
// Compares naive, optimized, and SIMD implementations

#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include <cstring>
#include <stepanov/matrix.hpp>
#include <stepanov/matrix_simd.hpp>
#include <stepanov/matrix_expressions.hpp>

using namespace std;
using namespace chrono;
using namespace stepanov;

// Naive matrix multiplication for comparison
void naive_multiply(const double* A, const double* B, double* C,
                    size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0;
            for (size_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Helper to time operations
template<typename F>
double time_ms(const string& name, F&& f, size_t iterations = 1) {
    // Warmup
    f();

    auto start = high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        f();
    }
    auto end = high_resolution_clock::now();

    double ms = duration_cast<microseconds>(end - start).count() / (1000.0 * iterations);
    cout << left << setw(30) << name << ": " << right << setw(10) << fixed
         << setprecision(3) << ms << " ms";
    return ms;
}

int main(int argc, char* argv[]) {
    cout << "=== SIMD AND OPENMP BENCHMARK ===\n\n";

    // Set number of threads for OpenMP
    int num_threads = omp_get_max_threads();
    cout << "OpenMP threads: " << num_threads << "\n";
    cout << "SIMD width: " << (sizeof(__m256d) / sizeof(double)) << " doubles\n\n";

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.1, 10.0);

    // Test different matrix sizes
    vector<size_t> sizes = {64, 128, 256, 512, 1024};

    for (size_t n : sizes) {
        cout << "=== Matrix Size: " << n << "x" << n << " ===\n";

        // Allocate aligned memory for matrices
        double* A = (double*)aligned_alloc(32, n * n * sizeof(double));
        double* B = (double*)aligned_alloc(32, n * n * sizeof(double));
        double* C = (double*)aligned_alloc(32, n * n * sizeof(double));

        // Initialize with random data
        for (size_t i = 0; i < n * n; ++i) {
            A[i] = dis(gen);
            B[i] = dis(gen);
        }

        // 1. Naive multiplication
        double naive_time = time_ms("Naive multiplication", [&]() {
            naive_multiply(A, B, C, n, n, n);
        }, 3);
        cout << "\n";

        // 2. Blocked multiplication (cache-friendly)
        double blocked_time = time_ms("Blocked multiplication", [&]() {
            matrix_multiply_blocked_simd(A, B, C, n, n, n);
        }, 3);
        cout << " (speedup: " << naive_time / blocked_time << "x)\n";

        // 3. AVX-optimized multiplication (for doubles)
        double avx_time = time_ms("AVX multiplication", [&]() {
            matrix_multiply_avx_double(A, B, C, n, n, n);
        }, 3);
        cout << " (speedup: " << naive_time / avx_time << "x)\n\n";

        // Verify correctness (spot check)
        double* C_verify = (double*)aligned_alloc(32, n * n * sizeof(double));
        naive_multiply(A, B, C_verify, n, n, n);

        double max_error = 0;
        for (size_t i = 0; i < min(size_t(100), n * n); ++i) {
            max_error = max(max_error, abs(C[i] - C_verify[i]));
        }
        cout << "Max error: " << scientific << max_error << fixed << "\n\n";

        free(A);
        free(B);
        free(C);
        free(C_verify);
    }

    // Special matrix operations benchmark
    cout << "=== SPECIAL MATRIX OPERATIONS ===\n\n";

    const size_t special_n = 1000;

    // 1. Diagonal matrix multiplication
    cout << "Diagonal Matrix Multiplication (" << special_n << "x" << special_n << "):\n";

    vector<double> diag(special_n);
    vector<double> M_data(special_n * special_n);
    vector<double> result(special_n * special_n);

    for (size_t i = 0; i < special_n; ++i) {
        diag[i] = dis(gen);
    }
    for (size_t i = 0; i < special_n * special_n; ++i) {
        M_data[i] = dis(gen);
    }

    // Naive diagonal multiplication
    double diag_naive_time = time_ms("Naive (treat as full)", [&]() {
        for (size_t i = 0; i < special_n; ++i) {
            for (size_t j = 0; j < special_n; ++j) {
                double sum = 0;
                for (size_t k = 0; k < special_n; ++k) {
                    double a_ik = (i == k) ? diag[i] : 0;
                    sum += a_ik * M_data[k * special_n + j];
                }
                result[i * special_n + j] = sum;
            }
        }
    }, 3);
    cout << "\n";

    // Optimized diagonal multiplication
    double diag_opt_time = time_ms("Optimized (O(n²))", [&]() {
        diagonal_multiply_simd(diag.data(), M_data.data(), result.data(),
                              special_n, special_n);
    }, 3);
    cout << " (speedup: " << diag_naive_time / diag_opt_time << "x)\n\n";

    // 2. Symmetric matrix operations
    cout << "Symmetric Matrix Operations (" << special_n << "x" << special_n << "):\n";

    // Memory comparison
    size_t full_memory = special_n * special_n * sizeof(double);
    size_t symmetric_memory = special_n * (special_n + 1) / 2 * sizeof(double);

    cout << "Full storage: " << full_memory / (1024.0 * 1024.0) << " MB\n";
    cout << "Symmetric storage: " << symmetric_memory / (1024.0 * 1024.0) << " MB\n";
    cout << "Memory savings: " << (1.0 - double(symmetric_memory) / full_memory) * 100 << "%\n\n";

    // 3. Expression templates
    cout << "Expression Templates (500x500):\n";

    const size_t expr_n = 500;
    matrix<double> A_expr(expr_n, expr_n);
    matrix<double> B_expr(expr_n, expr_n);
    matrix<double> C_expr(expr_n, expr_n);
    matrix<double> result_expr(expr_n, expr_n);

    // Initialize
    for (size_t i = 0; i < expr_n; ++i) {
        for (size_t j = 0; j < expr_n; ++j) {
            A_expr(i, j) = dis(gen);
            B_expr(i, j) = dis(gen);
            C_expr(i, j) = dis(gen);
        }
    }

    // Naive: Multiple passes
    double expr_naive_time = time_ms("Naive (4 passes)", [&]() {
        // Pass 1: 2.0 * A
        for (size_t i = 0; i < expr_n; ++i) {
            for (size_t j = 0; j < expr_n; ++j) {
                result_expr(i, j) = 2.0 * A_expr(i, j);
            }
        }
        // Pass 2: + B
        for (size_t i = 0; i < expr_n; ++i) {
            for (size_t j = 0; j < expr_n; ++j) {
                result_expr(i, j) += B_expr(i, j);
            }
        }
        // Pass 3: - 0.5 * C
        for (size_t i = 0; i < expr_n; ++i) {
            for (size_t j = 0; j < expr_n; ++j) {
                result_expr(i, j) -= 0.5 * C_expr(i, j);
            }
        }
    }, 10);
    cout << "\n";

    // SIMD: Single fused pass
    double expr_simd_time = time_ms("SIMD (fused)", [&]() {
        evaluate_expression_simd(
            A_expr.data(), B_expr.data(), C_expr.data(),
            2.0, 1.0, 0.5,
            result_expr.data(), expr_n * expr_n
        );
    }, 10);
    cout << " (speedup: " << expr_naive_time / expr_simd_time << "x)\n\n";

    cout << "=== SUMMARY ===\n\n";
    cout << "Key optimizations demonstrated:\n";
    cout << "• Cache blocking: Better memory locality\n";
    cout << "• SIMD vectorization: Process multiple elements per instruction\n";
    cout << "• OpenMP parallelization: Use all CPU cores\n";
    cout << "• Structure exploitation: O(n²) for diagonal instead of O(n³)\n";
    cout << "• Expression fusion: Single pass instead of multiple\n\n";

    cout << "With proper optimization, the Stepanov library can achieve\n";
    cout << "performance competitive with highly-tuned BLAS libraries!\n";

    return 0;
}