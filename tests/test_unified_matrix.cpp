/**
 * Test Suite for Unified Matrix Library
 * Verifies correctness and performance of the refactored implementation
 */

#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>
#include <cassert>
#include <stepanov/matrix_unified.hpp>
#include <stepanov/matrix_specialized.hpp>

using namespace std;
using namespace chrono;
using namespace stepanov;

// Timing utility
template<typename F>
double benchmark_ms(F&& f, size_t iterations = 10) {
    // Warmup
    f();

    auto start = high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        f();
    }
    auto end = high_resolution_clock::now();

    return duration_cast<microseconds>(end - start).count() / (1000.0 * iterations);
}

// Test correctness of basic operations
void test_basic_operations() {
    cout << "Testing basic matrix operations...\n";

    // Create test matrices
    dense_matrix<double> A(3, 3);
    dense_matrix<double> B(3, 3);

    // Initialize with known values
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            A(i, j) = i * 3 + j + 1;  // 1, 2, 3, 4, 5, 6, 7, 8, 9
            B(i, j) = (i == j) ? 1.0 : 0.0;  // Identity
        }
    }

    // Test multiplication with identity
    auto C = A * B;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            assert(abs(C(i, j) - A(i, j)) < 1e-10);
        }
    }

    // Test addition
    auto D = A + A;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            assert(abs(D(i, j) - 2 * A(i, j)) < 1e-10);
        }
    }

    // Test scalar multiplication
    auto E = A * 2.0;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            assert(abs(E(i, j) - 2 * A(i, j)) < 1e-10);
        }
    }

    cout << "  ✓ Basic operations correct\n";
}

// Test structure detection
void test_structure_detection() {
    cout << "Testing structure detection...\n";

    // Create diagonal matrix
    dense_matrix<double> D(100, 100);
    D.zero();
    for (size_t i = 0; i < 100; ++i) {
        D(i, i) = i + 1.0;
    }

    auto analysis = structure_analyzer<double>::analyze(D);
    assert(analysis.is_diagonal);
    assert(analysis.sparsity > 0.98);

    // Create sparse matrix
    dense_matrix<double> S(100, 100);
    S.zero();
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);

    for (size_t i = 0; i < 100; ++i) {
        for (size_t j = 0; j < 100; ++j) {
            if (dis(gen) < 0.05) {  // 5% density
                S(i, j) = dis(gen);
            }
        }
    }

    auto sparse_analysis = structure_analyzer<double>::analyze(S);
    assert(sparse_analysis.is_sparse);

    cout << "  ✓ Structure detection working\n";
}

// Benchmark diagonal matrix optimization
void benchmark_diagonal_performance() {
    cout << "\nBenchmarking diagonal matrix performance...\n";

    vector<size_t> sizes = {100, 500, 1000};

    cout << setw(10) << "Size"
         << setw(15) << "Dense (ms)"
         << setw(15) << "Diagonal (ms)"
         << setw(12) << "Speedup"
         << setw(15) << "Memory Saved\n";
    cout << string(67, '-') << "\n";

    for (size_t n : sizes) {
        // Create diagonal matrix
        dense_matrix<double> D_dense(n, n);
        D_dense.zero();
        for (size_t i = 0; i < n; ++i) {
            D_dense(i, i) = i + 1.0;
        }

        // Create general matrix for multiplication
        dense_matrix<double> M(n, n);
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(1.0, 10.0);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                M(i, j) = dis(gen);
            }
        }

        // Benchmark dense multiplication
        double dense_time = benchmark_ms([&]() {
            auto result = D_dense * M;
        });

        // Create optimized diagonal storage
        diagonal_storage<double> D_opt(n);
        for (size_t i = 0; i < n; ++i) {
            D_opt.diagonal()[i] = i + 1.0;
        }

        // Benchmark optimized multiplication
        double opt_time = benchmark_ms([&]() {
            auto result = diagonal_matrix_ops<double>::multiply_matrix(D_opt, M);
        });

        double speedup = dense_time / opt_time;
        size_t memory_saved = (n * n - n) * sizeof(double);

        cout << setw(10) << n
             << setw(15) << dense_time
             << setw(15) << opt_time
             << setw(12) << speedup << "x"
             << setw(15) << (memory_saved / 1024) << " KB\n";
    }
}

// Benchmark sparse matrix optimization
void benchmark_sparse_performance() {
    cout << "\nBenchmarking sparse matrix performance...\n";

    vector<size_t> sizes = {500, 1000};
    vector<double> sparsities = {0.95, 0.99};

    cout << setw(10) << "Size"
         << setw(12) << "Sparsity"
         << setw(15) << "Dense (ms)"
         << setw(15) << "Sparse (ms)"
         << setw(12) << "Speedup\n";
    cout << string(64, '-') << "\n";

    for (size_t n : sizes) {
        for (double sparsity : sparsities) {
            // Create sparse matrix
            dense_matrix<double> M_dense(n, n);
            csr_storage<double> M_sparse(n, n);

            random_device rd;
            mt19937 gen(rd());
            uniform_real_distribution<> dis(1.0, 10.0);
            uniform_real_distribution<> sparse_dis(0.0, 1.0);

            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    if (sparse_dis(gen) > sparsity) {
                        double val = dis(gen);
                        M_dense(i, j) = val;
                        M_sparse.add_entry(i, j, val);
                    } else {
                        M_dense(i, j) = 0.0;
                    }
                }
            }
            M_sparse.finalize();

            // Create vector for multiplication
            vector<double> x(n);
            for (size_t i = 0; i < n; ++i) {
                x[i] = dis(gen);
            }

            // Benchmark dense multiplication
            double dense_time = benchmark_ms([&]() {
                auto y = M_dense * x;
            });

            // Benchmark sparse multiplication
            double sparse_time = benchmark_ms([&]() {
                auto y = M_sparse.multiply_vector(x);
            });

            double speedup = dense_time / sparse_time;

            cout << setw(10) << n
                 << setw(12) << sparsity
                 << setw(15) << dense_time
                 << setw(15) << sparse_time
                 << setw(12) << speedup << "x\n";
        }
    }
}

// Test execution policy selection
void test_execution_policies() {
    cout << "\nTesting execution policies...\n";

    // Small matrix - should use sequential
    dense_matrix<double> small(50, 50);
    small.fill(1.0);

    auto policy_small = operation_traits<double>::select_policy(50);
    assert(policy_small == exec_policy::sequential);

    // Medium matrix - should use vectorized
    dense_matrix<double> medium(200, 200);
    medium.fill(1.0);

    auto policy_medium = operation_traits<double>::select_policy(200);
    assert(policy_medium == exec_policy::vectorized);

    // Large matrix - should use parallel
    dense_matrix<double> large(1000, 1000);
    large.fill(1.0);

    auto policy_large = operation_traits<double>::select_policy(1000);
    assert(policy_large == exec_policy::parallel);

    cout << "  ✓ Execution policy selection correct\n";
}

// Test memory efficiency
void test_memory_efficiency() {
    cout << "\nTesting memory efficiency...\n";

    size_t n = 1000;

    // Dense matrix memory
    dense_matrix<double> dense(n, n);
    size_t dense_memory = dense.memory_usage();

    // Diagonal matrix memory
    diagonal_storage<double> diag(n);
    size_t diag_memory = diag.memory_usage();

    // Symmetric matrix memory
    symmetric_storage<double> sym(n);
    size_t sym_memory = sym.memory_usage();

    // Sparse matrix memory (5% density)
    csr_storage<double> sparse(n, n);
    size_t nnz = n * n * 0.05;
    for (size_t k = 0; k < nnz; ++k) {
        sparse.add_entry(k / n, k % n, 1.0);
    }
    sparse.finalize();
    size_t sparse_memory = sparse.memory_usage();

    cout << "Matrix size: " << n << "×" << n << "\n";
    cout << "Dense memory:     " << dense_memory / (1024 * 1024) << " MB\n";
    cout << "Diagonal memory:  " << diag_memory / 1024 << " KB (saved "
         << (dense_memory - diag_memory) / (1024 * 1024) << " MB)\n";
    cout << "Symmetric memory: " << sym_memory / (1024 * 1024) << " MB (saved "
         << (dense_memory - sym_memory) / (1024 * 1024) << " MB)\n";
    cout << "Sparse memory:    " << sparse_memory / 1024 << " KB (saved "
         << (dense_memory - sparse_memory) / (1024 * 1024) << " MB)\n";
}

// Main test runner
int main() {
    cout << "=================================================\n";
    cout << "   Unified Matrix Library Test Suite\n";
    cout << "=================================================\n\n";

    try {
        // Run correctness tests
        test_basic_operations();
        test_structure_detection();
        test_execution_policies();

        // Run performance benchmarks
        benchmark_diagonal_performance();
        benchmark_sparse_performance();

        // Test memory efficiency
        test_memory_efficiency();

        cout << "\n=================================================\n";
        cout << "   All tests passed successfully! ✓\n";
        cout << "=================================================\n";

    } catch (const exception& e) {
        cerr << "Test failed with error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}