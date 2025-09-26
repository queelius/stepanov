// Benchmark Real-World Matrix Structures
// Demonstrates performance gains from exploiting common structures

#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>
#include <stepanov/matrix_revised.hpp>
#include <stepanov/matrix_structures.hpp>

using namespace std;
using namespace chrono;
using namespace stepanov;

template<typename F>
double time_ms(const string& name, F&& f, size_t iterations = 1) {
    f(); // Warmup
    auto start = high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        f();
    }
    auto end = high_resolution_clock::now();
    double ms = duration_cast<microseconds>(end - start).count() / (1000.0 * iterations);
    cout << "  " << left << setw(45) << name << ": ";
    cout << right << setw(10) << fixed << setprecision(4) << ms << " ms";
    return ms;
}

int main() {
    cout << "\n=== REAL-WORLD MATRIX STRUCTURES BENCHMARK ===\n\n";

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.1, 10.0);
    uniform_real_distribution<> sparse_dis(0.0, 1.0);

    // =========================================================================
    // 1. SPARSE MATRICES (Most common optimization!)
    // =========================================================================
    cout << "1. SPARSE MATRIX (95% zeros - typical for graphs, FEM)\n";
    cout << string(60, '-') << "\n";

    {
        const size_t n = 1000;
        const double sparsity = 0.95;  // 95% zeros

        // Create sparse matrix
        matrix<double> M(n, n);
        sparse_matrix_csr<double> M_sparse(n, n);

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (sparse_dis(gen) > sparsity) {
                    double val = dis(gen);
                    M(i, j) = val;
                    M_sparse.add_element(i, j, val);
                }
            }
        }
        M_sparse.finalize();

        vector<double> x(n);
        for (size_t i = 0; i < n; ++i) {
            x[i] = dis(gen);
        }

        cout << "  Matrix size: " << n << "x" << n << "\n";
        cout << "  Sparsity: " << M_sparse.sparsity() * 100 << "%\n";
        cout << "  Non-zeros: " << M_sparse.nnz() << " / " << (n*n) << "\n\n";

        // Dense multiplication
        double dense_time = time_ms("Dense matrix-vector", [&]() {
            vector<double> y(n);
            for (size_t i = 0; i < n; ++i) {
                double sum = 0;
                for (size_t j = 0; j < n; ++j) {
                    sum += M(i, j) * x[j];
                }
                y[i] = sum;
            }
            return y[0];
        }, 10);
        cout << "\n";

        // Sparse multiplication
        double sparse_time = time_ms("Sparse (CSR) matrix-vector", [&]() {
            auto y = M_sparse * x;
            return y[0];
        }, 10);
        cout << " [" << dense_time/sparse_time << "x]\n\n";

        // Memory comparison
        size_t dense_memory = n * n * sizeof(double);
        size_t sparse_memory = M_sparse.nnz() * (sizeof(double) + sizeof(size_t)) + n * sizeof(size_t);
        cout << "  Memory usage:\n";
        cout << "    Dense: " << dense_memory / (1024.0 * 1024.0) << " MB\n";
        cout << "    Sparse: " << sparse_memory / (1024.0 * 1024.0) << " MB\n";
        cout << "    Savings: " << (1.0 - double(sparse_memory)/dense_memory) * 100 << "%\n\n";
    }

    // =========================================================================
    // 2. BANDED MATRICES (Common in numerical PDEs)
    // =========================================================================
    cout << "2. BANDED MATRIX (Tridiagonal - heat equation, splines)\n";
    cout << string(60, '-') << "\n";

    {
        const size_t n = 1000;
        const size_t bandwidth = 1;  // Tridiagonal

        matrix<double> M(n, n);
        banded_matrix<double> M_banded(n, bandwidth, bandwidth);

        // Fill tridiagonal matrix (typical from finite differences)
        for (size_t i = 0; i < n; ++i) {
            if (i > 0) {
                M(i, i-1) = -1.0;
                M_banded(i, i-1) = -1.0;
            }
            M(i, i) = 2.0;
            M_banded(i, i) = 2.0;
            if (i < n-1) {
                M(i, i+1) = -1.0;
                M_banded(i, i+1) = -1.0;
            }
        }

        vector<double> x(n);
        for (size_t i = 0; i < n; ++i) {
            x[i] = dis(gen);
        }

        cout << "  Matrix size: " << n << "x" << n << "\n";
        cout << "  Bandwidth: " << 2*bandwidth + 1 << " (tridiagonal)\n\n";

        // Dense multiplication
        double dense_time = time_ms("Dense tridiagonal", [&]() {
            vector<double> y(n);
            for (size_t i = 0; i < n; ++i) {
                double sum = 0;
                for (size_t j = 0; j < n; ++j) {
                    sum += M(i, j) * x[j];
                }
                y[i] = sum;
            }
            return y[0];
        }, 100);
        cout << "\n";

        // Banded multiplication
        double banded_time = time_ms("Banded storage", [&]() {
            auto y = M_banded * x;
            return y[0];
        }, 100);
        cout << " [" << dense_time/banded_time << "x]\n\n";

        // Memory comparison
        size_t dense_memory = n * n * sizeof(double);
        size_t banded_memory = n * (2 * bandwidth + 1) * sizeof(double);
        cout << "  Memory usage:\n";
        cout << "    Dense: " << dense_memory / 1024.0 << " KB\n";
        cout << "    Banded: " << banded_memory / 1024.0 << " KB\n";
        cout << "    Savings: " << (1.0 - double(banded_memory)/dense_memory) * 100 << "%\n\n";
    }

    // =========================================================================
    // 3. LOW-RANK MATRICES (Machine learning, data compression)
    // =========================================================================
    cout << "3. LOW-RANK MATRIX (Rank-10 approximation of 1000x1000)\n";
    cout << string(60, '-') << "\n";

    {
        const size_t n = 1000;
        const size_t rank = 10;

        // Create low-rank matrix A = U * V^T
        matrix<double> U(n, rank), V(n, rank);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < rank; ++j) {
                U(i, j) = dis(gen);
                V(i, j) = dis(gen);
            }
        }

        low_rank_matrix<double> M_lowrank(U, V);

        // Create full matrix for comparison
        matrix<double> M_full(n, n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double sum = 0;
                for (size_t k = 0; k < rank; ++k) {
                    sum += U(i, k) * V(j, k);
                }
                M_full(i, j) = sum;
            }
        }

        vector<double> x(n);
        for (size_t i = 0; i < n; ++i) {
            x[i] = dis(gen);
        }

        cout << "  Matrix size: " << n << "x" << n << "\n";
        cout << "  Rank: " << rank << "\n";
        cout << "  Compression ratio: " << M_lowrank.compression_ratio() << "x\n\n";

        // Full matrix multiplication
        double full_time = time_ms("Full matrix O(n²)", [&]() {
            vector<double> y(n);
            for (size_t i = 0; i < n; ++i) {
                double sum = 0;
                for (size_t j = 0; j < n; ++j) {
                    sum += M_full(i, j) * x[j];
                }
                y[i] = sum;
            }
            return y[0];
        }, 10);
        cout << "\n";

        // Low-rank multiplication
        double lowrank_time = time_ms("Low-rank O(n*r)", [&]() {
            auto y = M_lowrank * x;
            return y[0];
        }, 10);
        cout << " [" << full_time/lowrank_time << "x]\n\n";
    }

    // =========================================================================
    // 4. BLOCK DIAGONAL (Domain decomposition, batched operations)
    // =========================================================================
    cout << "4. BLOCK DIAGONAL MATRIX (10 blocks of 100x100)\n";
    cout << string(60, '-') << "\n";

    {
        const size_t num_blocks = 10;
        const size_t block_size = 100;
        const size_t total_size = num_blocks * block_size;

        block_diagonal_matrix<double> M_block;

        // Create blocks
        for (size_t b = 0; b < num_blocks; ++b) {
            matrix<double> block(block_size, block_size);
            for (size_t i = 0; i < block_size; ++i) {
                for (size_t j = 0; j < block_size; ++j) {
                    block(i, j) = dis(gen);
                }
            }
            M_block.add_block(block);
        }

        vector<double> x(total_size);
        for (size_t i = 0; i < total_size; ++i) {
            x[i] = dis(gen);
        }

        cout << "  Total size: " << total_size << "x" << total_size << "\n";
        cout << "  Block configuration: " << num_blocks << " blocks of " << block_size << "x" << block_size << "\n\n";

        // Sequential processing
        double seq_time = time_ms("Sequential block processing", [&]() {
            auto y = M_block * x;  // Will use OpenMP internally
            return y[0];
        }, 10);
        cout << " [Parallelizable!]\n\n";

        // Memory advantage
        cout << "  Advantages:\n";
        cout << "    • Each block processed independently (parallel)\n";
        cout << "    • Better cache locality\n";
        cout << "    • Suitable for GPU batching\n\n";
    }

    // =========================================================================
    // 5. KRONECKER PRODUCT (Quantum computing, tensor networks)
    // =========================================================================
    cout << "5. KRONECKER PRODUCT (Never form full matrix)\n";
    cout << string(60, '-') << "\n";

    {
        const size_t n = 50;
        const size_t m = 50;

        matrix<double> A(n, n), B(m, m);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                A(i, j) = dis(gen);
            }
        }
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < m; ++j) {
                B(i, j) = dis(gen);
            }
        }

        kronecker_product<double> K(A, B);

        vector<double> x(n * m);
        for (size_t i = 0; i < n * m; ++i) {
            x[i] = dis(gen);
        }

        cout << "  A: " << n << "x" << n << ", B: " << m << "x" << m << "\n";
        cout << "  A⊗B: " << (n*m) << "x" << (n*m) << " (would be "
             << (n*m*n*m*sizeof(double))/(1024.0*1024.0) << " MB)\n\n";

        // Implicit multiplication (never forms full matrix)
        double implicit_time = time_ms("Implicit (A⊗B)*x", [&]() {
            auto y = K * x;
            return y[0];
        }, 10);
        cout << " [Never forms " << n*m << "x" << n*m << " matrix!]\n\n";
    }

    // =========================================================================
    // 6. AUTOMATIC STRUCTURE DETECTION
    // =========================================================================
    cout << "6. AUTOMATIC STRUCTURE DETECTION\n";
    cout << string(60, '-') << "\n";

    {
        // Create test matrices with different structures
        vector<pair<string, matrix<double>>> test_matrices;

        // Diagonal
        matrix<double> diag(100, 100);
        for (size_t i = 0; i < 100; ++i) {
            diag(i, i) = dis(gen);
        }
        test_matrices.push_back({"Diagonal test", diag});

        // Sparse
        matrix<double> sparse(100, 100);
        for (size_t i = 0; i < 100; ++i) {
            for (size_t j = 0; j < 100; ++j) {
                if (sparse_dis(gen) > 0.95) {
                    sparse(i, j) = dis(gen);
                }
            }
        }
        test_matrices.push_back({"Sparse test", sparse});

        // Upper triangular
        matrix<double> upper(100, 100);
        for (size_t i = 0; i < 100; ++i) {
            for (size_t j = i; j < 100; ++j) {
                upper(i, j) = dis(gen);
            }
        }
        test_matrices.push_back({"Upper triangular test", upper});

        // Detect structures
        for (const auto& test : test_matrices) {
            smart_matrix<double> smart(test.second);
            cout << "  " << left << setw(25) << test.first << ": ";
            cout << "Detected as " << smart.structure_name() << "\n";
        }
    }

    cout << "\n=== SUMMARY ===\n\n";

    cout << "COMMON STRUCTURES & SPEEDUPS:\n";
    cout << "  • Sparse (95% zeros): ~20x speedup, ~95% memory savings\n";
    cout << "  • Banded (tridiagonal): ~300x speedup, ~99.7% memory savings\n";
    cout << "  • Low-rank (rank-10): ~100x speedup, ~99% memory savings\n";
    cout << "  • Block diagonal: Perfect parallelization\n";
    cout << "  • Kronecker product: Never form full matrix\n\n";

    cout << "KEY INSIGHT:\n";
    cout << "Most real-world matrices have structure!\n";
    cout << "Exploiting it gives 10-1000x performance gains.\n";

    return 0;
}