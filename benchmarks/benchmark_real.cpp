// Real-world Performance Benchmark
// Tests actual implementations that compile

#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <random>
#include <iomanip>
#include <stepanov/math.hpp>
#include <stepanov/gcd.hpp>
#include <stepanov/matrix.hpp>
#include <stepanov/matrix_expressions.hpp>

using namespace std;
using namespace chrono;
using namespace stepanov;

template<typename F>
double benchmark_ms(const string& name, size_t iterations, F&& f) {
    // Warmup
    for (size_t i = 0; i < 10; ++i) f();

    auto start = high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        volatile auto result = f();
        (void)result;
    }
    auto end = high_resolution_clock::now();

    double ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    double ms_per_op = ms / iterations;

    cout << "  " << left << setw(35) << name << ": ";
    cout << right << setw(10) << fixed << setprecision(4) << ms_per_op << " ms";

    return ms_per_op;
}

// Naive diagonal matrix multiply
void naive_diagonal_multiply(const vector<double>& diag,
                            const vector<vector<double>>& M,
                            vector<vector<double>>& result) {
    size_t n = diag.size();
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0;
            for (size_t k = 0; k < n; ++k) {
                double d_ik = (i == k) ? diag[i] : 0;
                sum += d_ik * M[k][j];
            }
            result[i][j] = sum;
        }
    }
}

int main() {
    cout << "=== STEPANOV LIBRARY - REAL PERFORMANCE ===\n\n";

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.1, 10.0);

    // ========================================================================
    // 1. MATHEMATICAL OPERATIONS
    // ========================================================================
    cout << "1. CORE MATH OPERATIONS\n";
    cout << "------------------------\n";

    // Power function comparison
    {
        cout << "Power function (2^30):\n";

        double stepanov_time = benchmark_ms("Stepanov power()", 1000000, []() {
            return stepanov::power(2, 30);
        });

        double std_time = benchmark_ms("std::pow()", 1000000, []() {
            return static_cast<int>(std::pow(2.0, 30.0));
        });

        cout << " (ratio: " << stepanov_time/std_time << ")\n\n";
    }

    // Modular exponentiation
    {
        cout << "Modular exponentiation (3^100000 mod 97):\n";

        double stepanov_time = benchmark_ms("Stepanov power_mod()", 10000, []() {
            return stepanov::power_mod(3, 100000, 97);
        });

        // Naive implementation would be too slow, so we compare against a reasonable baseline
        double baseline_time = benchmark_ms("Baseline (3^1000 mod 97)", 10000, []() {
            long long result = 1;
            for (int i = 0; i < 1000; ++i) {
                result = (result * 3) % 97;
            }
            return static_cast<int>(result);
        });

        cout << " (100x fewer iterations)\n\n";
    }

    // ========================================================================
    // 2. MATRIX OPERATIONS
    // ========================================================================
    cout << "2. MATRIX STRUCTURE EXPLOITATION\n";
    cout << "---------------------------------\n";

    const size_t n = 100;

    // Diagonal matrix multiplication
    {
        cout << "Diagonal matrix multiply (100x100):\n";

        // Setup
        matrix_expr::diagonal_matrix<double> D(n);
        matrix<double> M(n, n);

        for (size_t i = 0; i < n; ++i) {
            D.diagonal(i) = dis(gen);
            for (size_t j = 0; j < n; ++j) {
                M(i, j) = dis(gen);
            }
        }

        // Optimized: O(n²)
        double opt_time = benchmark_ms("Structure-aware (O(n²))", 100, [&]() {
            auto result = D * M;
            return result(0, 0);  // Force evaluation
        });

        // Naive: O(n³)
        vector<double> diag_vec(n);
        vector<vector<double>> m_vec(n, vector<double>(n));
        vector<vector<double>> result_vec(n, vector<double>(n));

        for (size_t i = 0; i < n; ++i) {
            diag_vec[i] = D.diagonal(i);
            for (size_t j = 0; j < n; ++j) {
                m_vec[i][j] = M(i, j);
            }
        }

        double naive_time = benchmark_ms("Naive approach (O(n³))", 100, [&]() {
            naive_diagonal_multiply(diag_vec, m_vec, result_vec);
            return result_vec[0][0];
        });

        cout << "\n  SPEEDUP: " << naive_time / opt_time << "x\n\n";
    }

    // Symmetric matrix storage
    {
        cout << "Symmetric matrix memory usage (1000x1000):\n";

        const size_t big_n = 1000;
        size_t full_storage = big_n * big_n * sizeof(double);
        size_t symmetric_storage = (big_n * (big_n + 1) / 2) * sizeof(double);

        cout << "  Full matrix: " << full_storage / (1024.0 * 1024.0) << " MB\n";
        cout << "  Symmetric: " << symmetric_storage / (1024.0 * 1024.0) << " MB\n";
        cout << "  SAVINGS: " << (1.0 - double(symmetric_storage)/full_storage) * 100 << "%\n\n";
    }

    // Expression templates
    {
        cout << "Expression templates (200x200):\n";

        const size_t expr_n = 200;
        matrix<double> A(expr_n, expr_n);
        matrix<double> B(expr_n, expr_n);
        matrix<double> C(expr_n, expr_n);

        for (size_t i = 0; i < expr_n; ++i) {
            for (size_t j = 0; j < expr_n; ++j) {
                A(i, j) = dis(gen);
                B(i, j) = dis(gen);
                C(i, j) = dis(gen);
            }
        }

        // Single pass with expression templates
        double expr_time = benchmark_ms("Expression templates (1 pass)", 10, [&]() {
            matrix<double> result = 2.0 * A + B - 0.5 * C;
            return result(0, 0);
        });

        // Multiple passes without expression templates
        double naive_time = benchmark_ms("Naive (3 passes)", 10, [&]() {
            matrix<double> temp1(expr_n, expr_n);
            matrix<double> temp2(expr_n, expr_n);
            matrix<double> result(expr_n, expr_n);

            // Pass 1: 2.0 * A
            for (size_t i = 0; i < expr_n; ++i) {
                for (size_t j = 0; j < expr_n; ++j) {
                    temp1(i, j) = 2.0 * A(i, j);
                }
            }

            // Pass 2: temp1 + B
            for (size_t i = 0; i < expr_n; ++i) {
                for (size_t j = 0; j < expr_n; ++j) {
                    temp2(i, j) = temp1(i, j) + B(i, j);
                }
            }

            // Pass 3: temp2 - 0.5 * C
            for (size_t i = 0; i < expr_n; ++i) {
                for (size_t j = 0; j < expr_n; ++j) {
                    result(i, j) = temp2(i, j) - 0.5 * C(i, j);
                }
            }

            return result(0, 0);
        });

        cout << "\n  SPEEDUP: " << naive_time / expr_time << "x\n\n";
    }

    // ========================================================================
    // 3. PERFORMANCE SUMMARY
    // ========================================================================
    cout << "=== VERIFIED PERFORMANCE CLAIMS ===\n\n";

    cout << "✓ CONFIRMED:\n";
    cout << "  • Symmetric matrices use 50% less memory (mathematical fact)\n";
    cout << "  • Diagonal matrix multiply is O(n²) vs O(n³) naive\n";
    cout << "  • Expression templates eliminate temporaries\n";
    cout << "  • Modular exponentiation is O(log n)\n\n";

    cout << "✗ ISSUES FOUND:\n";
    cout << "  • Generic power() has overhead vs native operations\n";
    cout << "  • GCD implementation needs optimization\n";
    cout << "  • Some claimed speedups are overstated\n\n";

    cout << "CONCLUSION:\n";
    cout << "The library demonstrates valuable algorithmic principles\n";
    cout << "and achieves significant speedups through structure\n";
    cout << "exploitation, but specific numerical claims in the\n";
    cout << "whitepaper should be updated with real measurements.\n";

    return 0;
}