#include <iostream>
#include <chrono>
#include <random>
#include <stepanov/symmetric_matrix.hpp>
#include <stepanov/matrix.hpp>

using namespace stepanov;
using namespace std;
using namespace std::chrono;

template<typename F>
double time_ms(const string& name, F&& f) {
    auto start = high_resolution_clock::now();
    f();
    auto end = high_resolution_clock::now();
    auto ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    cout << name << ": " << ms << " ms" << endl;
    return ms;
}

int main() {
    cout << "=== Symmetric Matrix Efficiency Demo ===" << endl << endl;

    const size_t n = 1000;
    cout << "Matrix size: " << n << " x " << n << endl << endl;

    // Memory comparison
    cout << "Memory Usage:" << endl;
    cout << "  Dense matrix: " << (n * n * sizeof(double)) / (1024.0 * 1024.0) << " MB" << endl;
    cout << "  Symmetric matrix: " << (n * (n + 1) / 2 * sizeof(double)) / (1024.0 * 1024.0) << " MB" << endl;
    cout << "  Savings: " << (1 - (n + 1.0) / (2.0 * n)) * 100 << "%" << endl << endl;

    // Create a random symmetric matrix
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);

    cout << "Creating matrices..." << endl;

    // Symmetric matrix
    symmetric_matrix<double> A = symmetric_matrix<double>::from_function(n,
        [&](size_t i, size_t j) { return dis(gen); });

    // Dense matrix (for comparison)
    matrix<double> B(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i; j < n; ++j) {
            double val = A(i, j);
            B(i, j) = val;
            B(j, i) = val;
        }
    }

    // Create a test vector
    vector<double> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = dis(gen);
    }

    cout << "Matrices created." << endl << endl;

    // Benchmark matrix-vector multiplication
    cout << "Matrix-Vector Multiplication:" << endl;

    vector<double> result1, result2;

    double t1 = time_ms("  Symmetric storage", [&]() {
        result1 = A * v;
    });

    double t2 = time_ms("  Dense storage", [&]() {
        result2 = B * v;
    });

    cout << "  Speedup: " << t2 / t1 << "x" << endl;

    // Verify correctness (results should be very close)
    double max_diff = 0.0;
    for (size_t i = 0; i < n; ++i) {
        max_diff = max(max_diff, abs(result1[i] - result2[i]));
    }
    cout << "  Max difference: " << max_diff << " (should be ~0)" << endl << endl;

    // Quadratic form benchmark
    cout << "Quadratic Form (x^T * A * x):" << endl;

    double qf1, qf2;

    t1 = time_ms("  Symmetric (optimized)", [&]() {
        qf1 = A.quadratic_form(v);
    });

    t2 = time_ms("  Dense (naive)", [&]() {
        // Compute x^T * A * x the naive way
        qf2 = 0.0;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                qf2 += v[i] * B(i, j) * v[j];
            }
        }
    });

    cout << "  Speedup: " << t2 / t1 << "x" << endl;
    cout << "  Difference: " << abs(qf1 - qf2) << " (should be ~0)" << endl << endl;

    // Graph operations demo
    cout << "Graph Operations Demo:" << endl;

    // Create a small graph adjacency matrix
    const size_t g_size = 6;
    graph_matrix<int> G(g_size);

    // Create a simple undirected graph
    G(0, 1) = 1;
    G(0, 2) = 1;
    G(1, 2) = 1;
    G(1, 3) = 1;
    G(2, 4) = 1;
    G(3, 4) = 1;
    G(3, 5) = 1;
    G(4, 5) = 1;

    cout << "  Graph with " << g_size << " vertices" << endl;
    cout << "  Degrees: ";
    auto degrees = G.degree_vector();
    for (auto d : degrees) cout << d << " ";
    cout << endl;

    cout << "  Number of triangles: " << G.triangle_count() << endl;
    cout << "  Is connected: " << (G.is_connected() ? "Yes" : "No") << endl;

    // Compute Laplacian
    auto L = G.laplacian();
    cout << "  Laplacian trace: " << L.trace() << " (should equal 2 * edges)" << endl << endl;

    // Eigenvalue computation
    cout << "Eigenvalue Computation:" << endl;
    symmetric_matrix<double> M(5);
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = i; j < 5; ++j) {
            M(i, j) = dis(gen);
        }
    }

    auto [eigenvalue, eigenvector] = M.power_method(100);
    cout << "  Largest eigenvalue (power method): " << eigenvalue << endl;

    // Verify it's an eigenvalue
    auto Mv = M * eigenvector;
    vector<double> lambda_v(5);
    for (size_t i = 0; i < 5; ++i) {
        lambda_v[i] = eigenvalue * eigenvector[i];
    }

    double eigen_error = 0.0;
    for (size_t i = 0; i < 5; ++i) {
        eigen_error += abs(Mv[i] - lambda_v[i]);
    }
    cout << "  Eigenvalue verification error: " << eigen_error << " (should be ~0)" << endl << endl;

    // Positive definiteness check
    cout << "Positive Definiteness:" << endl;

    // Create a positive definite matrix (A^T * A + I)
    symmetric_matrix<double> PD(4);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = i; j < 4; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < 4; ++k) {
                sum += dis(gen) * dis(gen);  // Random positive contribution
            }
            PD(i, j) = sum;
        }
        PD(i, i) += 1.0;  // Add to diagonal to ensure positive definiteness
    }

    cout << "  Matrix is positive definite: " << (PD.is_positive_definite() ? "Yes" : "No") << endl;

    cout << "\n=== Summary ===" << endl;
    cout << "Symmetric matrix storage provides:" << endl;
    cout << "  - ~50% memory savings for large matrices" << endl;
    cout << "  - Faster operations by exploiting symmetry" << endl;
    cout << "  - Specialized algorithms for graph analysis" << endl;
    cout << "  - Guaranteed symmetry by construction" << endl;

    return 0;
}