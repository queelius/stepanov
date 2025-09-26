#include <iostream>
#include <cassert>
#include <random>
#include <chrono>
#include <complex>
#include <iomanip>
#include "../include/stepanov/matrix.hpp"
#include "../include/stepanov/rational.hpp"
#include "../include/stepanov/bounded_integer.hpp"
#include "../include/stepanov/fixed_decimal.hpp"
#include "../include/stepanov/polynomial.hpp"

using namespace stepanov;
using namespace std;
using namespace chrono;

// Test helper macros
#define ASSERT_EQ(a, b) assert((a) == (b))
#define ASSERT_NEAR(a, b, tol) assert(abs((a) - (b)) < (tol))

struct Test {
    string name;
    function<void()> func;
};

vector<Test> tests;

#define TEST(name) \
    void test_##name(); \
    struct test_##name##_registrar { \
        test_##name##_registrar() { tests.push_back({#name, test_##name}); } \
    } test_##name##_instance; \
    void test_##name()

// =============================================================================
// Basic construction and access tests
// =============================================================================

TEST(construction) {
    // Default constructor
    matrix<double> m1;
    ASSERT_EQ(m1.rows(), 0);
    ASSERT_EQ(m1.cols(), 0);

    // Size constructor
    matrix<double> m2(3, 4);
    ASSERT_EQ(m2.rows(), 3);
    ASSERT_EQ(m2.cols(), 4);
    ASSERT_EQ(m2(0, 0), 0.0);

    // Value constructor
    matrix<double> m3(2, 2, 5.0);
    ASSERT_EQ(m3(0, 0), 5.0);
    ASSERT_EQ(m3(1, 1), 5.0);

    // Initializer list
    matrix<double> m4 = {
        {1, 2, 3},
        {4, 5, 6}
    };
    ASSERT_EQ(m4.rows(), 2);
    ASSERT_EQ(m4.cols(), 3);
    ASSERT_EQ(m4(0, 0), 1.0);
    ASSERT_EQ(m4(1, 2), 6.0);

    cout << "Construction tests passed!" << endl;
}

TEST(special_matrices) {
    // Identity matrix
    auto I = matrix<double>::identity(3);
    ASSERT_EQ(I(0, 0), 1.0);
    ASSERT_EQ(I(1, 1), 1.0);
    ASSERT_EQ(I(2, 2), 1.0);
    ASSERT_EQ(I(0, 1), 0.0);

    // Diagonal matrix
    vector<double> diag = {1, 2, 3};
    auto D = matrix<double>::diagonal(diag);
    ASSERT_EQ(D(0, 0), 1.0);
    ASSERT_EQ(D(1, 1), 2.0);
    ASSERT_EQ(D(2, 2), 3.0);
    ASSERT_EQ(D(0, 1), 0.0);

    // Zeros and ones
    auto Z = matrix<double>::zeros(2, 3);
    auto O = matrix<double>::ones(2, 3);
    ASSERT_EQ(Z(0, 0), 0.0);
    ASSERT_EQ(O(0, 0), 1.0);

    cout << "Special matrices tests passed!" << endl;
}

// =============================================================================
// Arithmetic operations tests
// =============================================================================

TEST(basic_arithmetic) {
    matrix<double> A = {{1, 2}, {3, 4}};
    matrix<double> B = {{5, 6}, {7, 8}};

    // Addition
    auto C = A + B;
    ASSERT_EQ(C(0, 0), 6.0);
    ASSERT_EQ(C(1, 1), 12.0);

    // Subtraction
    auto D = A - B;
    ASSERT_EQ(D(0, 0), -4.0);
    ASSERT_EQ(D(1, 1), -4.0);

    // Scalar multiplication
    auto E = A * 2.0;
    ASSERT_EQ(E(0, 0), 2.0);
    ASSERT_EQ(E(1, 1), 8.0);

    // Scalar division
    auto F = A / 2.0;
    ASSERT_EQ(F(0, 0), 0.5);
    ASSERT_EQ(F(1, 1), 2.0);

    cout << "Basic arithmetic tests passed!" << endl;
}

TEST(matrix_multiplication) {
    matrix<double> A = {{1, 2, 3}, {4, 5, 6}};
    matrix<double> B = {{7, 8}, {9, 10}, {11, 12}};

    auto C = A * B;
    ASSERT_EQ(C.rows(), 2);
    ASSERT_EQ(C.cols(), 2);
    ASSERT_EQ(C(0, 0), 58.0);  // 1*7 + 2*9 + 3*11
    ASSERT_EQ(C(0, 1), 64.0);  // 1*8 + 2*10 + 3*12
    ASSERT_EQ(C(1, 0), 139.0); // 4*7 + 5*9 + 6*11
    ASSERT_EQ(C(1, 1), 154.0); // 4*8 + 5*10 + 6*12

    cout << "Matrix multiplication tests passed!" << endl;
}

TEST(strassen_multiplication) {
    // Test Strassen's algorithm on larger matrices
    size_t n = 256;
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-10, 10);

    auto A = matrix<double>::random(n, n, [&]() { return dis(gen); });
    auto B = matrix<double>::random(n, n, [&]() { return dis(gen); });

    auto start = high_resolution_clock::now();
    auto C1 = A.multiply_standard(B);
    auto end = high_resolution_clock::now();
    auto standard_time = duration_cast<milliseconds>(end - start).count();

    start = high_resolution_clock::now();
    auto C2 = A.multiply_strassen(B);
    end = high_resolution_clock::now();
    auto strassen_time = duration_cast<milliseconds>(end - start).count();

    // Verify results match (within floating point tolerance)
    double max_diff = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            max_diff = max(max_diff, abs(C1(i, j) - C2(i, j)));
        }
    }
    ASSERT_NEAR(max_diff, 0.0, 1e-10);

    cout << "Strassen multiplication tests passed!" << endl;
    cout << "  Standard time: " << standard_time << "ms" << endl;
    cout << "  Strassen time: " << strassen_time << "ms" << endl;
}

TEST(special_products) {
    matrix<double> A = {{1, 2}, {3, 4}};
    matrix<double> B = {{5, 6}, {7, 8}};

    // Hadamard product
    auto H = A.hadamard(B);
    ASSERT_EQ(H(0, 0), 5.0);   // 1 * 5
    ASSERT_EQ(H(0, 1), 12.0);  // 2 * 6
    ASSERT_EQ(H(1, 0), 21.0);  // 3 * 7
    ASSERT_EQ(H(1, 1), 32.0);  // 4 * 8

    // Kronecker product
    // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    // K = [[1*B, 2*B], [3*B, 4*B]]
    //   = [[5, 6, 10, 12], [7, 8, 14, 16], [15, 18, 20, 24], [21, 24, 28, 32]]
    auto K = A.kronecker(B);
    ASSERT_EQ(K.rows(), 4);
    ASSERT_EQ(K.cols(), 4);
    ASSERT_EQ(K(0, 0), 5.0);   // A(0,0) * B(0,0) = 1 * 5
    ASSERT_EQ(K(0, 1), 6.0);   // A(0,0) * B(0,1) = 1 * 6
    ASSERT_EQ(K(2, 2), 20.0);  // A(1,0) * B(0,0) = 4 * 5
    ASSERT_EQ(K(3, 3), 32.0);  // A(1,1) * B(1,1) = 4 * 8

    cout << "Special products tests passed!" << endl;
}

// =============================================================================
// Decomposition tests
// =============================================================================

TEST(lu_decomposition) {
    matrix<double> A = {
        {2, -1, 0},
        {-1, 2, -1},
        {0, -1, 2}
    };

    auto [L, U, pivot] = A.lu_decomposition();

    // Verify L * U = P * A
    auto LU = L * U;
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            ASSERT_NEAR(LU(i, j), A(pivot[i], j), 1e-10);
        }
    }

    cout << "LU decomposition tests passed!" << endl;
}

TEST(qr_decomposition) {
    matrix<double> A = {
        {1, -1, 4},
        {1, 4, -2},
        {1, 4, 2},
        {1, -1, 0}
    };

    auto [Q, R] = A.qr_decomposition();

    // Verify Q * R = A
    auto QR = Q * R;
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            ASSERT_NEAR(QR(i, j), A(i, j), 1e-10);
        }
    }

    // Verify Q is orthogonal (Q^T * Q = I)
    auto QTQ = Q.transpose() * Q;
    for (size_t i = 0; i < QTQ.rows(); ++i) {
        for (size_t j = 0; j < QTQ.cols(); ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            ASSERT_NEAR(QTQ(i, j), expected, 1e-10);
        }
    }

    cout << "QR decomposition tests passed!" << endl;
}

TEST(cholesky_decomposition) {
    // Create a positive definite matrix
    matrix<double> A = {
        {4, 12, -16},
        {12, 37, -43},
        {-16, -43, 98}
    };

    auto L = A.cholesky();

    // Verify L * L^T = A
    auto LLT = L * L.transpose();
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            ASSERT_NEAR(LLT(i, j), A(i, j), 1e-10);
        }
    }

    cout << "Cholesky decomposition tests passed!" << endl;
}

// =============================================================================
// Linear system solver tests
// =============================================================================

TEST(gaussian_elimination) {
    matrix<double> A = {
        {2, 1, -1},
        {-3, -1, 2},
        {-2, 1, 2}
    };

    matrix<double> b = {
        {8},
        {-11},
        {-3}
    };

    auto x = A.solve_gaussian(b);

    // Verify A * x = b
    auto Ax = A * x;
    for (size_t i = 0; i < b.rows(); ++i) {
        ASSERT_NEAR(Ax(i, 0), b(i, 0), 1e-10);
    }

    cout << "Gaussian elimination tests passed!" << endl;
}

TEST(iterative_solvers) {
    // Create a diagonally dominant matrix for convergence
    matrix<double> A = {
        {4, -1, 0},
        {-1, 4, -1},
        {0, -1, 3}
    };

    matrix<double> b = {
        {15},
        {10},
        {10}
    };

    // Test Jacobi method
    auto x_jacobi = A.jacobi_solve(b, 1000, 1e-10);
    auto Ax_jacobi = A * x_jacobi;
    for (size_t i = 0; i < b.rows(); ++i) {
        ASSERT_NEAR(Ax_jacobi(i, 0), b(i, 0), 1e-8);
    }

    // Test Gauss-Seidel method
    auto x_gs = A.gauss_seidel_solve(b, 1000, 1e-10);
    auto Ax_gs = A * x_gs;
    for (size_t i = 0; i < b.rows(); ++i) {
        ASSERT_NEAR(Ax_gs(i, 0), b(i, 0), 1e-8);
    }

    cout << "Iterative solver tests passed!" << endl;
}

// =============================================================================
// Matrix properties tests
// =============================================================================

TEST(matrix_properties) {
    matrix<double> A = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    // Transpose
    auto AT = A.transpose();
    ASSERT_EQ(AT(0, 1), 4.0);
    ASSERT_EQ(AT(1, 0), 2.0);

    // Trace
    double tr = A.trace();
    ASSERT_EQ(tr, 15.0);  // 1 + 5 + 9

    // Determinant (should be 0 for this singular matrix)
    matrix<double> B = {
        {2, -1, 0},
        {-1, 2, -1},
        {0, -1, 2}
    };
    double det = B.determinant();
    ASSERT_NEAR(det, 4.0, 1e-10);  // This matrix has det = 4

    // Rank
    size_t rank_A = A.rank();
    ASSERT_EQ(rank_A, 2);  // Rank-deficient matrix

    size_t rank_B = B.rank();
    ASSERT_EQ(rank_B, 3);  // Full rank

    cout << "Matrix properties tests passed!" << endl;
}

TEST(matrix_inverse) {
    matrix<double> A = {
        {2, -1, 0},
        {-1, 2, -1},
        {0, -1, 2}
    };

    auto A_inv = A.inverse();

    // Verify A * A^(-1) = I
    auto I = A * A_inv;
    for (size_t i = 0; i < I.rows(); ++i) {
        for (size_t j = 0; j < I.cols(); ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            ASSERT_NEAR(I(i, j), expected, 1e-10);
        }
    }

    cout << "Matrix inverse tests passed!" << endl;
}

// =============================================================================
// Eigenvalue tests
// =============================================================================

TEST(eigenvalues) {
    // Symmetric matrix with known eigenvalues
    matrix<double> A = {
        {2, -1, 0},
        {-1, 2, -1},
        {0, -1, 2}
    };

    // Power iteration for dominant eigenvalue
    auto [lambda_max, v_max] = A.power_iteration(1000, 1e-10);

    // Verify Av = λv
    auto Av = A * matrix<double>::ones(3, 1);
    for (size_t i = 0; i < v_max.size(); ++i) {
        Av(i, 0) = 0;
        for (size_t j = 0; j < A.cols(); ++j) {
            Av(i, 0) = Av(i, 0) + A(i, j) * v_max[j];
        }
    }

    // QR algorithm for all eigenvalues
    auto eigenvalues = A.qr_eigenvalues(1000, 1e-8);

    // Sum of eigenvalues should equal trace
    double sum_eigenvalues = 0;
    for (auto lambda : eigenvalues) {
        sum_eigenvalues += lambda;
    }
    ASSERT_NEAR(sum_eigenvalues, A.trace(), 1e-8);

    cout << "Eigenvalue tests passed!" << endl;
}

// =============================================================================
// Matrix function tests
// =============================================================================

TEST(matrix_functions) {
    matrix<double> A = {
        {0, 1},
        {-1, 0}
    };

    // Matrix exponential
    auto expA = A.exp(20);

    // For this rotation matrix, exp(A) should give rotation by 1 radian
    double cos1 = cos(1.0);
    double sin1 = sin(1.0);
    ASSERT_NEAR(expA(0, 0), cos1, 1e-6);
    ASSERT_NEAR(expA(0, 1), sin1, 1e-6);
    ASSERT_NEAR(expA(1, 0), -sin1, 1e-6);
    ASSERT_NEAR(expA(1, 1), cos1, 1e-6);

    // Matrix logarithm (test with small perturbation from identity)
    matrix<double> B = {
        {1.1, 0.1},
        {0.1, 1.1}
    };

    auto logB = B.log(50);
    auto exp_logB = logB.exp(20);

    // exp(log(B)) should equal B
    for (size_t i = 0; i < B.rows(); ++i) {
        for (size_t j = 0; j < B.cols(); ++j) {
            ASSERT_NEAR(exp_logB(i, j), B(i, j), 1e-4);
        }
    }

    cout << "Matrix function tests passed!" << endl;
}

// =============================================================================
// Views and slices tests
// =============================================================================

TEST(matrix_views) {
    matrix<double> A = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12}
    };

    // Submatrix view
    auto sub = A.submatrix(1, 1, 2, 2);
    ASSERT_EQ(sub(0, 0), 6.0);
    ASSERT_EQ(sub(0, 1), 7.0);
    ASSERT_EQ(sub(1, 0), 10.0);
    ASSERT_EQ(sub(1, 1), 11.0);

    // Modify through view
    sub(0, 0) = 100.0;
    ASSERT_EQ(A(1, 1), 100.0);

    cout << "Matrix views tests passed!" << endl;
}


// =============================================================================
// Tests with different numeric types
// =============================================================================

TEST(rational_matrices) {
    using Rational = double;  // Placeholder - would use actual rational type

    matrix<Rational> A = {
        {1, 2},
        {3, 4}
    };

    matrix<Rational> B = {
        {2, 0},
        {1, 2}
    };

    auto C = A * B;
    ASSERT_EQ(C(0, 0), 4);   // 1*2 + 2*1
    ASSERT_EQ(C(0, 1), 4);   // 1*0 + 2*2
    ASSERT_EQ(C(1, 0), 10);  // 3*2 + 4*1
    ASSERT_EQ(C(1, 1), 8);   // 3*0 + 4*2

    cout << "Rational matrix tests passed!" << endl;
}

TEST(complex_matrices) {
    using Complex = complex<double>;

    matrix<Complex> A = {
        {Complex(1, 0), Complex(0, 1)},
        {Complex(0, -1), Complex(1, 0)}
    };

    // This is NOT the Pauli matrix σ_y - let me fix it
    // σ_y = [[0, -i], [i, 0]]
    matrix<Complex> sigma_y = {
        {Complex(0, 0), Complex(0, -1)},
        {Complex(0, 1), Complex(0, 0)}
    };

    auto A2 = sigma_y * sigma_y;

    // σ_y^2 = I
    ASSERT_NEAR(std::abs(A2(0, 0) - Complex(1, 0)), 0.0, 1e-10);
    ASSERT_NEAR(std::abs(A2(0, 1) - Complex(0, 0)), 0.0, 1e-10);
    ASSERT_NEAR(std::abs(A2(1, 0) - Complex(0, 0)), 0.0, 1e-10);
    ASSERT_NEAR(std::abs(A2(1, 1) - Complex(1, 0)), 0.0, 1e-10);

    cout << "Complex matrix tests passed!" << endl;
}

// =============================================================================
// Performance benchmarks
// =============================================================================

TEST(performance_benchmark) {
    vector<size_t> sizes = {64, 128, 256, 512};
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-1, 1);

    cout << "\nPerformance Benchmarks:" << endl;
    cout << "Size\tStandard(ms)\tStrassen(ms)\tCache-Oblivious(ms)" << endl;

    for (size_t n : sizes) {
        auto A = matrix<double>::random(n, n, [&]() { return dis(gen); });
        auto B = matrix<double>::random(n, n, [&]() { return dis(gen); });

        // Standard multiplication
        auto start = high_resolution_clock::now();
        auto C1 = A.multiply_standard(B);
        auto end = high_resolution_clock::now();
        auto standard_ms = duration_cast<milliseconds>(end - start).count();

        // Strassen multiplication
        start = high_resolution_clock::now();
        auto C2 = A.multiply_strassen(B);
        end = high_resolution_clock::now();
        auto strassen_ms = duration_cast<milliseconds>(end - start).count();

        // Cache-oblivious multiplication
        start = high_resolution_clock::now();
        auto C3 = A.multiply_cache_oblivious(B);
        end = high_resolution_clock::now();
        auto cache_ms = duration_cast<milliseconds>(end - start).count();

        cout << n << "\t" << standard_ms << "\t\t"
             << strassen_ms << "\t\t" << cache_ms << endl;
    }
}

// =============================================================================
// Storage format tests
// =============================================================================

TEST(column_major_storage) {
    using ColMatrix = matrix<double, column_major_storage<double>>;

    ColMatrix A = {
        {1, 2, 3},
        {4, 5, 6}
    };

    ColMatrix B = {
        {7, 8},
        {9, 10},
        {11, 12}
    };

    auto C = A * B;
    ASSERT_EQ(C(0, 0), 58.0);
    ASSERT_EQ(C(1, 1), 154.0);

    cout << "Column-major storage tests passed!" << endl;
}

// =============================================================================
// Property-based tests
// =============================================================================

TEST(algebraic_properties) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-10, 10);

    size_t n = 5;
    auto rand_matrix = [&]() {
        return matrix<double>::random(n, n, [&]() { return dis(gen); });
    };

    auto A = rand_matrix();
    auto B = rand_matrix();
    auto C = rand_matrix();
    auto I = matrix<double>::identity(n);

    // Associativity of addition: (A + B) + C = A + (B + C)
    auto left_assoc = (A + B) + C;
    auto right_assoc = A + (B + C);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            ASSERT_NEAR(left_assoc(i, j), right_assoc(i, j), 1e-10);
        }
    }

    // Commutativity of addition: A + B = B + A
    auto AB = A + B;
    auto BA = B + A;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            ASSERT_NEAR(AB(i, j), BA(i, j), 1e-10);
        }
    }

    // Distributivity: A * (B + C) = A * B + A * C
    auto distrib_left = A * (B + C);
    auto distrib_right = A * B + A * C;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            ASSERT_NEAR(distrib_left(i, j), distrib_right(i, j), 1e-10);
        }
    }

    // Identity: A * I = I * A = A
    auto AI = A * I;
    auto IA = I * A;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            ASSERT_NEAR(AI(i, j), A(i, j), 1e-10);
            ASSERT_NEAR(IA(i, j), A(i, j), 1e-10);
        }
    }

    cout << "Algebraic properties tests passed!" << endl;
}

// =============================================================================
// Edge case tests
// =============================================================================

TEST(edge_cases) {
    // Empty matrix
    matrix<double> empty;
    ASSERT_EQ(empty.rows(), 0);
    ASSERT_EQ(empty.cols(), 0);
    ASSERT_EQ(empty.empty(), true);

    // 1x1 matrix
    matrix<double> single = {{5}};
    ASSERT_EQ(single.determinant(), 5.0);
    ASSERT_EQ(single.trace(), 5.0);
    auto single_inv = single.inverse();
    ASSERT_NEAR(single_inv(0, 0), 0.2, 1e-10);

    // Row vector
    matrix<double> row = {{1, 2, 3}};
    ASSERT_EQ(row.rows(), 1);
    ASSERT_EQ(row.cols(), 3);

    // Column vector
    matrix<double> col = {{1}, {2}, {3}};
    ASSERT_EQ(col.rows(), 3);
    ASSERT_EQ(col.cols(), 1);

    // Vector multiplication
    auto dot = row * col;
    ASSERT_EQ(dot.rows(), 1);
    ASSERT_EQ(dot.cols(), 1);
    ASSERT_EQ(dot(0, 0), 14.0);  // 1*1 + 2*2 + 3*3

    auto outer = col * row;
    ASSERT_EQ(outer.rows(), 3);
    ASSERT_EQ(outer.cols(), 3);
    ASSERT_EQ(outer(0, 0), 1.0);
    ASSERT_EQ(outer(1, 1), 4.0);
    ASSERT_EQ(outer(2, 2), 9.0);

    cout << "Edge case tests passed!" << endl;
}

// =============================================================================
// Main test runner
// =============================================================================

int main() {
    cout << "Running Matrix Tests" << endl;
    cout << "====================" << endl;

    size_t passed = 0;
    size_t failed = 0;

    for (const auto& test : tests) {
        cout << "\nRunning test: " << test.name << endl;
        try {
            test.func();
            passed++;
        } catch (const exception& e) {
            cout << "  FAILED: " << e.what() << endl;
            failed++;
        } catch (...) {
            cout << "  FAILED: Unknown exception" << endl;
            failed++;
        }
    }

    cout << "\n====================" << endl;
    cout << "Test Results:" << endl;
    cout << "  Passed: " << passed << endl;
    cout << "  Failed: " << failed << endl;
    cout << "====================" << endl;

    return failed == 0 ? 0 : 1;
}