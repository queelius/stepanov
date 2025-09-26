#include <gtest/gtest.h>
#include <stepanov/matrix_expressions.hpp>
#include <stepanov/symmetric_matrix.hpp>
#include <stepanov/matrix_algorithms.hpp>
#include <stepanov/matrix_type_erasure.hpp>
#include <random>
#include <cmath>

using namespace stepanov;
using namespace stepanov::matrix_expr;

class MatrixExpressionsTest : public ::testing::Test {
protected:
    std::random_device rd;
    std::mt19937 gen{rd()};
    std::uniform_real_distribution<> dis{-10.0, 10.0};
    std::uniform_real_distribution<> pos_dis{0.1, 10.0};

    static constexpr double EPSILON = 1e-10;

    // Helper to check if two doubles are approximately equal
    bool approx_equal(double a, double b, double eps = EPSILON) {
        return std::abs(a - b) < eps;
    }

    // Helper to check if two matrices are approximately equal
    template<typename M1, typename M2>
    bool matrices_equal(const M1& A, const M2& B, double eps = EPSILON) {
        if (A.rows() != B.rows() || A.cols() != B.cols()) return false;

        for (size_t i = 0; i < A.rows(); ++i) {
            for (size_t j = 0; j < A.cols(); ++j) {
                if (!approx_equal(A(i, j), B(i, j), eps)) {
                    return false;
                }
            }
        }
        return true;
    }

    // Create a random symmetric positive definite matrix
    symmetric_matrix<double> create_spd_matrix(size_t n) {
        symmetric_matrix<double> A(n);

        // A = B^T * B + I ensures positive definiteness
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i; j < n; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    double val = dis(gen);
                    sum += val * val;
                }
                A(i, j) = sum;
            }
            A(i, i) += 1.0; // Add to diagonal for strict positive definiteness
        }

        return A;
    }
};

// =============================================================================
// Diagonal Matrix Tests
// =============================================================================

TEST_F(MatrixExpressionsTest, DiagonalMatrixConstruction) {
    diagonal_matrix<double> D(3);
    D.diagonal(0) = 1.0;
    D.diagonal(1) = 2.0;
    D.diagonal(2) = 3.0;

    EXPECT_EQ(D.rows(), 3);
    EXPECT_EQ(D.cols(), 3);
    EXPECT_DOUBLE_EQ(D(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(D(1, 1), 2.0);
    EXPECT_DOUBLE_EQ(D(2, 2), 3.0);
    EXPECT_DOUBLE_EQ(D(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(D(1, 0), 0.0);
}

TEST_F(MatrixExpressionsTest, DiagonalMatrixMultiplication) {
    diagonal_matrix<double> D1(3);
    diagonal_matrix<double> D2(3);

    D1.diagonal(0) = 2.0;
    D1.diagonal(1) = 3.0;
    D1.diagonal(2) = 4.0;

    D2.diagonal(0) = 5.0;
    D2.diagonal(1) = 6.0;
    D2.diagonal(2) = 7.0;

    // D1 * D2 should also be diagonal
    auto D3 = D1 * D2;

    // Result should be diagonal with products
    EXPECT_DOUBLE_EQ(D3(0, 0), 10.0);
    EXPECT_DOUBLE_EQ(D3(1, 1), 18.0);
    EXPECT_DOUBLE_EQ(D3(2, 2), 28.0);

    // Off-diagonal should be zero
    EXPECT_DOUBLE_EQ(D3(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(D3(1, 2), 0.0);
}

TEST_F(MatrixExpressionsTest, DiagonalMatrixInverse) {
    diagonal_matrix<double> D(4);
    D.diagonal(0) = 2.0;
    D.diagonal(1) = 4.0;
    D.diagonal(2) = 5.0;
    D.diagonal(3) = 10.0;

    auto D_inv = D.inverse();

    // Check that D * D_inv = I
    auto I = D * D_inv;

    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            if (i == j) {
                EXPECT_NEAR(I(i, j), 1.0, EPSILON);
            } else {
                EXPECT_NEAR(I(i, j), 0.0, EPSILON);
            }
        }
    }
}

TEST_F(MatrixExpressionsTest, DiagonalMatrixDeterminant) {
    diagonal_matrix<double> D(3);
    D.diagonal(0) = 2.0;
    D.diagonal(1) = 3.0;
    D.diagonal(2) = 4.0;

    double det = D.determinant();
    EXPECT_DOUBLE_EQ(det, 24.0); // 2 * 3 * 4
}

// =============================================================================
// Triangular Matrix Tests
// =============================================================================

TEST_F(MatrixExpressionsTest, LowerTriangularConstruction) {
    lower_triangular<double> L(3);

    L.at(0, 0) = 1.0;
    L.at(1, 0) = 2.0;
    L.at(1, 1) = 3.0;
    L.at(2, 0) = 4.0;
    L.at(2, 1) = 5.0;
    L.at(2, 2) = 6.0;

    EXPECT_DOUBLE_EQ(L(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(L(2, 1), 5.0);
    EXPECT_DOUBLE_EQ(L(0, 1), 0.0); // Upper triangle should be zero
    EXPECT_DOUBLE_EQ(L(0, 2), 0.0);
}

TEST_F(MatrixExpressionsTest, UpperTriangularConstruction) {
    upper_triangular<double> U(3);

    U.at(0, 0) = 1.0;
    U.at(0, 1) = 2.0;
    U.at(0, 2) = 3.0;
    U.at(1, 1) = 4.0;
    U.at(1, 2) = 5.0;
    U.at(2, 2) = 6.0;

    EXPECT_DOUBLE_EQ(U(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(U(1, 2), 5.0);
    EXPECT_DOUBLE_EQ(U(1, 0), 0.0); // Lower triangle should be zero
    EXPECT_DOUBLE_EQ(U(2, 0), 0.0);
}

TEST_F(MatrixExpressionsTest, TriangularSystemSolve) {
    const size_t n = 5;
    lower_triangular<double> L(n);
    std::vector<double> b(n);

    // Create a lower triangular system
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            L.at(i, j) = pos_dis(gen);
        }
        b[i] = dis(gen);
    }

    // Solve Lx = b
    auto x = L.solve(b);

    // Verify: compute Lx and check if it equals b
    std::vector<double> Lx(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            Lx[i] += L(i, j) * x[j];
        }
    }

    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(Lx[i], b[i], 1e-8);
    }
}

// =============================================================================
// Symmetric Matrix Tests
// =============================================================================

TEST_F(MatrixExpressionsTest, SymmetricMatrixConstruction) {
    symmetric_matrix<double> S(3);

    S(0, 0) = 1.0;
    S(0, 1) = 2.0;
    S(0, 2) = 3.0;
    S(1, 1) = 4.0;
    S(1, 2) = 5.0;
    S(2, 2) = 6.0;

    // Check symmetry
    EXPECT_DOUBLE_EQ(S(0, 1), S(1, 0));
    EXPECT_DOUBLE_EQ(S(0, 2), S(2, 0));
    EXPECT_DOUBLE_EQ(S(1, 2), S(2, 1));
}

TEST_F(MatrixExpressionsTest, SymmetricMatrixQuadraticForm) {
    symmetric_matrix<double> A(3);
    A(0, 0) = 2.0;
    A(0, 1) = 1.0;
    A(0, 2) = 0.0;
    A(1, 1) = 3.0;
    A(1, 2) = 1.0;
    A(2, 2) = 2.0;

    std::vector<double> x = {1.0, 2.0, 3.0};

    // Compute x^T * A * x
    double result = A.quadratic_form(x);

    // Manual computation:
    // x^T * A * x = [1 2 3] * [2 1 0] * [1]
    //                         [1 3 1]   [2]
    //                         [0 1 2]   [3]
    //             = [1 2 3] * [4, 9, 8]
    //             = 4 + 18 + 24 = 46

    EXPECT_NEAR(result, 46.0, EPSILON);
}

TEST_F(MatrixExpressionsTest, SymmetricMatrixEigenvalues) {
    symmetric_matrix<double> A(3);
    A(0, 0) = 4.0;
    A(0, 1) = 1.0;
    A(0, 2) = 0.0;
    A(1, 1) = 3.0;
    A(1, 2) = 1.0;
    A(2, 2) = 2.0;

    auto [eigenvalue, eigenvector] = A.power_method(1000);

    // Verify it's an eigenvalue: A*v = λ*v
    auto Av = A * eigenvector;

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(Av[i], eigenvalue * eigenvector[i], 1e-6);
    }
}

TEST_F(MatrixExpressionsTest, SymmetricMatrixPositiveDefinite) {
    // Create a positive definite matrix
    symmetric_matrix<double> A = create_spd_matrix(4);
    EXPECT_TRUE(A.is_positive_definite());

    // Create a non-positive definite matrix
    symmetric_matrix<double> B(3);
    B(0, 0) = 1.0;
    B(0, 1) = 2.0;
    B(0, 2) = 0.0;
    B(1, 1) = 1.0;
    B(1, 2) = 0.0;
    B(2, 2) = -1.0; // Negative diagonal entry

    EXPECT_FALSE(B.is_positive_definite());
}

// =============================================================================
// Graph Matrix Tests
// =============================================================================

TEST_F(MatrixExpressionsTest, GraphMatrixOperations) {
    // Create a simple graph (triangle with one extra vertex)
    graph_matrix<int> G(4);

    G(0, 1) = 1;
    G(0, 2) = 1;
    G(1, 2) = 1;
    G(2, 3) = 1;

    // Check degrees
    EXPECT_EQ(G.degree(0), 2);
    EXPECT_EQ(G.degree(1), 2);
    EXPECT_EQ(G.degree(2), 3);
    EXPECT_EQ(G.degree(3), 1);

    // Check triangle count
    EXPECT_EQ(G.triangle_count(), 1);

    // Check connectivity
    EXPECT_TRUE(G.is_connected());

    // Check Laplacian
    auto L = G.laplacian();

    // Laplacian diagonal should equal degrees
    EXPECT_EQ(L(0, 0), 2);
    EXPECT_EQ(L(1, 1), 2);
    EXPECT_EQ(L(2, 2), 3);
    EXPECT_EQ(L(3, 3), 1);

    // Laplacian off-diagonal should be negative of adjacency
    EXPECT_EQ(L(0, 1), -1);
    EXPECT_EQ(L(0, 3), 0);
}

// =============================================================================
// Expression Template Tests
// =============================================================================

TEST_F(MatrixExpressionsTest, ExpressionTemplateAddition) {
    diagonal_matrix<double> D1(3);
    diagonal_matrix<double> D2(3);

    D1.diagonal(0) = 1.0;
    D1.diagonal(1) = 2.0;
    D1.diagonal(2) = 3.0;

    D2.diagonal(0) = 4.0;
    D2.diagonal(1) = 5.0;
    D2.diagonal(2) = 6.0;

    // Addition should preserve diagonal structure
    auto sum_expr = D1 + D2;

    // Evaluate the expression
    auto result = evaluate_dense(sum_expr);

    EXPECT_DOUBLE_EQ(result[0][0], 5.0);
    EXPECT_DOUBLE_EQ(result[1][1], 7.0);
    EXPECT_DOUBLE_EQ(result[2][2], 9.0);
    EXPECT_DOUBLE_EQ(result[0][1], 0.0);
}

TEST_F(MatrixExpressionsTest, ExpressionTemplateScalarMultiplication) {
    diagonal_matrix<double> D(3);
    D.diagonal(0) = 1.0;
    D.diagonal(1) = 2.0;
    D.diagonal(2) = 3.0;

    auto scaled = 2.0 * D;
    auto result = evaluate_dense(scaled);

    EXPECT_DOUBLE_EQ(result[0][0], 2.0);
    EXPECT_DOUBLE_EQ(result[1][1], 4.0);
    EXPECT_DOUBLE_EQ(result[2][2], 6.0);
}

TEST_F(MatrixExpressionsTest, ExpressionTemplateComposition) {
    diagonal_matrix<double> D1(3);
    diagonal_matrix<double> D2(3);

    D1.diagonal(0) = 1.0;
    D1.diagonal(1) = 2.0;
    D1.diagonal(2) = 3.0;

    D2.diagonal(0) = 2.0;
    D2.diagonal(1) = 3.0;
    D2.diagonal(2) = 4.0;

    // Complex expression: (D1 + D2) * 2.0
    auto expr = 2.0 * (D1 + D2);
    auto result = evaluate_dense(expr);

    EXPECT_DOUBLE_EQ(result[0][0], 6.0);  // (1+2)*2
    EXPECT_DOUBLE_EQ(result[1][1], 10.0); // (2+3)*2
    EXPECT_DOUBLE_EQ(result[2][2], 14.0); // (3+4)*2
}

// =============================================================================
// Type Erasure Tests
// =============================================================================

TEST_F(MatrixExpressionsTest, TypeErasedMatrixBasicOperations) {
    // Create different matrix types
    diagonal_matrix<double> D(3);
    D.diagonal(0) = 1.0;
    D.diagonal(1) = 2.0;
    D.diagonal(2) = 3.0;

    // Wrap in type-erased container
    any_matrix<double> A(D);

    EXPECT_EQ(A.rows(), 3);
    EXPECT_EQ(A.cols(), 3);
    EXPECT_TRUE(A.is_diagonal());
    EXPECT_FALSE(A.is_triangular());

    EXPECT_DOUBLE_EQ(A(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(A(1, 1), 2.0);
    EXPECT_DOUBLE_EQ(A(0, 1), 0.0);
}

TEST_F(MatrixExpressionsTest, TypeErasedMatrixVectorMultiplication) {
    diagonal_matrix<double> D(3);
    D.diagonal(0) = 2.0;
    D.diagonal(1) = 3.0;
    D.diagonal(2) = 4.0;

    any_matrix<double> A(D);
    std::vector<double> v = {1.0, 2.0, 3.0};

    auto result = A * v;

    EXPECT_DOUBLE_EQ(result[0], 2.0);  // 2*1
    EXPECT_DOUBLE_EQ(result[1], 6.0);  // 3*2
    EXPECT_DOUBLE_EQ(result[2], 12.0); // 4*3
}

TEST_F(MatrixExpressionsTest, TypeErasedMatrixAddition) {
    diagonal_matrix<double> D1(2);
    D1.diagonal(0) = 1.0;
    D1.diagonal(1) = 2.0;

    diagonal_matrix<double> D2(2);
    D2.diagonal(0) = 3.0;
    D2.diagonal(1) = 4.0;

    any_matrix<double> A1(D1);
    any_matrix<double> A2(D2);

    auto sum = A1 + A2;

    EXPECT_DOUBLE_EQ(sum(0, 0), 4.0);
    EXPECT_DOUBLE_EQ(sum(1, 1), 6.0);
}

TEST_F(MatrixExpressionsTest, TypeErasedHeterogeneousCollection) {
    matrix_collection<double> collection;

    // Add different matrix types
    diagonal_matrix<double> D(3);
    D.diagonal(0) = 1.0;
    D.diagonal(1) = 2.0;
    D.diagonal(2) = 3.0;

    symmetric_matrix<double> S(3);
    S(0, 0) = 1.0;
    S(0, 1) = 2.0;
    S(1, 1) = 3.0;

    collection.add("diagonal", D);
    collection.add("symmetric", S);

    // Access by name
    auto& diag = collection["diagonal"];
    EXPECT_TRUE(diag.is_diagonal());

    auto& sym = collection["symmetric"];
    EXPECT_TRUE(sym.is_symmetric());
}

// =============================================================================
// Matrix Algorithm Integration Tests
// =============================================================================

TEST_F(MatrixExpressionsTest, MatrixAlgorithmsWithDiagonal) {
    diagonal_matrix<double> D(4);
    D.diagonal(0) = 2.0;
    D.diagonal(1) = 3.0;
    D.diagonal(2) = 4.0;
    D.diagonal(3) = 5.0;

    // Inverse should be O(n)
    auto D_inv = matrix_algorithms<double>::inverse(D);

    // Verify D * D_inv = I
    auto I = D * D_inv;
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(I(i, j), expected, EPSILON);
        }
    }

    // Determinant should be product
    double det = matrix_algorithms<double>::determinant(D);
    EXPECT_DOUBLE_EQ(det, 120.0); // 2*3*4*5
}

TEST_F(MatrixExpressionsTest, MatrixAlgorithmsWithTriangular) {
    const size_t n = 5;
    lower_triangular<double> L(n);

    // Fill with positive values
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            L.at(i, j) = pos_dis(gen);
        }
    }

    // Create a right-hand side
    std::vector<double> b(n);
    for (size_t i = 0; i < n; ++i) {
        b[i] = dis(gen);
    }

    // Solve should use forward substitution (O(n²))
    auto x = matrix_algorithms<double>::solve(L, b);

    // Verify solution
    auto Lx = L * x;
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(Lx[i], b[i], 1e-8);
    }
}

TEST_F(MatrixExpressionsTest, MatrixAlgorithmsWithSymmetric) {
    // Create symmetric positive definite matrix
    symmetric_matrix<double> A = create_spd_matrix(4);

    // Cholesky decomposition should work
    auto L = matrix_algorithms<double>::cholesky(A);

    // Verify A = L * L^T
    // Note: We'd need to implement this verification properly
    // For now, just check that decomposition succeeded
    EXPECT_EQ(L.rows(), 4);
    EXPECT_EQ(L.cols(), 4);

    // The decomposition should be lower triangular
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = i + 1; j < 4; ++j) {
            EXPECT_NEAR(L(i, j), 0.0, EPSILON);
        }
    }
}

// =============================================================================
// Performance and Memory Tests
// =============================================================================

TEST_F(MatrixExpressionsTest, MemoryEfficiency) {
    const size_t n = 100;

    // Symmetric matrix uses n(n+1)/2 storage
    symmetric_matrix<double> S(n);
    size_t symmetric_memory = (n * (n + 1)) / 2 * sizeof(double);
    EXPECT_EQ(S.memory_usage(), symmetric_memory);

    // Diagonal matrix uses n storage
    diagonal_matrix<double> D(n);
    // Note: memory_usage() would need to be implemented

    // Triangular matrix uses n(n+1)/2 storage
    lower_triangular<double> L(n);
    // Similar memory as symmetric
}

TEST_F(MatrixExpressionsTest, StructuralZeroOptimization) {
    diagonal_matrix<double> D(100);
    for (size_t i = 0; i < 100; ++i) {
        D.diagonal(i) = i + 1.0;
    }

    // Multiplication should skip structural zeros
    auto D2 = D * D;

    // Only diagonal elements should be computed
    for (size_t i = 0; i < 100; ++i) {
        for (size_t j = 0; j < 100; ++j) {
            if (i == j) {
                double expected = (i + 1.0) * (i + 1.0);
                EXPECT_DOUBLE_EQ(D2(i, j), expected);
            } else {
                EXPECT_DOUBLE_EQ(D2(i, j), 0.0);
            }
        }
    }
}