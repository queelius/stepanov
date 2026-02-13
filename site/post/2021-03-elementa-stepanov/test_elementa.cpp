// test_elementa.cpp - Tests for elementa linear algebra library
//
// Uses Google Test and verifies all matrix operations.

#include <gtest/gtest.h>
#include <elementa.hpp>
#include <cmath>

using namespace elementa;

// =============================================================================
// Matrix Construction Tests
// =============================================================================

TEST(MatrixConstruction, DefaultCreatesEmpty) {
    matrix<double> m;
    EXPECT_EQ(m.rows(), 0);
    EXPECT_EQ(m.cols(), 0);
    EXPECT_TRUE(m.empty());
}

TEST(MatrixConstruction, SizeWithDefault) {
    matrix<double> m(3, 4);
    EXPECT_EQ(m.rows(), 3);
    EXPECT_EQ(m.cols(), 4);
    EXPECT_EQ(m.size(), 12);

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            EXPECT_EQ(m(i, j), 0.0);
        }
    }
}

TEST(MatrixConstruction, SizeWithFill) {
    matrix<double> m(2, 2, 5.0);
    EXPECT_EQ(m(0, 0), 5.0);
    EXPECT_EQ(m(0, 1), 5.0);
    EXPECT_EQ(m(1, 0), 5.0);
    EXPECT_EQ(m(1, 1), 5.0);
}

TEST(MatrixConstruction, FlatInitializerList) {
    matrix<double> m(2, 3, {1, 2, 3, 4, 5, 6});
    EXPECT_EQ(m(0, 0), 1);
    EXPECT_EQ(m(0, 1), 2);
    EXPECT_EQ(m(0, 2), 3);
    EXPECT_EQ(m(1, 0), 4);
    EXPECT_EQ(m(1, 1), 5);
    EXPECT_EQ(m(1, 2), 6);
}

TEST(MatrixConstruction, NestedInitializerList) {
    matrix<double> m{{1, 2, 3}, {4, 5, 6}};
    EXPECT_EQ(m.rows(), 2);
    EXPECT_EQ(m.cols(), 3);
    EXPECT_EQ(m(0, 0), 1);
    EXPECT_EQ(m(1, 2), 6);
}

TEST(MatrixConstruction, CopyAndMove) {
    matrix<double> m{{1, 2}, {3, 4}};

    // Copy
    matrix<double> m2 = m;
    EXPECT_EQ(m2(0, 0), 1);
    EXPECT_EQ(m2(1, 1), 4);

    // Modify copy doesn't affect original
    m2(0, 0) = 99;
    EXPECT_EQ(m(0, 0), 1);
    EXPECT_EQ(m2(0, 0), 99);

    // Move
    matrix<double> m3 = std::move(m2);
    EXPECT_EQ(m3(0, 0), 99);
}

// =============================================================================
// Factory Functions Tests
// =============================================================================

TEST(FactoryFunctions, Eye) {
    auto I = eye<double>(3);
    EXPECT_EQ(I.rows(), 3);
    EXPECT_EQ(I.cols(), 3);
    EXPECT_EQ(I(0, 0), 1);
    EXPECT_EQ(I(1, 1), 1);
    EXPECT_EQ(I(2, 2), 1);
    EXPECT_EQ(I(0, 1), 0);
    EXPECT_EQ(I(1, 0), 0);
}

TEST(FactoryFunctions, Zeros) {
    auto Z = zeros<double>(2, 3);
    EXPECT_EQ(Z.rows(), 2);
    EXPECT_EQ(Z.cols(), 3);
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(Z(i, j), 0);
        }
    }
}

TEST(FactoryFunctions, Ones) {
    auto O = ones<double>(2, 2);
    EXPECT_EQ(O(0, 0), 1);
    EXPECT_EQ(O(1, 1), 1);
}

TEST(FactoryFunctions, DiagCreate) {
    auto D = diag(std::vector<double>{1, 2, 3});
    EXPECT_EQ(D.rows(), 3);
    EXPECT_EQ(D.cols(), 3);
    EXPECT_EQ(D(0, 0), 1);
    EXPECT_EQ(D(1, 1), 2);
    EXPECT_EQ(D(2, 2), 3);
    EXPECT_EQ(D(0, 1), 0);
}

TEST(FactoryFunctions, DiagExtract) {
    matrix<double> A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto d = diag(A);
    EXPECT_EQ(d.size(), 3);
    EXPECT_EQ(d[0], 1);
    EXPECT_EQ(d[1], 5);
    EXPECT_EQ(d[2], 9);
}

// =============================================================================
// Arithmetic Operations Tests
// =============================================================================

TEST(MatrixArithmetic, Addition) {
    matrix<double> A{{1, 2}, {3, 4}};
    matrix<double> B{{5, 6}, {7, 8}};
    auto C = A + B;
    EXPECT_EQ(C(0, 0), 6);
    EXPECT_EQ(C(0, 1), 8);
    EXPECT_EQ(C(1, 0), 10);
    EXPECT_EQ(C(1, 1), 12);
}

TEST(MatrixArithmetic, Subtraction) {
    matrix<double> A{{1, 2}, {3, 4}};
    matrix<double> B{{5, 6}, {7, 8}};
    auto C = A - B;
    EXPECT_EQ(C(0, 0), -4);
    EXPECT_EQ(C(0, 1), -4);
    EXPECT_EQ(C(1, 0), -4);
    EXPECT_EQ(C(1, 1), -4);
}

TEST(MatrixArithmetic, UnaryNegation) {
    matrix<double> A{{1, 2}, {3, 4}};
    auto C = -A;
    EXPECT_EQ(C(0, 0), -1);
    EXPECT_EQ(C(1, 1), -4);
}

TEST(MatrixArithmetic, ScalarMultRight) {
    matrix<double> A{{1, 2}, {3, 4}};
    auto C = A * 2.0;
    EXPECT_EQ(C(0, 0), 2);
    EXPECT_EQ(C(1, 1), 8);
}

TEST(MatrixArithmetic, ScalarMultLeft) {
    matrix<double> A{{1, 2}, {3, 4}};
    auto C = 2.0 * A;
    EXPECT_EQ(C(0, 0), 2);
    EXPECT_EQ(C(1, 1), 8);
}

TEST(MatrixArithmetic, ScalarDivision) {
    matrix<double> A{{1, 2}, {3, 4}};
    auto C = A / 2.0;
    EXPECT_EQ(C(0, 0), 0.5);
    EXPECT_EQ(C(1, 1), 2.0);
}

TEST(MatrixArithmetic, CompoundAddAssign) {
    matrix<double> A{{1, 2}, {3, 4}};
    matrix<double> B{{5, 6}, {7, 8}};
    A += B;
    EXPECT_EQ(A(0, 0), 6);
    EXPECT_EQ(A(1, 1), 12);
}

TEST(MatrixArithmetic, CompoundSubAssign) {
    matrix<double> A{{1, 2}, {3, 4}};
    matrix<double> B{{5, 6}, {7, 8}};
    A -= B;
    EXPECT_EQ(A(0, 0), -4);
}

TEST(MatrixArithmetic, CompoundMulAssign) {
    matrix<double> A{{1, 2}, {3, 4}};
    A *= 3.0;
    EXPECT_EQ(A(0, 0), 3);
    EXPECT_EQ(A(1, 1), 12);
}

// =============================================================================
// Matrix Multiplication Tests
// =============================================================================

TEST(MatrixMul, TwoByTwo) {
    matrix<double> A{{1, 2}, {3, 4}};
    matrix<double> B{{5, 6}, {7, 8}};
    auto C = matmul(A, B);
    EXPECT_EQ(C(0, 0), 19);
    EXPECT_EQ(C(0, 1), 22);
    EXPECT_EQ(C(1, 0), 43);
    EXPECT_EQ(C(1, 1), 50);
}

TEST(MatrixMul, TwoByThreeTimesThreeByTwo) {
    matrix<double> A{{1, 2, 3}, {4, 5, 6}};
    matrix<double> B{{7, 8}, {9, 10}, {11, 12}};
    auto C = matmul(A, B);
    EXPECT_EQ(C.rows(), 2);
    EXPECT_EQ(C.cols(), 2);
    EXPECT_EQ(C(0, 0), 58);
}

TEST(MatrixMul, OperatorStar) {
    matrix<double> A{{1, 2}, {3, 4}};
    matrix<double> B{{5, 6}, {7, 8}};
    auto C = A * B;
    EXPECT_EQ(C(0, 0), 19);
}

TEST(MatrixMul, Identity) {
    matrix<double> A{{1, 2}, {3, 4}};
    auto I = eye<double>(2);
    auto C = A * I;
    EXPECT_TRUE(C == A);
    auto D = I * A;
    EXPECT_TRUE(D == A);
}

// =============================================================================
// Transpose Tests
// =============================================================================

TEST(Transpose, Square) {
    matrix<double> A{{1, 2}, {3, 4}};
    auto At = transpose(A);
    EXPECT_EQ(At(0, 0), 1);
    EXPECT_EQ(At(0, 1), 3);
    EXPECT_EQ(At(1, 0), 2);
    EXPECT_EQ(At(1, 1), 4);
}

TEST(Transpose, Rectangular) {
    matrix<double> A{{1, 2, 3}, {4, 5, 6}};
    auto At = transpose(A);
    EXPECT_EQ(At.rows(), 3);
    EXPECT_EQ(At.cols(), 2);
    EXPECT_EQ(At(0, 0), 1);
    EXPECT_EQ(At(2, 1), 6);
}

TEST(Transpose, DoubleIsIdentity) {
    matrix<double> A{{1, 2}, {3, 4}};
    auto Att = transpose(transpose(A));
    EXPECT_TRUE(Att == A);
}

// =============================================================================
// Reduction Tests
// =============================================================================

TEST(Reductions, Trace) {
    matrix<double> A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    EXPECT_EQ(trace(A), 15);
}

TEST(Reductions, Sum) {
    matrix<double> A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    EXPECT_EQ(sum(A), 45);
}

TEST(Reductions, Mean) {
    matrix<double> A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    EXPECT_EQ(mean(A), 5.0);
}

// =============================================================================
// Norm Tests
// =============================================================================

TEST(Norms, Frobenius) {
    matrix<double> A{{3, 4}, {0, 0}};
    EXPECT_NEAR(frobenius_norm(A), 5.0, 1e-10);
}

TEST(Norms, L1) {
    matrix<double> B{{1, 2}, {3, 4}};
    EXPECT_NEAR(l1_norm(B), 6.0, 1e-10);
}

TEST(Norms, LInf) {
    matrix<double> B{{1, 2}, {3, 4}};
    EXPECT_NEAR(linf_norm(B), 7.0, 1e-10);
}

// =============================================================================
// LU Decomposition Tests
// =============================================================================

TEST(LU, Simple2x2) {
    matrix<double> A{{4, 3}, {6, 3}};
    auto [L, U, perm, sign, singular] = lu(A);
    EXPECT_FALSE(singular);

    // Verify P * A = L * U
    matrix<double> PA(2, 2);
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            PA(i, j) = A(perm[i], j);
        }
    }
    auto LU = L * U;
    EXPECT_TRUE(approx_equal(PA, LU));
}

TEST(LU, ThreeByThreeWithPivoting) {
    matrix<double> A{{2, 1, 1}, {4, 3, 3}, {8, 7, 9}};
    auto [L, U, perm, sign, singular] = lu(A);
    EXPECT_FALSE(singular);

    matrix<double> PA(3, 3);
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            PA(i, j) = A(perm[i], j);
        }
    }
    auto LU = L * U;
    EXPECT_TRUE(approx_equal(PA, LU, 1e-10, 1e-10));
}

TEST(LU, SingularDetection) {
    matrix<double> A{{1, 2}, {2, 4}};
    auto [L, U, perm, sign, singular] = lu(A);
    EXPECT_TRUE(singular);
}

// =============================================================================
// Determinant Tests
// =============================================================================

TEST(Determinant, TwoByTwo) {
    matrix<double> A{{3, 8}, {4, 6}};
    EXPECT_NEAR(det(A), -14.0, 1e-10);
}

TEST(Determinant, ThreeByThree) {
    matrix<double> A{{6, 1, 1}, {4, -2, 5}, {2, 8, 7}};
    EXPECT_NEAR(det(A), -306.0, 1e-10);
}

TEST(Determinant, IdentityIsOne) {
    auto I = eye<double>(4);
    EXPECT_NEAR(det(I), 1.0, 1e-10);
}

TEST(Determinant, SingularIsZero) {
    matrix<double> A{{1, 2}, {2, 4}};
    EXPECT_NEAR(det(A), 0.0, 1e-10);
}

// =============================================================================
// Log Determinant Tests
// =============================================================================

TEST(LogDet, PositiveDefinite) {
    matrix<double> A{{4, 2}, {2, 3}};
    auto d = det(A);
    auto [sign, logabs] = logdet(A);
    EXPECT_EQ(sign, 1);
    EXPECT_NEAR(std::exp(logabs), std::abs(d), 1e-10);
}

TEST(LogDet, NegativeDeterminant) {
    matrix<double> A{{1, 2}, {3, 4}};
    auto [sign, logabs] = logdet(A);
    EXPECT_EQ(sign, -1);
    EXPECT_NEAR(std::exp(logabs), 2.0, 1e-10);
}

// =============================================================================
// Linear System Solve Tests
// =============================================================================

TEST(Solve, TwoByTwo) {
    matrix<double> A{{2, 1}, {1, 3}};
    matrix<double> b{{5}, {5}};
    auto x = solve(A, b);
    EXPECT_NEAR(x(0, 0), 2.0, 1e-10);
    EXPECT_NEAR(x(1, 0), 1.0, 1e-10);
}

TEST(Solve, ThreeByThree) {
    matrix<double> A{{1, 2, 3}, {4, 5, 6}, {7, 8, 10}};
    matrix<double> b{{14}, {32}, {50}};
    auto x = solve(A, b);
    auto Ax = A * x;
    EXPECT_TRUE(approx_equal(Ax, b));
}

TEST(Solve, MultipleRHS) {
    matrix<double> A{{2, 1}, {1, 3}};
    matrix<double> B{{5, 3}, {5, 7}};
    auto X = solve(A, B);
    auto AX = A * X;
    EXPECT_TRUE(approx_equal(AX, B));
}

// =============================================================================
// Matrix Inverse Tests
// =============================================================================

TEST(Inverse, TwoByTwo) {
    matrix<double> A{{4, 7}, {2, 6}};
    auto Ainv = inverse(A);
    auto I = A * Ainv;
    auto eye2 = eye<double>(2);
    EXPECT_TRUE(approx_equal(I, eye2));
    auto I2 = Ainv * A;
    EXPECT_TRUE(approx_equal(I2, eye2));
}

TEST(Inverse, ThreeByThree) {
    matrix<double> A{{1, 2, 3}, {0, 1, 4}, {5, 6, 0}};
    auto Ainv = inverse(A);
    auto I = A * Ainv;
    auto eye3 = eye<double>(3);
    EXPECT_TRUE(approx_equal(I, eye3));
}

// =============================================================================
// Element-wise Function Tests
// =============================================================================

TEST(Elementwise, Sqrt) {
    matrix<double> A{{1, 4}, {9, 16}};
    auto B = sqrt(A);
    EXPECT_NEAR(B(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(B(0, 1), 2.0, 1e-10);
    EXPECT_NEAR(B(1, 0), 3.0, 1e-10);
    EXPECT_NEAR(B(1, 1), 4.0, 1e-10);
}

TEST(Elementwise, ExpLog) {
    matrix<double> B{{1, 2}};
    auto expB = exp(B);
    auto logExpB = log(expB);
    EXPECT_TRUE(approx_equal(logExpB, B));
}

TEST(Elementwise, Pow) {
    matrix<double> A{{1, 4}, {9, 16}};
    auto B = pow(A, 0.5);
    EXPECT_NEAR(B(0, 1), 2.0, 1e-10);
}

TEST(Elementwise, Abs) {
    matrix<double> B{{-1, 2}, {-3, 4}};
    auto C = abs(B);
    EXPECT_EQ(C(0, 0), 1);
    EXPECT_EQ(C(1, 0), 3);
}

// =============================================================================
// Hadamard Product Tests
// =============================================================================

TEST(Hadamard, Basic) {
    matrix<double> A{{1, 2}, {3, 4}};
    matrix<double> B{{5, 6}, {7, 8}};
    auto C = hadamard(A, B);
    EXPECT_EQ(C(0, 0), 5);
    EXPECT_EQ(C(0, 1), 12);
    EXPECT_EQ(C(1, 0), 21);
    EXPECT_EQ(C(1, 1), 32);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(EdgeCases, OneByOne) {
    matrix<double> A{{5}};
    matrix<double> B{{3}};
    EXPECT_EQ((A + B)(0, 0), 8);
    EXPECT_EQ((A * B)(0, 0), 15);
    EXPECT_EQ(det(A), 5);
    EXPECT_NEAR(inverse(A)(0, 0), 0.2, 1e-10);
    EXPECT_EQ(trace(A), 5);
}

TEST(EdgeCases, LargeIdentityInverse) {
    auto I = eye<double>(10);
    auto Iinv = inverse(I);
    EXPECT_TRUE(approx_equal(I, Iinv));
}

// =============================================================================
// Concept Verification
// =============================================================================

TEST(Concept, Satisfaction) {
    static_assert(Matrix<matrix<double>>);
    static_assert(Matrix<matrix<float>>);
    static_assert(Matrix<matrix<int>>);

    auto generic_trace = []<Matrix M>(const M& m) {
        return trace(m);
    };

    matrix<double> A{{1, 2}, {3, 4}};
    EXPECT_EQ(generic_trace(A), 5);
}

// =============================================================================
// String Representation Tests
// =============================================================================

TEST(IO, ToString) {
    matrix<double> A{{1, 2}, {3, 4}};
    auto s = to_string(A);
    EXPECT_NE(s.find("1.0000"), std::string::npos);
    EXPECT_NE(s.find("4.0000"), std::string::npos);
}
