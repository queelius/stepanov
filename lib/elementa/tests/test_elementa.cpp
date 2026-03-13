// test_elementa.cpp - Tests for elementa linear algebra library
//
// Uses Catch2 for testing and verifies all matrix operations.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <elementa.hpp>
#include <cmath>

using namespace elementa;
using Catch::Approx;

// =============================================================================
// Matrix Construction Tests
// =============================================================================

TEST_CASE("Matrix construction", "[matrix][construction]") {
    SECTION("Default constructor creates empty matrix") {
        matrix<double> m;
        REQUIRE(m.rows() == 0);
        REQUIRE(m.cols() == 0);
        REQUIRE(m.empty());
    }

    SECTION("Size constructor with default value") {
        matrix<double> m(3, 4);
        REQUIRE(m.rows() == 3);
        REQUIRE(m.cols() == 4);
        REQUIRE(m.size() == 12);

        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                REQUIRE(m(i, j) == 0.0);
            }
        }
    }

    SECTION("Size constructor with fill value") {
        matrix<double> m(2, 2, 5.0);
        REQUIRE(m(0, 0) == 5.0);
        REQUIRE(m(0, 1) == 5.0);
        REQUIRE(m(1, 0) == 5.0);
        REQUIRE(m(1, 1) == 5.0);
    }

    SECTION("Initializer list constructor (flat)") {
        matrix<double> m(2, 3, {1, 2, 3, 4, 5, 6});
        REQUIRE(m(0, 0) == 1);
        REQUIRE(m(0, 1) == 2);
        REQUIRE(m(0, 2) == 3);
        REQUIRE(m(1, 0) == 4);
        REQUIRE(m(1, 1) == 5);
        REQUIRE(m(1, 2) == 6);
    }

    SECTION("Nested initializer list constructor") {
        matrix<double> m{{1, 2, 3}, {4, 5, 6}};
        REQUIRE(m.rows() == 2);
        REQUIRE(m.cols() == 3);
        REQUIRE(m(0, 0) == 1);
        REQUIRE(m(1, 2) == 6);
    }

    SECTION("Copy and move semantics") {
        matrix<double> m{{1, 2}, {3, 4}};

        // Copy
        matrix<double> m2 = m;
        REQUIRE(m2(0, 0) == 1);
        REQUIRE(m2(1, 1) == 4);

        // Modify copy doesn't affect original
        m2(0, 0) = 99;
        REQUIRE(m(0, 0) == 1);
        REQUIRE(m2(0, 0) == 99);

        // Move
        matrix<double> m3 = std::move(m2);
        REQUIRE(m3(0, 0) == 99);
    }
}

// =============================================================================
// Factory Functions Tests
// =============================================================================

TEST_CASE("Factory functions", "[matrix][factory]") {
    SECTION("eye creates identity matrix") {
        auto I = eye<double>(3);
        REQUIRE(I.rows() == 3);
        REQUIRE(I.cols() == 3);
        REQUIRE(I(0, 0) == 1);
        REQUIRE(I(1, 1) == 1);
        REQUIRE(I(2, 2) == 1);
        REQUIRE(I(0, 1) == 0);
        REQUIRE(I(1, 0) == 0);
    }

    SECTION("zeros creates zero matrix") {
        auto Z = zeros<double>(2, 3);
        REQUIRE(Z.rows() == 2);
        REQUIRE(Z.cols() == 3);
        for (std::size_t i = 0; i < 2; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                REQUIRE(Z(i, j) == 0);
            }
        }
    }

    SECTION("ones creates ones matrix") {
        auto O = ones<double>(2, 2);
        REQUIRE(O(0, 0) == 1);
        REQUIRE(O(1, 1) == 1);
    }

    SECTION("diag creates diagonal matrix") {
        auto D = diag(std::vector<double>{1, 2, 3});
        REQUIRE(D.rows() == 3);
        REQUIRE(D.cols() == 3);
        REQUIRE(D(0, 0) == 1);
        REQUIRE(D(1, 1) == 2);
        REQUIRE(D(2, 2) == 3);
        REQUIRE(D(0, 1) == 0);
    }

    SECTION("diag extracts diagonal") {
        matrix<double> A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        auto d = diag(A);
        REQUIRE(d.size() == 3);
        REQUIRE(d[0] == 1);
        REQUIRE(d[1] == 5);
        REQUIRE(d[2] == 9);
    }
}

// =============================================================================
// Arithmetic Operations Tests
// =============================================================================

TEST_CASE("Matrix arithmetic", "[matrix][arithmetic]") {
    matrix<double> A{{1, 2}, {3, 4}};
    matrix<double> B{{5, 6}, {7, 8}};

    SECTION("Addition") {
        auto C = A + B;
        REQUIRE(C(0, 0) == 6);
        REQUIRE(C(0, 1) == 8);
        REQUIRE(C(1, 0) == 10);
        REQUIRE(C(1, 1) == 12);
    }

    SECTION("Subtraction") {
        auto C = A - B;
        REQUIRE(C(0, 0) == -4);
        REQUIRE(C(0, 1) == -4);
        REQUIRE(C(1, 0) == -4);
        REQUIRE(C(1, 1) == -4);
    }

    SECTION("Unary negation") {
        auto C = -A;
        REQUIRE(C(0, 0) == -1);
        REQUIRE(C(1, 1) == -4);
    }

    SECTION("Scalar multiplication (right)") {
        auto C = A * 2.0;
        REQUIRE(C(0, 0) == 2);
        REQUIRE(C(1, 1) == 8);
    }

    SECTION("Scalar multiplication (left)") {
        auto C = 2.0 * A;
        REQUIRE(C(0, 0) == 2);
        REQUIRE(C(1, 1) == 8);
    }

    SECTION("Scalar division") {
        auto C = A / 2.0;
        REQUIRE(C(0, 0) == 0.5);
        REQUIRE(C(1, 1) == 2.0);
    }

    SECTION("Compound assignment +=") {
        auto C = A;
        C += B;
        REQUIRE(C(0, 0) == 6);
        REQUIRE(C(1, 1) == 12);
    }

    SECTION("Compound assignment -=") {
        auto C = A;
        C -= B;
        REQUIRE(C(0, 0) == -4);
    }

    SECTION("Compound assignment *=") {
        auto C = A;
        C *= 3.0;
        REQUIRE(C(0, 0) == 3);
        REQUIRE(C(1, 1) == 12);
    }
}

// =============================================================================
// Matrix Multiplication Tests
// =============================================================================

TEST_CASE("Matrix multiplication", "[matrix][matmul]") {
    SECTION("2x2 * 2x2") {
        matrix<double> A{{1, 2}, {3, 4}};
        matrix<double> B{{5, 6}, {7, 8}};

        auto C = matmul(A, B);
        // C = [1*5+2*7, 1*6+2*8]   = [19, 22]
        //     [3*5+4*7, 3*6+4*8]   = [43, 50]
        REQUIRE(C(0, 0) == 19);
        REQUIRE(C(0, 1) == 22);
        REQUIRE(C(1, 0) == 43);
        REQUIRE(C(1, 1) == 50);
    }

    SECTION("2x3 * 3x2") {
        matrix<double> A{{1, 2, 3}, {4, 5, 6}};
        matrix<double> B{{7, 8}, {9, 10}, {11, 12}};

        auto C = matmul(A, B);
        REQUIRE(C.rows() == 2);
        REQUIRE(C.cols() == 2);
        // C(0,0) = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
        REQUIRE(C(0, 0) == 58);
    }

    SECTION("Operator * for matmul") {
        matrix<double> A{{1, 2}, {3, 4}};
        matrix<double> B{{5, 6}, {7, 8}};

        auto C = A * B;
        REQUIRE(C(0, 0) == 19);
    }

    SECTION("Identity multiplication") {
        matrix<double> A{{1, 2}, {3, 4}};
        auto I = eye<double>(2);

        auto C = A * I;
        REQUIRE(C == A);

        auto D = I * A;
        REQUIRE(D == A);
    }
}

// =============================================================================
// Transpose Tests
// =============================================================================

TEST_CASE("Matrix transpose", "[matrix][transpose]") {
    SECTION("Square matrix") {
        matrix<double> A{{1, 2}, {3, 4}};
        auto At = transpose(A);

        REQUIRE(At(0, 0) == 1);
        REQUIRE(At(0, 1) == 3);
        REQUIRE(At(1, 0) == 2);
        REQUIRE(At(1, 1) == 4);
    }

    SECTION("Rectangular matrix") {
        matrix<double> A{{1, 2, 3}, {4, 5, 6}};
        auto At = transpose(A);

        REQUIRE(At.rows() == 3);
        REQUIRE(At.cols() == 2);
        REQUIRE(At(0, 0) == 1);
        REQUIRE(At(2, 1) == 6);
    }

    SECTION("Double transpose is identity") {
        matrix<double> A{{1, 2}, {3, 4}};
        auto Att = transpose(transpose(A));
        REQUIRE(Att == A);
    }
}

// =============================================================================
// Reduction Tests
// =============================================================================

TEST_CASE("Matrix reductions", "[matrix][reduction]") {
    matrix<double> A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    SECTION("Trace") {
        REQUIRE(trace(A) == 15);  // 1 + 5 + 9
    }

    SECTION("Sum") {
        REQUIRE(sum(A) == 45);  // 1+2+...+9
    }

    SECTION("Mean") {
        REQUIRE(mean(A) == 5.0);
    }
}

// =============================================================================
// Norm Tests
// =============================================================================

TEST_CASE("Matrix norms", "[matrix][norms]") {
    matrix<double> A{{3, 4}, {0, 0}};

    SECTION("Frobenius norm") {
        // sqrt(9 + 16 + 0 + 0) = 5
        REQUIRE(frobenius_norm(A) == Approx(5.0));
    }

    SECTION("L1 norm (max column sum)") {
        matrix<double> B{{1, 2}, {3, 4}};
        // Column 0: |1| + |3| = 4
        // Column 1: |2| + |4| = 6
        REQUIRE(l1_norm(B) == Approx(6.0));
    }

    SECTION("L-infinity norm (max row sum)") {
        matrix<double> B{{1, 2}, {3, 4}};
        // Row 0: |1| + |2| = 3
        // Row 1: |3| + |4| = 7
        REQUIRE(linf_norm(B) == Approx(7.0));
    }
}

// =============================================================================
// LU Decomposition Tests
// =============================================================================

TEST_CASE("LU decomposition", "[matrix][lu]") {
    SECTION("Simple 2x2") {
        matrix<double> A{{4, 3}, {6, 3}};
        auto [L, U, perm, sign, singular] = lu(A);

        REQUIRE_FALSE(singular);

        // Verify P * A = L * U
        // First reconstruct P * A
        matrix<double> PA(2, 2);
        for (std::size_t i = 0; i < 2; ++i) {
            for (std::size_t j = 0; j < 2; ++j) {
                PA(i, j) = A(perm[i], j);
            }
        }

        auto LU = L * U;
        REQUIRE(approx_equal(PA, LU));
    }

    SECTION("3x3 with pivoting") {
        matrix<double> A{{2, 1, 1}, {4, 3, 3}, {8, 7, 9}};
        auto [L, U, perm, sign, singular] = lu(A);

        REQUIRE_FALSE(singular);

        // Verify decomposition
        matrix<double> PA(3, 3);
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                PA(i, j) = A(perm[i], j);
            }
        }

        auto LU = L * U;
        REQUIRE(approx_equal(PA, LU, 1e-10, 1e-10));
    }

    SECTION("Singular matrix detection") {
        matrix<double> A{{1, 2}, {2, 4}};  // Rank 1
        auto [L, U, perm, sign, singular] = lu(A);
        REQUIRE(singular);
    }
}

// =============================================================================
// Determinant Tests
// =============================================================================

TEST_CASE("Determinant", "[matrix][det]") {
    SECTION("2x2 matrix") {
        matrix<double> A{{3, 8}, {4, 6}};
        // det = 3*6 - 8*4 = 18 - 32 = -14
        REQUIRE(det(A) == Approx(-14.0));
    }

    SECTION("3x3 matrix") {
        matrix<double> A{{6, 1, 1}, {4, -2, 5}, {2, 8, 7}};
        // det = 6*(-2*7 - 5*8) - 1*(4*7 - 5*2) + 1*(4*8 - (-2)*2)
        //     = 6*(-14-40) - 1*(28-10) + 1*(32+4)
        //     = 6*(-54) - 18 + 36
        //     = -324 - 18 + 36 = -306
        REQUIRE(det(A) == Approx(-306.0));
    }

    SECTION("Identity matrix has det = 1") {
        auto I = eye<double>(4);
        REQUIRE(det(I) == Approx(1.0));
    }

    SECTION("Singular matrix has det = 0") {
        matrix<double> A{{1, 2}, {2, 4}};
        REQUIRE(det(A) == Approx(0.0).margin(1e-10));
    }
}

// =============================================================================
// Log Determinant Tests
// =============================================================================

TEST_CASE("Log determinant", "[matrix][logdet]") {
    SECTION("Positive definite matrix") {
        matrix<double> A{{4, 2}, {2, 3}};  // Positive definite
        auto d = det(A);
        auto [sign, logabs] = logdet(A);

        REQUIRE(sign == 1);
        REQUIRE(std::exp(logabs) == Approx(std::abs(d)));
    }

    SECTION("Matrix with negative determinant") {
        matrix<double> A{{1, 2}, {3, 4}};  // det = -2
        auto [sign, logabs] = logdet(A);

        REQUIRE(sign == -1);
        REQUIRE(std::exp(logabs) == Approx(2.0));
    }
}

// =============================================================================
// Linear System Solve Tests
// =============================================================================

TEST_CASE("Solve linear system", "[matrix][solve]") {
    SECTION("2x2 system") {
        // 2x + y = 5
        // x + 3y = 5
        // Solution: x = 2, y = 1
        matrix<double> A{{2, 1}, {1, 3}};
        matrix<double> b{{5}, {5}};

        auto x = solve(A, b);
        REQUIRE(x(0, 0) == Approx(2.0));
        REQUIRE(x(1, 0) == Approx(1.0));
    }

    SECTION("3x3 system") {
        matrix<double> A{{1, 2, 3}, {4, 5, 6}, {7, 8, 10}};
        matrix<double> b{{14}, {32}, {50}};

        auto x = solve(A, b);

        // Verify A * x = b
        auto Ax = A * x;
        REQUIRE(approx_equal(Ax, b));
    }

    SECTION("Multiple right-hand sides") {
        matrix<double> A{{2, 1}, {1, 3}};
        matrix<double> B{{5, 3}, {5, 7}};

        auto X = solve(A, B);
        auto AX = A * X;
        REQUIRE(approx_equal(AX, B));
    }
}

// =============================================================================
// Matrix Inverse Tests
// =============================================================================

TEST_CASE("Matrix inverse", "[matrix][inverse]") {
    SECTION("2x2 inverse") {
        matrix<double> A{{4, 7}, {2, 6}};
        auto Ainv = inverse(A);

        // A * A^(-1) = I
        auto I = A * Ainv;
        auto eye2 = eye<double>(2);
        REQUIRE(approx_equal(I, eye2));

        // A^(-1) * A = I
        auto I2 = Ainv * A;
        REQUIRE(approx_equal(I2, eye2));
    }

    SECTION("3x3 inverse") {
        matrix<double> A{{1, 2, 3}, {0, 1, 4}, {5, 6, 0}};
        auto Ainv = inverse(A);

        auto I = A * Ainv;
        auto eye3 = eye<double>(3);
        REQUIRE(approx_equal(I, eye3));
    }
}

// =============================================================================
// Element-wise Function Tests
// =============================================================================

TEST_CASE("Element-wise functions", "[matrix][elementwise]") {
    matrix<double> A{{1, 4}, {9, 16}};

    SECTION("sqrt") {
        auto B = sqrt(A);
        REQUIRE(B(0, 0) == Approx(1.0));
        REQUIRE(B(0, 1) == Approx(2.0));
        REQUIRE(B(1, 0) == Approx(3.0));
        REQUIRE(B(1, 1) == Approx(4.0));
    }

    SECTION("exp and log") {
        matrix<double> B{{1, 2}};
        auto expB = exp(B);
        auto logExpB = log(expB);
        REQUIRE(approx_equal(logExpB, B));
    }

    SECTION("pow") {
        auto B = pow(A, 0.5);
        REQUIRE(B(0, 1) == Approx(2.0));
    }

    SECTION("abs") {
        matrix<double> B{{-1, 2}, {-3, 4}};
        auto C = abs(B);
        REQUIRE(C(0, 0) == 1);
        REQUIRE(C(1, 0) == 3);
    }
}

// =============================================================================
// Hadamard Product Tests
// =============================================================================

TEST_CASE("Hadamard product", "[matrix][hadamard]") {
    matrix<double> A{{1, 2}, {3, 4}};
    matrix<double> B{{5, 6}, {7, 8}};

    auto C = hadamard(A, B);
    REQUIRE(C(0, 0) == 5);   // 1 * 5
    REQUIRE(C(0, 1) == 12);  // 2 * 6
    REQUIRE(C(1, 0) == 21);  // 3 * 7
    REQUIRE(C(1, 1) == 32);  // 4 * 8
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_CASE("Edge cases", "[matrix][edge]") {
    SECTION("1x1 matrix operations") {
        matrix<double> A{{5}};
        matrix<double> B{{3}};

        REQUIRE((A + B)(0, 0) == 8);
        REQUIRE((A * B)(0, 0) == 15);
        REQUIRE(det(A) == 5);
        REQUIRE(inverse(A)(0, 0) == Approx(0.2));
        REQUIRE(trace(A) == 5);
    }

    SECTION("Large identity inverse") {
        auto I = eye<double>(10);
        auto Iinv = inverse(I);
        REQUIRE(approx_equal(I, Iinv));
    }
}

// =============================================================================
// Concept Verification
// =============================================================================

TEST_CASE("Matrix concept satisfaction", "[matrix][concept]") {
    // This test verifies at compile time that matrix<double> satisfies Matrix
    static_assert(Matrix<matrix<double>>);
    static_assert(Matrix<matrix<float>>);
    static_assert(Matrix<matrix<int>>);

    // Verify concept operations work generically
    auto generic_trace = []<Matrix M>(const M& m) {
        return trace(m);
    };

    matrix<double> A{{1, 2}, {3, 4}};
    REQUIRE(generic_trace(A) == 5);
}

// =============================================================================
// Cholesky Decomposition Tests
// =============================================================================

TEST_CASE("Cholesky decomposition", "[matrix][cholesky]") {
    SECTION("2x2 positive definite") {
        matrix<double> A{{4, 2}, {2, 3}};
        auto L = cholesky(A);

        // Verify L is lower triangular
        REQUIRE(L(0, 1) == Approx(0.0));
        REQUIRE(L(0, 0) > 0);
        REQUIRE(L(1, 1) > 0);

        // Verify L * L^T == A
        auto LLT = L * transpose(L);
        REQUIRE(approx_equal(LLT, A));
    }

    SECTION("3x3 positive definite") {
        matrix<double> A{{4, 2, 1}, {2, 5, 3}, {1, 3, 6}};
        auto L = cholesky(A);

        // Verify L * L^T == A
        auto LLT = L * transpose(L);
        REQUIRE(approx_equal(LLT, A));
    }

    SECTION("Identity matrix") {
        auto I = eye<double>(3);
        auto L = cholesky(I);

        // Cholesky of I is I
        REQUIRE(approx_equal(L, I));
    }

    SECTION("Diagonal matrix") {
        auto A = diag(std::vector<double>{4, 9, 16});
        auto L = cholesky(A);

        // L should be diag(2, 3, 4)
        REQUIRE(L(0, 0) == Approx(2.0));
        REQUIRE(L(1, 1) == Approx(3.0));
        REQUIRE(L(2, 2) == Approx(4.0));
    }

    SECTION("Non-positive-definite throws") {
        matrix<double> A{{1, 2}, {2, 1}};  // eigenvalues: 3, -1
        REQUIRE_THROWS_AS(cholesky(A), std::runtime_error);
    }
}

// =============================================================================
// String Representation Tests
// =============================================================================

TEST_CASE("String representation", "[matrix][io]") {
    matrix<double> A{{1, 2}, {3, 4}};
    auto s = to_string(A);

    REQUIRE(s.find("1.0000") != std::string::npos);
    REQUIRE(s.find("4.0000") != std::string::npos);
}
