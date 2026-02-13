#include <gtest/gtest.h>
#include <cmath>
#include <numbers>

#include "../peasant.hpp"
#include "../examples/mat2.hpp"
#include "../examples/perm.hpp"
#include "../examples/mod_int.hpp"
#include "../examples/tropical.hpp"
#include "../examples/str_repeat.hpp"
#include "../examples/affine.hpp"
#include "../examples/poly.hpp"
#include "../examples/dual.hpp"
#include "../examples/bool_matrix.hpp"
#include "../examples/dfa.hpp"
#include "../examples/quat.hpp"
#include "../examples/split_complex.hpp"
#include "../examples/recurrence.hpp"
#include "../examples/tree_graft.hpp"

using namespace peasant;
using namespace peasant::examples;

// Helper: repeated squaring power function (same pattern used throughout)
template<typename T>
T power_by_squaring(T base, int64_t exp) {
    T result = one(base);
    while (exp > 0) {
        if (exp & 1) result = result * base;
        base = base * base;
        exp >>= 1;
    }
    return result;
}

// =============================================================================
// mat2: Fibonacci via matrix exponentiation
// =============================================================================

TEST(Mat2Test, FibonacciSmall) {
    auto fib_direct = [](int64_t n) -> int64_t {
        if (n <= 0) return 0;
        mat2 result = one(mat2{});
        mat2 base = fib_matrix;
        while (n > 0) {
            if (n & 1) result = result * base;
            base = base * base;
            n >>= 1;
        }
        return result.b;
    };

    EXPECT_EQ(fib_direct(0), 0);
    EXPECT_EQ(fib_direct(1), 1);
    EXPECT_EQ(fib_direct(2), 1);
    EXPECT_EQ(fib_direct(3), 2);
    EXPECT_EQ(fib_direct(4), 3);
    EXPECT_EQ(fib_direct(5), 5);
    EXPECT_EQ(fib_direct(10), 55);
    EXPECT_EQ(fib_direct(20), 6765);
    EXPECT_EQ(fib_direct(50), 12586269025LL);
}

TEST(Mat2Test, MatrixMultiplication) {
    mat2 a{1, 2, 3, 4};
    mat2 b{5, 6, 7, 8};
    mat2 c = a * b;

    EXPECT_EQ(c.a, 19);
    EXPECT_EQ(c.b, 22);
    EXPECT_EQ(c.c, 43);
    EXPECT_EQ(c.d, 50);
}

TEST(Mat2Test, IdentityMatrix) {
    mat2 a{1, 2, 3, 4};
    mat2 id = one(mat2{});

    EXPECT_EQ(a * id, a);
    EXPECT_EQ(id * a, a);
}

// =============================================================================
// perm: Permutation exponentiation
// =============================================================================

TEST(PermTest, RotationPower) {
    auto rot = rotate_perm<4>();

    EXPECT_EQ(rot(0), 1);
    EXPECT_EQ(rot(1), 2);
    EXPECT_EQ(rot(2), 3);
    EXPECT_EQ(rot(3), 0);

    auto rot4 = rot * rot * rot * rot;
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(rot4(i), i);
    }
}

TEST(PermTest, Inverse) {
    auto rot = rotate_perm<4>();
    auto inv = rot.inverse();
    auto id = rot * inv;

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(id(i), i);
    }
}

TEST(PermTest, LargePower) {
    auto rot = rotate_perm<4>();

    perm<4> result = identity_perm<4>();
    perm<4> base = rot;
    int64_t n = 1000000;
    while (n > 0) {
        if (n & 1) result = result * base;
        base = base * base;
        n >>= 1;
    }

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(result(i), i);
    }
}

// =============================================================================
// mod_int: Modular arithmetic
// =============================================================================

TEST(ModIntTest, BasicArithmetic) {
    mod_1e9_7 a{5};
    mod_1e9_7 b{3};

    EXPECT_EQ((a + b).v, 8);
    EXPECT_EQ((a - b).v, 2);
    EXPECT_EQ((a * b).v, 15);
}

TEST(ModIntTest, ModularReduction) {
    mod_int<7> a{15};
    EXPECT_EQ(a.v, 1);

    mod_int<7> b{-2};
    EXPECT_EQ(b.v, 5);
}

TEST(ModIntTest, PowerMod) {
    mod_1e9_7 base{2};
    mod_1e9_7 result = one(mod_1e9_7{});
    mod_1e9_7 b = base;
    int64_t exp = 10;
    while (exp > 0) {
        if (exp & 1) result = result * b;
        b = b * b;
        exp >>= 1;
    }
    EXPECT_EQ(result.v, 1024);
}

TEST(ModIntTest, FermatLittleTheorem) {
    mod_int<7> a{3};
    mod_int<7> result = one(mod_int<7>{});
    mod_int<7> base = a;
    int64_t exp = 6;
    while (exp > 0) {
        if (exp & 1) result = result * base;
        base = base * base;
        exp >>= 1;
    }
    EXPECT_EQ(result.v, 1);
}

// =============================================================================
// tropical: Shortest paths via algebra!
// =============================================================================

TEST(TropicalTest, MinOperation) {
    tropical a{3.0};
    tropical b{5.0};

    EXPECT_EQ((a + b).v, 3.0);
}

TEST(TropicalTest, PlusOperation) {
    tropical a{3.0};
    tropical b{5.0};

    EXPECT_EQ((a * b).v, 8.0);
}

TEST(TropicalTest, ShortestPath) {
    trop_matrix<3> adj;
    adj.data[0][0] = tropical{0};
    adj.data[0][1] = tropical{2};
    adj.data[0][2] = tropical{10};
    adj.data[1][0] = tropical{2};
    adj.data[1][1] = tropical{0};
    adj.data[1][2] = tropical{3};
    adj.data[2][0] = tropical{10};
    adj.data[2][1] = tropical{3};
    adj.data[2][2] = tropical{0};

    auto a2 = adj * adj;

    EXPECT_EQ(a2.data[0][2].v, 5.0);
}

// =============================================================================
// str_repeat: String repetition
// =============================================================================

TEST(StrRepTest, BasicRepetition) {
    str_rep s{"ab"};
    auto s2 = s * s;
    EXPECT_EQ(s2.str(), "abab");

    auto s4 = s2 * s2;
    EXPECT_EQ(s4.str(), "abababab");
}

TEST(StrRepTest, LargeRepetition) {
    str_rep s{"x"};

    str_rep result = one(str_rep{});
    str_rep base = s;
    int n = 1024;
    while (n > 0) {
        if (n & 1) result = result * base;
        base = base * base;
        n >>= 1;
    }

    EXPECT_EQ(result.size(), 1024);
}

TEST(StrRepTest, EmptyIdentity) {
    str_rep s{"hello"};
    str_rep empty = one(str_rep{});

    EXPECT_EQ((s * empty).str(), "hello");
    EXPECT_EQ((empty * s).str(), "hello");
}

// =============================================================================
// Affine: Function composition / compound interest
// =============================================================================

TEST(AffineTest, Identity) {
    affine<double> id = one(affine<double>{});
    EXPECT_DOUBLE_EQ(id(5.0), 5.0);
    EXPECT_DOUBLE_EQ(id(100.0), 100.0);
}

TEST(AffineTest, Composition) {
    affine<double> f{2, 3};
    affine<double> g{4, 5};
    auto fg = f * g;

    EXPECT_DOUBLE_EQ(fg.a, 8.0);
    EXPECT_DOUBLE_EQ(fg.b, 13.0);
    EXPECT_DOUBLE_EQ(fg(1.0), 21.0);
}

TEST(AffineTest, CompoundInterest) {
    affine<double> yearly = compound_interest(0.05, 100.0);

    EXPECT_DOUBLE_EQ(yearly(1000.0), 1150.0);

    auto two_years = yearly * yearly;
    EXPECT_DOUBLE_EQ(two_years(1000.0), 1307.5);

    auto thirty_years = power_by_squaring(yearly, 30);
    double expected = 1000.0 * std::pow(1.05, 30) + 100.0 * (std::pow(1.05, 30) - 1) / 0.05;
    EXPECT_NEAR(thirty_years(1000.0), expected, 0.01);
}

// =============================================================================
// Poly: Polynomial powers / binomial coefficients
// =============================================================================

TEST(PolyTest, BinomialCoefficients) {
    auto binomial = binomial_generator<int64_t>();
    auto fifth = power_by_squaring(binomial, 5);

    EXPECT_EQ(fifth[0], 1);
    EXPECT_EQ(fifth[1], 5);
    EXPECT_EQ(fifth[2], 10);
    EXPECT_EQ(fifth[3], 10);
    EXPECT_EQ(fifth[4], 5);
    EXPECT_EQ(fifth[5], 1);
}

TEST(PolyTest, BinomialTenth) {
    auto binomial = binomial_generator<int64_t>();
    auto tenth = power_by_squaring(binomial, 10);

    EXPECT_EQ(tenth[5], 252);
    int64_t sum = 0;
    for (size_t i = 0; i <= 10; ++i) sum += tenth[i];
    EXPECT_EQ(sum, 1024);
}

TEST(PolyTest, Multiplication) {
    poly<int64_t> p1{1, 2};
    poly<int64_t> p2{3, 4};
    auto product = p1 * p2;

    EXPECT_EQ(product[0], 3);
    EXPECT_EQ(product[1], 10);
    EXPECT_EQ(product[2], 8);
}

// =============================================================================
// Dual: Automatic differentiation
// =============================================================================

TEST(DualTest, PowerRuleDerivative) {
    dual<double> x = variable(2.0);
    auto x5 = power_by_squaring(x, 5);

    EXPECT_DOUBLE_EQ(x5.val, 32.0);
    EXPECT_DOUBLE_EQ(x5.deriv, 80.0);
}

TEST(DualTest, ProductRule) {
    dual<double> x = variable(2.0);
    auto x2 = x * x;
    auto x3 = x2 * x;
    auto product = x2 * x3;

    EXPECT_DOUBLE_EQ(product.val, 32.0);
    EXPECT_DOUBLE_EQ(product.deriv, 80.0);
}

TEST(DualTest, ChainRuleViaExp) {
    dual<double> x = variable(1.0);
    auto ex = exp(x);

    EXPECT_NEAR(ex.val, std::exp(1.0), 1e-10);
    EXPECT_NEAR(ex.deriv, std::exp(1.0), 1e-10);
}

TEST(DualTest, TrigDerivatives) {
    dual<double> x = variable(std::numbers::pi / 4.0);
    auto s = sin(x);

    EXPECT_NEAR(s.val, std::sqrt(2.0) / 2.0, 1e-10);
    EXPECT_NEAR(s.deriv, std::sqrt(2.0) / 2.0, 1e-10);
}

// =============================================================================
// BoolMatrix: Graph reachability
// =============================================================================

TEST(BoolMatrixTest, DirectPaths) {
    bool_matrix<3> adj;
    adj.set(0, 1);
    adj.set(1, 2);

    EXPECT_TRUE(adj.get(0, 1));
    EXPECT_TRUE(adj.get(1, 2));
    EXPECT_FALSE(adj.get(0, 2));
}

TEST(BoolMatrixTest, TwoStepPaths) {
    bool_matrix<3> adj;
    adj.set(0, 1);
    adj.set(1, 2);

    auto adj2 = adj * adj;

    EXPECT_TRUE(adj2.get(0, 2));
    EXPECT_FALSE(adj2.get(0, 1));
}

TEST(BoolMatrixTest, CyclicGraph) {
    bool_matrix<3> adj;
    adj.set(0, 1);
    adj.set(1, 2);
    adj.set(2, 0);

    auto adj3 = power_by_squaring(adj, 3);
    EXPECT_TRUE(adj3.get(0, 0));
    EXPECT_TRUE(adj3.get(1, 1));
    EXPECT_TRUE(adj3.get(2, 2));
}

// =============================================================================
// DFA: Automaton fast-forward
// =============================================================================

TEST(DfaTest, CyclicTransition) {
    auto cycle = cyclic_transition<4>();

    EXPECT_EQ(cycle(0), 1);
    EXPECT_EQ(cycle(1), 2);
    EXPECT_EQ(cycle(2), 3);
    EXPECT_EQ(cycle(3), 0);
}

TEST(DfaTest, PowerOfCycle) {
    auto cycle = cyclic_transition<4>();

    auto cycle4 = power_by_squaring(cycle, 4);
    for (size_t s = 0; s < 4; ++s) {
        EXPECT_EQ(cycle4(s), s);
    }
}

TEST(DfaTest, LargePower) {
    auto cycle = cyclic_transition<5>();
    auto result = power_by_squaring(cycle, 1000000);

    for (size_t s = 0; s < 5; ++s) {
        EXPECT_EQ(result(s), s);
    }
}

TEST(DfaTest, NonCyclicTransition) {
    auto absorbing = make_transition<4>({{1, 2, 2, 2}});

    auto result = power_by_squaring(absorbing, 10);
    for (size_t s = 0; s < 4; ++s) {
        EXPECT_EQ(result(s), 2);
    }
}

// =============================================================================
// Quaternions: 3D rotations
// =============================================================================

TEST(QuatTest, Identity) {
    quat<double> id = one(quat<double>{});
    EXPECT_DOUBLE_EQ(id.w, 1.0);
    EXPECT_DOUBLE_EQ(id.x, 0.0);
    EXPECT_DOUBLE_EQ(id.y, 0.0);
    EXPECT_DOUBLE_EQ(id.z, 0.0);
}

TEST(QuatTest, UnitQuaternionNorm) {
    auto rot = rotation_x(0.5);
    EXPECT_NEAR(rot.norm(), 1.0, 1e-10);
}

TEST(QuatTest, RotationComposition) {
    double theta = std::numbers::pi / 4.0;
    auto rot45 = rotation_z(theta);
    auto rot90 = rot45 * rot45;

    auto direct90 = rotation_z(std::numbers::pi / 2.0);

    EXPECT_NEAR(rot90.w, direct90.w, 1e-10);
    EXPECT_NEAR(rot90.x, direct90.x, 1e-10);
    EXPECT_NEAR(rot90.y, direct90.y, 1e-10);
    EXPECT_NEAR(rot90.z, direct90.z, 1e-10);
}

TEST(QuatTest, PowerRotation) {
    double theta = std::numbers::pi / 18.0;
    auto rot10 = rotation_z(theta);
    auto rot360 = power_by_squaring(rot10, 36);

    EXPECT_NEAR(std::abs(rot360.w), 1.0, 1e-10);
    EXPECT_NEAR(rot360.x, 0.0, 1e-10);
    EXPECT_NEAR(rot360.y, 0.0, 1e-10);
    EXPECT_NEAR(rot360.z, 0.0, 1e-10);
}

TEST(QuatTest, NonCommutative) {
    auto rx = rotation_x(0.5);
    auto ry = rotation_y(0.5);

    auto xy = rx * ry;
    auto yx = ry * rx;

    bool equal = (std::abs(xy.w - yx.w) < 1e-10 &&
                  std::abs(xy.x - yx.x) < 1e-10 &&
                  std::abs(xy.y - yx.y) < 1e-10 &&
                  std::abs(xy.z - yx.z) < 1e-10);
    EXPECT_FALSE(equal);
}

// =============================================================================
// Split-complex: Hyperbolic geometry
// =============================================================================

TEST(SplitComplexTest, Identity) {
    split_complex<double> id = one(split_complex<double>{});
    EXPECT_DOUBLE_EQ(id.a, 1.0);
    EXPECT_DOUBLE_EQ(id.b, 0.0);
}

TEST(SplitComplexTest, JSquaredIsOne) {
    split_complex<double> j{0, 1};
    auto j2 = j * j;
    EXPECT_DOUBLE_EQ(j2.a, 1.0);
    EXPECT_DOUBLE_EQ(j2.b, 0.0);
}

TEST(SplitComplexTest, HyperbolicRotation) {
    double phi = 0.5;
    auto hyp = hyperbolic_rotation(phi);
    EXPECT_NEAR(hyp.modulus_sq(), 1.0, 1e-10);
}

TEST(SplitComplexTest, AdditionOfAngles) {
    double a = 0.3, b = 0.5;
    auto ha = hyperbolic_rotation(a);
    auto hb = hyperbolic_rotation(b);
    auto product = ha * hb;
    auto direct = hyperbolic_rotation(a + b);

    EXPECT_NEAR(product.a, direct.a, 1e-10);
    EXPECT_NEAR(product.b, direct.b, 1e-10);
}

TEST(SplitComplexTest, PowerOfRotation) {
    double phi = 0.1;
    auto h = hyperbolic_rotation(phi);
    auto h10 = power_by_squaring(h, 10);
    auto direct = hyperbolic_rotation(10 * phi);

    EXPECT_NEAR(h10.a, direct.a, 1e-10);
    EXPECT_NEAR(h10.b, direct.b, 1e-10);
}

// =============================================================================
// Recurrence: Companion matrices for linear recurrences
// =============================================================================

TEST(RecurrenceTest, Fibonacci) {
    auto fib_mat = fibonacci_companion();
    std::array<int64_t, 2> initial = {1, 0};

    EXPECT_EQ(compute_term(fib_mat, initial, 0), 0);
    EXPECT_EQ(compute_term(fib_mat, initial, 1), 1);
    EXPECT_EQ(compute_term(fib_mat, initial, 2), 1);
    EXPECT_EQ(compute_term(fib_mat, initial, 10), 55);
    EXPECT_EQ(compute_term(fib_mat, initial, 20), 6765);
}

TEST(RecurrenceTest, Tribonacci) {
    auto tri_mat = tribonacci_companion();
    std::array<int64_t, 3> initial = {1, 0, 0};

    EXPECT_EQ(compute_term(tri_mat, initial, 0), 0);
    EXPECT_EQ(compute_term(tri_mat, initial, 1), 0);
    EXPECT_EQ(compute_term(tri_mat, initial, 2), 1);
    EXPECT_EQ(compute_term(tri_mat, initial, 3), 1);
    EXPECT_EQ(compute_term(tri_mat, initial, 4), 2);
    EXPECT_EQ(compute_term(tri_mat, initial, 5), 4);
    EXPECT_EQ(compute_term(tri_mat, initial, 10), 81);
}

TEST(RecurrenceTest, Pell) {
    auto pell_mat = pell_companion();
    std::array<int64_t, 2> initial = {1, 0};

    EXPECT_EQ(compute_term(pell_mat, initial, 0), 0);
    EXPECT_EQ(compute_term(pell_mat, initial, 1), 1);
    EXPECT_EQ(compute_term(pell_mat, initial, 2), 2);
    EXPECT_EQ(compute_term(pell_mat, initial, 3), 5);
    EXPECT_EQ(compute_term(pell_mat, initial, 4), 12);
    EXPECT_EQ(compute_term(pell_mat, initial, 5), 29);
}

TEST(RecurrenceTest, Lucas) {
    auto lucas_mat = lucas_companion();
    std::array<int64_t, 2> initial = {1, 2};

    EXPECT_EQ(compute_term(lucas_mat, initial, 0), 2);
    EXPECT_EQ(compute_term(lucas_mat, initial, 1), 1);
    EXPECT_EQ(compute_term(lucas_mat, initial, 2), 3);
    EXPECT_EQ(compute_term(lucas_mat, initial, 3), 4);
    EXPECT_EQ(compute_term(lucas_mat, initial, 10), 123);
}

// =============================================================================
// TreeGraft: Fractal trees
// =============================================================================

TEST(TreeGraftTest, SingleLeaf) {
    tree_monoid<int> leaf;
    EXPECT_EQ(leaf.count_leaves(), 1);
    EXPECT_EQ(leaf.depth(), 0);
}

TEST(TreeGraftTest, BinaryBranch) {
    auto branch = binary_branch<int>();
    EXPECT_EQ(branch.count_leaves(), 2);
    EXPECT_EQ(branch.depth(), 1);
}

TEST(TreeGraftTest, TernaryBranch) {
    auto branch = ternary_branch<int>();
    EXPECT_EQ(branch.count_leaves(), 3);
    EXPECT_EQ(branch.depth(), 1);
}

TEST(TreeGraftTest, GraftingDoubles) {
    auto branch = binary_branch<int>();
    auto grafted = branch * branch;

    EXPECT_EQ(grafted.count_leaves(), 4);
    EXPECT_EQ(grafted.depth(), 2);
}

TEST(TreeGraftTest, PowerOfBinaryBranch) {
    auto branch = binary_branch<int>();

    auto tree3 = power_by_squaring(branch, 3);
    EXPECT_EQ(tree3.count_leaves(), 8);
    EXPECT_EQ(tree3.depth(), 3);

    auto tree5 = power_by_squaring(branch, 5);
    EXPECT_EQ(tree5.count_leaves(), 32);
    EXPECT_EQ(tree5.depth(), 5);
}

TEST(TreeGraftTest, PowerOfTernaryBranch) {
    auto branch = ternary_branch<int>();

    auto tree3 = power_by_squaring(branch, 3);
    EXPECT_EQ(tree3.count_leaves(), 27);
    EXPECT_EQ(tree3.depth(), 3);
}

TEST(TreeGraftTest, PowerOfNaryBranch) {
    auto branch = n_ary_branch<int>(4);

    auto tree3 = power_by_squaring(branch, 3);
    EXPECT_EQ(tree3.count_leaves(), 64);
    EXPECT_EQ(tree3.depth(), 3);
}
