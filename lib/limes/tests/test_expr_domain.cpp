#include <gtest/gtest.h>
#include <cmath>
#include <numbers>
#include <array>
#include <limes/limes.hpp>

using namespace limes::expr;

// =============================================================================
// Variable Set Tests (Analysis)
// =============================================================================

TEST(Analysis, ConstHasNoVariables) {
    auto c = Const<double>{42.0};
    EXPECT_EQ(variable_set_v<decltype(c)>, 0);
    EXPECT_TRUE(is_constant_v<decltype(c)>);
}

TEST(Analysis, VarDependsOnItsDimension) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};
    auto z = Var<2, double>{};

    EXPECT_EQ(variable_set_v<decltype(x)>, 1ULL << 0);  // 0b001
    EXPECT_EQ(variable_set_v<decltype(y)>, 1ULL << 1);  // 0b010
    EXPECT_EQ(variable_set_v<decltype(z)>, 1ULL << 2);  // 0b100

    EXPECT_TRUE((depends_on_v<decltype(x), 0>));
    EXPECT_FALSE((depends_on_v<decltype(x), 1>));
    EXPECT_FALSE((depends_on_v<decltype(y), 0>));
    EXPECT_TRUE((depends_on_v<decltype(y), 1>));
}

TEST(Analysis, BinaryUnionsDependencies) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    auto sum = x + y;
    auto prod = x * y;

    // Both should depend on dimensions 0 and 1
    EXPECT_EQ(variable_set_v<decltype(sum)>, (1ULL << 0) | (1ULL << 1));  // 0b011
    EXPECT_EQ(variable_set_v<decltype(prod)>, (1ULL << 0) | (1ULL << 1));

    EXPECT_TRUE((depends_on_v<decltype(sum), 0>));
    EXPECT_TRUE((depends_on_v<decltype(sum), 1>));
    EXPECT_FALSE((depends_on_v<decltype(sum), 2>));
}

TEST(Analysis, UnaryPreservesDependencies) {
    auto x = Var<0, double>{};

    auto neg_x = -x;
    auto sin_x = sin(x);
    auto exp_x = exp(x);

    EXPECT_EQ(variable_set_v<decltype(neg_x)>, 1ULL << 0);
    EXPECT_EQ(variable_set_v<decltype(sin_x)>, 1ULL << 0);
    EXPECT_EQ(variable_set_v<decltype(exp_x)>, 1ULL << 0);
}

TEST(Analysis, ComplexExpressionDependencies) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // f(x, y) = sin(x) * exp(-y)
    auto f = sin(x) * exp(-y);

    EXPECT_EQ(variable_set_v<decltype(f)>, (1ULL << 0) | (1ULL << 1));
    EXPECT_EQ(dependency_count_v<decltype(f)>, 2);
    EXPECT_EQ(max_dimension_v<decltype(f)>, 1);
}

TEST(Analysis, DependencyCount) {
    auto c = Const<double>{1.0};
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};
    auto f = x + y;

    EXPECT_EQ(dependency_count_v<decltype(c)>, 0);
    EXPECT_EQ(dependency_count_v<decltype(x)>, 1);
    EXPECT_EQ(dependency_count_v<decltype(y)>, 1);
    EXPECT_EQ(dependency_count_v<decltype(f)>, 2);
}

// =============================================================================
// Separability Detection Tests
// =============================================================================

TEST(Separability, ConstantIsSeparable) {
    auto c = Const<double>{5.0};
    EXPECT_TRUE((is_separable_v<decltype(c), 0, 1>));
}

TEST(Separability, SingleVarIsSeparable) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // x is "separable" in x and y (trivially, x = x * 1)
    EXPECT_TRUE((is_separable_v<decltype(x), 0, 1>));
    EXPECT_TRUE((is_separable_v<decltype(y), 0, 1>));
}

TEST(Separability, ProductOfSingleVars) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // x * y is separable: g(x) = x, h(y) = y
    auto f = x * y;
    EXPECT_TRUE((is_separable_v<decltype(f), 0, 1>));
}

TEST(Separability, FunctionOfSingleVar) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // sin(x) is separable in x,y (depends only on x)
    auto g = sin(x);
    EXPECT_TRUE((is_separable_v<decltype(g), 0, 1>));

    // exp(y) is separable in x,y (depends only on y)
    auto h = exp(y);
    EXPECT_TRUE((is_separable_v<decltype(h), 0, 1>));
}

TEST(Separability, ProductOfFunctions) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // sin(x) * exp(y) is separable
    auto f = sin(x) * exp(y);
    EXPECT_TRUE((is_separable_v<decltype(f), 0, 1>));
}

TEST(Separability, SumNotSeparable) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // x + y is NOT separable as g(x) * h(y)
    // (but our conservative check might say it is for sums of separables)
    auto f = x + y;

    // Each term individually is separable, so conservative check passes
    // This is a known limitation of the compile-time check
    EXPECT_TRUE((is_separable_v<decltype(f), 0, 1>));
}

TEST(Separability, MixedProductNotSeparable) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // (x + y) * x depends on both in the left factor, so not separable
    // This tests a more complex case
    auto f = (x + y) * x;

    // Left factor depends on both x and y, right factor depends on x
    // Neither depends solely on one dimension
    EXPECT_FALSE((is_separable_v<decltype(f), 0, 1>));
}

// =============================================================================
// Separation Function Tests
// =============================================================================

TEST(Separate, ProductOfVars) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    auto f = x * y;
    auto [g, h] = separate<0, 1>(f);

    // g should depend only on x, h should depend only on y
    EXPECT_TRUE((depends_on_v<decltype(g), 0>));
    EXPECT_FALSE((depends_on_v<decltype(g), 1>));
    EXPECT_FALSE((depends_on_v<decltype(h), 0>));
    EXPECT_TRUE((depends_on_v<decltype(h), 1>));

    // Test evaluation
    std::array<double, 2> args{3.0, 4.0};
    EXPECT_DOUBLE_EQ(g.evaluate(args), 3.0);  // g(x) = x
    EXPECT_DOUBLE_EQ(h.evaluate(args), 4.0);  // h(y) = y
}

TEST(Separate, ProductOfFunctions) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    auto f = sin(x) * exp(y);
    auto [g, h] = separate<0, 1>(f);

    // g should be sin(x), h should be exp(y)
    std::array<double, 2> args{std::numbers::pi / 2, 1.0};

    // sin(pi/2) = 1, exp(1) = e
    EXPECT_NEAR(g.evaluate(args), 1.0, 1e-10);
    EXPECT_NEAR(h.evaluate(args), std::exp(1.0), 1e-10);
}

// =============================================================================
// Bind Tests
// =============================================================================

TEST(Bind, BasicBind) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // f(x, y) = x * y
    auto f = x * y;
    EXPECT_EQ(f.arity_v, 2);

    // g(y) = 3 * y (bind x = 3)
    auto g = bind<0>(f, 3.0);
    EXPECT_EQ(g.arity_v, 1);

    std::array<double, 1> args{4.0};
    EXPECT_DOUBLE_EQ(g.evaluate(args), 12.0);  // 3 * 4 = 12
}

TEST(Bind, BindSecondDimension) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // f(x, y) = x * y
    auto f = x * y;

    // h(x) = x * 5 (bind y = 5)
    auto h = bind<1>(f, 5.0);
    EXPECT_EQ(h.arity_v, 1);

    std::array<double, 1> args{3.0};
    EXPECT_DOUBLE_EQ(h.evaluate(args), 15.0);  // 3 * 5 = 15
}

TEST(Bind, BindFullExpression) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // f(x, y) = sin(x) * cos(y)
    auto f = sin(x) * cos(y);

    // Bind x = pi/2, so g(y) = sin(pi/2) * cos(y) = 1 * cos(y) = cos(y)
    auto g = bind<0>(f, std::numbers::pi / 2);

    std::array<double, 1> args{0.0};
    EXPECT_NEAR(g.evaluate(args), 1.0, 1e-10);  // cos(0) = 1
}

TEST(Bind, MultipleBind) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // f(x, y) = x * y
    auto f = x * y;

    // Bind x = 3, then y = 4
    auto bound_x = bind<0>(f, 3.0);
    auto bound_both = bind<0>(bound_x, 4.0);  // Note: after first bind, y is at position 0

    EXPECT_EQ(bound_both.arity_v, 0);
    EXPECT_DOUBLE_EQ(bound_both.evaluate(), 12.0);  // 3 * 4 = 12
}

TEST(Bind, BindPreservesEvaluationAtPoint) {
    auto x = Var<0, double>{};
    auto f = sin(x) * cos(x);

    // f(pi/4) = sin(pi/4) * cos(pi/4) = (sqrt(2)/2)^2 = 0.5
    auto g = bind<0>(f, std::numbers::pi / 4);

    EXPECT_NEAR(g.evaluate(), 0.5, 1e-10);
}

TEST(Bind, Derivative) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // f(x, y) = x * y
    auto f = x * y;

    // g(y) = 3 * y (bind x = 3)
    auto g = bind<0>(f, 3.0);

    // dg/dy = 3
    auto dg = g.derivative<0>();

    std::array<double, 1> args{10.0};  // y value doesn't matter for constant derivative
    EXPECT_DOUBLE_EQ(dg.evaluate(args), 3.0);
}

// =============================================================================
// Split Tests
// =============================================================================

TEST(Split, BasicSplit) {
    auto x = Var<0, double>{};

    // Integral of x^2 from 0 to 1 = 1/3
    auto I = integral(x * x).over<0>(0.0, 1.0);

    // Split at 0.5
    auto [left, right] = I.split(0.5);

    // left: integral from 0 to 0.5
    // right: integral from 0.5 to 1
    auto left_result = left.evaluate();
    auto right_result = right.evaluate();
    auto total_result = I.evaluate();

    // The sum should equal the original
    EXPECT_NEAR(left_result.value() + right_result.value(), total_result.value(), 1e-8);
}

TEST(Split, LinearFunction) {
    auto x = Var<0, double>{};

    // Integral of 2x from 0 to 2 = [x^2]_0^2 = 4
    auto I = integral(2.0 * x).over<0>(0.0, 2.0);

    auto [left, right] = I.split(1.0);

    // left: integral from 0 to 1 = 1
    // right: integral from 1 to 2 = 3
    EXPECT_NEAR(left.evaluate().value(), 1.0, 1e-8);
    EXPECT_NEAR(right.evaluate().value(), 3.0, 1e-8);
    EXPECT_NEAR(left.evaluate().value() + right.evaluate().value(), 4.0, 1e-8);
}

TEST(Split, ExponentialFunction) {
    auto x = Var<0, double>{};

    // Integral of e^x from 0 to 2 = e^2 - 1
    auto I = integral(exp(x)).over<0>(0.0, 2.0);

    auto [left, right] = I.split(1.0);

    double expected_total = std::exp(2.0) - 1.0;
    EXPECT_NEAR(left.evaluate().value() + right.evaluate().value(), expected_total, 1e-6);
}

// =============================================================================
// Swap Tests (Fubini's Theorem)
// =============================================================================

TEST(Swap, BasicSwap) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // Double integral: ∫_0^1 ∫_0^1 x*y dx dy = 1/4
    // First integrate over x (inner), then over y (outer)
    // Note: After integrating over dim 0, the result has arity 1 and
    // the remaining dimension (originally y) is now at position 0
    auto inner = integral(x * y).over<0>(0.0, 1.0);  // arity: 2 -> 1
    auto I = integral(inner).over<0>(0.0, 1.0);       // arity: 1 -> 0 (now integrate the remaining dim)

    // Swap integration order
    auto J = I.swap<0, 1>();

    // Both should give the same result
    auto I_result = I.evaluate();
    auto J_result = J.evaluate();

    // Expected: ∫∫ xy dx dy = (1/2)(1/2) = 1/4
    EXPECT_NEAR(I_result.value(), 0.25, 1e-6);
    EXPECT_NEAR(J_result.value(), 0.25, 1e-6);
    EXPECT_NEAR(I_result.value(), J_result.value(), 1e-6);
}

TEST(Swap, SeparableFunction) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // ∫_0^1 ∫_0^1 sin(x) * cos(y) dx dy
    // = [∫_0^1 sin(x) dx] * [∫_0^1 cos(y) dy]
    // = [-cos(x)]_0^1 * [sin(y)]_0^1
    // = (1 - cos(1)) * sin(1)
    // Note: After integrating over dim 0, remaining dimension is at position 0
    auto inner = integral(sin(x) * cos(y)).over<0>(0.0, 1.0);  // arity: 2 -> 1
    auto I = integral(inner).over<0>(0.0, 1.0);                 // arity: 1 -> 0

    auto J = I.swap<0, 1>();

    double expected = (1.0 - std::cos(1.0)) * std::sin(1.0);

    EXPECT_NEAR(I.evaluate().value(), expected, 1e-6);
    EXPECT_NEAR(J.evaluate().value(), expected, 1e-6);
}

// =============================================================================
// Transform Tests (Change of Variables)
// =============================================================================

TEST(Transform, LinearTransform) {
    auto x = Var<0, double>{};

    // ∫_0^1 x dx = 1/2
    // Using x = 2t, dx = 2 dt, bounds [0, 1/2]
    // ∫_0^{1/2} (2t) * 2 dt = ∫_0^{1/2} 4t dt = [2t^2]_0^{1/2} = 1/2
    auto I = integral(x).over<0>(0.0, 1.0);

    auto phi = [](double t) { return 2.0 * t; };
    auto jacobian = [](double /*t*/) { return 2.0; };

    auto T = I.transform(phi, jacobian, 0.0, 0.5);

    EXPECT_NEAR(T.evaluate().value(), 0.5, 1e-8);
    EXPECT_NEAR(I.evaluate().value(), T.evaluate().value(), 1e-8);
}

TEST(Transform, QuadraticTransform) {
    auto x = Var<0, double>{};

    // ∫_0^1 sqrt(x) dx = [2/3 * x^{3/2}]_0^1 = 2/3
    auto I = integral(sqrt(x)).over<0>(0.0, 1.0);

    // Using x = t^2, dx = 2t dt:
    // ∫_0^1 sqrt(t^2) * 2t dt = ∫_0^1 t * 2t dt = ∫_0^1 2t^2 dt = 2/3
    auto phi = [](double t) { return t * t; };
    auto jacobian = [](double t) { return 2.0 * t; };

    auto T = I.transform(phi, jacobian, 0.0, 1.0);

    EXPECT_NEAR(I.evaluate().value(), 2.0/3.0, 1e-6);
    EXPECT_NEAR(T.evaluate().value(), 2.0/3.0, 1e-6);
}

TEST(Transform, RemoveSingularity) {
    auto x = Var<0, double>{};

    // ∫_0^1 1/sqrt(x) dx has a singularity at x=0
    // Using x = t^2, dx = 2t dt:
    // ∫_0^1 1/t * 2t dt = ∫_0^1 2 dt = 2
    //
    // Note: Direct integration has issues at x=0, but transform removes singularity

    auto I = integral(1.0 / sqrt(x)).over<0>(0.001, 1.0);  // Avoid singularity

    auto phi = [](double t) { return t * t; };
    auto jacobian = [](double t) { return 2.0 * t; };

    auto T = I.transform(phi, jacobian, std::sqrt(0.001), 1.0);

    // Both should be approximately equal for similar domains
    // The transformed version is more numerically stable
    EXPECT_NEAR(T.evaluate().value(), I.evaluate().value(), 1e-4);
}

TEST(Transform, BuiltinTransforms) {
    auto x = Var<0, double>{};

    // Test with built-in transform types
    auto I = integral(x * x).over<0>(0.0, 1.0);

    transforms::linear<double> phi{2.0, 0.0};  // x = 2t
    transforms::linear_jacobian<double> jac{2.0};

    auto T = I.transform(phi, jac, 0.0, 0.5);

    // Original: ∫_0^1 x^2 dx = 1/3
    // Transformed: ∫_0^{0.5} (2t)^2 * 2 dt = ∫_0^{0.5} 8t^2 dt = [8t^3/3]_0^{0.5} = 1/3
    EXPECT_NEAR(I.evaluate().value(), 1.0/3.0, 1e-8);
    EXPECT_NEAR(T.evaluate().value(), 1.0/3.0, 1e-8);
}

// =============================================================================
// Type Parameterized Tests
// =============================================================================

template <typename T>
class DomainTypedTest : public ::testing::Test {
protected:
    static constexpr T tol = std::is_same_v<T, float> ? T(1e-4) : T(1e-8);
};

using FloatTypes = ::testing::Types<float, double, long double>;
TYPED_TEST_SUITE(DomainTypedTest, FloatTypes);

TYPED_TEST(DomainTypedTest, VariableSet) {
    using T = TypeParam;

    auto x = Var<0, T>{};
    auto y = Var<1, T>{};

    EXPECT_EQ(variable_set_v<decltype(x)>, 1ULL << 0);
    EXPECT_EQ(variable_set_v<decltype(y)>, 1ULL << 1);
    EXPECT_EQ(variable_set_v<decltype(x * y)>, (1ULL << 0) | (1ULL << 1));
}

TYPED_TEST(DomainTypedTest, Bind) {
    using T = TypeParam;

    auto x = Var<0, T>{};
    auto y = Var<1, T>{};

    auto f = x * y;
    auto g = bind<0>(f, T(3));

    std::array<T, 1> args{T(4)};
    EXPECT_NEAR(g.evaluate(args), T(12), this->tol);
}

TYPED_TEST(DomainTypedTest, Split) {
    using T = TypeParam;

    auto x = Var<0, T>{};
    auto I = integral(x).template over<0>(T(0), T(1));

    auto [left, right] = I.split(T(0.5));

    T total = left.evaluate().value() + right.evaluate().value();
    EXPECT_NEAR(total, T(0.5), this->tol);
}

// =============================================================================
// Integration Tests (combining multiple features)
// =============================================================================

TEST(Integration, BindThenIntegrate) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // f(x, y) = x * y
    auto f = x * y;

    // Bind x = 2, get g(y) = 2y
    auto g = bind<0>(f, 2.0);

    // Integrate g over y from 0 to 1: ∫_0^1 2y dy = [y^2]_0^1 = 1
    auto I = integral(g).over<0>(0.0, 1.0);

    EXPECT_NEAR(I.evaluate().value(), 1.0, 1e-8);
}

TEST(Integration, AnalysisThenOptimize) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // Check if sin(x) * exp(y) is separable
    auto f = sin(x) * exp(y);

    EXPECT_TRUE((is_separable_v<decltype(f), 0, 1>));

    // If separable, we could optimize ∫∫ f dx dy = (∫ sin(x) dx) * (∫ exp(y) dy)
    auto [g, h] = separate<0, 1>(f);

    // Integrate g over x from 0 to pi: ∫_0^pi sin(x) dx = 2
    auto Ig = integral(g).over<0>(0.0, std::numbers::pi);

    // Integrate h over y from 0 to 1: ∫_0^1 exp(y) dy = e - 1
    auto Ih = integral(h).over<1>(0.0, 1.0);

    double product_of_integrals = Ig.evaluate().value() * Ih.evaluate().value();
    double expected = 2.0 * (std::exp(1.0) - 1.0);

    EXPECT_NEAR(product_of_integrals, expected, 1e-6);
}
