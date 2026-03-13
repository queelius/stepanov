#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <numbers>
#include <array>
#include <limes/limes.hpp>

using namespace limes::expr;

// =============================================================================
// Core Node Tests
// =============================================================================

TEST(ExprConst, Evaluate) {
    auto c = Const<double>{42.0};

    std::array<double, 0> args{};
    EXPECT_DOUBLE_EQ(c.evaluate(args), 42.0);
}

TEST(ExprConst, Arity) {
    auto c = Const<double>{1.0};
    EXPECT_EQ(c.arity_v, 0);
}

TEST(ExprConst, Derivative) {
    auto c = Const<double>{5.0};
    auto dc = c.derivative<0>();

    std::array<double, 0> args{};
    EXPECT_DOUBLE_EQ(dc.evaluate(args), 0.0);
}

TEST(ExprConst, ToString) {
    auto c = Const<double>{3.14};
    EXPECT_EQ(c.to_string(), "3.14");
}

TEST(ExprVar, Evaluate) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    std::array<double, 2> args{10.0, 20.0};
    EXPECT_DOUBLE_EQ(x.evaluate(args), 10.0);
    EXPECT_DOUBLE_EQ(y.evaluate(args), 20.0);
}

TEST(ExprVar, Arity) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};
    auto z = Var<2, double>{};

    EXPECT_EQ(x.arity_v, 1);
    EXPECT_EQ(y.arity_v, 2);
    EXPECT_EQ(z.arity_v, 3);
}

TEST(ExprVar, Derivative) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // d/dx[x] = 1
    auto dx_dx = x.derivative<0>();
    // d/dy[x] = 0
    auto dx_dy = x.derivative<1>();
    // d/dx[y] = 0
    auto dy_dx = y.derivative<0>();
    // d/dy[y] = 1
    auto dy_dy = y.derivative<1>();

    std::array<double, 2> args{1.0, 2.0};
    EXPECT_DOUBLE_EQ(dx_dx.evaluate(args), 1.0);
    EXPECT_DOUBLE_EQ(dx_dy.evaluate(args), 0.0);
    EXPECT_DOUBLE_EQ(dy_dx.evaluate(args), 0.0);
    EXPECT_DOUBLE_EQ(dy_dy.evaluate(args), 1.0);
}

TEST(ExprVar, ArgAlias) {
    // Test the arg<N> alias
    auto x = arg<0, double>;
    auto y = arg<1, double>;

    std::array<double, 2> args{5.0, 7.0};
    EXPECT_DOUBLE_EQ(x.evaluate(args), 5.0);
    EXPECT_DOUBLE_EQ(y.evaluate(args), 7.0);
}

TEST(ExprVar, ToString) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    EXPECT_EQ(x.to_string(), "x0");
    EXPECT_EQ(y.to_string(), "x1");
}

// =============================================================================
// Binary Operation Tests
// =============================================================================

TEST(ExprBinary, Addition) {
    auto x = Var<0, double>{};
    auto c = Const<double>{5.0};
    auto sum = x + c;

    std::array<double, 1> args{10.0};
    EXPECT_DOUBLE_EQ(sum.evaluate(args), 15.0);
    EXPECT_EQ(sum.arity_v, 1);
}

TEST(ExprBinary, Subtraction) {
    auto x = Var<0, double>{};
    auto c = Const<double>{3.0};
    auto diff = x - c;

    std::array<double, 1> args{10.0};
    EXPECT_DOUBLE_EQ(diff.evaluate(args), 7.0);
}

TEST(ExprBinary, Multiplication) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};
    auto prod = x * y;

    std::array<double, 2> args{3.0, 4.0};
    EXPECT_DOUBLE_EQ(prod.evaluate(args), 12.0);
    EXPECT_EQ(prod.arity_v, 2);
}

TEST(ExprBinary, Division) {
    auto x = Var<0, double>{};
    auto c = Const<double>{2.0};
    auto quot = x / c;

    std::array<double, 1> args{10.0};
    EXPECT_DOUBLE_EQ(quot.evaluate(args), 5.0);
}

TEST(ExprBinary, ScalarOperators) {
    auto x = Var<0, double>{};

    // expr + scalar
    auto sum1 = x + 5.0;
    // scalar + expr
    auto sum2 = 5.0 + x;
    // expr * scalar
    auto prod1 = x * 2.0;
    // scalar * expr
    auto prod2 = 2.0 * x;
    // expr / scalar
    auto quot1 = x / 2.0;
    // scalar / expr
    auto quot2 = 10.0 / x;

    std::array<double, 1> args{3.0};
    EXPECT_DOUBLE_EQ(sum1.evaluate(args), 8.0);
    EXPECT_DOUBLE_EQ(sum2.evaluate(args), 8.0);
    EXPECT_DOUBLE_EQ(prod1.evaluate(args), 6.0);
    EXPECT_DOUBLE_EQ(prod2.evaluate(args), 6.0);
    EXPECT_DOUBLE_EQ(quot1.evaluate(args), 1.5);
    EXPECT_NEAR(quot2.evaluate(args), 10.0/3.0, 1e-10);
}

TEST(ExprBinary, ComplexExpression) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // (x + y) * 2
    auto expr = (x + y) * 2.0;

    std::array<double, 2> args{3.0, 4.0};
    EXPECT_DOUBLE_EQ(expr.evaluate(args), 14.0);
}

TEST(ExprBinary, DerivativeAddition) {
    auto x = Var<0, double>{};
    auto c = Const<double>{5.0};
    auto sum = x + c;

    // d/dx[x + 5] = 1 + 0 = 1
    auto dsum = sum.derivative<0>();

    std::array<double, 1> args{10.0};
    EXPECT_DOUBLE_EQ(dsum.evaluate(args), 1.0);
}

TEST(ExprBinary, DerivativeProduct) {
    auto x = Var<0, double>{};

    // f(x) = x * x = x^2
    auto f = x * x;
    // d/dx[x^2] = 2x
    auto df = f.derivative<0>();

    std::array<double, 1> args{3.0};
    EXPECT_DOUBLE_EQ(df.evaluate(args), 6.0);  // 2 * 3 = 6
}

TEST(ExprBinary, DerivativeQuotient) {
    auto x = Var<0, double>{};
    auto c = Const<double>{1.0};

    // f(x) = 1/x
    auto f = c / x;
    // d/dx[1/x] = -1/x^2
    auto df = f.derivative<0>();

    std::array<double, 1> args{2.0};
    EXPECT_DOUBLE_EQ(df.evaluate(args), -0.25);  // -1/4
}

TEST(ExprBinary, ToString) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    auto sum = x + y;
    auto prod = x * y;

    EXPECT_EQ(sum.to_string(), "(+ x0 x1)");
    EXPECT_EQ(prod.to_string(), "(* x0 x1)");
}

// =============================================================================
// Unary Operation Tests
// =============================================================================

TEST(ExprUnary, Negation) {
    auto x = Var<0, double>{};
    auto neg_x = -x;

    std::array<double, 1> args{5.0};
    EXPECT_DOUBLE_EQ(neg_x.evaluate(args), -5.0);
}

TEST(ExprUnary, NegationDerivative) {
    auto x = Var<0, double>{};
    auto neg_x = -x;

    // d/dx[-x] = -1
    auto d_neg_x = neg_x.derivative<0>();

    std::array<double, 1> args{5.0};
    EXPECT_DOUBLE_EQ(d_neg_x.evaluate(args), -1.0);
}

TEST(ExprUnary, ToString) {
    auto x = Var<0, double>{};
    auto neg_x = -x;

    EXPECT_EQ(neg_x.to_string(), "(- x0)");
}

// =============================================================================
// Primitive Function Tests
// =============================================================================

TEST(ExprPrimitives, Exp) {
    auto x = Var<0, double>{};
    auto f = exp(x);

    std::array<double, 1> args{1.0};
    EXPECT_NEAR(f.evaluate(args), std::exp(1.0), 1e-10);
}

TEST(ExprPrimitives, Log) {
    auto x = Var<0, double>{};
    auto f = log(x);

    std::array<double, 1> args{std::numbers::e};
    EXPECT_NEAR(f.evaluate(args), 1.0, 1e-10);
}

TEST(ExprPrimitives, Sin) {
    auto x = Var<0, double>{};
    auto f = sin(x);

    std::array<double, 1> args{std::numbers::pi / 2};
    EXPECT_NEAR(f.evaluate(args), 1.0, 1e-10);
}

TEST(ExprPrimitives, Cos) {
    auto x = Var<0, double>{};
    auto f = cos(x);

    std::array<double, 1> args{0.0};
    EXPECT_NEAR(f.evaluate(args), 1.0, 1e-10);
}

TEST(ExprPrimitives, Sqrt) {
    auto x = Var<0, double>{};
    auto f = sqrt(x);

    std::array<double, 1> args{4.0};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 2.0);
}

TEST(ExprPrimitives, Abs) {
    auto x = Var<0, double>{};
    auto f = abs(x);

    std::array<double, 1> args1{-5.0};
    std::array<double, 1> args2{5.0};
    EXPECT_DOUBLE_EQ(f.evaluate(args1), 5.0);
    EXPECT_DOUBLE_EQ(f.evaluate(args2), 5.0);
}

TEST(ExprPrimitives, ExpDerivative) {
    auto x = Var<0, double>{};
    auto f = exp(x);

    // d/dx[exp(x)] = exp(x)
    auto df = f.derivative<0>();

    std::array<double, 1> args{2.0};
    EXPECT_NEAR(df.evaluate(args), std::exp(2.0), 1e-10);
}

TEST(ExprPrimitives, LogDerivative) {
    auto x = Var<0, double>{};
    auto f = log(x);

    // d/dx[log(x)] = 1/x
    auto df = f.derivative<0>();

    std::array<double, 1> args{2.0};
    EXPECT_NEAR(df.evaluate(args), 0.5, 1e-10);
}

TEST(ExprPrimitives, SinDerivative) {
    auto x = Var<0, double>{};
    auto f = sin(x);

    // d/dx[sin(x)] = cos(x)
    auto df = f.derivative<0>();

    std::array<double, 1> args{std::numbers::pi / 3};
    EXPECT_NEAR(df.evaluate(args), std::cos(std::numbers::pi / 3), 1e-10);
}

TEST(ExprPrimitives, CosDerivative) {
    auto x = Var<0, double>{};
    auto f = cos(x);

    // d/dx[cos(x)] = -sin(x)
    auto df = f.derivative<0>();

    std::array<double, 1> args{std::numbers::pi / 4};
    EXPECT_NEAR(df.evaluate(args), -std::sin(std::numbers::pi / 4), 1e-10);
}

TEST(ExprPrimitives, SqrtDerivative) {
    auto x = Var<0, double>{};
    auto f = sqrt(x);

    // d/dx[sqrt(x)] = 1/(2*sqrt(x))
    auto df = f.derivative<0>();

    std::array<double, 1> args{4.0};
    EXPECT_NEAR(df.evaluate(args), 0.25, 1e-10);  // 1/(2*2)
}

TEST(ExprPrimitives, ToString) {
    auto x = Var<0, double>{};

    EXPECT_EQ(exp(x).to_string(), "(exp x0)");
    EXPECT_EQ(log(x).to_string(), "(log x0)");
    EXPECT_EQ(sin(x).to_string(), "(sin x0)");
    EXPECT_EQ(cos(x).to_string(), "(cos x0)");
    EXPECT_EQ(sqrt(x).to_string(), "(sqrt x0)");
    EXPECT_EQ(abs(x).to_string(), "(abs x0)");
}

// =============================================================================
// Chain Rule Tests
// =============================================================================

TEST(ExprChainRule, SinOfSquare) {
    auto x = Var<0, double>{};

    // f(x) = sin(x^2)
    auto f = sin(x * x);

    // d/dx[sin(x^2)] = cos(x^2) * 2x
    auto df = f.derivative<0>();

    double val = 1.5;
    std::array<double, 1> args{val};

    double expected = std::cos(val * val) * 2 * val;
    EXPECT_NEAR(df.evaluate(args), expected, 1e-10);
}

TEST(ExprChainRule, ExpOfNegSquare) {
    auto x = Var<0, double>{};

    // f(x) = exp(-x^2)
    auto neg_x_sq = -(x * x);
    auto f = exp(neg_x_sq);

    // d/dx[exp(-x^2)] = exp(-x^2) * (-2x) = -2x * exp(-x^2)
    auto df = f.derivative<0>();

    double val = 0.5;
    std::array<double, 1> args{val};

    double expected = std::exp(-val * val) * (-2 * val);
    EXPECT_NEAR(df.evaluate(args), expected, 1e-10);
}

TEST(ExprChainRule, LogOfSin) {
    auto x = Var<0, double>{};

    // f(x) = log(sin(x))
    auto f = log(sin(x));

    // d/dx[log(sin(x))] = cos(x)/sin(x) = cot(x)
    auto df = f.derivative<0>();

    double val = std::numbers::pi / 4;  // 45 degrees
    std::array<double, 1> args{val};

    double expected = std::cos(val) / std::sin(val);  // cot(pi/4) = 1
    EXPECT_NEAR(df.evaluate(args), expected, 1e-10);
}

// =============================================================================
// Derivative Function Tests
// =============================================================================

TEST(ExprDerivative, CompileTimeDerivative) {
    auto x = Var<0, double>{};
    auto f = x * x;

    // Using free function
    auto df = derivative<0>(f);

    std::array<double, 1> args{3.0};
    EXPECT_DOUBLE_EQ(df.evaluate(args), 6.0);
}

TEST(ExprDerivative, RuntimeDerivative) {
    auto x = Var<0, double>{};
    auto f = x * x;

    // Using runtime dimension
    auto df = derivative(f, 0);

    std::array<double, 1> args{3.0};
    EXPECT_DOUBLE_EQ(df.evaluate(args), 6.0);
}

TEST(ExprDerivative, Gradient) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // f(x, y) = x^2 + y^2
    auto f = x * x + y * y;

    // grad(f) = (2x, 2y)
    auto grad = gradient(f);

    std::array<double, 2> args{3.0, 4.0};

    auto df_dx = std::get<0>(grad);
    auto df_dy = std::get<1>(grad);

    EXPECT_DOUBLE_EQ(df_dx.evaluate(args), 6.0);  // 2 * 3
    EXPECT_DOUBLE_EQ(df_dy.evaluate(args), 8.0);  // 2 * 4
}

TEST(ExprDerivative, HigherOrder) {
    auto x = Var<0, double>{};

    // f(x) = x^3
    auto f = x * x * x;

    // d/dx[x^3] = 3x^2
    auto df = derivative<0>(f);

    // d^2/dx^2[x^3] = 6x
    auto d2f = derivative<0>(df);

    std::array<double, 1> args{2.0};
    EXPECT_DOUBLE_EQ(d2f.evaluate(args), 12.0);  // 6 * 2
}

// =============================================================================
// DerivativeBuilder Tests (Phase 2: Fluent API)
// =============================================================================

TEST(DerivativeBuilder, BasicWrt) {
    auto x = Var<0, double>{};
    auto f = x * x;

    // derivative(f).wrt<0>() should equal derivative<0>(f)
    auto df = derivative(f).wrt<0>();

    std::array<double, 1> args{3.0};
    EXPECT_DOUBLE_EQ(df.eval(args), 6.0);  // d/dx[x^2] = 2x = 6
}

TEST(DerivativeBuilder, ChainedWrt) {
    auto x = Var<0, double>{};
    auto f = x * x * x;

    // derivative(f).wrt<0>().wrt<0>() = d^2/dx^2[x^3] = 6x
    auto d2f = derivative(f).wrt<0>().wrt<0>();

    std::array<double, 1> args{2.0};
    EXPECT_DOUBLE_EQ(d2f.eval(args), 12.0);  // 6 * 2 = 12
}

TEST(DerivativeBuilder, MultipleWrtTemplate) {
    auto x = Var<0, double>{};
    auto f = x * x * x;

    // derivative(f).wrt<0, 0>() = d^2/dx^2[x^3] = 6x (convenience syntax)
    auto d2f = derivative(f).wrt<0, 0>();

    std::array<double, 1> args{2.0};
    EXPECT_DOUBLE_EQ(d2f.eval(args), 12.0);  // 6 * 2 = 12
}

TEST(DerivativeBuilder, MixedPartials) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // f(x, y) = x^2 * y
    auto f = x * x * y;

    // d^2f/dx dy = d/dy[2xy] = 2x
    auto d2f = derivative(f).wrt<0>().wrt<1>();

    std::array<double, 2> args{3.0, 4.0};
    EXPECT_DOUBLE_EQ(d2f.eval(args), 6.0);  // 2 * 3 = 6
}

TEST(DerivativeBuilder, MixedPartialsConvenience) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // f(x, y) = x^2 * y
    auto f = x * x * y;

    // d^2f/dx dy using convenience syntax
    auto d2f = derivative(f).wrt<0, 1>();

    std::array<double, 2> args{3.0, 4.0};
    EXPECT_DOUBLE_EQ(d2f.eval(args), 6.0);  // 2 * 3 = 6
}

TEST(DerivativeBuilder, Gradient) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // f(x, y) = x^2 + y^2
    auto f = x * x + y * y;

    // gradient(f) = (2x, 2y) using builder
    auto grad = derivative(f).gradient();

    std::array<double, 2> args{3.0, 4.0};

    auto df_dx = std::get<0>(grad);
    auto df_dy = std::get<1>(grad);

    EXPECT_DOUBLE_EQ(df_dx.eval(args), 6.0);  // 2 * 3
    EXPECT_DOUBLE_EQ(df_dy.eval(args), 8.0);  // 2 * 4
}

TEST(DerivativeBuilder, Get) {
    auto x = Var<0, double>{};
    auto f = x * x;

    // Get the underlying expression
    auto df = derivative(f).wrt<0>().get();

    std::array<double, 1> args{3.0};
    EXPECT_DOUBLE_EQ(df.eval(args), 6.0);
}

TEST(DerivativeBuilder, ToString) {
    auto x = Var<0, double>{};
    auto f = sin(x);

    // derivative(sin(x)).wrt<0>() = cos(x) * 1 = cos(x)
    auto df = derivative(f).wrt<0>();
    std::string str = df.to_string();

    // The result should contain "cos"
    EXPECT_TRUE(str.find("cos") != std::string::npos);
}

TEST(DerivativeBuilder, WithPrimitives) {
    auto x = Var<0, double>{};

    // f(x) = sin(x^2)
    auto f = sin(x * x);

    // d/dx[sin(x^2)] = cos(x^2) * 2x
    auto df = derivative(f).wrt<0>();

    std::array<double, 1> args{1.0};
    double expected = std::cos(1.0) * 2.0;  // cos(1^2) * 2*1
    EXPECT_NEAR(df.eval(args), expected, 1e-10);
}

TEST(DerivativeBuilder, ThirdOrderPartial) {
    auto x = Var<0, double>{};

    // f(x) = x^4
    auto f = pow<4>(x);

    // d^3/dx^3[x^4] = 24x
    auto d3f = derivative(f).wrt<0, 0, 0>();

    std::array<double, 1> args{2.0};
    EXPECT_DOUBLE_EQ(d3f.eval(args), 48.0);  // 24 * 2 = 48
}

// =============================================================================
// Integration Tests
// =============================================================================

TEST(ExprIntegral, SimplePolynomial) {
    auto x = Var<0, double>{};

    // Integral of x^2 from 0 to 1 = 1/3
    auto I = integral(x * x).over<0>(0.0, 1.0);

    auto result = I.evaluate();
    EXPECT_NEAR(result.value(), 1.0/3.0, 1e-8);
}

TEST(ExprIntegral, Exponential) {
    auto x = Var<0, double>{};

    // Integral of e^x from 0 to 1 = e - 1
    auto I = integral(exp(x)).over<0>(0.0, 1.0);

    auto result = I.evaluate();
    EXPECT_NEAR(result.value(), std::exp(1.0) - 1.0, 1e-8);
}

TEST(ExprIntegral, Trigonometric) {
    auto x = Var<0, double>{};

    // Integral of sin(x) from 0 to pi = 2
    auto I = integral(sin(x)).over<0>(0.0, std::numbers::pi);

    auto result = I.evaluate();
    EXPECT_NEAR(result.value(), 2.0, 1e-8);
}

TEST(ExprIntegral, LinearFunction) {
    auto x = Var<0, double>{};

    // Integral of 2x + 3 from 0 to 1 = x^2 + 3x |_0^1 = 1 + 3 = 4
    auto f = 2.0 * x + 3.0;
    auto I = integral(f).over<0>(0.0, 1.0);

    auto result = I.evaluate();
    EXPECT_NEAR(result.value(), 4.0, 1e-8);
}

TEST(ExprIntegral, GaussianLike) {
    auto x = Var<0, double>{};

    // Integral of exp(-x^2) from -3 to 3 ≈ sqrt(pi) ≈ 1.7724538509
    // (slightly less due to truncation at +/-3)
    auto f = exp(-(x * x));
    auto I = integral(f).over<0>(-3.0, 3.0);

    auto result = I.evaluate();
    // Use wider bounds to get closer to sqrt(pi), or accept looser tolerance
    EXPECT_NEAR(result.value(), std::sqrt(std::numbers::pi), 1e-4);
}

TEST(ExprIntegral, Arity) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // Before integration: arity 2
    auto f = x * y;
    EXPECT_EQ(f.arity_v, 2);

    // After integrating x: arity 1
    auto Ix = integral(f).over<0>(0.0, 1.0);
    EXPECT_EQ(Ix.arity_v, 1);
}

TEST(ExprIntegral, ToString) {
    auto x = Var<0, double>{};
    auto f = x * x;

    auto I = integral(f).over<0>(0.0, 1.0);

    std::string str = I.to_string();
    EXPECT_TRUE(str.find("integral") != std::string::npos);
}

// =============================================================================
// Integration Method Tests (Phase 3: Methods as first-class objects)
// =============================================================================

TEST(IntegrationMethods, GaussLegendre) {
    using namespace limes::methods;

    auto x = Var<0, double>{};
    auto I = integral(x * x).over<0>(0.0, 1.0);

    // Use Gauss-Legendre with 7 points
    auto result = I.eval(gauss<7>());
    EXPECT_NEAR(result.value(), 1.0/3.0, 1e-10);
}

TEST(IntegrationMethods, GaussLegendreHighOrder) {
    using namespace limes::methods;

    auto x = Var<0, double>{};
    auto I = integral(sin(x)).over<0>(0.0, std::numbers::pi);

    // Use Gauss-Legendre with 15 points for high accuracy
    auto result = I.eval(gauss<15>());
    EXPECT_NEAR(result.value(), 2.0, 1e-12);
}

TEST(IntegrationMethods, Adaptive) {
    using namespace limes::methods;

    auto x = Var<0, double>{};
    auto I = integral(exp(x)).over<0>(0.0, 1.0);

    // Use adaptive method with explicit tolerance
    auto result = I.eval(adaptive_method(1e-12));
    EXPECT_NEAR(result.value(), std::exp(1.0) - 1.0, 1e-10);
}

TEST(IntegrationMethods, AdaptiveWithTolerance) {
    using namespace limes::methods;

    auto x = Var<0, double>{};
    auto I = integral(x * x).over<0>(0.0, 1.0);

    // Create method and configure tolerance
    auto method = adaptive<double>{}.with_tolerance(1e-14);
    auto result = I.eval(method);
    EXPECT_NEAR(result.value(), 1.0/3.0, 1e-12);
}

TEST(IntegrationMethods, MonteCarlo) {
    using namespace limes::methods;

    auto x = Var<0, double>{};
    auto I = integral(x * x).over<0>(0.0, 1.0);

    // Use Monte Carlo with fixed seed for reproducibility
    auto result = I.eval(monte_carlo<double>{100000}.with_seed(42));
    EXPECT_NEAR(result.value(), 1.0/3.0, 0.01);  // Monte Carlo has higher variance
}

TEST(IntegrationMethods, MonteCarloReproducibility) {
    using namespace limes::methods;

    auto x = Var<0, double>{};
    auto I = integral(x * x).over<0>(0.0, 1.0);

    // Same seed should give same result
    auto result1 = I.eval(monte_carlo<double>{10000}.with_seed(123));
    auto result2 = I.eval(monte_carlo<double>{10000}.with_seed(123));
    EXPECT_DOUBLE_EQ(result1.value(), result2.value());
}

TEST(IntegrationMethods, Simpson) {
    using namespace limes::methods;

    auto x = Var<0, double>{};
    auto I = integral(x * x).over<0>(0.0, 1.0);

    // Use Simpson's rule with 100 subdivisions
    auto result = I.eval(simpson<100>());
    EXPECT_NEAR(result.value(), 1.0/3.0, 1e-8);
}

TEST(IntegrationMethods, Trapezoidal) {
    using namespace limes::methods;

    auto x = Var<0, double>{};
    auto I = integral(x * x).over<0>(0.0, 1.0);

    // Use trapezoidal rule with 1000 subdivisions
    auto result = I.eval(trapezoidal<1000>());
    EXPECT_NEAR(result.value(), 1.0/3.0, 1e-5);  // Lower accuracy
}

TEST(IntegrationMethods, ComposedAdaptive) {
    using namespace limes::methods;

    auto x = Var<0, double>{};
    auto I = integral(sin(x) * sin(x)).over<0>(0.0, std::numbers::pi);

    // Compose adaptive scheme around Gauss-Legendre
    auto result = I.eval(make_adaptive(gauss<5>(), 1e-10));
    // Integral of sin^2(x) from 0 to pi = pi/2
    EXPECT_NEAR(result.value(), std::numbers::pi / 2.0, 1e-8);
}

TEST(IntegrationMethods, MethodFactory) {
    using namespace limes::methods;

    auto x = Var<0, double>{};
    auto I = integral(exp(-x * x)).over<0>(-2.0, 2.0);

    // Test factory functions
    auto g15 = gauss<15>();  // Use 15 points for better accuracy
    auto mc = monte_carlo_method<double>(50000);

    // Both should give reasonable approximations to sqrt(pi) * erf(2)
    double expected = std::sqrt(std::numbers::pi) * std::erf(2.0);

    auto r1 = I.eval(g15);
    EXPECT_NEAR(r1.value(), expected, 1e-6);

    // Monte Carlo has lower accuracy but should be close
    auto r2 = I.eval(mc.with_seed(42));
    EXPECT_NEAR(r2.value(), expected, 0.1);
}

TEST(IntegrationMethods, DefaultMethod) {
    auto x = Var<0, double>{};
    auto I = integral(x * x).over<0>(0.0, 1.0);

    // Default eval() uses adaptive method
    auto result = I.eval();
    EXPECT_NEAR(result.value(), 1.0/3.0, 1e-8);
}

// =============================================================================
// Named Variable Tests (Phase 4)
// =============================================================================

TEST(NamedVar, BasicEvaluation) {
    auto x = named<0>("x");
    auto y = named<1>("y");

    std::array<double, 2> args{3.0, 4.0};
    EXPECT_DOUBLE_EQ(x.eval(args), 3.0);
    EXPECT_DOUBLE_EQ(y.eval(args), 4.0);
}

TEST(NamedVar, ToString) {
    auto x = named<0>("position");
    auto y = named<1>("velocity");

    EXPECT_EQ(x.to_string(), "position");
    EXPECT_EQ(y.to_string(), "velocity");
}

TEST(NamedVar, Expressions) {
    auto x = named<0>("x");
    auto y = named<1>("y");

    auto f = x * x + y * y;

    std::array<double, 2> args{3.0, 4.0};
    EXPECT_DOUBLE_EQ(f.eval(args), 25.0);  // 3^2 + 4^2
}

TEST(NamedVar, ExpressionToString) {
    auto x = named<0>("x");
    auto y = named<1>("y");

    auto f = x * y;
    std::string str = f.to_string();

    // Should contain variable names instead of x0, x1
    EXPECT_TRUE(str.find("x") != std::string::npos);
    EXPECT_TRUE(str.find("y") != std::string::npos);
}

TEST(NamedVar, Derivative) {
    auto x = named<0>("x");

    auto f = x * x;
    auto df = f.derivative<0>();

    std::array<double, 1> args{3.0};
    EXPECT_DOUBLE_EQ(df.eval(args), 6.0);  // d/dx[x^2] = 2x
}

TEST(NamedVar, WithPrimitives) {
    auto x = named<0>("theta");

    auto f = sin(x);

    std::array<double, 1> args{std::numbers::pi / 2.0};
    EXPECT_NEAR(f.eval(args), 1.0, 1e-10);

    // to_string should use the name
    std::string str = f.to_string();
    EXPECT_TRUE(str.find("theta") != std::string::npos);
}

TEST(NamedVar, Integration) {
    auto t = named<0>("t");

    auto I = integral(t * t).over<0>(0.0, 1.0);
    auto result = I.eval();

    EXPECT_NEAR(result.value(), 1.0/3.0, 1e-8);
}

TEST(NamedVar, VarFactory) {
    // Runtime dimension with name
    auto x = var(0, "x");
    auto y = var(1, "y");

    std::array<double, 2> args{5.0, 7.0};
    EXPECT_DOUBLE_EQ(x.eval(args), 5.0);
    EXPECT_DOUBLE_EQ(y.eval(args), 7.0);

    EXPECT_EQ(x.to_string(), "x");
    EXPECT_EQ(y.to_string(), "y");
}

TEST(NamedVar, VarsXY) {
    auto [x, y] = vars_xy<double>();

    auto f = x + y;

    std::array<double, 2> args{10.0, 20.0};
    EXPECT_DOUBLE_EQ(f.eval(args), 30.0);
}

TEST(NamedVar, VarsXYZ) {
    auto [x, y, z] = vars_xyz<double>("pos_x", "pos_y", "pos_z");

    EXPECT_EQ(x.to_string(), "pos_x");
    EXPECT_EQ(y.to_string(), "pos_y");
    EXPECT_EQ(z.to_string(), "pos_z");

    auto f = x + y + z;
    std::array<double, 3> args{1.0, 2.0, 3.0};
    EXPECT_DOUBLE_EQ(f.eval(args), 6.0);
}

TEST(NamedVar, Arity) {
    auto x = named<0>("x");
    auto y = named<1>("y");
    auto z = named<5>("z");

    EXPECT_EQ(x.arity_v, 1);
    EXPECT_EQ(y.arity_v, 2);
    EXPECT_EQ(z.arity_v, 6);
}

TEST(NamedVar, MixedWithUnnamed) {
    // Named and unnamed variables can coexist
    auto x = named<0>("x");
    auto y = Var<1, double>{};  // unnamed

    auto f = x * y;

    std::array<double, 2> args{3.0, 4.0};
    EXPECT_DOUBLE_EQ(f.eval(args), 12.0);
}

// =============================================================================
// Complex Expression Tests
// =============================================================================

TEST(ExprComplex, SinCosIdentity) {
    auto x = Var<0, double>{};

    // sin^2(x) + cos^2(x) = 1
    auto sin_sq = sin(x) * sin(x);
    auto cos_sq = cos(x) * cos(x);
    auto identity = sin_sq + cos_sq;

    std::array<double, 1> args{0.7};  // Random angle
    EXPECT_NEAR(identity.evaluate(args), 1.0, 1e-10);
}

TEST(ExprComplex, ProductOfExpressions) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // f(x, y) = sin(x) * exp(-y)
    auto f = sin(x) * exp(-y);

    std::array<double, 2> args{std::numbers::pi / 2, 0.0};
    EXPECT_NEAR(f.evaluate(args), 1.0, 1e-10);  // sin(pi/2) * exp(0) = 1
}

TEST(ExprComplex, NestedFunctions) {
    auto x = Var<0, double>{};

    // f(x) = exp(sin(x))
    auto f = exp(sin(x));

    std::array<double, 1> args{0.0};
    EXPECT_NEAR(f.evaluate(args), 1.0, 1e-10);  // exp(sin(0)) = exp(0) = 1
}

// =============================================================================
// Type Parameterized Tests
// =============================================================================

template <typename T>
class ExprTypedTest : public ::testing::Test {
protected:
    static constexpr T tol = std::is_same_v<T, float> ? T(1e-5) : T(1e-10);
};

using FloatTypes = ::testing::Types<float, double, long double>;
TYPED_TEST_SUITE(ExprTypedTest, FloatTypes);

TYPED_TEST(ExprTypedTest, BasicArithmetic) {
    using T = TypeParam;

    auto x = Var<0, T>{};
    auto f = x * x + T(2) * x + T(1);  // (x + 1)^2

    std::array<T, 1> args{T(2)};
    T expected = T(9);  // (2 + 1)^2 = 9

    EXPECT_NEAR(f.evaluate(args), expected, this->tol);
}

TYPED_TEST(ExprTypedTest, Derivative) {
    using T = TypeParam;

    auto x = Var<0, T>{};
    auto f = x * x * x;  // x^3

    // d/dx[x^3] = 3x^2
    auto df = derivative<0>(f);

    std::array<T, 1> args{T(2)};
    T expected = T(12);  // 3 * 4 = 12

    EXPECT_NEAR(df.evaluate(args), expected, this->tol);
}

TYPED_TEST(ExprTypedTest, Integration) {
    using T = TypeParam;

    auto x = Var<0, T>{};

    // Integral of x from 0 to 1 = 0.5
    auto I = integral(x).template over<0>(T(0), T(1));

    auto result = I.evaluate();
    T expected = T(0.5);

    // Use a looser tolerance for integration
    T int_tol = std::is_same_v<T, float> ? T(1e-4) : T(1e-8);
    EXPECT_NEAR(result.value(), expected, int_tol);
}

// =============================================================================
// Expression Simplification Tests
// =============================================================================

TEST(ExprSimplification, ZeroMarkerType) {
    Zero<double> z;

    std::array<double, 0> args{};
    EXPECT_DOUBLE_EQ(z.evaluate(args), 0.0);
    EXPECT_EQ(z.to_string(), "0");

    // Derivative of Zero is Zero
    auto dz = z.derivative<0>();
    EXPECT_DOUBLE_EQ(dz.evaluate(args), 0.0);
}

TEST(ExprSimplification, OneMarkerType) {
    One<double> one;

    std::array<double, 0> args{};
    EXPECT_DOUBLE_EQ(one.evaluate(args), 1.0);
    EXPECT_EQ(one.to_string(), "1");

    // Derivative of One is Zero
    auto done = one.derivative<0>();
    EXPECT_DOUBLE_EQ(done.evaluate(args), 0.0);
}

TEST(ExprSimplification, VarDerivativeReturnsMarkerTypes) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // d/dx[x] should be One<T>
    auto dx_dx = x.derivative<0>();
    static_assert(is_one_v<decltype(dx_dx)>, "d/dx[x] should return One<T>");

    // d/dy[x] should be Zero<T>
    auto dx_dy = x.derivative<1>();
    static_assert(is_zero_v<decltype(dx_dy)>, "d/dy[x] should return Zero<T>");
}

TEST(ExprSimplification, ConstDerivativeReturnsZero) {
    auto c = Const<double>{5.0};

    // d/dx[c] should be Zero<T>
    auto dc = c.derivative<0>();
    static_assert(is_zero_v<decltype(dc)>, "d/dx[const] should return Zero<T>");
}

TEST(ExprSimplification, ZeroAdditionElimination) {
    auto x = Var<0, double>{};
    Zero<double> z;

    // 0 + x = x (returns type R, not Binary)
    auto result1 = z + x;
    static_assert(std::is_same_v<decltype(result1), Var<0, double>>, "0 + x should simplify to x");

    // x + 0 = x (returns type L, not Binary)
    auto result2 = x + z;
    static_assert(std::is_same_v<decltype(result2), Var<0, double>>, "x + 0 should simplify to x");
}

TEST(ExprSimplification, ZeroMultiplicationElimination) {
    auto x = Var<0, double>{};
    Zero<double> z;

    // 0 * x = 0
    auto result1 = z * x;
    static_assert(is_zero_v<decltype(result1)>, "0 * x should simplify to Zero");

    // x * 0 = 0
    auto result2 = x * z;
    static_assert(is_zero_v<decltype(result2)>, "x * 0 should simplify to Zero");
}

TEST(ExprSimplification, OneMultiplicationElimination) {
    auto x = Var<0, double>{};
    One<double> one;

    // 1 * x = x
    auto result1 = one * x;
    static_assert(std::is_same_v<decltype(result1), Var<0, double>>, "1 * x should simplify to x");

    // x * 1 = x
    auto result2 = x * one;
    static_assert(std::is_same_v<decltype(result2), Var<0, double>>, "x * 1 should simplify to x");
}

TEST(ExprSimplification, ZeroDivisionElimination) {
    auto x = Var<0, double>{};
    Zero<double> z;
    One<double> one;

    // 0 / x = 0
    auto result1 = z / x;
    static_assert(is_zero_v<decltype(result1)>, "0 / x should simplify to Zero");

    // x / 1 = x
    auto result2 = x / one;
    static_assert(std::is_same_v<decltype(result2), Var<0, double>>, "x / 1 should simplify to x");
}

TEST(ExprSimplification, ZeroSubtractionElimination) {
    auto x = Var<0, double>{};
    Zero<double> z;

    // x - 0 = x
    auto result = x - z;
    static_assert(std::is_same_v<decltype(result), Var<0, double>>, "x - 0 should simplify to x");
}

TEST(ExprSimplification, NegationOfZero) {
    Zero<double> z;

    // -0 = 0
    auto result = -z;
    static_assert(is_zero_v<decltype(result)>, "-Zero should simplify to Zero");
}

TEST(ExprSimplification, ProductRuleSimplification) {
    auto x = Var<0, double>{};

    // f(x) = x * x = x^2
    auto f = x * x;

    // d/dx[x^2] = 1*x + x*1 = x + x = 2*x (fully simplified via x+x=2*x rule)
    auto df = derivative<0>(f);

    // Check that the expression is simplified to 2*x
    std::string str = df.to_string();
    EXPECT_EQ(str, "(* 2 x0)");

    // Verify numerical correctness
    std::array<double, 1> args{3.0};
    EXPECT_DOUBLE_EQ(df.evaluate(args), 6.0);  // 2 * 3 = 6
}

TEST(ExprSimplification, ProductRuleWithConstant) {
    auto x = Var<0, double>{};
    auto c = Const<double>{2.0};

    // f(x) = 2 * x
    auto f = c * x;

    // d/dx[2*x] = 0*x + 2*1 = 2 (simplified)
    auto df = derivative<0>(f);

    std::string str = df.to_string();
    EXPECT_EQ(str, "2");  // Fully simplified to constant

    std::array<double, 1> args{5.0};
    EXPECT_DOUBLE_EQ(df.evaluate(args), 2.0);
}

TEST(ExprSimplification, SumRuleSimplification) {
    auto x = Var<0, double>{};
    auto c = Const<double>{5.0};

    // f(x) = x + 5
    auto f = x + c;

    // d/dx[x + 5] = 1 + 0 = 1 (simplified from (+ 1 0))
    auto df = derivative<0>(f);

    std::string str = df.to_string();
    EXPECT_EQ(str, "1");  // Simplified to just One

    std::array<double, 1> args{10.0};
    EXPECT_DOUBLE_EQ(df.evaluate(args), 1.0);
}

TEST(ExprSimplification, ConstantFolding) {
    auto c1 = Const<double>{2.0};
    auto c2 = Const<double>{3.0};

    // 2 + 3 = 5 (constant folding)
    auto sum = c1 + c2;
    static_assert(is_const_expr_v<decltype(sum)>, "Const + Const should fold to Const");

    std::array<double, 0> args{};
    EXPECT_DOUBLE_EQ(sum.evaluate(args), 5.0);
    EXPECT_EQ(sum.to_string(), "5");

    // 2 * 3 = 6 (constant folding)
    auto prod = c1 * c2;
    static_assert(is_const_expr_v<decltype(prod)>, "Const * Const should fold to Const");
    EXPECT_DOUBLE_EQ(prod.evaluate(args), 6.0);
    EXPECT_EQ(prod.to_string(), "6");
}

TEST(ExprSimplification, ChainRuleSimplification) {
    auto x = Var<0, double>{};

    // f(x) = sin(x)
    auto f = sin(x);

    // d/dx[sin(x)] = cos(x) * 1 = cos(x) (simplified)
    auto df = derivative<0>(f);

    // Should simplify to just cos(x), not (* cos(x) 1)
    std::string str = df.to_string();
    EXPECT_EQ(str, "(cos x0)");

    std::array<double, 1> args{0.0};
    EXPECT_NEAR(df.evaluate(args), 1.0, 1e-10);  // cos(0) = 1
}

TEST(ExprSimplification, HigherOrderDerivative) {
    auto x = Var<0, double>{};

    // f(x) = x^3 = x * x * x
    auto f = x * x * x;

    // d/dx[x^3] = d/dx[(x*x) * x]
    //           = d/dx[x*x] * x + (x*x) * 1
    //           = (x + x) * x + (x*x)
    auto df = derivative<0>(f);

    // d^2/dx^2[x^3] = 6x
    auto d2f = derivative<0>(df);

    std::array<double, 1> args{2.0};
    EXPECT_DOUBLE_EQ(d2f.evaluate(args), 12.0);  // 6 * 2 = 12
}

TEST(ExprSimplification, MixedExpressionSimplification) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // f(x, y) = x * y
    auto f = x * y;

    // d/dx[x*y] = 1*y + x*0 = y (simplified)
    auto df_dx = derivative<0>(f);
    // d/dy[x*y] = 0*y + x*1 = x (simplified)
    auto df_dy = derivative<1>(f);

    std::string str_dx = df_dx.to_string();
    std::string str_dy = df_dy.to_string();

    EXPECT_EQ(str_dx, "x1");  // Simplified to just y
    EXPECT_EQ(str_dy, "x0");  // Simplified to just x

    std::array<double, 2> args{3.0, 4.0};
    EXPECT_DOUBLE_EQ(df_dx.evaluate(args), 4.0);  // y = 4
    EXPECT_DOUBLE_EQ(df_dy.evaluate(args), 3.0);  // x = 3
}

// =============================================================================
// Algebraic Cancellation Tests (v2.1)
// =============================================================================

TEST(ExprAlgebraic, SubtractionCancellation) {
    auto x = Var<0, double>{};

    // x - x = 0
    auto diff = x - x;
    static_assert(is_zero_v<decltype(diff)>, "x - x should simplify to Zero");

    std::array<double, 1> args{5.0};
    EXPECT_DOUBLE_EQ(diff.evaluate(args), 0.0);
    EXPECT_EQ(diff.to_string(), "0");
}

TEST(ExprAlgebraic, DivisionCancellation) {
    auto x = Var<0, double>{};

    // x / x = 1
    auto quot = x / x;
    static_assert(is_one_v<decltype(quot)>, "x / x should simplify to One");

    std::array<double, 1> args{5.0};
    EXPECT_DOUBLE_EQ(quot.evaluate(args), 1.0);
    EXPECT_EQ(quot.to_string(), "1");
}

TEST(ExprAlgebraic, AdditionDoubling) {
    auto x = Var<0, double>{};

    // x + x = 2*x
    auto sum = x + x;

    std::array<double, 1> args{5.0};
    EXPECT_DOUBLE_EQ(sum.evaluate(args), 10.0);  // 2 * 5 = 10

    // Should be (* 2 x0)
    std::string str = sum.to_string();
    EXPECT_EQ(str, "(* 2 x0)");
}

TEST(ExprAlgebraic, DoubleNegation) {
    auto x = Var<0, double>{};

    // --x = x
    auto neg_x = -x;
    auto neg_neg_x = -neg_x;

    // Should simplify back to x
    static_assert(std::is_same_v<decltype(neg_neg_x), Var<0, double>>,
                  "--x should simplify to x");

    std::array<double, 1> args{7.0};
    EXPECT_DOUBLE_EQ(neg_neg_x.evaluate(args), 7.0);
    EXPECT_EQ(neg_neg_x.to_string(), "x0");
}

TEST(ExprAlgebraic, ComplexCancellation) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // (x + y) - (x + y) = 0
    auto sum = x + y;
    auto diff = sum - sum;
    static_assert(is_zero_v<decltype(diff)>, "(x+y) - (x+y) should be Zero");

    std::array<double, 2> args{3.0, 4.0};
    EXPECT_DOUBLE_EQ(diff.evaluate(args), 0.0);
}

// =============================================================================
// Power Expression Tests (v2.1)
// =============================================================================

TEST(ExprPow, BasicPower) {
    auto x = Var<0, double>{};

    // x^3
    auto f = pow<3>(x);

    std::array<double, 1> args{2.0};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 8.0);  // 2^3 = 8
    EXPECT_EQ(f.to_string(), "(^ x0 3)");
}

TEST(ExprPow, PowerZero) {
    auto x = Var<0, double>{};

    // x^0 = 1
    auto f = pow<0>(x);
    static_assert(is_one_v<decltype(f)>, "x^0 should simplify to One");

    std::array<double, 1> args{42.0};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 1.0);
}

TEST(ExprPow, PowerOne) {
    auto x = Var<0, double>{};

    // x^1 = x
    auto f = pow<1>(x);
    static_assert(std::is_same_v<decltype(f), Var<0, double>>, "x^1 should simplify to x");

    std::array<double, 1> args{5.0};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 5.0);
}

TEST(ExprPow, Square) {
    auto x = Var<0, double>{};

    // square(x) = x^2
    auto f = square(x);
    static_assert(is_pow_v<decltype(f)>, "square(x) should be Pow type");

    std::array<double, 1> args{4.0};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 16.0);  // 4^2 = 16
    EXPECT_EQ(f.to_string(), "(^ x0 2)");
}

TEST(ExprPow, Cube) {
    auto x = Var<0, double>{};

    // cube(x) = x^3
    auto f = cube(x);
    static_assert(is_pow_v<decltype(f)>, "cube(x) should be Pow type");

    std::array<double, 1> args{3.0};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 27.0);  // 3^3 = 27
    EXPECT_EQ(f.to_string(), "(^ x0 3)");
}

TEST(ExprPow, NegativeExponent) {
    auto x = Var<0, double>{};

    // x^(-2) = 1/x^2
    auto f = pow<-2>(x);

    std::array<double, 1> args{2.0};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 0.25);  // 1/4
    EXPECT_EQ(f.to_string(), "(^ x0 -2)");
}

TEST(ExprPow, DerivativePowerTwo) {
    auto x = Var<0, double>{};

    // f(x) = x^2
    auto f = square(x);

    // d/dx[x^2] = 2x
    auto df = derivative<0>(f);

    std::array<double, 1> args{3.0};
    EXPECT_DOUBLE_EQ(df.evaluate(args), 6.0);  // 2 * 3 = 6
}

TEST(ExprPow, DerivativePowerThree) {
    auto x = Var<0, double>{};

    // f(x) = x^3
    auto f = pow<3>(x);

    // d/dx[x^3] = 3x^2
    auto df = derivative<0>(f);

    std::array<double, 1> args{2.0};
    EXPECT_DOUBLE_EQ(df.evaluate(args), 12.0);  // 3 * 4 = 12
}

TEST(ExprPow, DerivativeHigherPower) {
    auto x = Var<0, double>{};

    // f(x) = x^5
    auto f = pow<5>(x);

    // d/dx[x^5] = 5x^4
    auto df = derivative<0>(f);

    std::array<double, 1> args{2.0};
    EXPECT_DOUBLE_EQ(df.evaluate(args), 80.0);  // 5 * 16 = 80
}

TEST(ExprPow, SecondDerivative) {
    auto x = Var<0, double>{};

    // f(x) = x^3
    auto f = pow<3>(x);

    // d/dx[x^3] = 3x^2
    auto df = derivative<0>(f);

    // d^2/dx^2[x^3] = 6x
    auto d2f = derivative<0>(df);

    std::array<double, 1> args{4.0};
    EXPECT_DOUBLE_EQ(d2f.evaluate(args), 24.0);  // 6 * 4 = 24
}

TEST(ExprPow, DerivativeOfConstantVariable) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // f(x, y) = x^2 (no y dependence)
    auto f = square(x);

    // d/dy[x^2] = 0
    auto df_dy = derivative<1>(f);
    static_assert(is_zero_v<decltype(df_dy)>, "d/dy[x^2] should be Zero");

    std::array<double, 2> args{3.0, 4.0};
    EXPECT_DOUBLE_EQ(df_dy.evaluate(args), 0.0);
}

TEST(ExprPow, ComposedPower) {
    auto x = Var<0, double>{};

    // f(x) = (x + 1)^2
    auto inner = x + Const<double>{1.0};
    auto f = square(inner);

    std::array<double, 1> args{2.0};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 9.0);  // (2+1)^2 = 9

    // d/dx[(x+1)^2] = 2(x+1) * 1 = 2(x+1)
    auto df = derivative<0>(f);
    EXPECT_DOUBLE_EQ(df.evaluate(args), 6.0);  // 2 * 3 = 6
}

TEST(ExprPow, IntegrationWithPower) {
    auto x = Var<0, double>{};

    // Integral of x^2 from 0 to 1 = 1/3
    auto f = square(x);
    auto I = integral(f).over<0>(0.0, 1.0);

    auto result = I.evaluate();
    EXPECT_NEAR(result.value(), 1.0/3.0, 1e-8);
}

TEST(ExprPow, TypeTraits) {
    auto x = Var<0, double>{};

    auto p2 = pow<2>(x);
    auto p3 = pow<3>(x);

    static_assert(is_pow_v<decltype(p2)>, "pow<2>(x) should be Pow");
    static_assert(is_pow_v<decltype(p3)>, "pow<3>(x) should be Pow");
    static_assert(!is_pow_v<Var<0, double>>, "Var should not be Pow");
    static_assert(!is_pow_v<Const<double>>, "Const should not be Pow");

    // Check exponent extraction
    static_assert(decltype(p2)::exponent == 2);
    static_assert(decltype(p3)::exponent == 3);
}

// =============================================================================
// Negation Type Trait Tests (v2.1)
// =============================================================================

TEST(ExprNegation, IsNegationTrait) {
    auto x = Var<0, double>{};
    auto neg_x = -x;

    static_assert(!is_negation_v<Var<0, double>>, "Var should not be negation");
    static_assert(!is_negation_v<Const<double>>, "Const should not be negation");
    static_assert(is_negation_v<decltype(neg_x)>, "-x should be negation");
}

TEST(ExprNegation, TripleNegation) {
    auto x = Var<0, double>{};

    // ---x = -x (two negations cancel)
    auto neg_x = -x;
    auto neg_neg_x = -neg_x;   // = x
    auto neg_neg_neg_x = -neg_neg_x;  // = -x

    static_assert(is_negation_v<decltype(neg_neg_neg_x)>, "---x should be -x");

    std::array<double, 1> args{5.0};
    EXPECT_DOUBLE_EQ(neg_neg_neg_x.evaluate(args), -5.0);
}

// =============================================================================
// Extended Primitive Function Tests (v2.2)
// =============================================================================

// Tan
TEST(ExprPrimitives, Tan) {
    auto x = Var<0, double>{};
    auto f = tan(x);

    std::array<double, 1> args{std::numbers::pi / 4};
    EXPECT_NEAR(f.evaluate(args), 1.0, 1e-10);  // tan(π/4) = 1
}

TEST(ExprPrimitives, TanDerivative) {
    auto x = Var<0, double>{};
    auto f = tan(x);
    auto df = derivative<0>(f);  // sec²(x)

    std::array<double, 1> args{0.0};
    EXPECT_NEAR(df.evaluate(args), 1.0, 1e-10);  // sec²(0) = 1
}

// Sinh
TEST(ExprPrimitives, Sinh) {
    auto x = Var<0, double>{};
    auto f = sinh(x);

    std::array<double, 1> args{0.0};
    EXPECT_NEAR(f.evaluate(args), 0.0, 1e-10);  // sinh(0) = 0
}

TEST(ExprPrimitives, SinhDerivative) {
    auto x = Var<0, double>{};
    auto f = sinh(x);
    auto df = derivative<0>(f);  // cosh(x)

    std::array<double, 1> args{0.0};
    EXPECT_NEAR(df.evaluate(args), 1.0, 1e-10);  // cosh(0) = 1
}

// Cosh
TEST(ExprPrimitives, Cosh) {
    auto x = Var<0, double>{};
    auto f = cosh(x);

    std::array<double, 1> args{0.0};
    EXPECT_NEAR(f.evaluate(args), 1.0, 1e-10);  // cosh(0) = 1
}

TEST(ExprPrimitives, CoshDerivative) {
    auto x = Var<0, double>{};
    auto f = cosh(x);
    auto df = derivative<0>(f);  // sinh(x)

    std::array<double, 1> args{0.0};
    EXPECT_NEAR(df.evaluate(args), 0.0, 1e-10);  // sinh(0) = 0
}

// Tanh
TEST(ExprPrimitives, Tanh) {
    auto x = Var<0, double>{};
    auto f = tanh(x);

    std::array<double, 1> args{0.0};
    EXPECT_NEAR(f.evaluate(args), 0.0, 1e-10);  // tanh(0) = 0
}

TEST(ExprPrimitives, TanhDerivative) {
    auto x = Var<0, double>{};
    auto f = tanh(x);
    auto df = derivative<0>(f);  // sech²(x) = 1 - tanh²(x)

    std::array<double, 1> args{0.0};
    EXPECT_NEAR(df.evaluate(args), 1.0, 1e-10);  // sech²(0) = 1
}

// Asin
TEST(ExprPrimitives, Asin) {
    auto x = Var<0, double>{};
    auto f = asin(x);

    std::array<double, 1> args{0.5};
    EXPECT_NEAR(f.evaluate(args), std::numbers::pi / 6, 1e-10);  // asin(0.5) = π/6
}

TEST(ExprPrimitives, AsinDerivative) {
    auto x = Var<0, double>{};
    auto f = asin(x);
    auto df = derivative<0>(f);  // 1/√(1-x²)

    std::array<double, 1> args{0.0};
    EXPECT_NEAR(df.evaluate(args), 1.0, 1e-10);  // 1/√(1-0) = 1
}

// Acos
TEST(ExprPrimitives, Acos) {
    auto x = Var<0, double>{};
    auto f = acos(x);

    std::array<double, 1> args{0.5};
    EXPECT_NEAR(f.evaluate(args), std::numbers::pi / 3, 1e-10);  // acos(0.5) = π/3
}

TEST(ExprPrimitives, AcosDerivative) {
    auto x = Var<0, double>{};
    auto f = acos(x);
    auto df = derivative<0>(f);  // -1/√(1-x²)

    std::array<double, 1> args{0.0};
    EXPECT_NEAR(df.evaluate(args), -1.0, 1e-10);  // -1/√(1-0) = -1
}

// Atan
TEST(ExprPrimitives, Atan) {
    auto x = Var<0, double>{};
    auto f = atan(x);

    std::array<double, 1> args{1.0};
    EXPECT_NEAR(f.evaluate(args), std::numbers::pi / 4, 1e-10);  // atan(1) = π/4
}

TEST(ExprPrimitives, AtanDerivative) {
    auto x = Var<0, double>{};
    auto f = atan(x);
    auto df = derivative<0>(f);  // 1/(1+x²)

    std::array<double, 1> args{0.0};
    EXPECT_NEAR(df.evaluate(args), 1.0, 1e-10);  // 1/(1+0) = 1
}

// ToString tests for extended primitives
TEST(ExprPrimitives, ExtendedToString) {
    auto x = Var<0, double>{};

    EXPECT_EQ(tan(x).to_string(), "(tan x0)");
    EXPECT_EQ(sinh(x).to_string(), "(sinh x0)");
    EXPECT_EQ(cosh(x).to_string(), "(cosh x0)");
    EXPECT_EQ(tanh(x).to_string(), "(tanh x0)");
    EXPECT_EQ(asin(x).to_string(), "(asin x0)");
    EXPECT_EQ(acos(x).to_string(), "(acos x0)");
    EXPECT_EQ(atan(x).to_string(), "(atan x0)");
}

// Chain rule with new primitives
TEST(ExprChainRule, TanOfSquare) {
    auto x = Var<0, double>{};
    auto f = tan(x * x);
    auto df = derivative<0>(f);  // sec²(x²) * 2x

    std::array<double, 1> args{0.0};
    EXPECT_NEAR(df.evaluate(args), 0.0, 1e-10);  // sec²(0) * 0 = 0
}

TEST(ExprChainRule, HyperbolicIdentity) {
    auto x = Var<0, double>{};
    // cosh²(x) - sinh²(x) = 1
    auto identity = cosh(x) * cosh(x) - sinh(x) * sinh(x);

    std::array<double, 1> args{1.5};
    EXPECT_NEAR(identity.evaluate(args), 1.0, 1e-10);
}

// =============================================================================
// Binary Function Tests (v2.3) - pow, max, min
// =============================================================================

// -----------------------------------------------------------------------------
// Runtime Pow Evaluation Tests
// -----------------------------------------------------------------------------

TEST(ExprBinaryFunc, PowExprScalar) {
    auto x = Var<0, double>{};

    // pow(x, 2.5) at x=4 → 4^2.5 = 32
    auto f = pow(x, 2.5);

    std::array<double, 1> args{4.0};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 32.0);
}

TEST(ExprBinaryFunc, PowExprExpr) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // pow(x, y) at (2, 3) → 2^3 = 8
    auto f = pow(x, y);

    std::array<double, 2> args{2.0, 3.0};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 8.0);
    EXPECT_EQ(f.arity_v, 2);
}

TEST(ExprBinaryFunc, PowScalarExpr) {
    auto x = Var<0, double>{};

    // pow(2.0, x) at x=3 → 2^3 = 8
    auto f = pow(2.0, x);

    std::array<double, 1> args{3.0};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 8.0);
}

TEST(ExprBinaryFunc, PowFractionalExponent) {
    auto x = Var<0, double>{};

    // pow(x, 0.5) = sqrt(x)
    auto f = pow(x, 0.5);

    std::array<double, 1> args{9.0};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 3.0);
}

// -----------------------------------------------------------------------------
// Max Evaluation Tests
// -----------------------------------------------------------------------------

TEST(ExprBinaryFunc, MaxWhenFirstGreater) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // max(x, y) when x > y
    auto f = max(x, y);

    std::array<double, 2> args{5.0, 3.0};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 5.0);
}

TEST(ExprBinaryFunc, MaxWhenSecondGreater) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // max(x, y) when y > x
    auto f = max(x, y);

    std::array<double, 2> args{3.0, 5.0};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 5.0);
}

TEST(ExprBinaryFunc, MaxWhenEqual) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // max(x, y) when x = y
    auto f = max(x, y);

    std::array<double, 2> args{4.0, 4.0};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 4.0);
}

TEST(ExprBinaryFunc, MaxWithScalar) {
    auto x = Var<0, double>{};

    // max(x, 0) - ReLU-like function
    auto f = max(x, 0.0);

    std::array<double, 1> args_pos{3.0};
    std::array<double, 1> args_neg{-3.0};

    EXPECT_DOUBLE_EQ(f.evaluate(args_pos), 3.0);
    EXPECT_DOUBLE_EQ(f.evaluate(args_neg), 0.0);
}

// -----------------------------------------------------------------------------
// Min Evaluation Tests
// -----------------------------------------------------------------------------

TEST(ExprBinaryFunc, MinWhenFirstSmaller) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // min(x, y) when x < y
    auto f = min(x, y);

    std::array<double, 2> args{3.0, 5.0};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 3.0);
}

TEST(ExprBinaryFunc, MinWhenSecondSmaller) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // min(x, y) when y < x
    auto f = min(x, y);

    std::array<double, 2> args{5.0, 3.0};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 3.0);
}

TEST(ExprBinaryFunc, MinWithScalar) {
    auto x = Var<0, double>{};

    // min(x, 5)
    auto f = min(x, 5.0);

    std::array<double, 1> args_low{3.0};
    std::array<double, 1> args_high{7.0};

    EXPECT_DOUBLE_EQ(f.evaluate(args_low), 3.0);
    EXPECT_DOUBLE_EQ(f.evaluate(args_high), 5.0);
}

// -----------------------------------------------------------------------------
// Pow Derivative Tests
// -----------------------------------------------------------------------------

TEST(ExprBinaryFunc, PowDerivativeConstantExponent) {
    auto x = Var<0, double>{};

    // f(x) = x^3.0
    // d/dx[x^n] = n*x^(n-1) (when n is constant)
    auto f = pow(x, 3.0);
    auto df = derivative<0>(f);

    // At x=2: 3 * 2^2 = 12
    std::array<double, 1> args{2.0};
    EXPECT_NEAR(df.evaluate(args), 12.0, 1e-10);
}

TEST(ExprBinaryFunc, PowDerivativeConstantBase) {
    auto x = Var<0, double>{};

    // f(x) = 2^x
    // d/dx[a^x] = a^x * ln(a) (when a is constant)
    auto f = pow(2.0, x);
    auto df = derivative<0>(f);

    // At x=3: 2^3 * ln(2) = 8 * ln(2)
    std::array<double, 1> args{3.0};
    double expected = 8.0 * std::log(2.0);
    EXPECT_NEAR(df.evaluate(args), expected, 1e-10);
}

TEST(ExprBinaryFunc, PowDerivativeGeneral) {
    auto x = Var<0, double>{};

    // f(x) = x^x
    // d/dx[x^x] = x^x * (1 + ln(x))
    auto f = pow(x, x);
    auto df = derivative<0>(f);

    // At x=2: 2^2 * (1 + ln(2)) = 4 * (1 + ln(2))
    std::array<double, 1> args{2.0};
    double expected = 4.0 * (1.0 + std::log(2.0));
    EXPECT_NEAR(df.evaluate(args), expected, 1e-10);
}

TEST(ExprBinaryFunc, PowDerivativeChainRule) {
    auto x = Var<0, double>{};

    // f(x) = sin(x)^2 (using runtime pow)
    // d/dx[sin(x)^2] = 2*sin(x)*cos(x)
    auto f = pow(sin(x), 2.0);
    auto df = derivative<0>(f);

    double val = std::numbers::pi / 4;
    std::array<double, 1> args{val};
    double expected = 2.0 * std::sin(val) * std::cos(val);
    EXPECT_NEAR(df.evaluate(args), expected, 1e-10);
}

// -----------------------------------------------------------------------------
// Max/Min Derivative Tests
// -----------------------------------------------------------------------------

TEST(ExprBinaryFunc, MaxDerivativeWhenFirstGreater) {
    auto x = Var<0, double>{};

    // f(x) = max(x, 0) - ReLU derivative
    // d/dx[max(x, 0)] = 1 when x > 0
    auto f = max(x, 0.0);
    auto df = derivative<0>(f);

    std::array<double, 1> args{2.0};
    EXPECT_NEAR(df.evaluate(args), 1.0, 1e-10);
}

TEST(ExprBinaryFunc, MaxDerivativeWhenSecondGreater) {
    auto x = Var<0, double>{};

    // f(x) = max(x, 0) - ReLU derivative
    // d/dx[max(x, 0)] = 0 when x < 0
    auto f = max(x, 0.0);
    auto df = derivative<0>(f);

    std::array<double, 1> args{-2.0};
    EXPECT_NEAR(df.evaluate(args), 0.0, 1e-10);
}

TEST(ExprBinaryFunc, MinDerivativeWhenFirstSmaller) {
    auto x = Var<0, double>{};

    // f(x) = min(x, 5)
    // d/dx[min(x, 5)] = 1 when x < 5
    auto f = min(x, 5.0);
    auto df = derivative<0>(f);

    std::array<double, 1> args{2.0};
    EXPECT_NEAR(df.evaluate(args), 1.0, 1e-10);
}

TEST(ExprBinaryFunc, MinDerivativeWhenSecondSmaller) {
    auto x = Var<0, double>{};

    // f(x) = min(x, 5)
    // d/dx[min(x, 5)] = 0 when x > 5
    auto f = min(x, 5.0);
    auto df = derivative<0>(f);

    std::array<double, 1> args{8.0};
    EXPECT_NEAR(df.evaluate(args), 0.0, 1e-10);
}

// -----------------------------------------------------------------------------
// Simplification Tests
// -----------------------------------------------------------------------------

TEST(ExprBinaryFunc, PowZeroSimplification) {
    auto x = Var<0, double>{};
    Zero<double> z;

    // pow(x, 0) = 1
    auto f = pow(x, z);
    static_assert(is_one_v<decltype(f)>, "pow(x, Zero) should simplify to One");

    std::array<double, 1> args{42.0};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 1.0);
}

TEST(ExprBinaryFunc, PowOneSimplification) {
    auto x = Var<0, double>{};
    One<double> one;

    // pow(x, 1) = x
    auto f = pow(x, one);
    static_assert(std::is_same_v<decltype(f), Var<0, double>>, "pow(x, One) should simplify to x");

    std::array<double, 1> args{7.0};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 7.0);
}

TEST(ExprBinaryFunc, PowOneBaseSimplification) {
    auto y = Var<0, double>{};
    One<double> one;

    // pow(1, y) = 1
    auto f = pow(one, y);
    static_assert(is_one_v<decltype(f)>, "pow(One, y) should simplify to One");

    std::array<double, 1> args{100.0};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 1.0);
}

TEST(ExprBinaryFunc, MaxSameTypeSimplification) {
    auto x = Var<0, double>{};

    // max(x, x) = x
    auto f = max(x, x);
    static_assert(std::is_same_v<decltype(f), Var<0, double>>, "max(x, x) should simplify to x");

    std::array<double, 1> args{5.0};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 5.0);
}

TEST(ExprBinaryFunc, MinSameTypeSimplification) {
    auto x = Var<0, double>{};

    // min(x, x) = x
    auto f = min(x, x);
    static_assert(std::is_same_v<decltype(f), Var<0, double>>, "min(x, x) should simplify to x");

    std::array<double, 1> args{5.0};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 5.0);
}

TEST(ExprBinaryFunc, PowConstantFolding) {
    auto c1 = Const<double>{2.0};
    auto c2 = Const<double>{3.0};

    // pow(2, 3) = 8
    auto f = pow(c1, c2);
    static_assert(is_const_expr_v<decltype(f)>, "pow(Const, Const) should fold to Const");

    std::array<double, 0> args{};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 8.0);
}

TEST(ExprBinaryFunc, MaxConstantFolding) {
    auto c1 = Const<double>{2.0};
    auto c2 = Const<double>{5.0};

    // max(2, 5) = 5
    auto f = max(c1, c2);
    static_assert(is_const_expr_v<decltype(f)>, "max(Const, Const) should fold to Const");

    std::array<double, 0> args{};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 5.0);
}

TEST(ExprBinaryFunc, MinConstantFolding) {
    auto c1 = Const<double>{2.0};
    auto c2 = Const<double>{5.0};

    // min(2, 5) = 2
    auto f = min(c1, c2);
    static_assert(is_const_expr_v<decltype(f)>, "min(Const, Const) should fold to Const");

    std::array<double, 0> args{};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 2.0);
}

// -----------------------------------------------------------------------------
// Edge Cases
// -----------------------------------------------------------------------------

TEST(ExprBinaryFunc, PowZeroBase) {
    auto x = Var<0, double>{};

    // pow(0, 2) = 0
    auto f = pow(0.0, x);

    std::array<double, 1> args{2.0};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 0.0);
}

TEST(ExprBinaryFunc, PowNegativeBase) {
    auto x = Var<0, double>{};

    // pow(-3, 2) = 9
    auto f = pow(x, 2.0);

    std::array<double, 1> args{-3.0};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 9.0);
}

TEST(ExprBinaryFunc, PowNegativeBaseFractionalExponent) {
    auto x = Var<0, double>{};

    // pow(-1, 2.5) behavior: negative base with fractional exponent
    // Note: Result is platform-defined (NaN or domain error).
    // With -ffast-math, std::isnan may not work correctly.
    // We verify the expression matches std::pow behavior.
    auto f = pow(x, 2.5);

    std::array<double, 1> args{-1.0};
    double expr_result = f.evaluate(args);
    double std_result = std::pow(-1.0, 2.5);

    // Both should behave the same way (either both NaN or both some value)
    // Use bit comparison to handle NaN correctly even with -ffast-math
    EXPECT_EQ(std::memcmp(&expr_result, &std_result, sizeof(double)), 0);
}

// -----------------------------------------------------------------------------
// to_string Tests
// -----------------------------------------------------------------------------

TEST(ExprBinaryFunc, PowToString) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    auto f = pow(x, y);
    EXPECT_EQ(f.to_string(), "(pow x0 x1)");
}

TEST(ExprBinaryFunc, MaxToString) {
    auto x = Var<0, double>{};
    auto f = max(x, 5.0);

    EXPECT_EQ(f.to_string(), "(max x0 5)");
}

TEST(ExprBinaryFunc, MinToString) {
    auto x = Var<0, double>{};
    auto f = min(x, 3.0);

    EXPECT_EQ(f.to_string(), "(min x0 3)");
}

// -----------------------------------------------------------------------------
// Type Traits Tests
// -----------------------------------------------------------------------------

TEST(ExprBinaryFunc, TypeTraits) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    auto p = pow(x, y);
    auto mx = max(x, y);
    auto mn = min(x, y);

    // is_binary_func_v
    static_assert(is_binary_func_v<decltype(p)>, "pow should be BinaryFunc");
    static_assert(is_binary_func_v<decltype(mx)>, "max should be BinaryFunc");
    static_assert(is_binary_func_v<decltype(mn)>, "min should be BinaryFunc");
    static_assert(!is_binary_func_v<Var<0, double>>, "Var should not be BinaryFunc");
    static_assert(!is_binary_func_v<Const<double>>, "Const should not be BinaryFunc");

    // is_runtime_pow_v
    static_assert(is_runtime_pow_v<decltype(p)>, "pow should be RuntimePow");
    static_assert(!is_runtime_pow_v<decltype(mx)>, "max should not be RuntimePow");
    static_assert(!is_runtime_pow_v<decltype(mn)>, "min should not be RuntimePow");

    // is_max_v
    static_assert(is_max_v<decltype(mx)>, "max should be Max");
    static_assert(!is_max_v<decltype(p)>, "pow should not be Max");
    static_assert(!is_max_v<decltype(mn)>, "min should not be Max");

    // is_min_v
    static_assert(is_min_v<decltype(mn)>, "min should be Min");
    static_assert(!is_min_v<decltype(p)>, "pow should not be Min");
    static_assert(!is_min_v<decltype(mx)>, "max should not be Min");
}

// -----------------------------------------------------------------------------
// Integration Tests
// -----------------------------------------------------------------------------

TEST(ExprBinaryFunc, IntegralOfMaxReLU) {
    auto x = Var<0, double>{};

    // Integral of max(x, 0) from -1 to 1
    // = Integral of 0 from -1 to 0 + Integral of x from 0 to 1
    // = 0 + 0.5 = 0.5
    auto f = max(x, 0.0);
    auto I = integral(f).over<0>(-1.0, 1.0);

    auto result = I.evaluate();
    EXPECT_NEAR(result.value(), 0.5, 1e-6);
}

TEST(ExprBinaryFunc, IntegralOfMin) {
    auto x = Var<0, double>{};

    // Integral of min(x, 0.5) from 0 to 1
    // = Integral of x from 0 to 0.5 + Integral of 0.5 from 0.5 to 1
    // = 0.125 + 0.25 = 0.375
    auto f = min(x, 0.5);
    auto I = integral(f).over<0>(0.0, 1.0);

    auto result = I.evaluate();
    EXPECT_NEAR(result.value(), 0.375, 1e-6);
}

TEST(ExprBinaryFunc, IntegralOfPow) {
    auto x = Var<0, double>{};

    // Integral of x^2.0 from 0 to 1 = 1/3
    auto f = pow(x, 2.0);
    auto I = integral(f).over<0>(0.0, 1.0);

    auto result = I.evaluate();
    EXPECT_NEAR(result.value(), 1.0/3.0, 1e-8);
}

// -----------------------------------------------------------------------------
// Type-Parameterized Binary Function Tests
// -----------------------------------------------------------------------------

TYPED_TEST(ExprTypedTest, BinaryFuncPow) {
    using T = TypeParam;

    auto x = Var<0, T>{};
    auto f = pow(x, T(2.5));

    std::array<T, 1> args{T(4)};
    T expected = T(32);  // 4^2.5 = 32

    EXPECT_NEAR(f.evaluate(args), expected, this->tol);
}

TYPED_TEST(ExprTypedTest, BinaryFuncMax) {
    using T = TypeParam;

    auto x = Var<0, T>{};
    auto f = max(x, T(3));

    std::array<T, 1> args_low{T(1)};
    std::array<T, 1> args_high{T(5)};

    EXPECT_NEAR(f.evaluate(args_low), T(3), this->tol);
    EXPECT_NEAR(f.evaluate(args_high), T(5), this->tol);
}

TYPED_TEST(ExprTypedTest, BinaryFuncMin) {
    using T = TypeParam;

    auto x = Var<0, T>{};
    auto f = min(x, T(3));

    std::array<T, 1> args_low{T(1)};
    std::array<T, 1> args_high{T(5)};

    EXPECT_NEAR(f.evaluate(args_low), T(1), this->tol);
    EXPECT_NEAR(f.evaluate(args_high), T(3), this->tol);
}

// =============================================================================
// Inverse Hyperbolic Function Tests (v2.4)
// =============================================================================

// Asinh tests
TEST(ExprPrimitives, Asinh) {
    auto x = Var<0, double>{};
    auto f = asinh(x);

    std::array<double, 1> args0{0.0};
    EXPECT_NEAR(f.evaluate(args0), 0.0, 1e-10);  // asinh(0) = 0

    std::array<double, 1> args1{1.0};
    EXPECT_NEAR(f.evaluate(args1), std::asinh(1.0), 1e-10);  // asinh(1) ≈ 0.8814
}

TEST(ExprPrimitives, AsinhDerivative) {
    auto x = Var<0, double>{};
    auto f = asinh(x);
    auto df = derivative<0>(f);  // 1/√(1+x²)

    std::array<double, 1> args0{0.0};
    EXPECT_NEAR(df.evaluate(args0), 1.0, 1e-10);  // 1/√(1+0) = 1

    std::array<double, 1> args1{1.0};
    double expected = 1.0 / std::sqrt(2.0);  // 1/√2
    EXPECT_NEAR(df.evaluate(args1), expected, 1e-10);
}

// Acosh tests
TEST(ExprPrimitives, Acosh) {
    auto x = Var<0, double>{};
    auto f = acosh(x);

    std::array<double, 1> args1{1.0};
    EXPECT_NEAR(f.evaluate(args1), 0.0, 1e-10);  // acosh(1) = 0

    std::array<double, 1> args2{2.0};
    EXPECT_NEAR(f.evaluate(args2), std::acosh(2.0), 1e-10);  // acosh(2) ≈ 1.3170
}

TEST(ExprPrimitives, AcoshDerivative) {
    auto x = Var<0, double>{};
    auto f = acosh(x);
    auto df = derivative<0>(f);  // 1/√(x²-1)

    std::array<double, 1> args{2.0};
    double expected = 1.0 / std::sqrt(3.0);  // 1/√(4-1) = 1/√3
    EXPECT_NEAR(df.evaluate(args), expected, 1e-10);
}

// Atanh tests
TEST(ExprPrimitives, Atanh) {
    auto x = Var<0, double>{};
    auto f = atanh(x);

    std::array<double, 1> args0{0.0};
    EXPECT_NEAR(f.evaluate(args0), 0.0, 1e-10);  // atanh(0) = 0

    std::array<double, 1> args05{0.5};
    EXPECT_NEAR(f.evaluate(args05), std::atanh(0.5), 1e-10);  // atanh(0.5) ≈ 0.5493
}

TEST(ExprPrimitives, AtanhDerivative) {
    auto x = Var<0, double>{};
    auto f = atanh(x);
    auto df = derivative<0>(f);  // 1/(1-x²)

    std::array<double, 1> args0{0.0};
    EXPECT_NEAR(df.evaluate(args0), 1.0, 1e-10);  // 1/(1-0) = 1

    std::array<double, 1> args05{0.5};
    double expected = 1.0 / (1.0 - 0.25);  // 1/(1-0.25) = 4/3
    EXPECT_NEAR(df.evaluate(args05), expected, 1e-10);
}

// ToString tests for inverse hyperbolic functions
TEST(ExprPrimitives, InverseHyperbolicToString) {
    auto x = Var<0, double>{};

    EXPECT_EQ(asinh(x).to_string(), "(asinh x0)");
    EXPECT_EQ(acosh(x).to_string(), "(acosh x0)");
    EXPECT_EQ(atanh(x).to_string(), "(atanh x0)");
}

// Chain rule with inverse hyperbolic functions
TEST(ExprChainRule, AsinhOfSquare) {
    auto x = Var<0, double>{};
    auto f = asinh(x * x);
    auto df = derivative<0>(f);  // 1/√(1+(x²)²) * 2x

    std::array<double, 1> args{1.0};
    double u = 1.0;  // x²
    double expected = (1.0 / std::sqrt(1.0 + u*u)) * 2.0;
    EXPECT_NEAR(df.evaluate(args), expected, 1e-10);
}

// Integration of inverse hyperbolic functions
TEST(ExprIntegral, AsinhIntegration) {
    auto x = Var<0, double>{};

    // Integral of asinh(x) from 0 to 1
    // ∫asinh(x)dx = x*asinh(x) - √(1+x²) + C
    // From 0 to 1: (1*asinh(1) - √2) - (0 - 1) = asinh(1) - √2 + 1
    auto f = asinh(x);
    auto I = integral(f).over<0>(0.0, 1.0);

    double expected = std::asinh(1.0) - std::sqrt(2.0) + 1.0;
    auto result = I.evaluate();
    EXPECT_NEAR(result.value(), expected, 1e-6);
}

// Type aliases tests
TEST(ExprPrimitives, InverseHyperbolicTypeAliases) {
    auto x = Var<0, double>{};

    auto f1 = asinh(x);
    auto f2 = acosh(x);
    auto f3 = atanh(x);

    static_assert(std::is_same_v<decltype(f1), Asinh<Var<0, double>>>,
                  "asinh(x) should have type Asinh<Var>");
    static_assert(std::is_same_v<decltype(f2), Acosh<Var<0, double>>>,
                  "acosh(x) should have type Acosh<Var>");
    static_assert(std::is_same_v<decltype(f3), Atanh<Var<0, double>>>,
                  "atanh(x) should have type Atanh<Var>");
}

// =============================================================================
// Conditional/Piecewise Expression Tests (v2.4)
// =============================================================================

// Basic if_then_else evaluation
TEST(ExprConditional, BasicEvaluation) {
    auto x = Var<0, double>{};

    // if x > 0 then x² else -x
    auto f = if_then_else(x, x * x, -x);

    std::array<double, 1> args_pos{2.0};
    std::array<double, 1> args_neg{-2.0};
    std::array<double, 1> args_zero{0.0};

    EXPECT_DOUBLE_EQ(f.evaluate(args_pos), 4.0);   // x=2: x² = 4
    EXPECT_DOUBLE_EQ(f.evaluate(args_neg), 2.0);   // x=-2: -x = 2
    EXPECT_DOUBLE_EQ(f.evaluate(args_zero), 0.0);  // x=0: -x = 0 (else branch)
}

// Conditional with scalar then/else
TEST(ExprConditional, ScalarBranches) {
    auto x = Var<0, double>{};

    // if x > 0 then 1 else -1
    auto f = if_then_else(x, 1.0, -1.0);

    std::array<double, 1> args_pos{5.0};
    std::array<double, 1> args_neg{-5.0};

    EXPECT_DOUBLE_EQ(f.evaluate(args_pos), 1.0);
    EXPECT_DOUBLE_EQ(f.evaluate(args_neg), -1.0);
}

// Heaviside function
TEST(ExprConditional, Heaviside) {
    auto x = Var<0, double>{};
    auto H = heaviside(x);

    std::array<double, 1> args_pos{1.0};
    std::array<double, 1> args_neg{-1.0};
    std::array<double, 1> args_zero{0.0};

    EXPECT_DOUBLE_EQ(H.evaluate(args_pos), 1.0);
    EXPECT_DOUBLE_EQ(H.evaluate(args_neg), 0.0);
    EXPECT_DOUBLE_EQ(H.evaluate(args_zero), 0.0);  // Convention: H(0) = 0
}

// Ramp (ReLU) function
TEST(ExprConditional, Ramp) {
    auto x = Var<0, double>{};
    auto relu = ramp(x);

    std::array<double, 1> args_pos{3.0};
    std::array<double, 1> args_neg{-3.0};
    std::array<double, 1> args_zero{0.0};

    EXPECT_DOUBLE_EQ(relu.evaluate(args_pos), 3.0);
    EXPECT_DOUBLE_EQ(relu.evaluate(args_neg), 0.0);
    EXPECT_DOUBLE_EQ(relu.evaluate(args_zero), 0.0);
}

// Sign function
TEST(ExprConditional, Sign) {
    auto x = Var<0, double>{};
    auto sgn = sign(x);

    std::array<double, 1> args_pos{5.0};
    std::array<double, 1> args_neg{-5.0};
    std::array<double, 1> args_zero{0.0};

    EXPECT_DOUBLE_EQ(sgn.evaluate(args_pos), 1.0);
    EXPECT_DOUBLE_EQ(sgn.evaluate(args_neg), -1.0);
    EXPECT_DOUBLE_EQ(sgn.evaluate(args_zero), 0.0);  // Convention: sign(0) = 0
}

// Clamp function
TEST(ExprConditional, Clamp) {
    auto x = Var<0, double>{};
    auto f = clamp(x, -1.0, 1.0);

    std::array<double, 1> args_mid{0.5};
    std::array<double, 1> args_low{-5.0};
    std::array<double, 1> args_high{5.0};

    EXPECT_DOUBLE_EQ(f.evaluate(args_mid), 0.5);   // Within bounds
    EXPECT_DOUBLE_EQ(f.evaluate(args_low), -1.0);  // Clamped to lower
    EXPECT_DOUBLE_EQ(f.evaluate(args_high), 1.0);  // Clamped to upper
}

// Conditional with multivariate expressions
TEST(ExprConditional, Multivariate) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // if (x - y) > 0 then x else y (i.e., max(x, y))
    auto f = if_then_else(x - y, x, y);

    std::array<double, 2> args1{5.0, 3.0};
    std::array<double, 2> args2{3.0, 5.0};

    EXPECT_DOUBLE_EQ(f.evaluate(args1), 5.0);  // x > y, return x
    EXPECT_DOUBLE_EQ(f.evaluate(args2), 5.0);  // y > x, return y
}

// Derivative of conditional
TEST(ExprConditional, DerivativeBasic) {
    auto x = Var<0, double>{};

    // f(x) = if x > 0 then x² else 0
    auto f = if_then_else(x, x * x, Zero<double>{});
    auto df = derivative<0>(f);

    // For x > 0: df/dx = 2x
    // For x <= 0: df/dx = 0
    std::array<double, 1> args_pos{3.0};
    std::array<double, 1> args_neg{-3.0};

    EXPECT_DOUBLE_EQ(df.evaluate(args_pos), 6.0);  // 2*3
    EXPECT_DOUBLE_EQ(df.evaluate(args_neg), 0.0);  // 0
}

// Derivative of ramp (ReLU derivative)
TEST(ExprConditional, RampDerivative) {
    auto x = Var<0, double>{};
    auto relu = ramp(x);
    auto drelu = derivative<0>(relu);

    // d/dx[ramp(x)] = 1 for x > 0, 0 for x <= 0
    std::array<double, 1> args_pos{2.0};
    std::array<double, 1> args_neg{-2.0};

    EXPECT_DOUBLE_EQ(drelu.evaluate(args_pos), 1.0);
    EXPECT_DOUBLE_EQ(drelu.evaluate(args_neg), 0.0);
}

// to_string tests
TEST(ExprConditional, ToString) {
    auto x = Var<0, double>{};
    auto f = if_then_else(x, x * x, -x);

    std::string str = f.to_string();
    EXPECT_TRUE(str.find("if") != std::string::npos);
    EXPECT_TRUE(str.find("x0") != std::string::npos);
}

TEST(ExprConditional, HeavisideToString) {
    auto x = Var<0, double>{};
    auto H = heaviside(x);

    std::string str = H.to_string();
    EXPECT_EQ(str, "(if x0 1 0)");
}

// Type trait tests
TEST(ExprConditional, TypeTraits) {
    auto x = Var<0, double>{};
    auto f = if_then_else(x, x, -x);

    static_assert(is_conditional_v<decltype(f)>, "if_then_else should be Conditional");
    static_assert(!is_conditional_v<Var<0, double>>, "Var should not be Conditional");
    static_assert(!is_conditional_v<Const<double>>, "Const should not be Conditional");
}

// Arity tests
TEST(ExprConditional, Arity) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};
    auto z = Var<2, double>{};

    // f depends on x only
    auto f1 = if_then_else(x, x * x, -x);
    EXPECT_EQ(f1.arity_v, 1);

    // f depends on x and y
    auto f2 = if_then_else(x, y, -x);
    EXPECT_EQ(f2.arity_v, 2);

    // f depends on x, y, and z
    auto f3 = if_then_else(x - y, y, z);
    EXPECT_EQ(f3.arity_v, 3);
}

// Integration of piecewise function
TEST(ExprConditional, Integration) {
    auto x = Var<0, double>{};

    // Integral of ramp(x) = max(x, 0) from -1 to 1
    // = integral of 0 from -1 to 0 + integral of x from 0 to 1
    // = 0 + 0.5 = 0.5
    auto f = ramp(x);
    auto I = integral(f).over<0>(-1.0, 1.0);

    auto result = I.evaluate();
    EXPECT_NEAR(result.value(), 0.5, 1e-6);
}

// Integration of heaviside
TEST(ExprConditional, HeavisideIntegration) {
    auto x = Var<0, double>{};

    // Integral of H(x) from -1 to 2
    // = integral of 0 from -1 to 0 + integral of 1 from 0 to 2
    // = 0 + 2 = 2
    auto H = heaviside(x);
    auto I = integral(H).over<0>(-1.0, 2.0);

    auto result = I.evaluate();
    EXPECT_NEAR(result.value(), 2.0, 1e-6);
}

// Nested conditionals
TEST(ExprConditional, NestedConditionals) {
    auto x = Var<0, double>{};

    // Piecewise function:
    // f(x) = -1 if x < -1
    //      = x  if -1 <= x <= 1
    //      = 1  if x > 1
    // Implemented as: if (x + 1) > 0 then (if (1 - x) > 0 then x else 1) else -1
    auto lo_check = x + Const<double>{1.0};  // x + 1 > 0 means x > -1
    auto hi_check = Const<double>{1.0} - x;  // 1 - x > 0 means x < 1
    auto inner = if_then_else(hi_check, x, One<double>{});
    auto f = if_then_else(lo_check, inner, Const<double>{-1.0});

    std::array<double, 1> args_low{-2.0};
    std::array<double, 1> args_mid{0.5};
    std::array<double, 1> args_high{2.0};

    EXPECT_DOUBLE_EQ(f.evaluate(args_low), -1.0);
    EXPECT_DOUBLE_EQ(f.evaluate(args_mid), 0.5);
    EXPECT_DOUBLE_EQ(f.evaluate(args_high), 1.0);
}

// =============================================================================
// Finite Sum/Product Tests (v2.4)
// =============================================================================

// Basic sum: Σᵢ₌₁¹⁰ i = 55
TEST(ExprSum, BasicSum) {
    auto i = Var<0, double>{};

    // Sum of i from 1 to 10
    auto f = sum<0>(i, 1, 10);

    std::array<double, 0> args{};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 55.0);  // 1+2+...+10 = 55
}

// Sum of squares: Σᵢ₌₁⁵ i² = 55
TEST(ExprSum, SumOfSquares) {
    auto i = Var<0, double>{};

    // Sum of i² from 1 to 5
    auto f = sum<0>(i * i, 1, 5);

    std::array<double, 0> args{};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 55.0);  // 1+4+9+16+25 = 55
}

// Sum with external variable: Σᵢ₌₁³ x*i = x*(1+2+3) = 6x
TEST(ExprSum, SumWithExternalVar) {
    auto x = Var<0, double>{};
    auto i = Var<1, double>{};

    // Sum of x*i from i=1 to 3
    auto f = sum<1>(x * i, 1, 3);

    std::array<double, 1> args{2.0};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 12.0);  // 2*(1+2+3) = 12
}

// Sum of constants
TEST(ExprSum, SumOfConstants) {
    auto c = Const<double>{3.0};

    // Sum of 3 from 1 to 5 = 15
    auto f = sum<0>(c, 1, 5);

    std::array<double, 0> args{};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 15.0);  // 3*5 = 15
}

// Basic product: Πᵢ₌₁⁵ i = 5! = 120
TEST(ExprProduct, Factorial) {
    auto i = Var<0, double>{};

    // Product of i from 1 to 5
    auto f = product<0>(i, 1, 5);

    std::array<double, 0> args{};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 120.0);  // 5! = 120
}

// Product with external variable: Πᵢ₌₁³ x = x³
TEST(ExprProduct, ProductOfX) {
    auto x = Var<0, double>{};

    // Product of x from i=1 to 3 (x repeated 3 times)
    auto f = product<1>(x, 1, 3);

    std::array<double, 1> args{2.0};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 8.0);  // 2^3 = 8
}

// Empty sum (hi < lo)
TEST(ExprSum, EmptySum) {
    auto i = Var<0, double>{};

    // Sum from 1 to 0 (empty)
    auto f = sum<0>(i, 1, 0);

    std::array<double, 0> args{};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 0.0);
}

// Empty product (hi < lo)
TEST(ExprProduct, EmptyProduct) {
    auto i = Var<0, double>{};

    // Product from 1 to 0 (empty)
    auto f = product<0>(i, 1, 0);

    std::array<double, 0> args{};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 1.0);  // Empty product = 1
}

// Sum derivative: d/dx[Σᵢ x*i] = Σᵢ i
TEST(ExprSum, Derivative) {
    auto x = Var<0, double>{};
    auto i = Var<1, double>{};

    // f(x) = Σᵢ₌₁³ x*i = 6x
    auto f = sum<1>(x * i, 1, 3);

    // df/dx = Σᵢ₌₁³ i = 6
    auto df = derivative<0>(f);

    std::array<double, 1> args{5.0};
    EXPECT_DOUBLE_EQ(df.evaluate(args), 6.0);  // 1+2+3 = 6
}

// Sum to_string
TEST(ExprSum, ToString) {
    auto i = Var<0, double>{};
    auto f = sum<0>(i, 1, 10);

    std::string str = f.to_string();
    EXPECT_TRUE(str.find("sum") != std::string::npos);
    EXPECT_TRUE(str.find("1..10") != std::string::npos);
}

// Product to_string
TEST(ExprProduct, ToString) {
    auto i = Var<0, double>{};
    auto f = product<0>(i, 1, 5);

    std::string str = f.to_string();
    EXPECT_TRUE(str.find("prod") != std::string::npos);
    EXPECT_TRUE(str.find("1..5") != std::string::npos);
}

// Type trait tests
TEST(ExprSumProduct, TypeTraits) {
    auto i = Var<0, double>{};

    auto s = sum<0>(i, 1, 5);
    auto p = product<0>(i, 1, 5);

    static_assert(is_finite_sum_v<decltype(s)>, "sum should be FiniteSum");
    static_assert(!is_finite_sum_v<decltype(p)>, "product should not be FiniteSum");
    static_assert(is_finite_product_v<decltype(p)>, "product should be FiniteProduct");
    static_assert(!is_finite_product_v<decltype(s)>, "sum should not be FiniteProduct");
    static_assert(!is_finite_sum_v<Var<0, double>>, "Var should not be FiniteSum");
}

// Arity tests
TEST(ExprSum, Arity) {
    auto i = Var<0, double>{};
    auto x = Var<0, double>{};
    auto j = Var<1, double>{};

    // Sum over only index variable: arity 0
    auto f1 = sum<0>(i, 1, 5);
    EXPECT_EQ(f1.arity_v, 0);

    // Sum with external variable: arity 1
    auto f2 = sum<1>(x * j, 1, 5);
    EXPECT_EQ(f2.arity_v, 1);
}

// Geometric series: Σᵢ₌₀⁴ x^i ≈ (1-x⁵)/(1-x) for x ≠ 1
TEST(ExprSum, GeometricSeries) {
    auto x = Var<0, double>{};
    auto i = Var<1, double>{};

    // Σᵢ₌₀⁴ x^i = 1 + x + x² + x³ + x⁴
    auto f = sum<1>(pow(x, i), 0, 4);

    std::array<double, 1> args{2.0};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 31.0);  // 1+2+4+8+16 = 31

    std::array<double, 1> args_half{0.5};
    double expected = 1.0 + 0.5 + 0.25 + 0.125 + 0.0625;  // 1.9375
    EXPECT_DOUBLE_EQ(f.evaluate(args_half), expected);
}

// Integration of a sum
TEST(ExprSum, Integration) {
    auto x = Var<0, double>{};
    auto i = Var<1, double>{};

    // f(x) = Σᵢ₌₁² x*i = x + 2x = 3x
    // Integral from 0 to 1 = (3/2)x² |₀¹ = 1.5
    auto f = sum<1>(x * i, 1, 2);
    auto I = integral(f).over<0>(0.0, 1.0);

    auto result = I.evaluate();
    EXPECT_NEAR(result.value(), 1.5, 1e-8);
}

// Nested sum: ΣᵢΣⱼ i*j
TEST(ExprSum, NestedSum) {
    auto i = Var<0, double>{};
    auto j = Var<1, double>{};

    // Inner sum: Σⱼ₌₁² i*j = i*(1+2) = 3i
    auto inner = sum<1>(i * j, 1, 2);

    // Outer sum: Σᵢ₌₁³ 3i = 3*(1+2+3) = 18
    auto f = sum<0>(inner, 1, 3);

    std::array<double, 0> args{};
    EXPECT_DOUBLE_EQ(f.evaluate(args), 18.0);
}

// =============================================================================
// Symbolic Antiderivative Tests (v2.4)
// =============================================================================

// has_antiderivative trait tests
TEST(ExprAntiderivative, HasAntiderivativeConst) {
    static_assert(has_antiderivative_v<Const<double>, 0>,
                  "Const should have antiderivative");
    static_assert(has_antiderivative_v<Zero<double>, 0>,
                  "Zero should have antiderivative");
    static_assert(has_antiderivative_v<One<double>, 0>,
                  "One should have antiderivative");
}

TEST(ExprAntiderivative, HasAntiderivativeVar) {
    // x has antiderivative when integrating over x
    static_assert(has_antiderivative_v<Var<0, double>, 0>,
                  "Var<0> should have antiderivative over dim 0");
    // x does NOT have antiderivative when integrating over y
    static_assert(!has_antiderivative_v<Var<0, double>, 1>,
                  "Var<0> should not have antiderivative over dim 1");
}

TEST(ExprAntiderivative, HasAntiderivativePow) {
    auto x = Var<0, double>{};
    auto x2 = square(x);

    static_assert(has_antiderivative_v<decltype(x2), 0>,
                  "x^2 should have antiderivative");
}

TEST(ExprAntiderivative, HasAntiderivativeTrig) {
    auto x = Var<0, double>{};
    auto sin_x = sin(x);
    auto cos_x = cos(x);

    static_assert(has_antiderivative_v<decltype(sin_x), 0>,
                  "sin(x) should have antiderivative");
    static_assert(has_antiderivative_v<decltype(cos_x), 0>,
                  "cos(x) should have antiderivative");
}

TEST(ExprAntiderivative, HasAntiderivativeExp) {
    auto x = Var<0, double>{};
    auto exp_x = exp(x);

    static_assert(has_antiderivative_v<decltype(exp_x), 0>,
                  "exp(x) should have antiderivative");
}

// antiderivative computation tests

// ∫ c dx = c*x
TEST(ExprAntiderivative, ConstantIntegral) {
    auto c = Const<double>{3.0};
    auto F = antiderivative<0>(c);

    // F(x) = 3x
    std::array<double, 1> args{2.0};
    EXPECT_DOUBLE_EQ(F.evaluate(args), 6.0);  // 3*2 = 6
}

// ∫ 0 dx = 0
TEST(ExprAntiderivative, ZeroIntegral) {
    auto z = Zero<double>{};
    auto F = antiderivative<0>(z);

    std::array<double, 1> args{5.0};
    EXPECT_DOUBLE_EQ(F.evaluate(args), 0.0);
}

// ∫ 1 dx = x
TEST(ExprAntiderivative, OneIntegral) {
    auto one = One<double>{};
    auto F = antiderivative<0>(one);

    std::array<double, 1> args{7.0};
    EXPECT_DOUBLE_EQ(F.evaluate(args), 7.0);
}

// ∫ x dx = x²/2
TEST(ExprAntiderivative, VarIntegral) {
    auto x = Var<0, double>{};
    auto F = antiderivative<0>(x);

    // F(x) = x²/2
    std::array<double, 1> args{4.0};
    EXPECT_DOUBLE_EQ(F.evaluate(args), 8.0);  // 16/2 = 8
}

// ∫ x^2 dx = x³/3
TEST(ExprAntiderivative, SquareIntegral) {
    auto x = Var<0, double>{};
    auto x2 = square(x);
    auto F = antiderivative<0>(x2);

    // F(x) = x³/3
    std::array<double, 1> args{3.0};
    EXPECT_DOUBLE_EQ(F.evaluate(args), 9.0);  // 27/3 = 9
}

// ∫ x^3 dx = x⁴/4
TEST(ExprAntiderivative, CubeIntegral) {
    auto x = Var<0, double>{};
    auto x3 = cube(x);
    auto F = antiderivative<0>(x3);

    // F(x) = x⁴/4
    std::array<double, 1> args{2.0};
    EXPECT_DOUBLE_EQ(F.evaluate(args), 4.0);  // 16/4 = 4
}

// ∫ sin(x) dx = -cos(x)
TEST(ExprAntiderivative, SinIntegral) {
    auto x = Var<0, double>{};
    auto sin_x = sin(x);
    auto F = antiderivative<0>(sin_x);

    // F(x) = -cos(x)
    std::array<double, 1> args{0.0};
    EXPECT_DOUBLE_EQ(F.evaluate(args), -1.0);  // -cos(0) = -1
}

// ∫ cos(x) dx = sin(x)
TEST(ExprAntiderivative, CosIntegral) {
    auto x = Var<0, double>{};
    auto cos_x = cos(x);
    auto F = antiderivative<0>(cos_x);

    // F(x) = sin(x)
    std::array<double, 1> args{std::numbers::pi / 2};
    EXPECT_NEAR(F.evaluate(args), 1.0, 1e-10);  // sin(π/2) = 1
}

// ∫ exp(x) dx = exp(x)
TEST(ExprAntiderivative, ExpIntegral) {
    auto x = Var<0, double>{};
    auto exp_x = exp(x);
    auto F = antiderivative<0>(exp_x);

    // F(x) = exp(x)
    std::array<double, 1> args{1.0};
    EXPECT_NEAR(F.evaluate(args), std::exp(1.0), 1e-10);
}

// ∫ (f + g) dx = ∫f dx + ∫g dx
TEST(ExprAntiderivative, SumIntegral) {
    auto x = Var<0, double>{};
    auto f = x + Const<double>{2.0};  // x + 2
    auto F = antiderivative<0>(f);

    // F(x) = x²/2 + 2x
    std::array<double, 1> args{4.0};
    EXPECT_DOUBLE_EQ(F.evaluate(args), 16.0);  // 8 + 8 = 16
}

// ∫ c*f dx = c * ∫f dx
TEST(ExprAntiderivative, ScalarMultipleIntegral) {
    auto x = Var<0, double>{};
    auto f = Const<double>{3.0} * x;  // 3x
    auto F = antiderivative<0>(f);

    // F(x) = 3 * x²/2 = 3x²/2
    std::array<double, 1> args{2.0};
    EXPECT_DOUBLE_EQ(F.evaluate(args), 6.0);  // 3 * 4/2 = 6
}

// ∫ -f dx = -∫f dx
TEST(ExprAntiderivative, NegationIntegral) {
    auto x = Var<0, double>{};
    auto f = -x;
    auto F = antiderivative<0>(f);

    // F(x) = -x²/2
    std::array<double, 1> args{4.0};
    EXPECT_DOUBLE_EQ(F.evaluate(args), -8.0);  // -16/2 = -8
}

// Definite integral tests using Fundamental Theorem of Calculus

// ∫[0,1] x dx = 1/2
TEST(ExprAntiderivative, DefiniteIntegralVar) {
    auto x = Var<0, double>{};
    double result = definite_integral_symbolic<0>(x, 0.0, 1.0);
    EXPECT_DOUBLE_EQ(result, 0.5);
}

// ∫[0,1] x^2 dx = 1/3
TEST(ExprAntiderivative, DefiniteIntegralSquare) {
    auto x = Var<0, double>{};
    auto x2 = square(x);
    double result = definite_integral_symbolic<0>(x2, 0.0, 1.0);
    EXPECT_NEAR(result, 1.0/3.0, 1e-10);
}

// ∫[0,π] sin(x) dx = 2
TEST(ExprAntiderivative, DefiniteIntegralSin) {
    auto x = Var<0, double>{};
    auto sin_x = sin(x);
    double result = definite_integral_symbolic<0>(sin_x, 0.0, std::numbers::pi);
    EXPECT_NEAR(result, 2.0, 1e-10);
}

// ∫[0,1] exp(x) dx = e - 1
TEST(ExprAntiderivative, DefiniteIntegralExp) {
    auto x = Var<0, double>{};
    auto exp_x = exp(x);
    double result = definite_integral_symbolic<0>(exp_x, 0.0, 1.0);
    EXPECT_NEAR(result, std::exp(1.0) - 1.0, 1e-10);
}

// ∫[1,2] 3x^2 + 2x + 1 dx
TEST(ExprAntiderivative, DefiniteIntegralPolynomial) {
    auto x = Var<0, double>{};
    auto f = Const<double>{3.0} * square(x) + Const<double>{2.0} * x + One<double>{};

    // F(x) = x³ + x² + x
    // F(2) - F(1) = (8 + 4 + 2) - (1 + 1 + 1) = 14 - 3 = 11
    double result = definite_integral_symbolic<0>(f, 1.0, 2.0);
    EXPECT_NEAR(result, 11.0, 1e-10);
}

// Compare symbolic vs numerical
TEST(ExprAntiderivative, SymbolicVsNumerical) {
    auto x = Var<0, double>{};

    // ∫[0,1] x^2 dx
    auto x2 = square(x);

    // Symbolic result
    double symbolic = definite_integral_symbolic<0>(x2, 0.0, 1.0);

    // Numerical result
    auto I = integral(x2).over<0>(0.0, 1.0);
    auto numerical = I.evaluate();

    EXPECT_NEAR(symbolic, numerical.value(), 1e-8);
}

// =============================================================================
// BoxIntegral Tests (Phase 5)
// =============================================================================

// Test 2D box integral with constant function
TEST(BoxIntegral, Constant2D) {
    auto one = One<double>{};

    // ∫∫ 1 dxdy over [0,1]×[0,1] = 1
    auto I = integral(one).over_box(
        std::make_pair(0.0, 1.0),
        std::make_pair(0.0, 1.0)
    );

    auto result = I.eval(limes::methods::monte_carlo<double>{100000, 42});
    EXPECT_NEAR(result.value(), 1.0, 0.01);
}

// Test 2D box integral with x*y
TEST(BoxIntegral, XTimesY2D) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};
    auto f = x * y;

    // ∫∫ xy dxdy over [0,1]×[0,1] = (1/2)(1/2) = 1/4
    auto I = integral(f).over_box(
        std::make_pair(0.0, 1.0),
        std::make_pair(0.0, 1.0)
    );

    auto result = I.eval(limes::methods::monte_carlo<double>{100000, 42});
    EXPECT_NEAR(result.value(), 0.25, 0.01);
}

// Test 3D box integral (unit cube)
TEST(BoxIntegral, Constant3D) {
    auto one = One<double>{};

    // ∫∫∫ 1 dxdydz over [0,1]³ = 1
    auto I = integral(one).over_box(
        std::make_pair(0.0, 1.0),
        std::make_pair(0.0, 1.0),
        std::make_pair(0.0, 1.0)
    );

    auto result = I.eval(limes::methods::monte_carlo<double>{100000, 42});
    EXPECT_NEAR(result.value(), 1.0, 0.01);
}

// Test 3D box integral with x*y*z
TEST(BoxIntegral, XYZ3D) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};
    auto z = Var<2, double>{};
    auto f = x * y * z;

    // ∫∫∫ xyz dxdydz over [0,1]³ = (1/2)³ = 1/8
    auto I = integral(f).over_box(
        std::make_pair(0.0, 1.0),
        std::make_pair(0.0, 1.0),
        std::make_pair(0.0, 1.0)
    );

    auto result = I.eval(limes::methods::monte_carlo<double>{100000, 42});
    EXPECT_NEAR(result.value(), 0.125, 0.01);
}

// Test box integral with non-unit bounds
TEST(BoxIntegral, NonUnitBounds) {
    auto one = One<double>{};

    // ∫∫ 1 dxdy over [0,2]×[0,3] = 6
    auto I = integral(one).over_box(
        std::make_pair(0.0, 2.0),
        std::make_pair(0.0, 3.0)
    );

    auto result = I.eval(limes::methods::monte_carlo<double>{100000, 42});
    EXPECT_NEAR(result.value(), 6.0, 0.1);
}

// Test box integral default eval (10000 samples)
TEST(BoxIntegral, DefaultEval) {
    auto one = One<double>{};

    auto I = integral(one).over_box(
        std::make_pair(0.0, 1.0),
        std::make_pair(0.0, 1.0)
    );

    auto result = I.eval();  // Uses default 10000 samples
    EXPECT_NEAR(result.value(), 1.0, 0.05);
}

// Test box integral eval with sample count
TEST(BoxIntegral, EvalWithSampleCount) {
    auto one = One<double>{};

    auto I = integral(one).over_box(
        std::make_pair(0.0, 1.0),
        std::make_pair(0.0, 1.0)
    );

    auto result = I.eval(50000);
    EXPECT_NEAR(result.value(), 1.0, 0.02);
}

// Test box integral to_string
TEST(BoxIntegral, ToString) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};
    auto f = x + y;

    auto I = integral(f).over_box(
        std::make_pair(0.0, 1.0),
        std::make_pair(0.0, 2.0)
    );

    std::string s = I.to_string();
    EXPECT_TRUE(s.find("box-integral") != std::string::npos);
    EXPECT_TRUE(s.find("[0, 1]") != std::string::npos);
    EXPECT_TRUE(s.find("[0, 2]") != std::string::npos);
}

// Test constrained box integral (triangle: y < x)
TEST(BoxIntegral, ConstrainedTriangle) {
    auto one = One<double>{};

    // ∫∫ 1 dxdy over triangle where y < x in [0,1]×[0,1]
    // Area = 1/2
    auto I = integral(one)
        .over_box(std::make_pair(0.0, 1.0), std::make_pair(0.0, 1.0))
        .where([](double x, double y) { return y < x; });

    auto result = I.eval(limes::methods::monte_carlo<double>{100000, 42});
    EXPECT_NEAR(result.value(), 0.5, 0.02);
}

// Test constrained box integral (circle)
TEST(BoxIntegral, ConstrainedCircle) {
    auto one = One<double>{};

    // ∫∫ 1 dxdy over unit circle in [-1,1]×[-1,1]
    // Area = π
    auto I = integral(one)
        .over_box(std::make_pair(-1.0, 1.0), std::make_pair(-1.0, 1.0))
        .where([](double x, double y) { return x*x + y*y <= 1.0; });

    auto result = I.eval(limes::methods::monte_carlo<double>{200000, 42});
    EXPECT_NEAR(result.value(), std::numbers::pi, 0.05);
}

// Test constrained box integral to_string
TEST(BoxIntegral, ConstrainedToString) {
    auto x = Var<0, double>{};

    auto I = integral(x)
        .over_box(std::make_pair(0.0, 1.0), std::make_pair(0.0, 1.0))
        .where([](double, double) { return true; });

    std::string s = I.to_string();
    EXPECT_TRUE(s.find("constrained-box-integral") != std::string::npos);
    EXPECT_TRUE(s.find("<constraint>") != std::string::npos);
}

// Test box integral using array syntax
TEST(BoxIntegral, ArraySyntax) {
    auto one = One<double>{};

    std::array<std::pair<double, double>, 2> bounds = {{{0.0, 1.0}, {0.0, 1.0}}};
    auto I = integral(one).over_box(bounds);

    auto result = I.eval(limes::methods::monte_carlo<double>{50000, 42});
    EXPECT_NEAR(result.value(), 1.0, 0.02);
}

// Test convenience functions box2d and box3d
TEST(BoxIntegral, ConvenienceFunctions) {
    auto one = One<double>{};

    // box2d
    auto I2 = box2d(one, 0.0, 1.0, 0.0, 1.0);
    auto r2 = I2.eval(limes::methods::monte_carlo<double>{50000, 42});
    EXPECT_NEAR(r2.value(), 1.0, 0.02);

    // box3d
    auto I3 = box3d(one, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    auto r3 = I3.eval(limes::methods::monte_carlo<double>{50000, 42});
    EXPECT_NEAR(r3.value(), 1.0, 0.03);
}

// Test type traits
TEST(BoxIntegral, TypeTraits) {
    auto x = Var<0, double>{};

    auto I = integral(x).over_box(std::make_pair(0.0, 1.0), std::make_pair(0.0, 1.0));
    EXPECT_TRUE(is_box_integral_v<decltype(I)>);
    EXPECT_FALSE(is_constrained_box_integral_v<decltype(I)>);

    auto J = I.where([](double, double) { return true; });
    EXPECT_FALSE(is_box_integral_v<decltype(J)>);
    EXPECT_TRUE(is_constrained_box_integral_v<decltype(J)>);
}

// Test monte_carlo with_seed
TEST(BoxIntegral, MontCarloWithSeed) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};
    auto f = x * y;

    auto I = integral(f).over_box(std::make_pair(0.0, 1.0), std::make_pair(0.0, 1.0));

    // Two evaluations with same seed should give same result
    auto m = limes::methods::monte_carlo<double>{10000}.with_seed(123);
    auto r1 = I.eval(m);
    auto r2 = I.eval(m);

    EXPECT_EQ(r1.value(), r2.value());
}

// Test high-dimensional integration
TEST(BoxIntegral, HighDimensional) {
    auto one = One<double>{};

    // 4D unit hypercube volume = 1
    std::array<std::pair<double, double>, 4> bounds4d = {
        {{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}}
    };
    auto I4 = over_box(one, bounds4d);
    auto r4 = I4.eval(limes::methods::monte_carlo<double>{100000, 42});
    EXPECT_NEAR(r4.value(), 1.0, 0.02);
}

// =============================================================================
// ProductIntegral Tests (Phase 6: Separable Integral Composition)
// =============================================================================

// Test basic product of two independent integrals
TEST(ProductIntegral, BasicProduct) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // I = ∫[0,1] x dx = 1/2
    auto I = integral(x).over<0>(0.0, 1.0);

    // J = ∫[0,1] y dy = 1/2
    auto J = integral(y).over<1>(0.0, 1.0);

    // I * J = 1/2 * 1/2 = 1/4
    auto IJ = I * J;

    auto result = IJ.eval();
    EXPECT_NEAR(result.value(), 0.25, 1e-8);
}

// Test product with different functions
TEST(ProductIntegral, DifferentFunctions) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // I = ∫[0,1] x^2 dx = 1/3
    auto I = integral(x * x).over<0>(0.0, 1.0);

    // J = ∫[0,1] 2y dy = 1
    auto J = integral(Const<double>{2.0} * y).over<1>(0.0, 1.0);

    auto IJ = I * J;
    auto result = IJ.eval();
    EXPECT_NEAR(result.value(), 1.0/3.0, 1e-8);
}

// Test product of three integrals
TEST(ProductIntegral, ThreeIntegrals) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};
    auto z = Var<2, double>{};

    // I = ∫[0,1] x dx = 1/2
    auto I = integral(x).over<0>(0.0, 1.0);

    // J = ∫[0,1] y dy = 1/2
    auto J = integral(y).over<1>(0.0, 1.0);

    // K = ∫[0,1] z dz = 1/2
    auto K = integral(z).over<2>(0.0, 1.0);

    // I * J * K = 1/8
    auto IJK = I * J * K;

    auto result = IJK.eval();
    EXPECT_NEAR(result.value(), 0.125, 1e-7);
}

// Test product using product() function
TEST(ProductIntegral, ProductFunction) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    auto I = integral(x).over<0>(0.0, 1.0);
    auto J = integral(y).over<1>(0.0, 1.0);

    auto IJ = product(I, J);

    auto result = IJ.eval();
    EXPECT_NEAR(result.value(), 0.25, 1e-8);
}

// Test variadic product function
TEST(ProductIntegral, VariadicProduct) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};
    auto z = Var<2, double>{};

    auto I = integral(x).over<0>(0.0, 1.0);
    auto J = integral(y).over<1>(0.0, 1.0);
    auto K = integral(z).over<2>(0.0, 1.0);

    auto IJK = product(I, J, K);

    auto result = IJK.eval();
    EXPECT_NEAR(result.value(), 0.125, 1e-7);
}

// Test type traits
TEST(ProductIntegral, TypeTraits) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    auto I = integral(x).over<0>(0.0, 1.0);
    auto J = integral(y).over<1>(0.0, 1.0);
    auto IJ = I * J;

    EXPECT_TRUE(is_product_integral_v<decltype(IJ)>);
    EXPECT_FALSE(is_product_integral_v<decltype(I)>);
    EXPECT_FALSE(is_product_integral_v<decltype(J)>);
}

// Test independence check
TEST(ProductIntegral, IndependenceCheck) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    auto I = integral(x).over<0>(0.0, 1.0);
    auto J = integral(y).over<1>(0.0, 1.0);

    // These should be independent
    constexpr bool ij_independent = are_independent_integrals_v<decltype(I), decltype(J)>;
    EXPECT_TRUE(ij_independent);

    // Same-dimension integrals would not be independent
    auto K = integral(x * x).over<0>(0.0, 1.0);
    constexpr bool ik_independent = are_independent_integrals_v<decltype(I), decltype(K)>;
    EXPECT_FALSE(ik_independent);
}

// Test to_string
TEST(ProductIntegral, ToString) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    auto I = integral(x).over<0>(0.0, 1.0);
    auto J = integral(y).over<1>(0.0, 1.0);
    auto IJ = I * J;

    std::string s = IJ.to_string();
    EXPECT_TRUE(s.find("product-integral") != std::string::npos);
}

// Test with trigonometric functions
TEST(ProductIntegral, Trigonometric) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // I = ∫[0,π] sin(x) dx = 2
    auto I = integral(sin(x)).over<0>(0.0, std::numbers::pi);

    // J = ∫[0,π/2] cos(y) dy = 1
    auto J = integral(cos(y)).over<1>(0.0, std::numbers::pi / 2.0);

    auto IJ = I * J;
    auto result = IJ.eval();
    EXPECT_NEAR(result.value(), 2.0, 1e-7);
}

// Test error propagation
TEST(ProductIntegral, ErrorPropagation) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    auto I = integral(x).over<0>(0.0, 1.0);
    auto J = integral(y).over<1>(0.0, 1.0);
    auto IJ = I * J;

    auto result = IJ.eval();

    // Error should be non-negative
    EXPECT_GE(result.error(), 0.0);

    // Evaluations should be sum of both
    auto r1 = I.eval();
    auto r2 = J.eval();
    EXPECT_EQ(result.evaluations(), r1.evaluations() + r2.evaluations());
}

// Test product commutativity
TEST(ProductIntegral, Commutativity) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    auto I = integral(x * x).over<0>(0.0, 1.0);
    auto J = integral(y).over<1>(0.0, 2.0);

    auto IJ = I * J;
    auto JI = J * I;

    auto r1 = IJ.eval();
    auto r2 = JI.eval();

    // Results should be equal (commutativity)
    EXPECT_NEAR(r1.value(), r2.value(), 1e-10);
}

// Test with exponentials
TEST(ProductIntegral, Exponential) {
    auto x = Var<0, double>{};
    auto y = Var<1, double>{};

    // I = ∫[0,1] exp(x) dx = e - 1
    auto I = integral(exp(x)).over<0>(0.0, 1.0);

    // J = ∫[0,1] exp(-y) dy = 1 - 1/e
    auto J = integral(exp(-y)).over<1>(0.0, 1.0);

    auto IJ = I * J;
    auto result = IJ.eval();

    double expected = (std::exp(1.0) - 1.0) * (1.0 - std::exp(-1.0));
    EXPECT_NEAR(result.value(), expected, 1e-7);
}
