#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <limes/algorithms/quadrature/quadrature.hpp>

using namespace limes::algorithms::quadrature;

// Apply a quadrature rule to integrate f over [-1, 1]
template <typename T, typename Rule, typename F>
T apply_rule(const Rule& rule, F&& f) {
    T sum = T(0);
    for (size_t i = 0; i < rule.size(); ++i) {
        sum += rule.weight(i) * f(rule.abscissa(i));
    }
    return sum;
}

template <typename T>
T integrate_polynomial(const auto& rule, int degree) {
    return apply_rule<T>(rule, [degree](T x) -> T {
        return std::pow(x, degree);
    });
}

// Calculate exact integral of x^n from -1 to 1
template <typename T>
T exact_polynomial_integral(int n) {
    if (n % 2 == 1) return T(0); // Odd powers integrate to 0
    return T(2) / T(n + 1); // Even powers
}

// Test fixture for quadrature rules
template <typename T>
class QuadratureTest : public ::testing::Test {
protected:
    // Use a tolerance that accounts for the fact that implementations use double-precision constants
    // For long double, we cannot exceed double precision due to constant storage
    static constexpr T tol = std::is_same_v<T, long double>
        ? T(std::numeric_limits<double>::epsilon() * 100)
        : std::numeric_limits<T>::epsilon() * T(100);

    void test_polynomial_exactness(const auto& rule, int max_exact_degree) {
        for (int degree = 0; degree <= max_exact_degree; ++degree) {
            T computed = integrate_polynomial<T>(rule, degree);
            T exact = exact_polynomial_integral<T>(degree);

            EXPECT_NEAR(computed, exact, tol)
                << "Failed for polynomial degree " << degree
                << ", computed = " << computed << ", exact = " << exact;
        }

        // Should not be exact for higher even degrees (odd degrees integrate to 0, so they're trivially "exact")
        if (max_exact_degree < 20) {
            // Use an even degree for testing inexactness (odd powers integrate to 0 by symmetry)
            // Also add 2 to max_exact_degree to ensure we're testing beyond the exact range
            int test_degree = max_exact_degree + 2;
            if (test_degree % 2 == 1) {
                test_degree++;  // Make it even to avoid the trivial zero case
            }
            if (test_degree > 24) return;  // Don't test beyond reasonable bounds

            T computed = integrate_polynomial<T>(rule, test_degree);
            T exact = exact_polynomial_integral<T>(test_degree);

            // Allow for some accuracy but not machine precision
            // For float, we need more tolerance due to lower precision
            T error = std::abs(computed - exact);
            T min_expected_error = std::is_same_v<T, float> ? tol : tol * T(10);
            EXPECT_GT(error, min_expected_error)
                << "Unexpectedly exact for degree " << test_degree;
        }
    }

    void test_weight_sum(const auto& rule) {
        T sum = T(0);
        for (size_t i = 0; i < rule.size(); ++i) {
            sum += rule.weight(i);
        }
        // Weights should sum to 2 for interval [-1, 1]
        EXPECT_NEAR(sum, T(2), tol);
    }

    void test_node_bounds(const auto& rule) {
        for (size_t i = 0; i < rule.size(); ++i) {
            T node = rule.abscissa(i);
            EXPECT_GE(node, T(-1));
            EXPECT_LE(node, T(1));
        }
    }

    void test_symmetry(const auto& rule) {
        size_t n = rule.size();
        for (size_t i = 0; i < n / 2; ++i) {
            // Nodes should be symmetric
            EXPECT_NEAR(rule.abscissa(i), -rule.abscissa(n - 1 - i), tol);
            // Weights should be symmetric
            EXPECT_NEAR(rule.weight(i), rule.weight(n - 1 - i), tol);
        }
    }
};

using FloatTypes = ::testing::Types<float, double, long double>;
TYPED_TEST_SUITE(QuadratureTest, FloatTypes);

// Test Gauss-Legendre rules
TYPED_TEST(QuadratureTest, GaussLegendre2) {
    using T = TypeParam;
    gauss_legendre<T, 2> rule;

    EXPECT_EQ(rule.size(), 2);
    this->test_weight_sum(rule);
    this->test_node_bounds(rule);
    this->test_symmetry(rule);
    this->test_polynomial_exactness(rule, 3); // 2n-1 = 3
}

TYPED_TEST(QuadratureTest, GaussLegendre3) {
    using T = TypeParam;
    gauss_legendre<T, 3> rule;

    EXPECT_EQ(rule.size(), 3);
    this->test_weight_sum(rule);
    this->test_node_bounds(rule);
    this->test_symmetry(rule);
    this->test_polynomial_exactness(rule, 5); // 2n-1 = 5
}

TYPED_TEST(QuadratureTest, GaussLegendre5) {
    using T = TypeParam;
    gauss_legendre<T, 5> rule;

    EXPECT_EQ(rule.size(), 5);
    this->test_weight_sum(rule);
    this->test_node_bounds(rule);
    this->test_symmetry(rule);
    this->test_polynomial_exactness(rule, 9); // 2n-1 = 9
}

// Test Gauss-Kronrod rules
TYPED_TEST(QuadratureTest, GaussKronrod15) {
    using T = TypeParam;
    gauss_kronrod_15<T> rule;

    EXPECT_EQ(rule.size(), 15);
    this->test_weight_sum(rule);
    this->test_node_bounds(rule);
    this->test_symmetry(rule);

    // G-K 15 should integrate polynomials up to degree 22 exactly
    this->test_polynomial_exactness(rule, 22);
}

// Test Gauss-Kronrod embedded rules
TYPED_TEST(QuadratureTest, GaussKronrodEmbedded) {
    using T = TypeParam;
    gauss_kronrod_15<T> rule;

    // Test embedded Gauss rule
    EXPECT_EQ(rule.gauss_size, 7);

    // Test weight sums
    T gauss_weight_sum = T(0);
    for (size_t i = 0; i < rule.gauss_size; ++i) {
        gauss_weight_sum += rule.gauss_weights[i];
    }
    EXPECT_NEAR(gauss_weight_sum, T(2), this->tol);

    T kronrod_weight_sum = T(0);
    for (size_t i = 0; i < rule.size(); ++i) {
        kronrod_weight_sum += rule.weight(i);
    }
    EXPECT_NEAR(kronrod_weight_sum, T(2), this->tol);
}

// Test Clenshaw-Curtis rule
TYPED_TEST(QuadratureTest, ClenshawCurtis) {
    using T = TypeParam;

    // Test fixed size Clenshaw-Curtis
    clenshaw_curtis<T, 9> rule;

    EXPECT_EQ(rule.size(), 9);
    this->test_weight_sum(rule);
    this->test_node_bounds(rule);
    this->test_symmetry(rule);

    // Clenshaw-Curtis integrates polynomials up to degree n-1 exactly
    this->test_polynomial_exactness(rule, 8);
}

// Test Simpson's rule
TYPED_TEST(QuadratureTest, SimpsonsRule) {
    using T = TypeParam;
    simpson_rule<T> rule;

    EXPECT_EQ(rule.size(), 3);

    // Check specific nodes and weights
    EXPECT_NEAR(rule.abscissa(0), T(-1), this->tol);
    EXPECT_NEAR(rule.abscissa(1), T(0), this->tol);
    EXPECT_NEAR(rule.abscissa(2), T(1), this->tol);

    EXPECT_NEAR(rule.weight(0), T(1)/T(3), this->tol);
    EXPECT_NEAR(rule.weight(1), T(4)/T(3), this->tol);
    EXPECT_NEAR(rule.weight(2), T(1)/T(3), this->tol);

    this->test_weight_sum(rule);
    this->test_polynomial_exactness(rule, 3); // Simpson's rule is exact for cubics
}

// Test Trapezoidal rule
TYPED_TEST(QuadratureTest, TrapezoidalRule) {
    using T = TypeParam;
    trapezoidal_rule<T> rule;

    EXPECT_EQ(rule.size(), 2);

    // Check specific nodes and weights
    EXPECT_NEAR(rule.abscissa(0), T(-1), this->tol);
    EXPECT_NEAR(rule.abscissa(1), T(1), this->tol);

    EXPECT_NEAR(rule.weight(0), T(1), this->tol);
    EXPECT_NEAR(rule.weight(1), T(1), this->tol);

    this->test_weight_sum(rule);
    this->test_polynomial_exactness(rule, 1); // Trapezoidal rule is exact for linear
}

// Test Midpoint rule
TYPED_TEST(QuadratureTest, MidpointRule) {
    using T = TypeParam;
    midpoint_rule<T> rule;

    EXPECT_EQ(rule.size(), 1);

    // Check specific node and weight
    EXPECT_NEAR(rule.abscissa(0), T(0), this->tol);
    EXPECT_NEAR(rule.weight(0), T(2), this->tol);

    this->test_weight_sum(rule);
    this->test_polynomial_exactness(rule, 1); // Midpoint rule is exact for linear
}

// Test Tanh-Sinh quadrature nodes
TYPED_TEST(QuadratureTest, TanhSinhQuadrature) {
    using T = TypeParam;

    tanh_sinh_nodes<T> nodes;

    // Test a few nodes at level 3
    size_t level = 3;
    for (size_t i = 0; i < 5; ++i) {
        T x = nodes.abscissa(level, i);
        T w = nodes.weight(level, i);

        // Nodes should be in [-1, 1]
        EXPECT_GE(x, T(-1));
        EXPECT_LE(x, T(1));

        // Weights should be positive
        EXPECT_GT(w, T(0));
    }
}

// Test integration of specific functions
TEST(QuadratureIntegration, ExponentialFunction) {
    auto f = [](double x) { return std::exp(x); };
    double exact = std::exp(1.0) - std::exp(-1.0);

    gauss_legendre<double, 5> gl5;
    EXPECT_NEAR(apply_rule<double>(gl5, f), exact, 1e-9);

    gauss_kronrod_15<double> gk15;
    EXPECT_NEAR(apply_rule<double>(gk15, f), exact, 1e-14);
}

TEST(QuadratureIntegration, TrigonometricFunction) {
    auto f = [](double x) { return std::sin(M_PI * x); };

    gauss_legendre<double, 3> rule;
    EXPECT_NEAR(apply_rule<double>(rule, f), 0.0, 1e-14);
}

TEST(QuadratureIntegration, RationalFunction) {
    auto f = [](double x) { return 1.0 / (1.0 + x * x); };
    double exact = M_PI / 2.0;

    gauss_legendre<double, 5> rule;
    EXPECT_NEAR(apply_rule<double>(rule, f), exact, 1e-3);
}

// Test error estimation with embedded rules
TEST(QuadratureErrorEstimation, GaussKronrodError) {
    auto f = [](double x) { return std::exp(-x * x); };
    gauss_kronrod_15<double> rule;

    // Compute with embedded Gauss rule
    double gauss_result = 0.0;
    for (size_t i = 0; i < rule.gauss_size; ++i) {
        size_t idx = rule.gauss_indices[i];
        gauss_result += rule.gauss_weights[i] * f(rule.abscissa(idx));
    }

    double kronrod_result = apply_rule<double>(rule, f);
    double error_estimate = std::abs(kronrod_result - gauss_result);

    EXPECT_LT(error_estimate, 1e-6);

    // Kronrod should be more accurate than Gauss
    double exact = 1.4936482656248540; // erf(1) * sqrt(pi)
    EXPECT_LT(std::abs(kronrod_result - exact), std::abs(gauss_result - exact));
}

// Test quadrature with endpoint singularity: sqrt(1-x^2) (semicircle)
TEST(QuadratureAdaptive, SingularFunction) {
    auto f = [](double x) -> double {
        if (std::abs(x) >= 1.0) return 0.0;
        return std::sqrt(1.0 - x * x);
    };

    gauss_kronrod_15<double> rule;
    EXPECT_NEAR(apply_rule<double>(rule, f), M_PI / 2.0, 0.01);
}

// Test custom quadrature rule
TEST(CustomQuadrature, UserDefinedRule) {
    // Create a custom 3-point rule
    struct custom_rule {
        static constexpr size_t size() { return 3; }

        double abscissa(size_t i) const {
            static const double nodes[3] = {-0.5, 0.0, 0.5};
            return nodes[i];
        }

        double weight(size_t i) const {
            static const double weights[3] = {2.0/3.0, 2.0/3.0, 2.0/3.0};
            return weights[i];
        }
    };

    custom_rule rule;

    // Test weight sum
    double sum = 0.0;
    for (size_t i = 0; i < rule.size(); ++i) {
        sum += rule.weight(i);
    }
    EXPECT_NEAR(sum, 2.0, 1e-14);
}
