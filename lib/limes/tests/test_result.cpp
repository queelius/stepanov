#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <limes/algorithms/core/result.hpp>

using namespace limes::algorithms;

// Test fixture for integration_result tests
template <typename T>
class IntegrationResultTest : public ::testing::Test {
protected:
    static constexpr T eps = std::numeric_limits<T>::epsilon();
    static constexpr T tol = eps * T(100);
};

using FloatTypes = ::testing::Types<float, double, long double>;
TYPED_TEST_SUITE(IntegrationResultTest, FloatTypes);

// Test default construction
TYPED_TEST(IntegrationResultTest, DefaultConstruction) {
    using T = TypeParam;
    integration_result<T> result;

    EXPECT_EQ(result.value(), T(0));
    EXPECT_EQ(result.error(), T(0));
    EXPECT_EQ(result.iterations(), 0u);
    EXPECT_EQ(result.evaluations(), 0u);
    EXPECT_TRUE(result.converged());
}

// Test parameterized construction (3 params)
TYPED_TEST(IntegrationResultTest, ThreeParamConstruction) {
    using T = TypeParam;

    T val = T(3.14159);
    T err = T(1e-6);
    size_t iters = 100;

    integration_result<T> result(val, err, iters);

    EXPECT_EQ(result.value(), val);
    EXPECT_EQ(result.error(), err);
    EXPECT_EQ(result.iterations(), iters);
    EXPECT_EQ(result.evaluations(), iters); // Should default to iterations
    EXPECT_TRUE(result.converged()); // Default should be true
}

// Test parameterized construction (4 params)
TYPED_TEST(IntegrationResultTest, FourParamConstruction) {
    using T = TypeParam;

    T val = T(2.718);
    T err = T(1e-8);
    size_t iters = 50;
    size_t evals = 500;

    integration_result<T> result(val, err, iters, evals);

    EXPECT_EQ(result.value(), val);
    EXPECT_EQ(result.error(), err);
    EXPECT_EQ(result.iterations(), iters);
    EXPECT_EQ(result.evaluations(), evals);
}

// Test copy construction
TYPED_TEST(IntegrationResultTest, CopyConstruction) {
    using T = TypeParam;

    integration_result<T> original(T(1.5), T(1e-5), 200, 2000);
    integration_result<T> copy(original);

    EXPECT_EQ(copy.value(), original.value());
    EXPECT_EQ(copy.error(), original.error());
    EXPECT_EQ(copy.iterations(), original.iterations());
    EXPECT_EQ(copy.evaluations(), original.evaluations());
    EXPECT_EQ(copy.converged(), original.converged());
}

// Test move construction
TYPED_TEST(IntegrationResultTest, MoveConstruction) {
    using T = TypeParam;

    integration_result<T> original(T(1.5), T(1e-5), 200, 2000);
    T orig_val = original.value();
    T orig_err = original.error();
    size_t orig_iters = original.iterations();
    size_t orig_evals = original.evaluations();
    bool orig_conv = original.converged();

    integration_result<T> moved(std::move(original));

    EXPECT_EQ(moved.value(), orig_val);
    EXPECT_EQ(moved.error(), orig_err);
    EXPECT_EQ(moved.iterations(), orig_iters);
    EXPECT_EQ(moved.evaluations(), orig_evals);
    EXPECT_EQ(moved.converged(), orig_conv);
}

// Test assignment operators
TYPED_TEST(IntegrationResultTest, Assignment) {
    using T = TypeParam;

    integration_result<T> result1(T(1.0), T(1e-6), 100);
    integration_result<T> result2(T(2.0), T(1e-7), 200);

    result1 = result2;

    EXPECT_EQ(result1.value(), result2.value());
    EXPECT_EQ(result1.error(), result2.error());
    EXPECT_EQ(result1.iterations(), result2.iterations());
}

// Test addition operator
TYPED_TEST(IntegrationResultTest, Addition) {
    using T = TypeParam;

    integration_result<T> r1(T(1.0), T(1e-6), 100, 1000);
    integration_result<T> r2(T(2.0), T(2e-6), 200, 2000);

    auto sum = r1 + r2;

    EXPECT_NEAR(sum.value(), T(3.0), this->tol);
    EXPECT_NEAR(sum.error(), T(3e-6), this->tol); // Errors add
    EXPECT_EQ(sum.iterations(), 300u); // Iterations add
    EXPECT_EQ(sum.evaluations(), 3000u); // Evaluations add
    EXPECT_TRUE(sum.converged()); // Both converged, so sum converged
}

// Test compound addition
TYPED_TEST(IntegrationResultTest, CompoundAddition) {
    using T = TypeParam;

    integration_result<T> result(T(1.0), T(1e-6), 100, 1000);
    integration_result<T> to_add(T(0.5), T(0.5e-6), 50, 500);

    result += to_add;

    EXPECT_NEAR(result.value(), T(1.5), this->tol);
    EXPECT_NEAR(result.error(), T(1.5e-6), this->tol);
    EXPECT_EQ(result.iterations(), 150u);
    EXPECT_EQ(result.evaluations(), 1500u);
}

// Test multiplication by scalar
TYPED_TEST(IntegrationResultTest, ScalarMultiplication) {
    using T = TypeParam;

    integration_result<T> result(T(2.0), T(1e-6), 100, 1000);

    auto scaled = result * T(3.0);

    EXPECT_NEAR(scaled.value(), T(6.0), this->tol);
    EXPECT_NEAR(scaled.error(), T(3e-6), this->tol); // Error scales
    EXPECT_EQ(scaled.iterations(), 100u); // Iterations unchanged
    EXPECT_EQ(scaled.evaluations(), 1000u); // Evaluations unchanged
}

TYPED_TEST(IntegrationResultTest, ScalarMultiplicationCommutative) {
    using T = TypeParam;

    integration_result<T> result(T(2.0), T(1e-6), 100, 1000);

    auto scaled1 = result * T(3.0);
    auto scaled2 = T(3.0) * result;

    EXPECT_NEAR(scaled1.value(), scaled2.value(), this->tol);
    EXPECT_NEAR(scaled1.error(), scaled2.error(), this->tol);
    EXPECT_EQ(scaled1.iterations(), scaled2.iterations());
    EXPECT_EQ(scaled1.evaluations(), scaled2.evaluations());
}

// Test compound multiplication
TYPED_TEST(IntegrationResultTest, CompoundMultiplication) {
    using T = TypeParam;

    integration_result<T> result(T(2.0), T(1e-6), 100, 1000);

    result *= T(3.0);

    EXPECT_NEAR(result.value(), T(6.0), this->tol);
    EXPECT_NEAR(result.error(), T(3e-6), this->tol);
    EXPECT_EQ(result.iterations(), 100u);
    EXPECT_EQ(result.evaluations(), 1000u);
}

// Test conversion operators
TYPED_TEST(IntegrationResultTest, ConversionOperators) {
    using T = TypeParam;

    integration_result<T> result(T(3.14), T(1e-6), 100);

    // Implicit conversion to T
    T value = result;
    EXPECT_NEAR(value, T(3.14), this->tol);

    // Explicit conversion to bool
    EXPECT_TRUE(static_cast<bool>(result));

    // Non-converged result
    integration_result<T> non_converged;
    non_converged.converged_ = false;
    EXPECT_FALSE(static_cast<bool>(non_converged));
}

// Test relative error calculation
TYPED_TEST(IntegrationResultTest, RelativeError) {
    using T = TypeParam;

    {
        integration_result<T> result(T(100.0), T(1.0), 100);
        EXPECT_NEAR(result.relative_error(), T(0.01), this->tol);
    }

    {
        integration_result<T> result(T(0.0), T(1e-6), 100);
        EXPECT_EQ(result.relative_error(), T(1e-6)); // Absolute error when value is 0
    }

    {
        integration_result<T> result(T(-50.0), T(2.5), 100);
        EXPECT_NEAR(result.relative_error(), T(0.05), this->tol);
    }
}

// Test with NaN and infinity
// NOTE: These tests verify that integration_result can handle special floating-point values.
// When -ffast-math is enabled, std::isnan() and std::isinf() are unreliable because
// the compiler assumes no NaN/inf values exist. We skip this test in that case.
#if !defined(__FAST_MATH__) && (!defined(__FINITE_MATH_ONLY__) || __FINITE_MATH_ONLY__ == 0)
TYPED_TEST(IntegrationResultTest, SpecialValues) {
    using T = TypeParam;

    // Use volatile to prevent constexpr optimization that might eliminate NaN/inf
    volatile T nan_val = std::numeric_limits<T>::quiet_NaN();
    volatile T inf_val = std::numeric_limits<T>::infinity();

    // NaN value
    {
        integration_result<T> result(static_cast<T>(nan_val), T(1e-6), 100);
        EXPECT_TRUE(std::isnan(result.value()));
        // relative_error for NaN value should also be NaN
        EXPECT_TRUE(std::isnan(result.relative_error()));
    }

    // Infinite value
    {
        integration_result<T> result(static_cast<T>(inf_val), T(1e-6), 100);
        EXPECT_TRUE(std::isinf(result.value()));
        EXPECT_EQ(result.relative_error(), T(0)); // 1e-6/inf = 0
    }

    // Infinite error
    {
        integration_result<T> result(T(1.0), static_cast<T>(inf_val), 100);
        EXPECT_TRUE(std::isinf(result.error()));
        EXPECT_TRUE(std::isinf(result.relative_error()));
    }
}
#endif

// Test precision preservation
TYPED_TEST(IntegrationResultTest, PrecisionPreservation) {
    using T = TypeParam;

    // Very small values
    {
        T tiny = std::numeric_limits<T>::min();
        integration_result<T> result(tiny, tiny / T(10), 100);

        EXPECT_EQ(result.value(), tiny);
        EXPECT_EQ(result.error(), tiny / T(10));
    }

    // Very large values
    {
        T huge = std::numeric_limits<T>::max() / T(10);
        integration_result<T> result(huge, huge / T(1000), 100);

        EXPECT_EQ(result.value(), huge);
        EXPECT_EQ(result.error(), huge / T(1000));
    }
}

// Test chaining operations
TYPED_TEST(IntegrationResultTest, OperationChaining) {
    using T = TypeParam;

    integration_result<T> r1(T(1.0), T(1e-6), 100, 1000);
    integration_result<T> r2(T(2.0), T(2e-6), 200, 2000);
    integration_result<T> r3(T(3.0), T(3e-6), 300, 3000);

    // Chain multiple operations: (r1 + r2) * 2
    auto result = (r1 + r2) * T(2.0);

    EXPECT_NEAR(result.value(), T(6.0), this->tol * T(10));
    // Error propagation: (1e-6 + 2e-6) * 2 = 6e-6
    EXPECT_NEAR(result.error(), T(6e-6), this->tol * T(100));
    EXPECT_EQ(result.iterations(), 300u);
    EXPECT_EQ(result.evaluations(), 3000u);
}

// Test optional fields
TEST(IntegrationResultOptional, OptionalFields) {
    integration_result<double> result(3.14, 1e-6, 100);

    // Check that optional fields are not set by default
    EXPECT_FALSE(result.variance_.has_value());
    EXPECT_FALSE(result.intermediate_values_.has_value());

    // Set optional fields
    result.variance_ = 1e-12;
    EXPECT_TRUE(result.variance_.has_value());
    EXPECT_EQ(result.variance_.value(), 1e-12);

    result.intermediate_values_ = std::vector<double>{1.0, 2.0, 3.0};
    EXPECT_TRUE(result.intermediate_values_.has_value());
    EXPECT_EQ(result.intermediate_values_.value().size(), 3u);
}

// Test deduction guides
TEST(IntegrationResultDeduction, DeductionGuides) {
    // Three parameter deduction
    {
        auto result = integration_result(3.14, 1e-6, 100u);
        static_assert(std::is_same_v<decltype(result), integration_result<double>>);
        EXPECT_EQ(result.value(), 3.14);
    }

    // Four parameter deduction
    {
        auto result = integration_result(3.14f, 1e-6f, 100u, 1000u);
        static_assert(std::is_same_v<decltype(result), integration_result<float>>);
        EXPECT_EQ(result.value(), 3.14f);
    }
}

// Test convergence flag behavior
TEST(IntegrationResultConvergence, ConvergenceBehavior) {
    integration_result<double> r1(1.0, 1e-6, 100);
    integration_result<double> r2(2.0, 1e-7, 200);

    // Both converged by default
    EXPECT_TRUE(r1.converged());
    EXPECT_TRUE(r2.converged());

    auto sum = r1 + r2;
    EXPECT_TRUE(sum.converged());

    // Make one non-converged
    r2.converged_ = false;
    EXPECT_FALSE(r2.converged());

    auto sum2 = r1 + r2;
    EXPECT_FALSE(sum2.converged()); // Sum should not be converged if any component isn't
}
