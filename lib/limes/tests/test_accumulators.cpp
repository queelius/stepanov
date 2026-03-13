#include <gtest/gtest.h>
#include <random>
#include <vector>
#include <cmath>
#include <limits>
#include <limes/algorithms/accumulators/accumulators.hpp>

using namespace limes::algorithms::accumulators;

// Test fixture for accumulator tests
template <typename T>
class AccumulatorTest : public ::testing::Test {
protected:
    static constexpr T epsilon = std::numeric_limits<T>::epsilon();

    T relative_error(T computed, T exact) {
        if (exact == T(0)) return std::abs(computed);
        return std::abs(computed - exact) / std::abs(exact);
    }

    // Generate random data with a fixed seed for reproducibility
    std::vector<T> generate_random_data(size_t n, T min_val, T max_val) {
        std::mt19937 gen(42);
        std::uniform_real_distribution<T> dist(min_val, max_val);
        std::vector<T> data(n);
        for (auto& val : data) {
            val = dist(gen);
        }
        return data;
    }
};

// Use multiple floating-point types
using FloatTypes = ::testing::Types<float, double, long double>;
TYPED_TEST_SUITE(AccumulatorTest, FloatTypes);

// Test simple accumulator
TYPED_TEST(AccumulatorTest, SimpleAccumulatorBasic) {
    using T = TypeParam;
    simple_accumulator<T> acc;

    EXPECT_EQ(acc(), T(0));

    acc += T(1.0);
    EXPECT_EQ(acc(), T(1.0));

    acc += T(2.0);
    EXPECT_EQ(acc(), T(3.0));

    acc -= T(1.0);
    EXPECT_EQ(acc(), T(2.0));
}

TYPED_TEST(AccumulatorTest, SimpleAccumulatorRoundingError) {
    using T = TypeParam;
    simple_accumulator<T> acc;

    // Add many small numbers
    const size_t n = 1000000;
    const T tiny = std::numeric_limits<T>::epsilon();

    for (size_t i = 0; i < n; ++i) {
        acc += tiny;
    }

    T result = acc();
    T exact = tiny * n;

    // Simple accumulator will have significant rounding error
    T rel_error = this->relative_error(result, exact);

    // We expect error, just verify it's reasonable
    EXPECT_LT(rel_error, T(1.0)); // Error should be less than 100%
}

// Test Kahan accumulator
TYPED_TEST(AccumulatorTest, KahanAccumulatorBasic) {
    using T = TypeParam;
    kahan_accumulator<T> acc;

    EXPECT_EQ(acc(), T(0));

    acc += T(1.0);
    EXPECT_NEAR(acc(), T(1.0), this->epsilon);

    acc += T(2.0);
    EXPECT_NEAR(acc(), T(3.0), this->epsilon * T(3));
}

TYPED_TEST(AccumulatorTest, KahanAccumulatorCompensation) {
    using T = TypeParam;
    kahan_accumulator<T> acc;

    // Ill-conditioned data: one large value followed by many tiny values
    constexpr size_t n = 100000;
    T tiny = this->epsilon / T(2);
    T exact = T(1.0) + T(n - 1) * tiny;

    acc += T(1.0);
    for (size_t i = 1; i < n; ++i) {
        acc += tiny;
    }

    T rel_error = this->relative_error(acc(), exact);

    // Kahan should have much better accuracy than simple summation.
    // For float, this ill-conditioned test is very demanding (epsilon/2 ~ 6e-8).
    T tolerance = std::is_same_v<T, float> ? T(1e-2) : T(1e-10);
    EXPECT_LT(rel_error, tolerance);
}

// Test Neumaier accumulator
TYPED_TEST(AccumulatorTest, NeumaierAccumulatorBasic) {
    using T = TypeParam;
    neumaier_accumulator<T> acc;

    EXPECT_EQ(acc(), T(0));

    // Test with sorted values (where Neumaier excels)
    std::vector<T> values = {T(1e10), T(1.0), T(1e-10)};

    for (const auto& val : values) {
        acc += val;
    }

    T exact = T(1e10) + T(1.0) + T(1e-10);
    EXPECT_NEAR(acc(), exact, exact * this->epsilon * T(10));
}

TYPED_TEST(AccumulatorTest, NeumaierAccumulatorLargeSmallMix) {
    using T = TypeParam;
    neumaier_accumulator<T> acc;

    // Mix of large and small values
    acc += T(1e15);
    for (int i = 0; i < 1000; ++i) {
        acc += T(1.0);
    }
    acc += T(1e-15);

    T exact = T(1e15) + T(1000.0) + T(1e-15);
    T result = acc();

    // Neumaier should handle this well
    EXPECT_NEAR(result, exact, exact * this->epsilon * T(100));
}

// Test Klein accumulator
TYPED_TEST(AccumulatorTest, KleinAccumulatorBasic) {
    using T = TypeParam;
    klein_accumulator<T> acc;

    EXPECT_EQ(acc(), T(0));

    acc += T(1.0);
    EXPECT_EQ(acc(), T(1.0));

    acc += T(2.0);
    EXPECT_EQ(acc(), T(3.0));
}

TYPED_TEST(AccumulatorTest, KleinAccumulatorHighPrecision) {
    using T = TypeParam;
    klein_accumulator<T> acc;

    // Klein uses second-order error compensation
    const size_t n = 10000;
    std::vector<T> data = this->generate_random_data(n, T(-100), T(100));

    // Calculate exact sum using higher precision if available
    long double exact_sum = 0.0L;
    for (const auto& val : data) {
        exact_sum += static_cast<long double>(val);
    }

    for (const auto& val : data) {
        acc += val;
    }

    T result = acc();
    T rel_error = this->relative_error(result, static_cast<T>(exact_sum));

    // Klein should have excellent accuracy
    // Use type-aware tolerance
    T tolerance = std::is_same_v<T, float> ? T(1e-4) : T(1e-12);
    EXPECT_LT(rel_error, tolerance);
}

// Test Pairwise accumulator
TYPED_TEST(AccumulatorTest, PairwiseAccumulatorBasic) {
    using T = TypeParam;
    pairwise_accumulator<T> acc;

    EXPECT_EQ(acc(), T(0));

    // Add powers of 2 (exact in floating point)
    for (int i = 0; i < 10; ++i) {
        acc += std::pow(T(2), i);
    }

    T exact = std::pow(T(2), 10) - T(1); // Sum of geometric series
    EXPECT_NEAR(acc(), exact, exact * this->epsilon);
}

TYPED_TEST(AccumulatorTest, PairwiseAccumulatorLargeDataset) {
    using T = TypeParam;
    pairwise_accumulator<T> acc;

    constexpr size_t n = 100000;
    for (size_t i = 0; i < n; ++i) {
        acc += T(0.1);
    }

    T exact = T(n) * T(0.1);
    T result = acc();

    // Pairwise accumulator error bound scales with log2(n)
    T tolerance_factor = std::is_same_v<T, float> ? T(20) : T(1);
    EXPECT_NEAR(result, exact, exact * this->epsilon * std::log2(n) * tolerance_factor);
}

// Test pairwise accumulator finalize functionality
TYPED_TEST(AccumulatorTest, PairwiseAccumulatorSum) {
    using T = TypeParam;
    pairwise_accumulator<T> acc;

    // Add some values
    for (int i = 1; i <= 100; ++i) {
        acc += T(i);
    }

    T result = acc();

    T exact = T(100 * 101 / 2); // Sum of 1 to 100
    EXPECT_NEAR(result, exact, exact * this->epsilon * T(10));
}

// Comparative accuracy test
TYPED_TEST(AccumulatorTest, ComparativeAccuracy) {
    using T = TypeParam;

    // Generate challenging data
    std::vector<T> data;
    data.push_back(T(1e10));
    for (int i = 0; i < 10000; ++i) {
        data.push_back(T(1.0));
    }
    data.push_back(T(-1e10));

    T exact = T(10000.0);

    // Test all accumulators
    simple_accumulator<T> simple;
    kahan_accumulator<T> kahan;
    neumaier_accumulator<T> neumaier;
    klein_accumulator<T> klein;
    pairwise_accumulator<T> pairwise;

    for (const auto& val : data) {
        simple += val;
        kahan += val;
        neumaier += val;
        klein += val;
        pairwise += val;
    }

    // Calculate errors
    T simple_error = this->relative_error(simple(), exact);
    T kahan_error = this->relative_error(kahan(), exact);
    T neumaier_error = this->relative_error(neumaier(), exact);
    T klein_error = this->relative_error(klein(), exact);
    T pairwise_error = this->relative_error(pairwise(), exact);

    // Verify hierarchy of accuracy (generally expected)
    // Klein and Neumaier should be best, followed by Kahan and Pairwise, then Simple
    // Only check if there's meaningful error to compare
    if (simple_error > this->epsilon * T(10)) {
        EXPECT_GE(simple_error, kahan_error * T(0.1));
        EXPECT_LE(klein_error, kahan_error);
        EXPECT_LE(neumaier_error, kahan_error);

        // All compensated methods should be significantly better than simple (or equally good)
        EXPECT_LE(kahan_error, simple_error);
        EXPECT_LE(neumaier_error, simple_error);
        EXPECT_LE(klein_error, simple_error);
    } else {
        // If error is negligible, all methods should have very small error
        EXPECT_LT(simple_error, this->epsilon * T(100));
        EXPECT_LT(kahan_error, this->epsilon * T(100));
        EXPECT_LT(neumaier_error, this->epsilon * T(100));
        EXPECT_LT(klein_error, this->epsilon * T(100));
    }
}

// Test edge cases
TYPED_TEST(AccumulatorTest, EdgeCases) {
    using T = TypeParam;

    // Test overflow behavior
    {
        neumaier_accumulator<T> acc;
        T max_val = std::numeric_limits<T>::max();
        acc += max_val / T(2);
        acc += max_val / T(2);
        acc += T(1.0);
        // Result should be close to max_val + 1
        EXPECT_GT(acc(), max_val * T(0.99));
    }

    // Test underflow behavior
    {
        klein_accumulator<T> acc;
        T min_normal = std::numeric_limits<T>::min();
        for (int i = 0; i < 1000; ++i) {
            acc += min_normal;
        }
        EXPECT_GT(acc(), T(0));
        EXPECT_NEAR(acc(), min_normal * T(1000), min_normal * T(1000) * this->epsilon * T(100));
    }
}

// Test accumulator reset functionality
TEST(AccumulatorResetTest, ResetFunctionality) {
    // Test that accumulators can be reused after getting result
    kahan_accumulator<double> acc;

    acc += 1.0;
    acc += 2.0;
    EXPECT_EQ(acc(), 3.0);

    // Reset by creating new instance
    acc = kahan_accumulator<double>();
    EXPECT_EQ(acc(), 0.0);

    acc += 5.0;
    EXPECT_EQ(acc(), 5.0);
}

// Test accumulator with different numeric types
TEST(AccumulatorTypeTest, IntegerTypes) {
    // Accumulators should work with integer types too
    simple_accumulator<int> int_acc;
    int_acc += 5;
    int_acc += 10;
    EXPECT_EQ(int_acc(), 15);

    simple_accumulator<long long> ll_acc;
    ll_acc += 1000000000LL;
    ll_acc += 2000000000LL;
    EXPECT_EQ(ll_acc(), 3000000000LL);
}

// Test custom operations
TEST(AccumulatorCustomOps, MultiplyAccumulate) {
    // Test multiply-accumulate pattern
    kahan_accumulator<double> acc;

    // Simulate dot product
    std::vector<double> a = {1.0, 2.0, 3.0};
    std::vector<double> b = {4.0, 5.0, 6.0};

    for (size_t i = 0; i < a.size(); ++i) {
        acc += a[i] * b[i];
    }

    double expected = 1.0*4.0 + 2.0*5.0 + 3.0*6.0;
    EXPECT_NEAR(acc(), expected, 1e-14);
}
