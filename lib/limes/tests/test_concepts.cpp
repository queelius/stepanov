#include <gtest/gtest.h>
#include <type_traits>
#include <complex>
#include <functional>
#include <limes/algorithms/concepts/concepts.hpp>
#include <limes/algorithms/accumulators/accumulators.hpp>
#include <limes/algorithms/core/result.hpp>

using namespace limes::algorithms::concepts;

TEST(ConceptsTest, FieldTypes) {
    EXPECT_TRUE(Field<float>);
    EXPECT_TRUE(Field<double>);
    EXPECT_TRUE(Field<long double>);
    EXPECT_TRUE(Field<std::complex<double>>);
}

TEST(ConceptsTest, FunctionTypes) {
    // Lambda
    auto lambda = [](double x) { return x * x; };
    EXPECT_TRUE((UnivariateFunction<decltype(lambda), double>));

    // Function pointer
    double (*func_ptr)(double) = [](double x) { return std::sin(x); };
    EXPECT_TRUE((UnivariateFunction<decltype(func_ptr), double>));

    // std::function
    std::function<double(double)> std_func = [](double x) { return x; };
    EXPECT_TRUE((UnivariateFunction<decltype(std_func), double>));

    // Functor
    struct Functor {
        double operator()(double x) const { return x; }
    };
    EXPECT_TRUE((UnivariateFunction<Functor, double>));
}

TEST(ConceptsTest, ConstrainedTemplates) {
    auto add_fields = []<typename T>(T a, T b) -> T
        requires Field<T> {
        return a + b;
    };
    EXPECT_EQ(add_fields(3.0, 4.0), 7.0);

    auto evaluate_at_zero = []<typename F>(F f) -> double
        requires UnivariateFunction<F, double> {
        return f(0.0);
    };

    auto constant_func = [](double) { return 5.0; };
    EXPECT_EQ(evaluate_at_zero(constant_func), 5.0);
}

TEST(ConceptsTest, LibraryTypes) {
    using SimpleAcc = limes::algorithms::accumulators::simple_accumulator<double>;
    EXPECT_TRUE((Accumulator<SimpleAcc, double>));

    EXPECT_TRUE(Field<double>);
    limes::algorithms::integration_result<double> result(1.0, 1e-6, 100);
    EXPECT_EQ(result.value(), 1.0);
}
