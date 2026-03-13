#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <limes/limes.hpp>

using namespace limes::algorithms;

template<typename T>
void print_result(const char* name, const integration_result<T>& result, T exact = 0) {
    std::cout << std::setw(25) << name << ": "
              << std::setprecision(12) << result.value()
              << " +/- " << std::scientific << result.error()
              << " (" << result.evaluations() << " evals)";

    if (exact != 0) {
        T actual_error = std::abs(result.value() - exact);
        std::cout << " | actual error: " << actual_error;
    }
    std::cout << "\n";
}

int main() {
    using T = double;
    constexpr T pi = std::numbers::pi_v<T>;

    std::cout << "=== limes Library Demo ===\n\n";

    // 1. Simple integral: int_0^1 x^2 dx = 1/3
    {
        std::cout << "1. Simple polynomial: int_0^1 x^2 dx = 1/3\n";

        auto f = [](T x) { return x * x; };
        T exact = T(1) / T(3);

        // Adaptive Gauss-Kronrod
        auto adaptive = adaptive_integrator<T>{};
        auto r1 = adaptive(f, T(0), T(1));
        print_result("Adaptive G-K", r1, exact);

        // Romberg
        auto romberg = romberg_integrator<T>{};
        auto r2 = romberg(f, T(0), T(1));
        print_result("Romberg", r2, exact);

        // Direct quadrature
        using rule = quadrature::gauss_legendre<T, 5>;
        quadrature_integrator<T, rule> direct{rule{}};
        auto r3 = direct(f, T(0), T(1));
        print_result("Gauss-Legendre 5", r3, exact);

        std::cout << "\n";
    }

    // 2. Trigonometric: int_0^pi sin(x) dx = 2
    {
        std::cout << "2. Trigonometric: int_0^pi sin(x) dx = 2\n";

        auto f = [](T x) { return std::sin(x); };
        T exact = T(2);

        auto adaptive = adaptive_integrator<T>{};
        auto result = adaptive(f, T(0), pi);
        print_result("Adaptive", result, exact);

        std::cout << "\n";
    }

    // 3. Gaussian: int_{-5}^{5} exp(-x^2) dx ~ sqrt(pi)
    {
        std::cout << "3. Gaussian: int_{-5}^{5} exp(-x^2) dx ~ sqrt(pi)\n";

        auto f = [](T x) { return std::exp(-x * x); };
        T exact = std::sqrt(pi);

        auto adaptive = adaptive_integrator<T>{};
        auto result = adaptive(f, T(-5), T(5), T(1e-10));
        print_result("Adaptive", result, exact);

        std::cout << "\n";
    }

    // 4. Exponential decay: int_0^inf exp(-x) dx = 1 (approximated with large upper bound)
    {
        std::cout << "4. Exponential: int_0^30 exp(-x) dx ~ 1\n";

        auto f = [](T x) { return std::exp(-x); };
        T exact = T(1);

        auto adaptive = adaptive_integrator<T>{};
        auto result = adaptive(f, T(0), T(30), T(1e-10));
        print_result("Adaptive", result, exact);

        std::cout << "\n";
    }

    // 5. Using different accumulators
    {
        std::cout << "5. Accumulator comparison (sum of 1e-16, 1e6 times)\n";

        auto f = [](T x) { return T(1e-16); };
        std::size_t n = 1000000;
        T a = 0, b = n;
        T exact = T(1e-16) * n;

        auto integrate_with = [&](auto acc) {
            using Acc = decltype(acc);
            using Rule = quadrature::midpoint_rule<T>;
            quadrature_integrator<T, Rule, Acc> integrator{Rule{}, acc};
            return integrator(f, a, b);
        };

        print_result("Simple accumulator", integrate_with(accumulators::simple_accumulator<T>{}), exact);
        print_result("Kahan accumulator",  integrate_with(accumulators::kahan_accumulator<T>{}),  exact);
        print_result("Klein accumulator",  integrate_with(accumulators::klein_accumulator<T>{}),  exact);

        std::cout << "\n";
    }

    // 6. Composing different quadrature rules
    {
        std::cout << "6. Composing different quadrature rules\n";

        auto f = [](T x) { return std::exp(-x) * std::cos(x); };
        T a = 0, b = 10;
        T tol = T(1e-8);

        auto integrate_with = [&](auto rule) {
            using Rule = decltype(rule);
            quadrature_integrator<T, Rule> integrator{rule};
            return integrator(f, a, b, tol);
        };

        print_result("Simpson (adaptive)",  integrate_with(quadrature::simpson_rule<T>{}));
        print_result("Gauss-Kronrod 15",    integrate_with(quadrature::gauss_kronrod_15<T>{}));
        print_result("Clenshaw-Curtis 17",  integrate_with(quadrature::clenshaw_curtis<T, 17>{}));

        std::cout << "\n";
    }

    std::cout << "=== Demo Complete ===\n";
}
