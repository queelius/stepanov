#include <iostream>
#include <cassert>
#include <vector>
#include <chrono>
#include <iomanip>
#include <numeric>

// Include all headers from the library
#include <stepanov/concepts.hpp>
#include <stepanov/builtin_adaptors.hpp>
#include <stepanov/math.hpp>
#include <stepanov/gcd.hpp>
#include <stepanov/polynomial.hpp>
#include <stepanov/bounded_nat.hpp>
#include <stepanov/algorithms.hpp>
#include <stepanov/type_erasure.hpp>

using namespace stepanov;
using namespace std::chrono;

// Helper function for timing
template <typename F>
auto measure_time(F&& f, const std::string& name) {
    auto start = high_resolution_clock::now();
    auto result = f();
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    std::cout << name << " took " << duration.count() << " microseconds\n";
    return result;
}

// Test basic mathematical operations
void test_math_operations() {
    std::cout << "\n=== Testing Math Operations ===\n";

    // Test with built-in types
    assert(power_accumulate(2, 10, 1) == 1024);
    assert(power_mod(3, 100, 7) == 4);
    assert(binary_gcd(48, 18) == 6);

    // Test inner product
    std::vector<int> v1{1, 2, 3, 4};
    std::vector<int> v2{5, 6, 7, 8};
    assert(inner_product(v1.begin(), v1.end(), v2.begin(), 0) == 70);

    std::cout << "Math operations tests passed!\n";
}

// Test GCD and number theory algorithms
void test_gcd_algorithms() {
    std::cout << "\n=== Testing GCD Algorithms ===\n";

    // Basic GCD - use binary_gcd for built-in integers
    assert(binary_gcd(48, 18) == 6);

    using nat = bounded_nat<8>;
    assert(lcm(nat(12), nat(18)) == nat(36));

    // Extended GCD with bounded_nat for demonstration
    using nat = bounded_nat<8>;
    auto [g, x, y] = extended_gcd(nat(30), nat(18));
    assert(g == nat(6));
    assert(nat(30) * x + nat(18) * y == g);

    // Multiple GCD with bounded_nat
    assert(gcd(nat(12), nat(18), nat(24)) == nat(6));
    assert(lcm(nat(4), nat(6), nat(8)) == nat(24));

    // Coprime check
    assert(coprime(nat(15), nat(28)) == true);
    assert(coprime(nat(15), nat(30)) == false);

    // Modular inverse
    auto inv = mod_inverse(nat(3), nat(11));
    assert(inv.has_value());
    assert((nat(3) * inv.value()) % nat(11) == nat(1));

    std::cout << "GCD algorithms tests passed!\n";
}

// Test polynomial operations
void test_polynomials() {
    std::cout << "\n=== Testing Polynomials ===\n";

    using poly = polynomial<double>;

    // Create polynomial: x^2 + 2x + 1
    poly p1{{0, 1.0}, {1, 2.0}, {2, 1.0}};

    // Evaluate at x = 3: 3^2 + 2*3 + 1 = 16
    assert(std::abs(p1(3.0) - 16.0) < 1e-10);

    // Create polynomial: x - 1
    poly p2{{0, -1.0}, {1, 1.0}};

    // Multiply: (x^2 + 2x + 1)(x - 1) = x^3 + x^2 - x - 1
    auto p3 = p1 * p2;
    assert(std::abs(p3[0] - (-1.0)) < 1e-10);
    assert(std::abs(p3[1] - (-1.0)) < 1e-10);
    assert(std::abs(p3[2] - 1.0) < 1e-10);
    assert(std::abs(p3[3] - 1.0) < 1e-10);

    // Test derivative: d/dx(x^2 + 2x + 1) = 2x + 2
    auto dp = derivative(p1);
    assert(std::abs(dp[0] - 2.0) < 1e-10);
    assert(std::abs(dp[1] - 2.0) < 1e-10);

    // Test root finding
    poly p4{{0, -2.0}, {1, 0.0}, {2, 1.0}};  // x^2 - 2
    newton_solver<double> solver(1e-10);
    auto root = solver.find_root(p4, 1.0);
    assert(root.has_value());
    assert(std::abs(root.value() - std::sqrt(2.0)) < 1e-9);

    std::cout << "Polynomial tests passed!\n";
}

// Test bounded natural numbers
void test_bounded_nat() {
    std::cout << "\n=== Testing Bounded Natural Numbers ===\n";

    using nat32 = bounded_nat<4>;  // 32-bit natural numbers

    nat32 a(100);
    nat32 b(200);

    // Test arithmetic
    assert(a + b == nat32(300));
    assert(b - a == nat32(100));
    assert(a * nat32(2) == b);

    // Test bit operations
    assert(even(nat32(4)) == true);
    assert(even(nat32(5)) == false);
    assert(twice(nat32(5)) == nat32(10));
    assert(half(nat32(10)) == nat32(5));

    // Test comparison
    assert(a < b);
    assert(b > a);
    assert(a <= a);
    assert(b >= b);

    // Test GCD with bounded_nat
    assert(gcd(nat32(48), nat32(18)) == nat32(6));

    std::cout << "Bounded natural number tests passed!\n";
}

// Test generic algorithms
void test_algorithms() {
    std::cout << "\n=== Testing Generic Algorithms ===\n";

    // Test orbit and cycle detection
    auto collatz = [](int n) -> int {
        return (n % 2 == 0) ? n / 2 : 3 * n + 1;
    };

    // Collatz sequence from 27
    auto seq = orbit(27, 20, collatz);
    assert(seq.size() == 21);  // Initial value + 20 iterations

    // Test accumulation
    std::vector<int> nums{1, 2, 3, 4, 5};
    auto sum = accumulate(nums.begin(), nums.end(), 0,
                          [](int a, int b) { return a + b; });
    assert(sum == 15);

    // Test three-way partition
    std::vector<int> data{3, 1, 4, 1, 5, 9, 2, 6, 5, 3};
    auto [less_end, equal_end] = partition_3way(data.begin(), data.end(), 4);

    // Check partitioning
    for (auto it = data.begin(); it != less_end; ++it) {
        assert(*it < 4);
    }
    for (auto it = less_end; it != equal_end; ++it) {
        assert(*it == 4);
    }
    for (auto it = equal_end; it != data.end(); ++it) {
        assert(*it > 4);
    }

    // Test function composition
    auto add_one = [](int x) { return x + 1; };
    auto double_it = [](int x) { return x * 2; };
    auto f = compose(add_one, double_it);  // f(x) = 2x + 1
    assert(f(3) == 7);

    std::cout << "Generic algorithms tests passed!\n";
}

// Test type erasure
void test_type_erasure() {
    std::cout << "\n=== Testing Type Erasure ===\n";

    // Test any_regular
    any_regular r1(42);
    any_regular r2(3.14);
    any_regular r3(42);

    assert(r1 == r3);
    assert(r1 != r2);
    assert(r1.holds<int>());
    assert(!r1.holds<double>());
    assert(*r1.get<int>() == 42);

    // Test any_algebraic
    any_algebraic a1(5);
    any_algebraic a2(3);
    auto a3 = a1 + a2;
    auto a4 = a1 * a2;

    // Note: We can't directly check the value without knowing the type
    // but we can verify operations work
    assert(a3 == any_algebraic(8));
    assert(a4 == any_algebraic(15));

    // Test any_function
    any_function<int(int, int)> f1 = [](int a, int b) { return a + b; };
    assert(f1(3, 4) == 7);

    any_function<int(int, int)> f2 = [](int a, int b) { return a * b; };
    assert(f2(3, 4) == 12);

    std::cout << "Type erasure tests passed!\n";
}

// Benchmark generic vs specific implementations
void benchmark_algorithms() {
    std::cout << "\n=== Benchmarking Algorithms ===\n";

    const int N = 1000000;
    std::vector<int> data(N);
    for (int i = 0; i < N; ++i) {
        data[i] = i;
    }

    // Benchmark generic accumulate
    auto sum1 = measure_time([&]() {
        return accumulate(data.begin(), data.end(), 0,
                         [](int a, int b) { return a + b; });
    }, "Generic accumulate");

    // Benchmark std::accumulate
    auto sum2 = measure_time([&]() {
        return std::accumulate(data.begin(), data.end(), 0);
    }, "std::accumulate");

    assert(sum1 == sum2);

    // Benchmark GCD algorithms
    int a = 987654321, b = 123456789;

    using nat64 = bounded_nat<16>;
    auto gcd1 = measure_time([&]() {
        return gcd(nat64(a), nat64(b));
    }, "Euclidean GCD");

    auto gcd2 = measure_time([&]() {
        return binary_gcd(a, b);
    }, "Binary GCD");

    assert(gcd1 == nat64(gcd2));
}

// Demonstrate elegant API usage
void demonstrate_api() {
    std::cout << "\n=== API Demonstration ===\n";

    // Elegant polynomial composition
    polynomial<double> p{{0, -1}, {2, 1}};  // x^2 - 1
    polynomial<double> q{{1, 1}};           // x

    auto roots = stationary_points(p * q);  // Find where d/dx[(x^2 - 1)x] = 0
    std::cout << "Stationary points found: " << roots.size() << "\n";

    // Elegant algorithm chaining
    std::vector<int> numbers{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // Find sum of squares of even numbers
    auto sum_even_squares = accumulate(
        numbers.begin(), numbers.end(), 0,
        [](int acc, int x) {
            return (x % 2 == 0) ? acc + x * x : acc;
        });

    std::cout << "Sum of squares of even numbers: " << sum_even_squares << "\n";

    // Type erasure for heterogeneous collections
    std::vector<any_algebraic> values;
    values.emplace_back(5);
    values.emplace_back(3.14);
    values.emplace_back(bounded_nat<2>(100));

    std::cout << "Heterogeneous collection created with "
              << values.size() << " elements\n";
}

int main() {
    std::cout << "Generic Math Library Test Suite\n";
    std::cout << "================================\n";

    try {
        test_math_operations();
        test_gcd_algorithms();
        test_polynomials();
        test_bounded_nat();
        test_algorithms();
        test_type_erasure();
        benchmark_algorithms();
        demonstrate_api();

        std::cout << "\n=== ALL TESTS PASSED ===\n";
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}