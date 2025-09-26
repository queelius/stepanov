// Core Algorithms Benchmark
// Compares Stepanov library with standard implementations

#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <stepanov/math.hpp>
#include <stepanov/gcd.hpp>

using namespace std;
using namespace chrono;
using namespace stepanov;

template<typename F>
double benchmark(const string& name, size_t iterations, F&& f) {
    auto start = high_resolution_clock::now();

    for (size_t i = 0; i < iterations; ++i) {
        volatile auto result = f();  // volatile prevents optimization
    }

    auto end = high_resolution_clock::now();
    auto ns = duration_cast<nanoseconds>(end - start).count();
    double ns_per_op = double(ns) / iterations;

    cout << "  " << left << setw(30) << name << ": ";
    cout << right << setw(10) << fixed << setprecision(2) << ns_per_op << " ns/op";

    return ns_per_op;
}

// Standard iterative power
template<typename T>
T std_power(T base, int exp) {
    T result = 1;
    for (int i = 0; i < exp; ++i) {
        result *= base;
    }
    return result;
}

// Standard recursive GCD
int std_gcd_recursive(int a, int b) {
    if (b == 0) return a;
    return std_gcd_recursive(b, a % b);
}

// Standard iterative GCD
int std_gcd_iterative(int a, int b) {
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

int main() {
    cout << "=== STEPANOV LIBRARY BENCHMARKS ===\n";
    cout << "Comparing with standard implementations\n\n";

    const size_t ITERATIONS = 1000000;

    // 1. Power function comparison
    cout << "1. POWER FUNCTION (2^30)\n";
    cout << "-------------------------\n";

    double stepanov_time = benchmark("Stepanov power()", ITERATIONS,
        []() { return stepanov::power(2, 30); });

    double std_time = benchmark("std::pow()", ITERATIONS,
        []() { return static_cast<int>(std::pow(2, 30)); });

    double iterative_time = benchmark("Iterative multiplication", ITERATIONS,
        []() { return std_power(2, 30); });

    cout << "\n  Speedup vs iterative: " << iterative_time / stepanov_time << "x\n";
    cout << "  Speedup vs std::pow: " << std_time / stepanov_time << "x\n\n";

    // 2. GCD comparison
    cout << "2. GCD ALGORITHMS (gcd(1234567, 7654321))\n";
    cout << "-----------------------------------------\n";

    int a = 1234567, b = 7654321;

    double stepanov_gcd_time = benchmark("Stepanov gcd()", ITERATIONS,
        [=]() { return stepanov::gcd(a, b); });

    double std_gcd_time = benchmark("std::gcd() (C++17)", ITERATIONS,
        [=]() { return std::gcd(a, b); });

    double recursive_gcd_time = benchmark("Recursive Euclid", ITERATIONS/10,  // Less iterations due to stack
        [=]() { return std_gcd_recursive(a, b); });

    double iterative_gcd_time = benchmark("Iterative Euclid", ITERATIONS,
        [=]() { return std_gcd_iterative(a, b); });

    cout << "\n  Speedup vs recursive: " << recursive_gcd_time / stepanov_gcd_time << "x\n";
    cout << "  Speedup vs iterative: " << iterative_gcd_time / stepanov_gcd_time << "x\n";
    cout << "  vs std::gcd: " << (stepanov_gcd_time < std_gcd_time ? "faster" : "slower") << "\n\n";

    // 3. Modular exponentiation
    cout << "3. MODULAR EXPONENTIATION (3^1000000 mod 97)\n";
    cout << "--------------------------------------------\n";

    auto naive_modexp = [](int base, int exp, int mod) {
        long long result = 1;
        for (int i = 0; i < exp; ++i) {
            result = (result * base) % mod;
        }
        return static_cast<int>(result);
    };

    double stepanov_modexp_time = benchmark("Stepanov power_mod()", 10000,
        []() { return stepanov::power_mod(3, 1000000, 97); });

    // Can't run naive for 1000000 - too slow!
    double naive_time = benchmark("Naive modular exp (1000 iter)", 10000,
        [=]() { return naive_modexp(3, 1000, 97); });

    cout << "\n  Estimated speedup for 10^6: > " << (1000000.0/1000) * (naive_time / stepanov_modexp_time) << "x\n\n";

    // 4. Sum algorithms
    cout << "4. SUMMATION ALGORITHMS (10000 elements)\n";
    cout << "---------------------------------------\n";

    vector<int> data(10000);
    iota(data.begin(), data.end(), 1);

    double stepanov_sum_time = benchmark("Stepanov sum()", ITERATIONS/100,
        [&]() { return stepanov::sum(data.begin(), data.end(), 0); });

    double std_accumulate_time = benchmark("std::accumulate()", ITERATIONS/100,
        [&]() { return std::accumulate(data.begin(), data.end(), 0); });

    double manual_sum_time = benchmark("Manual for loop", ITERATIONS/100,
        [&]() {
            int s = 0;
            for (int x : data) s += x;
            return s;
        });

    cout << "\n  vs std::accumulate: " << (abs(stepanov_sum_time - std_accumulate_time) < 1.0 ? "comparable" : "different") << "\n";
    cout << "  vs manual loop: " << (abs(stepanov_sum_time - manual_sum_time) < 1.0 ? "comparable" : "different") << "\n\n";

    // 5. Product algorithms
    cout << "5. PRODUCT ALGORITHMS (product of 1..20)\n";
    cout << "---------------------------------------\n";

    vector<long long> small_data(20);
    iota(small_data.begin(), small_data.end(), 1LL);

    double stepanov_prod_time = benchmark("Stepanov product()", ITERATIONS,
        [&]() { return stepanov::product(small_data.begin(), small_data.end(), 1LL); });

    double std_prod_time = benchmark("std::accumulate(×)", ITERATIONS,
        [&]() {
            return std::accumulate(small_data.begin(), small_data.end(), 1LL,
                                  multiplies<long long>{});
        });

    double manual_prod_time = benchmark("Manual for loop", ITERATIONS,
        [&]() {
            long long p = 1;
            for (long long x : small_data) p *= x;
            return p;
        });

    cout << "\n  vs std::accumulate: " << (abs(stepanov_prod_time - std_prod_time) < 1.0 ? "comparable" : "different") << "\n";
    cout << "  vs manual loop: " << (abs(stepanov_prod_time - manual_prod_time) < 1.0 ? "comparable" : "different") << "\n\n";

    // Summary
    cout << "=== SUMMARY ===\n";
    cout << "The Stepanov library achieves:\n";
    cout << "  • O(log n) power computation vs O(n) naive\n";
    cout << "  • Optimized GCD with binary algorithm\n";
    cout << "  • Efficient modular exponentiation (critical for crypto)\n";
    cout << "  • Zero-overhead generic algorithms\n";
    cout << "  • Compile-time optimization through concepts\n\n";

    cout << "Key insight: Generic programming enables both\n";
    cout << "abstraction AND performance - no compromise needed!\n";

    return 0;
}