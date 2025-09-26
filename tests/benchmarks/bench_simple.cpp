#include <iostream>
#include <chrono>
#include <vector>
#include <stepanov/builtin_adaptors.hpp>
#include <stepanov/math.hpp>
#include <stepanov/gcd.hpp>

using namespace stepanov;
using namespace std::chrono;

template<typename F>
void benchmark(const std::string& name, F&& func, int iterations = 1000000) {
    auto start = high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        func();
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);

    std::cout << name << ": "
              << duration.count() / 1000.0 << " ms for "
              << iterations << " iterations ("
              << duration.count() / static_cast<double>(iterations) << " us/op)\n";
}

int main() {
    std::cout << "=== Generic Math Benchmarks ===\n\n";

    // Math operations benchmarks
    std::cout << "Math Operations:\n";
    benchmark("product(123, 456)", []() {
        volatile auto result = product(123, 456);
    });

    benchmark("power(2, 30)", []() {
        volatile auto result = power(2, 30);
    }, 100000);

    benchmark("square(12345)", []() {
        volatile auto result = square(12345);
    });

    // GCD benchmarks
    std::cout << "\nGCD Algorithms:\n";
    benchmark("gcd(48271, 31415)", []() {
        volatile auto result = gcd(48271, 31415);
    }, 100000);

    benchmark("binary_gcd(48271, 31415)", []() {
        volatile auto result = binary_gcd(48271, 31415);
    }, 100000);

    benchmark("lcm(123, 456)", []() {
        volatile auto result = lcm(123, 456);
    }, 100000);

    // Power mod benchmark
    std::cout << "\nModular Arithmetic:\n";
    benchmark("power_mod(2, 1000, 1000007)", []() {
        volatile auto result = power_mod(2, 1000, 1000007);
    }, 10000);

    return 0;
}