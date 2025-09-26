// Simple test to validate optimizations compile and work
#include <iostream>
#include <vector>
#include <chrono>
#include <stepanov/gcd_optimized.hpp>
#include <stepanov/math_optimized.hpp>

using namespace std;
using namespace stepanov;

int main() {
    cout << "Testing Optimizations...\n\n";

    // Test 1: Binary GCD
    cout << "1. Binary GCD Test:\n";
    int a = 48, b = 18;
    cout << "   gcd(" << a << ", " << b << ") = " << gcd(a, b) << "\n";
    cout << "   Expected: 6\n\n";

    // Test 2: Power optimization
    cout << "2. Power Function Test:\n";
    int base = 2, exp = 10;
    cout << "   power_optimized(" << base << ", " << exp << ") = " << power_optimized(base, exp) << "\n";
    cout << "   Expected: 1024\n\n";

    // Test 3: Modular power
    cout << "3. Modular Power Test:\n";
    cout << "   power_mod_optimized(3, 4, 7) = " << power_mod_optimized(3, 4, 7) << "\n";
    cout << "   Expected: 4 (3^4 = 81, 81 mod 7 = 4)\n\n";

    // Test 4: Performance comparison
    cout << "4. Performance Comparison:\n";
    const int iterations = 1000000;

    // GCD performance
    {
        auto start = chrono::high_resolution_clock::now();
        volatile int result = 0;
        for (int i = 0; i < iterations; ++i) {
            result = gcd(12345, 67890);
        }
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
        cout << "   Binary GCD: " << duration / 1000.0 << " ms for " << iterations << " iterations\n";
        cout << "   Result: " << result << "\n";
    }

    cout << "\nAll tests completed!\n";
    return 0;
}