// Core Mathematical Operations Demo
// Demonstrates Stepanov's generic programming principles

#include <iostream>
#include <stepanov/math.hpp>
#include <stepanov/concepts.hpp>

using namespace stepanov;
using namespace std;

// Custom type implementing required operations
struct modular_int {
    int value;
    int modulus;

    modular_int(int v = 0, int m = 7) : value(v % m), modulus(m) {}

    modular_int& operator+=(const modular_int& x) {
        value = (value + x.value) % modulus;
        return *this;
    }

    modular_int& operator*=(const modular_int& x) {
        value = (value * x.value) % modulus;
        return *this;
    }

    bool operator==(const modular_int& x) const {
        return value == x.value && modulus == x.modulus;
    }
};

modular_int operator+(modular_int a, const modular_int& b) {
    return a += b;
}

modular_int operator*(modular_int a, const modular_int& b) {
    return a *= b;
}

modular_int twice(const modular_int& x) {
    return x + x;
}

modular_int half(const modular_int& x) {
    // For demo - proper implementation would use modular inverse
    return modular_int(x.value / 2, x.modulus);
}

bool even(const modular_int& x) {
    return x.value % 2 == 0;
}

int main() {
    cout << "=== Core Mathematical Operations Demo ===\n\n";

    // 1. Generic power function works with any semiring
    cout << "1. Generic Power Function\n";
    cout << "   Works with any type satisfying semiring concept:\n\n";

    // With integers
    cout << "   Integer: 2^10 = " << power(2, 10) << "\n";

    // With doubles
    cout << "   Double: 1.5^8 = " << power(1.5, 8) << "\n";

    // With custom modular arithmetic type
    modular_int base(3, 7);
    auto result = power(base, 5);
    cout << "   Modular: 3^5 (mod 7) = " << result.value << "\n\n";

    // 2. Efficient algorithms using primitive operations
    cout << "2. Efficient Algorithms via Primitives\n";
    cout << "   Using twice(), half(), even() for efficiency:\n\n";

    // Fast multiplication using Russian peasant algorithm
    int a = 37, b = 41;
    cout << "   multiply_accumulate(" << a << ", " << b << ", 0) = ";
    cout << multiply_accumulate(a, b, 0) << "\n";
    cout << "   (uses only addition and bit operations)\n\n";

    // 3. Power with accumulation
    cout << "3. Power with Accumulation\n";
    cout << "   Compute a^n * m efficiently:\n\n";

    cout << "   power_accumulate(2, 10, 100) = ";
    cout << power_accumulate(2, 10, 100) << "\n";
    cout << "   (2^10 * 100 = 1024 * 100 = 102400)\n\n";

    // 4. Modular exponentiation
    cout << "4. Modular Exponentiation\n";
    cout << "   Critical for cryptography:\n\n";

    cout << "   power_mod(3, 1000000, 97) = ";
    cout << power_mod(3, 1000000, 97) << "\n";
    cout << "   (3^1000000 mod 97, computed efficiently)\n\n";

    // 5. Generic sum function
    cout << "5. Generic Sum Function\n";
    cout << "   Works with any additive monoid:\n\n";

    vector<int> nums = {1, 2, 3, 4, 5};
    cout << "   sum([1,2,3,4,5]) = " << sum(nums.begin(), nums.end(), 0) << "\n";

    vector<string> words = {"Hello", ", ", "World", "!"};
    cout << "   sum([\"Hello\", \", \", \"World\", \"!\"]) = ";
    cout << sum(words.begin(), words.end(), string{}) << "\n\n";

    // 6. Demonstrating the power of abstraction
    cout << "6. Power of Generic Programming\n";
    cout << "   Same algorithm, different algebraic structures:\n\n";

    // Matrix multiplication (if we had matrices)
    cout << "   power(matrix, n) computes matrix^n\n";
    cout << "   power(polynomial, n) computes polynomial^n\n";
    cout << "   power(permutation, n) computes permutation^n\n";
    cout << "   All using the SAME generic power() function!\n\n";

    cout << "Key Insight: By programming to mathematical abstractions\n";
    cout << "(groups, rings, fields), we write algorithms once that\n";
    cout << "work correctly for infinitely many types.\n";

    return 0;
}