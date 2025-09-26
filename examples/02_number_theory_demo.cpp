// Number Theory Algorithms Demo
// Showcases GCD, modular arithmetic, and primality testing

#include <iostream>
#include <chrono>
#include <stepanov/gcd.hpp>
#include <stepanov/mod.hpp>
#include <stepanov/primality.hpp>

using namespace stepanov;
using namespace std;
using namespace chrono;

int main() {
    cout << "=== Number Theory Algorithms Demo ===\n\n";

    // 1. GCD Algorithms
    cout << "1. GCD ALGORITHMS\n";
    cout << "-----------------\n";

    // Basic GCD
    cout << "gcd(48, 18) = " << gcd(48, 18) << "\n";
    cout << "gcd(1071, 462) = " << gcd(1071, 462) << "\n\n";

    // Extended GCD (Bezout coefficients)
    int x, y;
    int g = gcd_extended(240, 46, x, y);
    cout << "Extended GCD of 240 and 46:\n";
    cout << "  gcd = " << g << "\n";
    cout << "  Bezout: " << x << "*240 + " << y << "*46 = " << g << "\n";
    cout << "  Verify: " << (x*240 + y*46) << " = " << g << " ✓\n\n";

    // GCD works with any Euclidean domain (e.g., polynomials)
    cout << "Note: gcd() is generic - works with polynomials too!\n\n";

    // 2. Modular Arithmetic
    cout << "2. MODULAR ARITHMETIC\n";
    cout << "--------------------\n";

    // Modular inverse
    int inv = mod_inverse(7, 26);
    cout << "Modular inverse of 7 (mod 26) = " << inv << "\n";
    cout << "Verify: 7 * " << inv << " ≡ " << (7 * inv) % 26 << " (mod 26)\n\n";

    // Chinese Remainder Theorem
    vector<int> remainders = {2, 3, 2};
    vector<int> moduli = {3, 5, 7};
    int solution = chinese_remainder(remainders, moduli);
    cout << "Chinese Remainder Theorem:\n";
    cout << "  x ≡ 2 (mod 3)\n";
    cout << "  x ≡ 3 (mod 5)\n";
    cout << "  x ≡ 2 (mod 7)\n";
    cout << "  Solution: x = " << solution << "\n";
    cout << "  Verify: " << solution << " mod 3 = " << solution % 3;
    cout << ", mod 5 = " << solution % 5;
    cout << ", mod 7 = " << solution % 7 << " ✓\n\n";

    // 3. Primality Testing
    cout << "3. PRIMALITY TESTING\n";
    cout << "-------------------\n";

    // Fermat primality test
    cout << "Fermat primality test (probabilistic):\n";
    vector<int> test_nums = {17, 91, 97, 561, 1009};

    for (int n : test_nums) {
        bool is_prime = fermat_test(n, 10);  // 10 iterations
        cout << "  " << n << " is " << (is_prime ? "probably prime" : "composite");

        // Note about Carmichael numbers
        if (n == 561) {
            cout << " (561 is a Carmichael number!)";
        }
        cout << "\n";
    }
    cout << "\n";

    // Miller-Rabin test (more reliable)
    cout << "Miller-Rabin test (deterministic for small n):\n";
    for (int n : test_nums) {
        bool is_prime = miller_rabin(n);
        cout << "  " << n << " is " << (is_prime ? "prime" : "composite") << "\n";
    }
    cout << "\n";

    // 4. Performance comparison
    cout << "4. PERFORMANCE COMPARISON\n";
    cout << "------------------------\n";

    const int large_num = 1000000007;  // Large prime

    auto start = high_resolution_clock::now();
    bool fermat_result = fermat_test(large_num, 20);
    auto fermat_time = duration_cast<microseconds>(
        high_resolution_clock::now() - start).count();

    start = high_resolution_clock::now();
    bool miller_result = miller_rabin(large_num);
    auto miller_time = duration_cast<microseconds>(
        high_resolution_clock::now() - start).count();

    cout << "Testing " << large_num << ":\n";
    cout << "  Fermat test: " << fermat_time << " μs";
    cout << " (result: " << (fermat_result ? "prime" : "composite") << ")\n";
    cout << "  Miller-Rabin: " << miller_time << " μs";
    cout << " (result: " << (miller_result ? "prime" : "composite") << ")\n\n";

    // 5. Applications
    cout << "5. APPLICATIONS\n";
    cout << "---------------\n";

    // RSA-like example
    cout << "RSA-style encryption (simplified):\n";
    int p = 61, q = 53;
    int n = p * q;
    int phi = (p-1) * (q-1);
    int e = 17;  // Common choice
    int d = mod_inverse(e, phi);

    cout << "  p=" << p << ", q=" << q << ", n=" << n << "\n";
    cout << "  Public key: (n=" << n << ", e=" << e << ")\n";
    cout << "  Private key: d=" << d << "\n";

    int message = 123;
    int encrypted = power_mod(message, e, n);
    int decrypted = power_mod(encrypted, d, n);

    cout << "  Message: " << message << "\n";
    cout << "  Encrypted: " << encrypted << "\n";
    cout << "  Decrypted: " << decrypted << " ✓\n\n";

    cout << "Key Insight: These algorithms form the foundation\n";
    cout << "of modern cryptography and computational algebra.\n";

    return 0;
}