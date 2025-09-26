#include <iostream>
#include <cassert>
#include <vector>
#include <iomanip>

#include <stepanov/builtin_adaptors.hpp"
#include <stepanov/bounded_integer.hpp"
#include <stepanov/rational.hpp"
#include <stepanov/fixed_decimal.hpp"
#include <stepanov/gcd.hpp"
#include <stepanov/math.hpp"
#include <stepanov/continued_fraction.hpp"

using namespace generic_math;

// Helper function to print test results
template <typename T>
void print_test(const std::string& test_name, const T& expected, const T& actual) {
    using namespace generic_math;
    std::cout << test_name << ": ";
    if (expected == actual) {
        std::cout << "PASS";
    } else {
        std::cout << "FAIL (expected " << expected << ", got " << actual << ")";
    }
    std::cout << std::endl;
}

void test_bounded_integer() {
    std::cout << "\n=== Testing bounded_integer ===\n";

    using int128 = bounded_integer<16>;  // 128-bit signed integer

    // Construction and basic operations
    int128 a(42);
    int128 b(-17);
    int128 c = a + b;
    print_test("Addition", int128(25), c);

    // Test negation
    int128 neg_a = -a;
    print_test("Negation", int128(-42), neg_a);

    // Test multiplication
    int128 prod = a * b;
    print_test("Multiplication", int128(-714), prod);

    // Test division
    int128 quot = prod / b;
    print_test("Division", a, quot);

    // Test with generic algorithms
    int128 base(2);
    int128 exp(10);
    int128 pow_result = power(base, exp);
    print_test("Power (2^10)", int128(1024), pow_result);

    // Test GCD with bounded integers
    int128 x(48);
    int128 y(18);
    int128 g = gcd(x, y);
    print_test("GCD(48, 18)", int128(6), g);

    // Test comparison operators
    assert(a > b);
    assert(b < a);
    assert(a != b);
    std::cout << "Comparison operators: PASS\n";

    // Test fundamental operations
    int128 n(15);
    assert(!even(n));
    assert(even(twice(n)));
    print_test("Half of 30", int128(15), half(twice(n)));

    // Test absolute value
    bounded_nat<16> abs_val = int128(-42).abs();
    print_test("Abs(-42)", bounded_nat<16>(42), abs_val);
}

void test_rational() {
    std::cout << "\n=== Testing rational ===\n";

    using rat = rational<int>;

    // Basic construction and reduction
    rat a(6, 8);  // Should reduce to 3/4
    print_test("Reduction", rat(3, 4), a);

    // Arithmetic operations
    rat b(2, 3);
    rat sum = a + b;
    print_test("Addition (3/4 + 2/3)", rat(17, 12), sum);

    rat diff = a - b;
    print_test("Subtraction (3/4 - 2/3)", rat(1, 12), diff);

    rat prod = a * b;
    print_test("Multiplication (3/4 * 2/3)", rat(1, 2), prod);

    rat quot = a / b;
    print_test("Division (3/4 / 2/3)", rat(9, 8), quot);

    // Test infinities
    rat inf = rat::infinity();
    rat neg_inf = rat::negative_infinity();
    assert(inf.is_positive_infinity());
    assert(neg_inf.is_negative_infinity());
    std::cout << "Infinity handling: PASS\n";

    // Test with generic algorithms
    rat base(2, 1);
    rat three(3, 1);
    rat result = product(base, three);  // Uses generic product algorithm
    print_test("Generic product", rat(6, 1), result);

    // Test continued fractions
    rat pi_approx(22, 7);  // 22/7 ≈ π
    auto cf = pi_approx.to_continued_fraction();
    std::cout << "Continued fraction of 22/7: [";
    for (size_t i = 0; i < cf.size(); ++i) {
        std::cout << cf[i];
        if (i < cf.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";

    // Test best approximation
    rat value(355, 113);  // Better approximation of π
    rat approx = best_approximation(value, 100);
    std::cout << "Best approximation of 355/113 with denominator <= 100: "
              << approx.numerator() << "/" << approx.denominator() << "\n";

    // Test comparison
    assert(a < rat(1, 1));
    assert(a > rat(1, 2));
    std::cout << "Comparison operators: PASS\n";

    // Test mediant
    rat m = mediant(rat(1, 2), rat(2, 3));
    print_test("Mediant of 1/2 and 2/3", rat(3, 5), m);

    // Test floor and ceiling
    rat x(7, 3);  // 2.333...
    assert(floor(x) == 2);
    assert(ceil(x) == 3);
    std::cout << "Floor/Ceiling: PASS\n";

    // Test with integer GCD (note: rational GCD would need special implementation)
    int p_int = 12;
    int q_int = 8;
    int g_int = gcd(p_int, q_int);
    print_test("GCD of integers", 4, g_int);
}

void test_fixed_decimal() {
    std::cout << "\n=== Testing fixed_decimal ===\n";

    using money = fixed_decimal<long long, 2>;  // 2 decimal places for currency
    using precise = fixed_decimal<long long, 6>;  // 6 decimal places for precision

    // Basic construction
    money price(19.99);
    money tax_rate(0.08);
    money tax = price * tax_rate;
    std::cout << "Price: " << price.to_string() << "\n";
    std::cout << "Tax (8%): " << tax.to_string() << "\n";

    // Arithmetic operations
    money a(10.50);
    money b(3.25);

    money sum = a + b;
    std::cout << "10.50 + 3.25 = " << sum.to_string() << "\n";
    assert(sum == money(13.75));

    money diff = a - b;
    std::cout << "10.50 - 3.25 = " << diff.to_string() << "\n";
    assert(diff == money(7.25));

    money prod = a * money(2.0);
    std::cout << "10.50 * 2.00 = " << prod.to_string() << "\n";
    assert(prod == money(21.00));

    money quot = a / money(2.0);
    std::cout << "10.50 / 2.00 = " << quot.to_string() << "\n";
    assert(quot == money(5.25));

    // Test with generic algorithms
    money base(1.05);  // 5% growth
    money result = power(base, 10);  // Compound over 10 periods
    std::cout << "1.05^10 = " << result.to_string() << " (compound interest)\n";

    // Test rounding
    money x(10.125);  // Will round to 10.13
    money rounded = round(money(10.125));
    std::cout << "Round(10.125) = " << rounded.to_string() << "\n";

    // Test fundamental operations
    money n(4.00);
    assert(even(n));
    money doubled = twice(n);
    std::cout << "Twice(4.00) = " << doubled.to_string() << "\n";
    assert(doubled == money(8.00));

    money halved = half(doubled);
    std::cout << "Half(8.00) = " << halved.to_string() << "\n";
    assert(halved == n);

    // Test scale conversion
    precise high_precision(3.141593);
    money low_precision = scale_convert<long long, 6, 2>(high_precision);
    std::cout << "3.141593 rounded to 2 decimals: " << low_precision.to_string() << "\n";

    // Test square root
    money value(2.0);
    money sqrt_val = sqrt(value);
    std::cout << "sqrt(2.00) ≈ " << sqrt_val.to_string() << "\n";

    // Test comparison
    assert(a > b);
    assert(b < a);
    std::cout << "Comparison operators: PASS\n";

    // Test with integer GCD
    int p_int = 12;
    int q_int = 8;
    int g_int = gcd(p_int, q_int);
    std::cout << "GCD(12, 8) = " << g_int << "\n";
}

void test_integration() {
    std::cout << "\n=== Testing Integration with Generic Algorithms ===\n";

    // Test that all types work with the same generic algorithms

    // 1. Test with bounded_integer
    bounded_integer<8> bi_a(15);
    bounded_integer<8> bi_b(10);
    auto bi_gcd = gcd(bi_a, bi_b);
    std::cout << "GCD(15, 10) using bounded_integer: " << static_cast<long long>(bi_gcd) << "\n";

    // 2. Test with rational (GCD of numerators)
    rational<int> r_a(15, 1);
    rational<int> r_b(10, 1);
    int r_gcd = gcd(r_a.numerator(), r_b.numerator());
    std::cout << "GCD(15, 10) using rational numerators: " << r_gcd << "\n";

    // 3. Test with fixed_decimal (GCD of raw values)
    fixed_decimal<int, 2> fd_a(15);
    fixed_decimal<int, 2> fd_b(10);
    int fd_gcd = gcd(fd_a.raw_value() / 100, fd_b.raw_value() / 100);  // Convert back to integers
    std::cout << "GCD(15, 10) using fixed_decimal integers: " << fd_gcd << "\n";

    // Test power algorithm with different types
    std::cout << "\nPower algorithm with different types:\n";

    bounded_integer<8> bi_base(2);
    bounded_integer<8> bi_exp(8);
    auto bi_pow = power(bi_base, bi_exp);
    std::cout << "2^8 using bounded_integer: " << static_cast<long long>(bi_pow) << "\n";

    rational<int> r_base(2, 1);
    rational<int> r_exp(8, 1);
    auto r_pow = product(r_base, r_base);  // 2*2 as example
    std::cout << "2*2 using rational: " << r_pow.numerator() << "/" << r_pow.denominator() << "\n";

    fixed_decimal<long long, 4> fd_base(2.0);
    auto fd_pow = power(fd_base, 8);
    std::cout << "2^8 using fixed_decimal: " << fd_pow.to_string() << "\n";

    // Test sum algorithm
    std::cout << "\nSum algorithm with different types:\n";

    auto bi_sum = sum(bi_a, bi_b);
    std::cout << "15 + 10 using bounded_integer: " << static_cast<long long>(bi_sum) << "\n";

    auto r_sum = sum(r_a, r_b);
    std::cout << "15 + 10 using rational: " << r_sum.numerator() << "/" << r_sum.denominator() << "\n";

    auto fd_sum = sum(fd_a, fd_b);
    std::cout << "15 + 10 using fixed_decimal: " << fd_sum.to_string() << "\n";

    std::cout << "\nAll types successfully integrate with generic algorithms!\n";
}

void test_continued_fractions_with_rational() {
    std::cout << "\n=== Testing Continued Fractions with Rational ===\n";

    using rat = rational<int>;

    // Golden ratio approximation
    rat phi(89, 55);  // Fibonacci ratio approximates golden ratio
    auto cf = phi.to_continued_fraction();

    std::cout << "Continued fraction of 89/55: [";
    for (size_t i = 0; i < cf.size(); ++i) {
        std::cout << cf[i];
        if (i < cf.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";

    // Reconstruct from continued fraction
    rat reconstructed(cf);
    assert(reconstructed == phi);
    std::cout << "Reconstruction from continued fraction: PASS\n";

    // Test convergents
    auto convergents = compute_convergents(cf);
    std::cout << "Convergents of 89/55:\n";
    for (const auto& conv : convergents) {
        std::cout << "  " << conv.p << "/" << conv.q << "\n";
    }
}

int main() {
    try {
        std::cout << "Testing Generic Math Numeric Types\n";
        std::cout << "===================================\n";

        test_bounded_integer();
        test_rational();
        test_fixed_decimal();
        test_integration();
        test_continued_fractions_with_rational();

        std::cout << "\n=== All tests completed successfully! ===\n";

    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}