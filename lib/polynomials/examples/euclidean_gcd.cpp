/**
 * @file euclidean_gcd.cpp
 * @brief Demonstrating that GCD works the same for integers and polynomials
 *
 * THIS IS THE KEY INSIGHT of this library.
 */

#include <polynomials/polynomial.hpp>
#include <iostream>
#include <iomanip>

using namespace poly;

// GCD for integers (Euclidean algorithm)
int integer_gcd(int a, int b) {
    std::cout << "  gcd(" << a << ", " << b << ")\n";
    while (b != 0) {
        int r = a % b;
        std::cout << "    " << a << " = " << b << " * " << (a/b) << " + " << r << "\n";
        a = b;
        b = r;
    }
    return a;
}

void print_poly(const char* name, const polynomial<double>& p) {
    std::cout << name << " = ";
    if (p.is_zero()) {
        std::cout << "0";
    } else {
        bool first = true;
        for (int d = p.degree(); d >= 0; --d) {
            double c = p[d];
            if (std::abs(c) < 1e-10) continue;

            if (!first && c > 0) std::cout << " + ";
            if (!first && c < 0) { std::cout << " - "; c = -c; }

            if (d == 0 || std::abs(c - 1.0) > 1e-10) std::cout << c;
            if (d > 0) std::cout << "x";
            if (d > 1) std::cout << "^" << d;

            first = false;
        }
    }
    std::cout << "\n";
}

// Polynomial GCD with tracing
polynomial<double> polynomial_gcd_traced(polynomial<double> a, polynomial<double> b) {
    std::cout << "  gcd(";
    print_poly("a", a);
    std::cout << "      ";
    print_poly("b", b);
    std::cout << "  )\n";

    while (!b.is_zero()) {
        auto [q, r] = divmod(a, b);
        std::cout << "    deg(a)=" << a.degree() << ", deg(b)=" << b.degree()
                  << ", deg(r)=" << r.degree() << "\n";
        a = b;
        b = r;
    }

    // Normalize to monic
    if (!a.is_zero()) {
        a = a / a.leading_coefficient();
    }
    return a;
}

int main() {
    std::cout << std::fixed << std::setprecision(2);

    std::cout << "============================================================\n";
    std::cout << "THE EUCLIDEAN ALGORITHM: Same Structure, Different Types\n";
    std::cout << "============================================================\n\n";

    // Integer example
    std::cout << "1. INTEGERS: gcd(48, 18)\n";
    std::cout << "   Euclidean norm: |n|\n\n";

    int g_int = integer_gcd(48, 18);
    std::cout << "\n  Result: " << g_int << "\n\n";

    // Polynomial example
    std::cout << "2. POLYNOMIALS: gcd(x^3 - 1, x^2 - 1)\n";
    std::cout << "   Euclidean norm: degree(p)\n\n";

    // x^3 - 1 = (x-1)(x^2 + x + 1)
    // x^2 - 1 = (x-1)(x+1)
    // GCD should be (x-1)

    polynomial<double> p{-1.0, 0.0, 0.0, 1.0};  // x^3 - 1
    polynomial<double> q{-1.0, 0.0, 1.0};       // x^2 - 1

    auto g_poly = polynomial_gcd_traced(p, q);
    std::cout << "\n  Result: ";
    print_poly("gcd", g_poly);

    // The insight
    std::cout << "\n============================================================\n";
    std::cout << "THE INSIGHT\n";
    std::cout << "============================================================\n\n";

    std::cout << "Both algorithms have the same structure:\n\n";

    std::cout << "  while b != 0:\n";
    std::cout << "      r = a mod b\n";
    std::cout << "      a = b\n";
    std::cout << "      b = r\n";
    std::cout << "  return a\n\n";

    std::cout << "This works because both integers and polynomials are\n";
    std::cout << "EUCLIDEAN DOMAINS - algebraic structures where:\n\n";
    std::cout << "  1. There exists a norm function n(x)\n";
    std::cout << "  2. Division with remainder exists\n";
    std::cout << "  3. The remainder is 'smaller': n(r) < n(b)\n\n";

    std::cout << "| Structure   | Norm          | Termination Guarantee |\n";
    std::cout << "|-------------|---------------|----------------------|\n";
    std::cout << "| Integers    | |n|           | |a mod b| < |b|      |\n";
    std::cout << "| Polynomials | degree(p)     | deg(a mod b) < deg(b)|\n\n";

    // Extended GCD example
    std::cout << "============================================================\n";
    std::cout << "EXTENDED GCD: Bezout's Identity\n";
    std::cout << "============================================================\n\n";

    std::cout << "For any a, b, there exist s, t such that:\n";
    std::cout << "  gcd(a, b) = a*s + b*t\n\n";

    polynomial<double> a{-1.0, 0.0, 1.0};     // x^2 - 1
    polynomial<double> b{-2.0, 1.0, 1.0};     // x^2 + x - 2

    print_poly("a", a);
    print_poly("b", b);

    auto [g, s, t] = extended_gcd(a, b);

    std::cout << "\nExtended GCD result:\n";
    print_poly("gcd", g);
    print_poly("s", s);
    print_poly("t", t);

    // Verify
    auto check = a * s + b * t;
    std::cout << "\nVerification: a*s + b*t = ";
    print_poly("", check);

    std::cout << "\n============================================================\n";
    std::cout << "APPLICATIONS\n";
    std::cout << "============================================================\n\n";

    std::cout << "1. Finding common factors of polynomials\n";
    std::cout << "2. Testing if polynomials are coprime\n";
    std::cout << "3. Simplifying rational functions\n";
    std::cout << "4. Partial fraction decomposition\n";
    std::cout << "5. Error-correcting codes (over finite fields)\n";
    std::cout << "6. Cryptography (polynomial arithmetic over Galois fields)\n\n";

    return 0;
}
