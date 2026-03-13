/**
 * @file basic.cpp
 * @brief Basic polynomial operations demonstration
 */

#include <polynomials/polynomial.hpp>
#include <iostream>
#include <iomanip>

using namespace poly;

// Helper to print a polynomial
void print(const char* name, const polynomial<double>& p) {
    std::cout << name << " = ";
    if (p.is_zero()) {
        std::cout << "0";
    } else {
        bool first = true;
        for (int d = p.degree(); d >= 0; --d) {
            double c = p[d];
            if (c == 0.0) continue;

            if (!first && c > 0) std::cout << " + ";
            if (!first && c < 0) { std::cout << " - "; c = -c; }

            if (d == 0 || c != 1.0) std::cout << c;
            if (d > 0) std::cout << "x";
            if (d > 1) std::cout << "^" << d;

            first = false;
        }
    }
    std::cout << "\n";
}

int main() {
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "=== Polynomial Arithmetic ===\n\n";

    // Creating polynomials
    polynomial<double> p{1.0, -2.0, 1.0};  // 1 - 2x + x^2 = (x-1)^2
    polynomial<double> q{-1.0, 1.0};       // -1 + x = (x-1)

    print("p", p);
    print("q", q);

    // Arithmetic
    std::cout << "\nArithmetic:\n";
    print("p + q", p + q);
    print("p - q", p - q);
    print("p * q", p * q);

    // Division
    std::cout << "\nDivision:\n";
    auto [quot, rem] = divmod(p, q);
    print("p / q (quotient)", quot);
    print("p % q (remainder)", rem);

    // Verify: p = q * quot + rem
    auto reconstructed = q * quot + rem;
    std::cout << "Verification: p == q*quot + rem? "
              << (reconstructed == p ? "yes" : "no") << "\n";

    // Evaluation
    std::cout << "\n=== Evaluation ===\n\n";

    std::cout << "Evaluating p at various points:\n";
    for (double x : {0.0, 1.0, 2.0, 3.0}) {
        std::cout << "  p(" << x << ") = " << evaluate(p, x) << "\n";
    }

    // Calculus
    std::cout << "\n=== Calculus ===\n\n";

    polynomial<double> f{1.0, -3.0, 3.0, -1.0};  // (x-1)^3
    print("f", f);
    print("f'", derivative(f));
    print("f''", derivative(derivative(f)));

    auto F = antiderivative(f);
    print("integral(f)", F);

    double integral_01 = definite_integral(f, 0.0, 1.0);
    std::cout << "integral from 0 to 1: " << integral_01 << "\n";

    // Root finding
    std::cout << "\n=== Root Finding ===\n\n";

    polynomial<double> g{-6.0, 1.0, 1.0};  // x^2 + x - 6 = (x-2)(x+3)
    print("g", g);

    auto roots = find_roots(g, -10.0, 10.0);
    std::cout << "Roots of g:\n";
    for (double r : roots) {
        std::cout << "  x = " << r << " (verification: g(" << r << ") = "
                  << evaluate(g, r) << ")\n";
    }

    // Building from factors
    std::cout << "\n=== Building from Factors ===\n\n";

    auto x = polynomial<double>::x();
    auto factor1 = x - polynomial<double>{2.0};   // (x - 2)
    auto factor2 = x + polynomial<double>{3.0};   // (x + 3)
    auto product = factor1 * factor2;

    print("(x-2)", factor1);
    print("(x+3)", factor2);
    print("(x-2)(x+3)", product);

    return 0;
}
