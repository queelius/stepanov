#pragma once

#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include "concepts.hpp"
#include "gcd.hpp"

namespace stepanov {

/**
 * p-adic Numbers
 *
 * A different notion of "closeness" based on divisibility by prime p.
 * Two numbers are close if their difference is highly divisible by p.
 *
 * Key properties:
 * - Completes rationals differently than reals
 * - Every non-zero p-adic has multiplicative inverse
 * - Provides natural framework for modular arithmetic
 * - Hensel's lemma enables lifting solutions
 *
 * Trade-offs:
 * - Natural for number theory problems
 * - Finite precision in implementation
 * - No total ordering (unlike reals)
 */

template<typename T>
requires integral_domain<T>
class padic {
private:
    T prime;           // The prime p
    std::vector<T> digits;  // p-adic digits (least significant first)
    int valuation;     // p-adic valuation (power of p factored out)
    static constexpr size_t MAX_PRECISION = 32;

    // Normalize by removing trailing zeros and adjusting valuation
    void normalize() {
        while (!digits.empty() && digits.back() == T(0)) {
            digits.pop_back();
        }
        if (digits.empty()) {
            valuation = 0;
        }
    }

    // Ensure at least n digits
    void ensure_precision(size_t n) {
        while (digits.size() < n && digits.size() < MAX_PRECISION) {
            digits.push_back(T(0));
        }
    }

public:
    // Constructors
    padic(T p) : prime(p), valuation(0) {
        if (p < T(2)) throw std::invalid_argument("Prime must be >= 2");
    }

    padic(T p, T value) : prime(p), valuation(0) {
        if (p < T(2)) throw std::invalid_argument("Prime must be >= 2");

        // Convert integer to p-adic representation
        if (value == T(0)) {
            return;
        }

        // Factor out powers of p
        while (value != T(0) && remainder(value, prime) == T(0)) {
            value = quotient(value, prime);
            valuation++;
        }

        // Extract p-adic digits
        while (value != T(0) && digits.size() < MAX_PRECISION) {
            digits.push_back(remainder(value, prime));
            value = quotient(value, prime);
        }
    }

    // Factory method for rational p-adic
    static padic from_rational(T p, T num, T den) {
        if (den == T(0)) throw std::invalid_argument("Denominator cannot be zero");

        padic result(p);

        // Factor out powers of p from numerator and denominator
        while (remainder(num, p) == T(0)) {
            num = quotient(num, p);
            result.valuation++;
        }
        while (remainder(den, p) == T(0)) {
            den = quotient(den, p);
            result.valuation--;
        }

        // Use Hensel lifting to compute p-adic expansion
        // Find multiplicative inverse of den mod p^k
        T current = remainder(num, p);
        T den_inv = modular_inverse(den, p);

        for (size_t i = 0; i < MAX_PRECISION; ++i) {
            T digit = remainder(current * den_inv, p);
            result.digits.push_back(digit);

            current = current - digit * den;
            current = quotient(current, p);

            if (current == T(0)) break;
        }

        return result;
    }

    // Getters
    T get_prime() const { return prime; }
    int get_valuation() const { return valuation; }
    const std::vector<T>& get_digits() const { return digits; }

    // Convert to rational approximation (truncated)
    std::pair<T, T> to_rational(size_t precision = 10) const {
        if (digits.empty()) return {T(0), T(1)};

        T num = T(0);
        T den = T(1);

        // Reconstruct from p-adic digits
        for (size_t i = std::min(precision, digits.size()); i > 0; --i) {
            num = num * prime + digits[i - 1];
        }

        // Apply valuation
        if (valuation >= 0) {
            for (int i = 0; i < valuation; ++i) {
                num = num * prime;
            }
        } else {
            for (int i = 0; i < -valuation; ++i) {
                den = den * prime;
            }
        }

        return {num, den};
    }

    // Arithmetic operations
    padic operator+(const padic& other) const {
        if (prime != other.prime) {
            throw std::invalid_argument("Cannot add p-adics with different primes");
        }

        padic result(prime);
        result.valuation = std::min(valuation, other.valuation);

        // Align valuations
        int shift1 = valuation - result.valuation;
        int shift2 = other.valuation - result.valuation;

        T carry = T(0);
        size_t max_size = std::max(digits.size() + shift1, other.digits.size() + shift2);

        for (size_t i = 0; i < max_size && i < MAX_PRECISION; ++i) {
            T digit1 = (i >= static_cast<size_t>(shift1) && i - shift1 < digits.size())
                       ? digits[i - shift1] : T(0);
            T digit2 = (i >= static_cast<size_t>(shift2) && i - shift2 < other.digits.size())
                       ? other.digits[i - shift2] : T(0);

            T sum = digit1 + digit2 + carry;
            result.digits.push_back(remainder(sum, prime));
            carry = quotient(sum, prime);
        }

        result.normalize();
        return result;
    }

    padic operator-(const padic& other) const {
        return *this + (-other);
    }

    padic operator-() const {
        padic result(prime);
        result.valuation = valuation;

        // Compute p-adic negative
        T borrow = T(0);
        for (size_t i = 0; i < MAX_PRECISION; ++i) {
            T digit = (i < digits.size()) ? digits[i] : T(0);
            T neg_digit = (i == 0 && digit != T(0)) ? prime - digit : (prime - T(1)) - digit;
            neg_digit = neg_digit - borrow;

            if (neg_digit < T(0)) {
                neg_digit = neg_digit + prime;
                borrow = T(1);
            } else {
                borrow = T(0);
            }

            result.digits.push_back(neg_digit);
            if (i >= digits.size() && borrow == T(0)) break;
        }

        result.normalize();
        return result;
    }

    padic operator*(const padic& other) const {
        if (prime != other.prime) {
            throw std::invalid_argument("Cannot multiply p-adics with different primes");
        }

        padic result(prime);
        result.valuation = valuation + other.valuation;

        // Multiply digit by digit with carry
        std::vector<T> temp(MAX_PRECISION, T(0));

        for (size_t i = 0; i < digits.size() && i < MAX_PRECISION; ++i) {
            for (size_t j = 0; j < other.digits.size() && i + j < MAX_PRECISION; ++j) {
                T prod = digits[i] * other.digits[j];
                size_t k = i + j;

                while (prod != T(0) && k < MAX_PRECISION) {
                    temp[k] = temp[k] + prod;
                    prod = quotient(temp[k], prime);
                    temp[k] = remainder(temp[k], prime);
                    k++;
                }
            }
        }

        result.digits = temp;
        result.normalize();
        return result;
    }

    // p-adic norm (ultrametric)
    double norm() const {
        if (digits.empty()) return 0.0;
        return std::pow(static_cast<double>(prime), -static_cast<double>(valuation));
    }

    // p-adic distance (ultrametric)
    double distance(const padic& other) const {
        return (*this - other).norm();
    }

    // Check if unit (invertible)
    bool is_unit() const {
        return !digits.empty() && valuation == 0 && digits[0] != T(0);
    }

    // Multiplicative inverse (for units)
    padic inverse() const {
        if (!is_unit()) {
            throw std::domain_error("Cannot invert non-unit p-adic");
        }

        padic result(prime);
        result.valuation = -valuation;

        // Use Hensel lifting to compute inverse
        T a0 = digits[0];
        T inv = modular_inverse(a0, prime);

        result.digits.push_back(inv);

        // Lift solution iteratively
        for (size_t k = 1; k < MAX_PRECISION; ++k) {
            // Compute next digit of inverse
            T sum = T(0);
            for (size_t j = 1; j <= k && j < digits.size(); ++j) {
                sum = sum + digits[j] * result.digits[k - j];
            }
            T next_digit = remainder(-sum * inv, prime);
            result.digits.push_back(next_digit);
        }

        return result;
    }

    padic operator/(const padic& other) const {
        return *this * other.inverse();
    }

    // Equality (up to precision)
    bool operator==(const padic& other) const {
        if (prime != other.prime || valuation != other.valuation) return false;

        size_t max_size = std::max(digits.size(), other.digits.size());
        for (size_t i = 0; i < max_size; ++i) {
            T d1 = (i < digits.size()) ? digits[i] : T(0);
            T d2 = (i < other.digits.size()) ? other.digits[i] : T(0);
            if (d1 != d2) return false;
        }
        return true;
    }

    bool operator!=(const padic& other) const {
        return !(*this == other);
    }

private:
    // Compute modular inverse using extended GCD
    static T modular_inverse(T a, T m) {
        T g, x, y;
        extended_gcd(a, m, g, x, y);
        if (g != T(1)) {
            throw std::domain_error("Modular inverse does not exist");
        }
        return remainder(x + m, m);  // Ensure positive
    }

    // Extended GCD helper
    static void extended_gcd(T a, T b, T& g, T& x, T& y) {
        if (b == T(0)) {
            g = a;
            x = T(1);
            y = T(0);
            return;
        }
        T x1, y1;
        extended_gcd(b, remainder(a, b), g, x1, y1);
        x = y1;
        y = x1 - quotient(a, b) * y1;
    }
};

// Hensel's Lemma - lift solutions from mod p to mod p^k
template<typename T>
requires integral_domain<T>
struct hensel_lifting {
    // Lift a root of f(x) ≡ 0 (mod p) to f(x) ≡ 0 (mod p^k)
    static T lift_root(T root_mod_p, T p, size_t k,
                       std::function<T(T)> f,
                       std::function<T(T)> f_derivative) {
        T current = root_mod_p;
        T p_power = p;

        for (size_t i = 1; i < k; ++i) {
            T f_val = f(current);
            T f_prime = f_derivative(current);

            if (remainder(f_prime, p) == T(0)) {
                throw std::runtime_error("Cannot lift: derivative is zero mod p");
            }

            // Newton's method in p-adic setting
            T inv = modular_inverse_impl(f_prime, p);
            T correction = remainder(-f_val * inv, p_power * p);
            current = current + correction * p_power;

            p_power = p_power * p;
        }

        return current;
    }

private:
    static T modular_inverse_impl(T a, T m) {
        T g = gcd(a, m);
        if (g != T(1)) {
            throw std::domain_error("Modular inverse does not exist");
        }
        // Simplified inverse computation
        for (T x = T(1); x < m; x = x + T(1)) {
            if (remainder(a * x, m) == T(1)) {
                return x;
            }
        }
        return T(0);
    }
};

// Type aliases for common p-adic fields
using padic_2 = padic<int>;   // 2-adic numbers
using padic_3 = padic<int>;   // 3-adic numbers
using padic_5 = padic<int>;   // 5-adic numbers

// p-adic valuation function
template<typename T>
requires integral_domain<T>
int padic_valuation(T n, T p) {
    if (n == T(0)) return -1;  // Convention: v_p(0) = ∞

    int val = 0;
    while (remainder(n, p) == T(0)) {
        n = quotient(n, p);
        val++;
    }
    return val;
}

} // namespace stepanov