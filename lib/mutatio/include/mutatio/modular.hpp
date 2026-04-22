/**
 * Modular Arithmetic - Cyclic Group Operations
 * 
 * Transform: ℤ → ℤ/nℤ
 * Operations wrap around modulo n
 * 
 * Trade-off:
 *   Gain: Bounded values, cryptographic properties
 *   Loss: Ordering loses meaning, division complex
 * 
 * Applications:
 *   - Cryptography (RSA, Diffie-Hellman)
 *   - Hash functions
 *   - Random number generation
 *   - Clock arithmetic
 */

#pragma once
#include <iostream>
#include <numeric>

namespace mutatio {

template<typename T, T Modulus>
class modular {
    static_assert(std::is_integral_v<T>, "modular requires integral type");
    static_assert(Modulus > 0, "Modulus must be positive");
    
private:
    T value_;
    
    static T mod(T x) {
        T result = x % Modulus;
        return result < 0 ? result + Modulus : result;
    }
    
    // Extended Euclidean algorithm for modular inverse
    static T extended_gcd(T a, T b, T& x, T& y) {
        if (b == 0) {
            x = 1;
            y = 0;
            return a;
        }
        T x1, y1;
        T gcd = extended_gcd(b, a % b, x1, y1);
        x = y1;
        y = x1 - (a / b) * y1;
        return gcd;
    }
    
public:
    // Constructors
    modular() : value_(0) {}
    explicit modular(T value) : value_(mod(value)) {}
    
    // Getters
    T value() const { return value_; }
    static constexpr T modulus() { return Modulus; }
    
    // Arithmetic operations
    modular operator+(const modular& other) const {
        return modular(value_ + other.value_);
    }
    
    modular operator-(const modular& other) const {
        return modular(value_ - other.value_);
    }
    
    modular operator*(const modular& other) const {
        return modular(static_cast<long long>(value_) * other.value_);
    }
    
    modular operator/(const modular& other) const {
        return *this * other.inverse();
    }
    
    modular operator-() const {
        return modular(-value_);
    }
    
    // Modular exponentiation (fast power)
    modular pow(T exponent) const {
        if (exponent < 0) {
            return inverse().pow(-exponent);
        }
        
        modular result(1);
        modular base = *this;
        
        while (exponent > 0) {
            if (exponent & 1) {
                result = result * base;
            }
            base = base * base;
            exponent >>= 1;
        }
        
        return result;
    }
    
    // Modular inverse (exists only if gcd(value, Modulus) = 1)
    modular inverse() const {
        T x, y;
        T gcd = extended_gcd(value_, Modulus, x, y);
        if (gcd != 1) {
            throw std::runtime_error("Modular inverse does not exist");
        }
        return modular(x);
    }
    
    // Check if invertible
    bool is_unit() const {
        return std::gcd(value_, Modulus) == 1;
    }
    
    // Comparison
    bool operator==(const modular& other) const {
        return value_ == other.value_;
    }
    
    bool operator!=(const modular& other) const {
        return value_ != other.value_;
    }
    
    // Note: < and > don't have mathematical meaning in modular arithmetic
    // but can be useful for data structures
    bool operator<(const modular& other) const {
        return value_ < other.value_;
    }
    
    // Output
    friend std::ostream& operator<<(std::ostream& os, const modular& m) {
        return os << m.value_ << " (mod " << Modulus << ")";
    }
};

// Common modular arithmetic types
using mod7 = modular<int, 7>;
using mod256 = modular<int, 256>;
using mod_prime = modular<int, 1000000007>;  // Common in competitive programming

// Dynamic modulus version
template<typename T>
class modular_dynamic {
    static_assert(std::is_integral_v<T>, "modular requires integral type");
    
private:
    T value_;
    T modulus_;
    
    T mod(T x) const {
        T result = x % modulus_;
        return result < 0 ? result + modulus_ : result;
    }
    
public:
    modular_dynamic(T value, T modulus) 
        : value_(0), modulus_(modulus) {
        if (modulus_ <= 0) {
            throw std::invalid_argument("Modulus must be positive");
        }
        value_ = mod(value);
    }
    
    T value() const { return value_; }
    T modulus() const { return modulus_; }
    
    modular_dynamic operator+(const modular_dynamic& other) const {
        if (modulus_ != other.modulus_) {
            throw std::invalid_argument("Moduli must match");
        }
        return modular_dynamic(value_ + other.value_, modulus_);
    }
    
    modular_dynamic operator*(const modular_dynamic& other) const {
        if (modulus_ != other.modulus_) {
            throw std::invalid_argument("Moduli must match");
        }
        return modular_dynamic(static_cast<long long>(value_) * other.value_, 
                              modulus_);
    }
    
    friend std::ostream& operator<<(std::ostream& os, const modular_dynamic& m) {
        return os << m.value_ << " (mod " << m.modulus_ << ")";
    }
};

} // namespace mutatio