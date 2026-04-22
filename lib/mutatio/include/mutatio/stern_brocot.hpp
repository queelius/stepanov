/**
 * Stern-Brocot Transform - Exact Rational Arithmetic
 * 
 * Transform: ℝ → Tree position in Stern-Brocot tree
 * Representation: All positive rationals in lowest terms
 * 
 * Trade-off:
 *   Gain: Exact rational arithmetic, optimal approximations
 *   Loss: Irrational numbers require infinite representation
 * 
 * Applications:
 *   - Computer algebra systems
 *   - Music theory (frequency ratios)
 *   - Continued fractions
 *   - Farey sequences
 */

#pragma once
#include <numeric>
#include <vector>
#include <iostream>
#include <string>
#include <cmath>
#include <limits>
#include <type_traits>

namespace mutatio {

template<typename T>
class stern_brocot {
    static_assert(std::is_integral_v<T>, "stern_brocot requires integral type");
    
private:
    T num_, den_;
    
    void reduce() {
        T g = std::gcd(num_, den_);
        if (g > 1) {
            num_ /= g;
            den_ /= g;
        }
        if (den_ < 0) {
            num_ = -num_;
            den_ = -den_;
        }
    }
    
public:
    // Constructors
    stern_brocot() : num_(0), den_(1) {}
    
    stern_brocot(T n, T d) : num_(n), den_(d) {
        if (den_ == 0) throw std::invalid_argument("Denominator cannot be zero");
        reduce();
    }
    
    explicit stern_brocot(T n) : num_(n), den_(1) {}
    
    // Getters
    T numerator() const { return num_; }
    T denominator() const { return den_; }
    
    double to_double() const {
        return static_cast<double>(num_) / den_;
    }
    
    // Arithmetic
    stern_brocot operator+(const stern_brocot& other) const {
        return stern_brocot(num_ * other.den_ + other.num_ * den_,
                           den_ * other.den_);
    }
    
    stern_brocot operator-(const stern_brocot& other) const {
        return stern_brocot(num_ * other.den_ - other.num_ * den_,
                           den_ * other.den_);
    }
    
    stern_brocot operator*(const stern_brocot& other) const {
        return stern_brocot(num_ * other.num_, den_ * other.den_);
    }
    
    stern_brocot operator/(const stern_brocot& other) const {
        if (other.num_ == 0) throw std::invalid_argument("Division by zero");
        return stern_brocot(num_ * other.den_, den_ * other.num_);
    }
    
    // Mediant (fundamental Stern-Brocot operation)
    stern_brocot mediant(const stern_brocot& other) const {
        return stern_brocot(num_ + other.num_, den_ + other.den_);
    }
    
    // Find best rational approximation with denominator ≤ max_den
    static stern_brocot approximate(double x, T max_den) {
        if (max_den <= 0) {
            throw std::invalid_argument("max_den must be positive");
        }
        if (!std::isfinite(x)) {
            throw std::invalid_argument("Cannot approximate NaN or infinity");
        }
        
        if (x == 0.0) {
            return stern_brocot(0, 1);
        }
        
        bool negative = x < 0;
        long double value = std::abs(static_cast<long double>(x));
        const long double epsilon = std::numeric_limits<long double>::epsilon();
        
        // Continued fraction-based search through Stern-Brocot tree.
        T prev_num = 0;
        T prev_den = 1;
        T curr_num = 1;
        T curr_den = 0;
        long double cf_value = value;
        
        for (int iterations = 0; iterations < 256; ++iterations) {
            T a = static_cast<T>(std::floor(cf_value));
            T next_num = a * curr_num + prev_num;
            T next_den = a * curr_den + prev_den;
            
            if (next_den > max_den) {
                if (curr_den != 0) {
                    T t = (max_den - prev_den) / curr_den;
                    if (t > 0) {
                        T candidate_num = t * curr_num + prev_num;
                        T candidate_den = t * curr_den + prev_den;
                        
                        long double candidate_val = static_cast<long double>(candidate_num) / candidate_den;
                        long double current_val = static_cast<long double>(curr_num) / curr_den;
                        
                        auto candidate_error = std::abs(value - candidate_val);
                        auto current_error = std::abs(value - current_val);
                        
                        if (candidate_den > 0 && candidate_error < current_error) {
                            curr_num = candidate_num;
                            curr_den = candidate_den;
                        }
                    }
                }
                break;
            }
            
            prev_num = curr_num;
            prev_den = curr_den;
            curr_num = next_num;
            curr_den = next_den;
            
            long double remainder = cf_value - static_cast<long double>(a);
            if (std::abs(remainder) <= epsilon) {
                break;
            }
            
            cf_value = 1.0L / remainder;
            if (!std::isfinite(static_cast<double>(cf_value))) {
                break;
            }
        }
        
        if (curr_den == 0) {
            curr_den = 1;
            curr_num = 0;
        }
        
        auto result = stern_brocot(curr_num, curr_den);
        return negative ? stern_brocot(-result.num_, result.den_) : result;
    }
    
    // Continued fraction representation
    std::vector<T> to_continued_fraction() const {
        std::vector<T> cf;
        T n = num_, d = den_;
        
        while (d != 0) {
            T q = n / d;
            cf.push_back(q);
            T temp = d;
            d = n - q * d;
            n = temp;
        }
        
        return cf;
    }
    
    // Comparison
    bool operator==(const stern_brocot& other) const {
        return num_ * other.den_ == other.num_ * den_;
    }
    
    bool operator<(const stern_brocot& other) const {
        return num_ * other.den_ < other.num_ * den_;
    }
    
    // Output
    friend std::ostream& operator<<(std::ostream& os, const stern_brocot& r) {
        os << r.num_;
        if (r.den_ != 1) os << "/" << r.den_;
        return os;
    }
};

} // namespace mutatio
