/**
 * @file logarithmic.hpp
 * @brief Logarithmic Transform - The Canonical Computational Basis Transform
 * @author CBT Framework Contributors
 * @date 2024
 * 
 * @details
 * The logarithmic transform is the canonical example of a Computational Basis Transform (CBT).
 * It maps multiplication in the original domain to addition in the transformed domain,
 * fundamentally changing the computational complexity of certain operations.
 * 
 * Mathematical Foundation:
 * - Transform: φ(x) = log(x)
 * - Inverse: φ⁻¹(y) = exp(y)
 * - Homomorphism: φ(a × b) = φ(a) + φ(b)
 * 
 * @see tropical.hpp for related log-domain algebra
 */

#pragma once
#include <cmath>
#include <limits>
#include <iostream>

/// @namespace mutatio
/// @brief Core namespace for Computational Basis Transforms
namespace mutatio {

/**
 * @class lg
 * @brief Logarithmic transform implementation
 * @tparam T Underlying floating-point type (float, double, long double)
 * 
 * @details
 * The lg class implements a logarithmic computational basis where:
 * - Multiplication becomes addition: lg(a) * lg(b) = lg(a×b)
 * - Division becomes subtraction: lg(a) / lg(b) = lg(a÷b)  
 * - Exponentiation becomes multiplication: lg(a)^n = lg(a^n)
 * 
 * <b>Trade-offs:</b>
 * - Gains:
 *   - O(n²) multiplication → O(n) addition
 *   - Extended range: e^(-∞) to e^(+∞) without overflow
 *   - Accumulated rounding error reduction
 * - Losses:
 *   - No direct addition operation
 *   - Domain restricted to positive reals
 * 
 * <b>Example Usage:</b>
 * @code
 * cbt::lgd a(1000), b(2000);
 * auto product = a * b;        // Internally: log(1000) + log(2000)
 * auto huge = lgd::from_log(800); // Represents e^800 without overflow
 * @endcode
 */
template<typename T>
class lg {
    static_assert(std::is_floating_point_v<T>, "lg requires floating-point type");
    
private:
    T log_value_;
    
public:
    /// @brief Default constructor - represents zero
    constexpr lg() : log_value_(-std::numeric_limits<T>::infinity()) {}
    
    /// @brief Construct from a real value
    /// @param value The positive real value to transform
    /// @note Values ≤ 0 map to -∞ in log domain
    explicit constexpr lg(T value) 
        : log_value_(value > 0 ? std::log(value) : -std::numeric_limits<T>::infinity()) {}
    
    /// @brief Factory method to create directly from log value
    /// @param log_val The logarithmic value
    /// @return lg instance with the specified log value
    /// @note Useful for avoiding overflow when working with extreme values
    static constexpr lg from_log(T log_val) {
        lg result;
        result.log_value_ = log_val;
        return result;
    }
    
    /// @brief Convert back to real domain
    /// @return The exponential of the internal log value
    /// @warning May overflow for large log values
    constexpr T value() const {
        return std::exp(log_value_);
    }
    
    /// @brief Get the internal log representation
    /// @return The logarithmic value
    /// @note Safe to use even for values that would overflow in real domain
    constexpr T log() const {
        return log_value_;
    }
    
    // Arithmetic (multiplication in original domain)
    constexpr lg operator*(const lg& other) const {
        return from_log(log_value_ + other.log_value_);
    }
    
    constexpr lg operator/(const lg& other) const {
        return from_log(log_value_ - other.log_value_);
    }
    
    constexpr lg pow(T exponent) const {
        return from_log(log_value_ * exponent);
    }
    
    constexpr lg sqrt() const {
        return from_log(log_value_ / 2);
    }
    
    // Comparison
    constexpr bool operator==(const lg& other) const {
        return log_value_ == other.log_value_;
    }
    
    constexpr bool operator<(const lg& other) const {
        return log_value_ < other.log_value_;
    }
    
    constexpr bool operator<=(const lg& other) const {
        return log_value_ <= other.log_value_;
    }
    
    constexpr bool operator>(const lg& other) const {
        return log_value_ > other.log_value_;
    }
    
    constexpr bool operator>=(const lg& other) const {
        return log_value_ >= other.log_value_;
    }
    
    // Output
    friend std::ostream& operator<<(std::ostream& os, const lg& val) {
        return os << "lg(" << val.value() << ")";
    }
};

/// @typedef lgf
/// @brief Single-precision logarithmic transform
using lgf = lg<float>;

/// @typedef lgd  
/// @brief Double-precision logarithmic transform
using lgd = lg<double>;

} // namespace mutatio