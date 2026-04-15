/**
 * Tropical Algebra - Min-Plus and Max-Plus Semirings
 * 
 * Transform: (ℝ, +, ×) → (ℝ ∪ {∞}, min, +) or (ℝ ∪ {-∞}, max, +)
 * 
 * Trade-off:
 *   Gain: Linear algebra becomes piecewise linear, optimization problems simplify
 *   Loss: No additive inverse (not a ring)
 * 
 * Applications:
 *   - Shortest path algorithms
 *   - Task scheduling
 *   - Discrete event systems
 *   - Algebraic geometry
 */

#pragma once
#include <algorithm>
#include <iostream>
#include <limits>

namespace mutatio {

// Min-plus tropical semiring
template<typename T>
class tropical_min {
    static_assert(std::is_floating_point_v<T>, "tropical requires floating-point type");
    
private:
    T value_;
    
public:
    static constexpr T infinity() { 
        return std::numeric_limits<T>::infinity(); 
    }
    
    // Constructors
    tropical_min() : value_(infinity()) {}  // Additive identity
    explicit tropical_min(T value) : value_(value) {}
    
    // Factory methods
    static tropical_min zero() { return tropical_min(); }  // Tropical zero = ∞
    static tropical_min one() { return tropical_min(0); }   // Tropical one = 0
    
    T value() const { return value_; }
    bool is_infinite() const { return std::isinf(value_); }
    
    // Tropical addition = classical minimum
    tropical_min operator+(const tropical_min& other) const {
        return tropical_min(std::min(value_, other.value_));
    }
    
    // Tropical multiplication = classical addition
    tropical_min operator*(const tropical_min& other) const {
        if (is_infinite() || other.is_infinite()) {
            return tropical_min();  // ∞ + anything = ∞
        }
        return tropical_min(value_ + other.value_);
    }
    
    // Tropical power = classical multiplication
    tropical_min pow(T n) const {
        if (is_infinite()) return *this;
        return tropical_min(value_ * n);
    }
    
    // Comparison
    bool operator==(const tropical_min& other) const {
        return value_ == other.value_;
    }
    
    bool operator<(const tropical_min& other) const {
        return value_ < other.value_;
    }
    
    // Output
    friend std::ostream& operator<<(std::ostream& os, const tropical_min& t) {
        if (t.is_infinite()) {
            os << "∞";
        } else {
            os << t.value_;
        }
        return os;
    }
};

// Max-plus tropical semiring
template<typename T>
class tropical_max {
    static_assert(std::is_floating_point_v<T>, "tropical requires floating-point type");
    
private:
    T value_;
    
public:
    static constexpr T neg_infinity() { 
        return -std::numeric_limits<T>::infinity(); 
    }
    
    // Constructors
    tropical_max() : value_(neg_infinity()) {}  // Additive identity
    explicit tropical_max(T value) : value_(value) {}
    
    // Factory methods
    static tropical_max zero() { return tropical_max(); }  // Tropical zero = -∞
    static tropical_max one() { return tropical_max(0); }   // Tropical one = 0
    
    T value() const { return value_; }
    bool is_infinite() const { return std::isinf(value_); }
    
    // Tropical addition = classical maximum
    tropical_max operator+(const tropical_max& other) const {
        return tropical_max(std::max(value_, other.value_));
    }
    
    // Tropical multiplication = classical addition
    tropical_max operator*(const tropical_max& other) const {
        if (is_infinite() || other.is_infinite()) {
            return tropical_max();  // -∞ + anything = -∞
        }
        return tropical_max(value_ + other.value_);
    }
    
    // Tropical power = classical multiplication
    tropical_max pow(T n) const {
        if (is_infinite()) return *this;
        return tropical_max(value_ * n);
    }
    
    // Comparison
    bool operator==(const tropical_max& other) const {
        return value_ == other.value_;
    }
    
    bool operator<(const tropical_max& other) const {
        return value_ < other.value_;
    }
    
    // Output
    friend std::ostream& operator<<(std::ostream& os, const tropical_max& t) {
        if (t.is_infinite()) {
            os << "-∞";
        } else {
            os << t.value_;
        }
        return os;
    }
};

// Matrix operations in tropical algebra (for shortest paths)
template<typename T, size_t N>
class tropical_matrix {
    tropical_min<T> data_[N][N];
    
public:
    tropical_matrix() {
        // Initialize to tropical zero (infinity)
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                data_[i][j] = tropical_min<T>::zero();
            }
        }
    }
    
    // Set edge weight
    void set(size_t i, size_t j, T weight) {
        data_[i][j] = tropical_min<T>(weight);
    }
    
    // Get shortest path length
    T get(size_t i, size_t j) const {
        return data_[i][j].value();
    }
    
    // Tropical matrix multiplication (computes shortest paths)
    tropical_matrix operator*(const tropical_matrix& other) const {
        tropical_matrix result;
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                for (size_t k = 0; k < N; ++k) {
                    // min(result[i][j], data[i][k] + other[k][j])
                    result.data_[i][j] = result.data_[i][j] + 
                                        (data_[i][k] * other.data_[k][j]);
                }
            }
        }
        return result;
    }
};

// Type aliases
template<typename T>
using tmin = tropical_min<T>;

template<typename T>
using tmax = tropical_max<T>;

} // namespace mutatio