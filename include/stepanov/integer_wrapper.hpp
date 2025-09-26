#pragma once

#include <concepts>
#include "concepts.hpp"

namespace stepanov {

/**
 * Wrapper for built-in integer types to make them properly model euclidean_domain
 * This ensures all required operations are found via ADL
 */
template<std::integral T>
class integer {
private:
    T value_;

public:
    using value_type = T;

    // Constructors
    integer() : value_(0) {}
    explicit integer(T val) : value_(val) {}

    // Conversion
    operator T() const { return value_; }
    T value() const { return value_; }

    // Arithmetic operators
    integer operator+(const integer& other) const {
        return integer(value_ + other.value_);
    }

    integer operator-(const integer& other) const {
        return integer(value_ - other.value_);
    }

    integer operator*(const integer& other) const {
        return integer(value_ * other.value_);
    }

    integer operator/(const integer& other) const {
        return integer(value_ / other.value_);
    }

    integer operator%(const integer& other) const {
        return integer(value_ % other.value_);
    }

    integer operator-() const {
        return integer(-value_);
    }

    // Comparison operators
    bool operator==(const integer& other) const {
        return value_ == other.value_;
    }

    bool operator!=(const integer& other) const {
        return value_ != other.value_;
    }

    bool operator<(const integer& other) const {
        return value_ < other.value_;
    }

    bool operator<=(const integer& other) const {
        return value_ <= other.value_;
    }

    bool operator>(const integer& other) const {
        return value_ > other.value_;
    }

    bool operator>=(const integer& other) const {
        return value_ >= other.value_;
    }

    // Compound assignments
    integer& operator+=(const integer& other) {
        value_ += other.value_;
        return *this;
    }

    integer& operator-=(const integer& other) {
        value_ -= other.value_;
        return *this;
    }

    integer& operator*=(const integer& other) {
        value_ *= other.value_;
        return *this;
    }

    integer& operator/=(const integer& other) {
        value_ /= other.value_;
        return *this;
    }

    integer& operator%=(const integer& other) {
        value_ %= other.value_;
        return *this;
    }

    // Friend functions for euclidean_domain concept
    friend integer quotient(const integer& a, const integer& b) {
        return integer(a.value_ / b.value_);
    }

    friend integer remainder(const integer& a, const integer& b) {
        return integer(a.value_ % b.value_);
    }

    friend T norm(const integer& a) {
        return a.value_ < 0 ? -a.value_ : a.value_;
    }

    // Operations for generic algorithms
    friend bool even(const integer& x) {
        return (x.value_ & 1) == 0;
    }

    friend integer twice(const integer& x) {
        return integer(x.value_ << 1);
    }

    friend integer half(const integer& x) {
        return integer(x.value_ >> 1);
    }

    friend integer increment(const integer& x) {
        return integer(x.value_ + 1);
    }

    friend integer decrement(const integer& x) {
        return integer(x.value_ - 1);
    }
};

// Type aliases for common integer types
using Int = integer<int>;
using Long = integer<long>;
using LongLong = integer<long long>;

// Output operator
template<std::integral T>
std::ostream& operator<<(std::ostream& os, const integer<T>& i) {
    return os << i.value();
}

} // namespace stepanov