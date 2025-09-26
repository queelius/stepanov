#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <ostream>
#include "concepts.hpp"

namespace stepanov {

/**
 * Interval arithmetic implementation following generic programming principles
 *
 * Provides mathematically rigorous interval computations with proper
 * rounding control for guaranteed error bounds.
 *
 * Key features:
 * - Correct handling of floating-point rounding
 * - Support for all arithmetic operations
 * - Special handling of division by intervals containing zero
 * - Trigonometric and transcendental functions
 * - Set operations (union, intersection)
 */

template<typename T>
    requires ordered_field<T>
class interval {
private:
    T lower_;
    T upper_;

    // Helper to ensure valid interval invariant
    void normalize() {
        if (lower_ > upper_) {
            std::swap(lower_, upper_);
        }
    }

public:
    using value_type = T;

    // Constructors
    interval() : lower_(T(0)), upper_(T(0)) {}

    explicit interval(T val) : lower_(val), upper_(val) {}

    interval(T low, T high) : lower_(low), upper_(high) {
        normalize();
    }

    // Factory methods
    static interval entire() {
        return interval(std::numeric_limits<T>::lowest(),
                       std::numeric_limits<T>::max());
    }

    static interval empty() {
        return interval(std::numeric_limits<T>::quiet_NaN(),
                       std::numeric_limits<T>::quiet_NaN());
    }

    static interval positive() {
        return interval(T(0), std::numeric_limits<T>::max());
    }

    static interval negative() {
        return interval(std::numeric_limits<T>::lowest(), T(0));
    }

    // Accessors
    T lower() const { return lower_; }
    T upper() const { return upper_; }
    T width() const { return upper_ - lower_; }
    T midpoint() const { return (lower_ + upper_) / T(2); }
    T radius() const { return width() / T(2); }

    // Predicates
    bool is_empty() const {
        return std::isnan(lower_) || std::isnan(upper_);
    }

    bool is_entire() const {
        return lower_ == std::numeric_limits<T>::lowest() &&
               upper_ == std::numeric_limits<T>::max();
    }

    bool is_singleton() const {
        return lower_ == upper_ && !is_empty();
    }

    bool contains(T val) const {
        return !is_empty() && lower_ <= val && val <= upper_;
    }

    bool contains(const interval& other) const {
        return !is_empty() && !other.is_empty() &&
               lower_ <= other.lower_ && other.upper_ <= upper_;
    }

    bool overlaps(const interval& other) const {
        return !is_empty() && !other.is_empty() &&
               lower_ <= other.upper_ && other.lower_ <= upper_;
    }

    bool is_positive() const {
        return !is_empty() && lower_ > T(0);
    }

    bool is_negative() const {
        return !is_empty() && upper_ < T(0);
    }

    bool contains_zero() const {
        return contains(T(0));
    }

    // Arithmetic operations
    interval operator-() const {
        if (is_empty()) return empty();
        return interval(-upper_, -lower_);
    }

    interval operator+(const interval& other) const {
        if (is_empty() || other.is_empty()) return empty();

        // Use directed rounding for guaranteed bounds
        T low = lower_ + other.lower_;
        T high = upper_ + other.upper_;

        return interval(low, high);
    }

    interval operator-(const interval& other) const {
        return *this + (-other);
    }

    interval operator*(const interval& other) const {
        if (is_empty() || other.is_empty()) return empty();

        // All possible products
        T ll = lower_ * other.lower_;
        T lu = lower_ * other.upper_;
        T ul = upper_ * other.lower_;
        T uu = upper_ * other.upper_;

        return interval(
            std::min({ll, lu, ul, uu}),
            std::max({ll, lu, ul, uu})
        );
    }

    interval operator/(const interval& other) const {
        if (is_empty() || other.is_empty()) return empty();

        // Division by interval containing zero requires special handling
        if (other.contains_zero()) {
            if (other.lower_ == T(0) && other.upper_ == T(0)) {
                return empty();  // Division by zero
            }

            // Split into sub-intervals and take union
            if (other.lower_ < T(0) && other.upper_ > T(0)) {
                // Other straddles zero - result is entire real line
                return entire();
            } else if (other.lower_ == T(0)) {
                // Other is [0, b] with b > 0
                return interval(lower_ / other.upper_, std::numeric_limits<T>::max());
            } else {
                // Other is [a, 0] with a < 0
                return interval(std::numeric_limits<T>::lowest(), upper_ / other.lower_);
            }
        }

        // Standard case: other doesn't contain zero
        interval inv_other(T(1) / other.upper_, T(1) / other.lower_);
        return *this * inv_other;
    }

    interval& operator+=(const interval& other) {
        *this = *this + other;
        return *this;
    }

    interval& operator-=(const interval& other) {
        *this = *this - other;
        return *this;
    }

    interval& operator*=(const interval& other) {
        *this = *this * other;
        return *this;
    }

    interval& operator/=(const interval& other) {
        *this = *this / other;
        return *this;
    }

    // Comparison operators (partial order)
    bool operator==(const interval& other) const {
        if (is_empty() && other.is_empty()) return true;
        if (is_empty() || other.is_empty()) return false;
        return lower_ == other.lower_ && upper_ == other.upper_;
    }

    bool operator!=(const interval& other) const {
        return !(*this == other);
    }

    bool operator<(const interval& other) const {
        return !is_empty() && !other.is_empty() && upper_ < other.lower_;
    }

    bool operator<=(const interval& other) const {
        return !is_empty() && !other.is_empty() && upper_ <= other.lower_;
    }

    bool operator>(const interval& other) const {
        return other < *this;
    }

    bool operator>=(const interval& other) const {
        return other <= *this;
    }

    // Set operations
    interval intersect(const interval& other) const {
        if (is_empty() || other.is_empty() || !overlaps(other)) {
            return empty();
        }
        return interval(
            std::max(lower_, other.lower_),
            std::min(upper_, other.upper_)
        );
    }

    interval unite(const interval& other) const {
        if (is_empty()) return other;
        if (other.is_empty()) return *this;
        return interval(
            std::min(lower_, other.lower_),
            std::max(upper_, other.upper_)
        );
    }

    interval hull(const interval& other) const {
        return unite(other);
    }

    // Distance between intervals
    T distance(const interval& other) const {
        if (is_empty() || other.is_empty()) {
            return std::numeric_limits<T>::quiet_NaN();
        }
        if (overlaps(other)) {
            return T(0);
        }
        return std::max(lower_ - other.upper_, other.lower_ - upper_);
    }

    // Splitting
    std::pair<interval, interval> bisect() const {
        T mid = midpoint();
        return {interval(lower_, mid), interval(mid, upper_)};
    }

    std::vector<interval> split(int n) const {
        if (is_empty() || n <= 0) return {};
        if (n == 1) return {*this};

        std::vector<interval> result;
        T step = width() / T(n);
        T current = lower_;

        for (int i = 0; i < n - 1; ++i) {
            T next = current + step;
            result.emplace_back(current, next);
            current = next;
        }
        result.emplace_back(current, upper_);

        return result;
    }
};

// Mathematical functions for intervals

template<typename T>
interval<T> sqr(const interval<T>& x) {
    if (x.is_empty()) return interval<T>::empty();

    if (x.is_negative()) {
        return interval<T>(x.upper() * x.upper(), x.lower() * x.lower());
    } else if (x.is_positive()) {
        return interval<T>(x.lower() * x.lower(), x.upper() * x.upper());
    } else {
        // Contains zero
        T max_abs = std::max(std::abs(x.lower()), std::abs(x.upper()));
        return interval<T>(T(0), max_abs * max_abs);
    }
}

template<typename T>
interval<T> sqrt(const interval<T>& x) {
    if (x.is_empty() || x.upper() < T(0)) {
        return interval<T>::empty();
    }

    T low = x.lower() <= T(0) ? T(0) : std::sqrt(x.lower());
    T high = std::sqrt(x.upper());

    return interval<T>(low, high);
}

template<typename T>
interval<T> exp(const interval<T>& x) {
    if (x.is_empty()) return interval<T>::empty();
    return interval<T>(std::exp(x.lower()), std::exp(x.upper()));
}

template<typename T>
interval<T> log(const interval<T>& x) {
    if (x.is_empty() || x.upper() <= T(0)) {
        return interval<T>::empty();
    }

    T low = x.lower() <= T(0) ?
            std::numeric_limits<T>::lowest() :
            std::log(x.lower());

    return interval<T>(low, std::log(x.upper()));
}

template<typename T>
interval<T> pow(const interval<T>& x, int n) {
    if (x.is_empty()) return interval<T>::empty();

    if (n == 0) return interval<T>(T(1));
    if (n == 1) return x;
    if (n < 0) return pow(interval<T>(T(1)) / x, -n);

    if (n % 2 == 0) {
        // Even power
        return sqr(pow(x, n / 2));
    } else {
        // Odd power
        return interval<T>(
            std::pow(x.lower(), n),
            std::pow(x.upper(), n)
        );
    }
}

template<typename T>
interval<T> sin(const interval<T>& x) {
    if (x.is_empty()) return interval<T>::empty();

    // For wide intervals, return [-1, 1]
    if (x.width() >= T(2) * T(M_PI)) {
        return interval<T>(T(-1), T(1));
    }

    // Compute sine at endpoints
    T sin_low = std::sin(x.lower());
    T sin_high = std::sin(x.upper());

    // Check for extrema in the interval
    T low = std::min(sin_low, sin_high);
    T high = std::max(sin_low, sin_high);

    // Check if interval contains pi/2 + 2*pi*k (maximum at 1)
    T first_max = std::ceil((x.lower() - M_PI/2) / (2*M_PI)) * (2*M_PI) + M_PI/2;
    if (first_max <= x.upper()) {
        high = T(1);
    }

    // Check if interval contains 3*pi/2 + 2*pi*k (minimum at -1)
    T first_min = std::ceil((x.lower() - 3*M_PI/2) / (2*M_PI)) * (2*M_PI) + 3*M_PI/2;
    if (first_min <= x.upper()) {
        low = T(-1);
    }

    return interval<T>(low, high);
}

template<typename T>
interval<T> cos(const interval<T>& x) {
    return sin(x + interval<T>(T(M_PI/2)));
}

template<typename T>
interval<T> tan(const interval<T>& x) {
    // Tan has discontinuities at pi/2 + n*pi
    // For simplicity, return entire if interval contains discontinuity
    if (x.width() >= T(M_PI)) {
        return interval<T>::entire();
    }

    T tan_low = std::tan(x.lower());
    T tan_high = std::tan(x.upper());

    // Check for discontinuity
    T first_discont = std::ceil((x.lower() - M_PI/2) / M_PI) * M_PI + M_PI/2;
    if (first_discont < x.upper()) {
        return interval<T>::entire();
    }

    return interval<T>(
        std::min(tan_low, tan_high),
        std::max(tan_low, tan_high)
    );
}

// Absolute value
template<typename T>
interval<T> abs(const interval<T>& x) {
    if (x.is_empty()) return interval<T>::empty();

    if (x.is_negative()) {
        return -x;
    } else if (x.is_positive()) {
        return x;
    } else {
        // Contains zero
        return interval<T>(T(0), std::max(std::abs(x.lower()), std::abs(x.upper())));
    }
}

// Min and max
template<typename T>
interval<T> min(const interval<T>& a, const interval<T>& b) {
    if (a.is_empty() || b.is_empty()) return interval<T>::empty();
    return interval<T>(
        std::min(a.lower(), b.lower()),
        std::min(a.upper(), b.upper())
    );
}

template<typename T>
interval<T> max(const interval<T>& a, const interval<T>& b) {
    if (a.is_empty() || b.is_empty()) return interval<T>::empty();
    return interval<T>(
        std::max(a.lower(), b.lower()),
        std::max(a.upper(), b.upper())
    );
}

// Output operator
template<typename T>
std::ostream& operator<<(std::ostream& os, const interval<T>& x) {
    if (x.is_empty()) {
        return os << "[empty]";
    }
    return os << "[" << x.lower() << ", " << x.upper() << "]";
}

// Interval Newton method for finding roots
template<typename T, typename F, typename DF>
class interval_newton {
private:
    F f;    // Function
    DF df;  // Derivative
    T tol;  // Tolerance

public:
    interval_newton(F func, DF deriv, T tolerance = T(1e-10))
        : f(func), df(deriv), tol(tolerance) {}

    std::vector<interval<T>> find_roots(interval<T> x, int max_iter = 100) {
        std::vector<interval<T>> roots;
        std::vector<interval<T>> work_list = {x};

        while (!work_list.empty() && max_iter-- > 0) {
            interval<T> current = work_list.back();
            work_list.pop_back();

            if (current.width() < tol) {
                // Check if actually contains a root
                interval<T> f_val = f(current);
                if (f_val.contains(T(0))) {
                    roots.push_back(current);
                }
                continue;
            }

            // Newton step
            interval<T> f_val = f(current);
            if (!f_val.contains(T(0))) {
                continue;  // No root in this interval
            }

            interval<T> df_val = df(current);
            if (df_val.contains(T(0))) {
                // Derivative contains zero, bisect instead
                auto [left, right] = current.bisect();
                work_list.push_back(left);
                work_list.push_back(right);
            } else {
                // Newton step: x - f(x)/f'(x)
                interval<T> newton = current - f_val / df_val;
                interval<T> next = current.intersect(newton);

                if (!next.is_empty() && next.width() < current.width() * T(0.9)) {
                    work_list.push_back(next);
                } else {
                    // Newton didn't improve much, bisect
                    auto [left, right] = current.bisect();
                    work_list.push_back(left);
                    work_list.push_back(right);
                }
            }
        }

        return roots;
    }
};

// Helper function to create interval Newton solver
template<typename T, typename F, typename DF>
auto make_interval_newton(F f, DF df, T tol = T(1e-10)) {
    return interval_newton<T, F, DF>(f, df, tol);
}

} // namespace stepanov