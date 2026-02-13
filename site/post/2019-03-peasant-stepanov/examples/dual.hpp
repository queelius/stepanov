#pragma once

/**
 * @file dual.hpp
 * @brief Dual numbers - automatic differentiation falls out of algebra!
 *
 * A dual number is a + bε where ε² = 0 (nilpotent).
 *
 * The multiplication rule encodes the product rule of calculus:
 *   (a + bε)(c + dε) = ac + (ad + bc)ε
 *
 * Mind-blowing insight:
 *   If you compute power(dual{x, 1}, n), you get dual{x^n, n*x^(n-1)}
 *   The DERIVATIVE falls out automatically!
 *
 * This is forward-mode automatic differentiation via pure algebra.
 */

#include <compare>
#include <cmath>

namespace peasant::examples {

template<typename T = double>
struct dual {
    T val;    // f(x) - the value
    T deriv;  // f'(x) - the derivative

    constexpr dual() : val{0}, deriv{0} {}
    constexpr dual(T v) : val{v}, deriv{0} {}
    constexpr dual(T v, T d) : val{v}, deriv{d} {}

    constexpr bool operator==(dual const&) const = default;
    constexpr auto operator<=>(dual const&) const = default;

    constexpr dual operator+(dual o) const {
        return {val + o.val, deriv + o.deriv};
    }

    constexpr dual operator-() const {
        return {-val, -deriv};
    }

    constexpr dual operator-(dual o) const {
        return {val - o.val, deriv - o.deriv};
    }

    // Product rule encoded in multiplication!
    // (f * g)' = f' * g + f * g'
    constexpr dual operator*(dual o) const {
        return {val * o.val, deriv * o.val + val * o.deriv};
    }

    // Quotient rule (when needed)
    constexpr dual operator/(dual o) const {
        T v = val / o.val;
        T d = (deriv * o.val - val * o.deriv) / (o.val * o.val);
        return {v, d};
    }
};

// ADL functions
template<typename T> constexpr dual<T> zero(dual<T>) { return {T{0}, T{0}}; }
template<typename T> constexpr dual<T> one(dual<T>)  { return {T{1}, T{0}}; }

template<typename T> constexpr dual<T> twice(dual<T> const& d) { return d * d; }
template<typename T> constexpr dual<T> half(dual<T> const& d)  { return d; }
template<typename T> constexpr bool even(dual<T> const&)       { return true; }

template<typename T> constexpr dual<T> increment(dual<T> const& d) { return d; }
template<typename T> constexpr dual<T> decrement(dual<T> const& d) { return d; }

// Factory: Create a dual number representing "x" with derivative 1
// Use this as input to get automatic differentiation
template<typename T = double>
constexpr dual<T> variable(T x) {
    return {x, T{1}};  // d/dx[x] = 1
}

// Factory: Create a constant (derivative = 0)
template<typename T = double>
constexpr dual<T> constant(T c) {
    return {c, T{0}};
}

// Math functions extended to dual numbers (chain rule)
template<typename T>
dual<T> sqrt(dual<T> const& d) {
    T s = std::sqrt(d.val);
    return {s, d.deriv / (T{2} * s)};
}

template<typename T>
dual<T> exp(dual<T> const& d) {
    T e = std::exp(d.val);
    return {e, d.deriv * e};
}

template<typename T>
dual<T> log(dual<T> const& d) {
    return {std::log(d.val), d.deriv / d.val};
}

template<typename T>
dual<T> sin(dual<T> const& d) {
    return {std::sin(d.val), d.deriv * std::cos(d.val)};
}

template<typename T>
dual<T> cos(dual<T> const& d) {
    return {std::cos(d.val), -d.deriv * std::sin(d.val)};
}

} // namespace peasant::examples
