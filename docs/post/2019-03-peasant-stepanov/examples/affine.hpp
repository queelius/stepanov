#pragma once

/**
 * @file affine.hpp
 * @brief Affine functions f(x) = ax + b - compound interest in O(log n)
 *
 * An affine transformation is f(x) = ax + b. Composing two such functions:
 *   (f ∘ g)(x) = f(g(x)) = a*(c*x + d) + b = (a*c)*x + (a*d + b)
 *
 * This forms a monoid under composition!
 *
 * Mind-blowing use case: Compound interest with regular deposits.
 *   - Each year: balance *= 1.05; balance += 100  (5% interest, $100 deposit)
 *   - This is f(x) = 1.05*x + 100
 *   - power(f, 30) gives the 30-year transformation in O(log n)!
 */

#include <compare>

namespace peasant::examples {

template<typename T = double>
struct affine {
    T a, b;  // f(x) = ax + b

    constexpr bool operator==(affine const&) const = default;
    constexpr auto operator<=>(affine const&) const = default;

    // Composition: (f * g)(x) = f(g(x)) = a*(c*x + d) + b = ac*x + (ad + b)
    constexpr affine operator*(affine g) const {
        return {a * g.a, a * g.b + b};
    }

    // Apply the function to a value
    constexpr T operator()(T x) const {
        return a * x + b;
    }

    // Required for algebraic concept
    constexpr affine operator+(affine const& o) const { return *this * o; }
    constexpr affine operator-() const { return *this; }
};

// ADL functions
template<typename T> constexpr affine<T> zero(affine<T>) { return {T{1}, T{0}}; }  // Identity
template<typename T> constexpr affine<T> one(affine<T>)  { return {T{1}, T{0}}; }  // Identity: f(x) = x

template<typename T> constexpr affine<T> twice(affine<T> const& f) { return f * f; }
template<typename T> constexpr affine<T> half(affine<T> const& f)  { return f; }
template<typename T> constexpr bool even(affine<T> const&)         { return true; }

template<typename T> constexpr affine<T> increment(affine<T> const& f) { return f; }
template<typename T> constexpr affine<T> decrement(affine<T> const& f) { return f; }

// Factory functions
template<typename T = double>
constexpr affine<T> compound_interest(T rate, T deposit) {
    return {T{1} + rate, deposit};  // f(x) = (1+rate)*x + deposit
}

template<typename T = double>
constexpr affine<T> linear_map(T scale, T offset) {
    return {scale, offset};
}

} // namespace peasant::examples
