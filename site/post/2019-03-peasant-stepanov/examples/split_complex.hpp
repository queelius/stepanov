#pragma once

/**
 * @file split_complex.hpp
 * @brief Split-complex numbers - hyperbolic geometry / Lorentz boosts
 *
 * Split-complex numbers: a + bj where j² = +1 (not -1 like i!)
 *
 * Product: (a + bj)(c + dj) = (ac + bd) + (ad + bc)j
 *
 * These numbers parameterize hyperbolic rotations (Lorentz boosts).
 * In special relativity, spacetime transformations use this algebra!
 *
 * Mind-blowing insight:
 *   hyperbolic_rotation(φ)^n = hyperbolic_rotation(n*φ)
 *   power(boost(φ), n) gives the n-fold boost in O(log n)
 *
 * The "unit circle" is actually a hyperbola: a² - b² = 1
 */

#include <compare>
#include <cmath>

namespace peasant::examples {

template<typename T = double>
struct split_complex {
    T a, b;  // a + bj where j² = +1

    constexpr split_complex() : a{1}, b{0} {}
    constexpr split_complex(T a_, T b_) : a{a_}, b{b_} {}

    constexpr bool operator==(split_complex const&) const = default;
    constexpr auto operator<=>(split_complex const&) const = default;

    constexpr split_complex operator+(split_complex const& o) const {
        return {a + o.a, b + o.b};
    }

    constexpr split_complex operator-() const {
        return {-a, -b};
    }

    constexpr split_complex operator-(split_complex const& o) const {
        return *this + (-o);
    }

    // (a + bj)(c + dj) = (ac + bd) + (ad + bc)j
    // Note: +bd because j² = +1
    constexpr split_complex operator*(split_complex const& o) const {
        return {a*o.a + b*o.b, a*o.b + b*o.a};
    }

    // Conjugate: (a + bj)* = a - bj
    constexpr split_complex conjugate() const {
        return {a, -b};
    }

    // "Modulus squared": a² - b² (can be negative!)
    // For unit split-complex: a² - b² = 1 (a hyperbola)
    constexpr T modulus_sq() const {
        return a*a - b*b;
    }

    // Division: z/w = z * w* / |w|²
    constexpr split_complex operator/(split_complex const& o) const {
        T denom = o.modulus_sq();
        split_complex c = o.conjugate();
        return {(a*c.a + b*c.b) / denom, (a*c.b + b*c.a) / denom};
    }
};

// ADL functions
template<typename T> constexpr split_complex<T> zero(split_complex<T>) { return {T{1}, T{0}}; }
template<typename T> constexpr split_complex<T> one(split_complex<T>)  { return {T{1}, T{0}}; }

template<typename T> constexpr split_complex<T> twice(split_complex<T> const& s) { return s * s; }
template<typename T> constexpr split_complex<T> half(split_complex<T> const& s)  { return s; }
template<typename T> constexpr bool even(split_complex<T> const&)                { return true; }

template<typename T> constexpr split_complex<T> increment(split_complex<T> const& s) { return s; }
template<typename T> constexpr split_complex<T> decrement(split_complex<T> const& s) { return s; }

// Factory: Hyperbolic rotation by angle φ
// Like exp(jφ) = cosh(φ) + j*sinh(φ)
template<typename T = double>
split_complex<T> hyperbolic_rotation(T phi) {
    return {std::cosh(phi), std::sinh(phi)};
}

// Factory: Unit split-complex on the right branch of hyperbola
template<typename T = double>
split_complex<T> unit_split_complex(T t) {
    // Parameterize a² - b² = 1 as (cosh(t), sinh(t))
    return hyperbolic_rotation(t);
}

} // namespace peasant::examples
