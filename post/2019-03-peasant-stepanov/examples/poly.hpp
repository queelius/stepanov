#pragma once

/**
 * @file poly.hpp
 * @brief Polynomials - binomial coefficients via power((1+x), n)
 *
 * Polynomials form a ring under addition and multiplication (convolution).
 * The polynomial (1+x)^n has binomial coefficients as its coefficients!
 *
 * power(poly{1,1}, 10) = [1,10,45,120,210,252,210,120,45,10,1]
 *
 * This is Pascal's triangle computed algebraically.
 */

#include <vector>
#include <compare>
#include <algorithm>
#include <cstdint>

namespace peasant::examples {

template<typename T = int64_t>
struct poly {
    std::vector<T> coeffs;  // coeffs[i] = coefficient of x^i

    poly() : coeffs{T{0}} {}
    poly(std::initializer_list<T> cs) : coeffs(cs) { normalize(); }
    explicit poly(std::vector<T> cs) : coeffs(std::move(cs)) { normalize(); }
    explicit poly(T c) : coeffs{c} { normalize(); }

    // Remove trailing zeros
    void normalize() {
        while (coeffs.size() > 1 && coeffs.back() == T{0}) {
            coeffs.pop_back();
        }
        if (coeffs.empty()) coeffs.push_back(T{0});
    }

    size_t degree() const { return coeffs.size() - 1; }

    T operator[](size_t i) const {
        return i < coeffs.size() ? coeffs[i] : T{0};
    }

    bool operator==(poly const& o) const { return coeffs == o.coeffs; }
    auto operator<=>(poly const& o) const { return coeffs <=> o.coeffs; }

    // Addition
    poly operator+(poly const& o) const {
        size_t n = std::max(coeffs.size(), o.coeffs.size());
        std::vector<T> result(n);
        for (size_t i = 0; i < n; ++i) {
            result[i] = (*this)[i] + o[i];
        }
        poly p(std::move(result));
        p.normalize();
        return p;
    }

    poly operator-() const {
        std::vector<T> result(coeffs.size());
        for (size_t i = 0; i < coeffs.size(); ++i) {
            result[i] = -coeffs[i];
        }
        return poly(std::move(result));
    }

    poly operator-(poly const& o) const {
        return *this + (-o);
    }

    // Multiplication (convolution)
    poly operator*(poly const& o) const {
        if (coeffs.size() == 1 && coeffs[0] == T{0}) return *this;
        if (o.coeffs.size() == 1 && o.coeffs[0] == T{0}) return o;

        size_t n = coeffs.size() + o.coeffs.size() - 1;
        std::vector<T> result(n, T{0});
        for (size_t i = 0; i < coeffs.size(); ++i) {
            for (size_t j = 0; j < o.coeffs.size(); ++j) {
                result[i + j] += coeffs[i] * o.coeffs[j];
            }
        }
        poly p(std::move(result));
        p.normalize();
        return p;
    }

    // Evaluate at a point
    T operator()(T x) const {
        T result{0};
        T power{1};
        for (auto c : coeffs) {
            result += c * power;
            power *= x;
        }
        return result;
    }
};

// ADL functions
template<typename T> poly<T> zero(poly<T>) { return poly<T>{T{0}}; }
template<typename T> poly<T> one(poly<T>)  { return poly<T>{T{1}}; }

template<typename T> poly<T> twice(poly<T> const& p) { return p * p; }
template<typename T> poly<T> half(poly<T> const& p)  { return p; }
template<typename T> bool even(poly<T> const&)       { return true; }

template<typename T> poly<T> increment(poly<T> const& p) { return p; }
template<typename T> poly<T> decrement(poly<T> const& p) { return p; }

// Factory: (1 + x) - raising this to power n gives binomial coefficients
template<typename T = int64_t>
poly<T> binomial_generator() {
    return poly<T>{T{1}, T{1}};  // 1 + x
}

// Factory: x - raising this to power n gives x^n
template<typename T = int64_t>
poly<T> x_monomial() {
    return poly<T>{T{0}, T{1}};  // x
}

} // namespace peasant::examples
