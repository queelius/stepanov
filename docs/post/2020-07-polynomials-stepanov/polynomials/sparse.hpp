#pragma once

/**
 * @file sparse.hpp
 * @brief Sparse polynomial representation
 *
 * THIS MODULE TEACHES: Polynomials as algebraic objects.
 *
 * A polynomial is not just a sequence of coefficients - it's an element of a ring.
 * The same operations (+, -, *) work regardless of the coefficient type:
 * integers, rationals, modular arithmetic, or other polynomials.
 *
 * Sparse representation stores only non-zero terms, making it efficient for
 * polynomials like x^1000 + 1 (only 2 terms, not 1001 coefficients).
 */

#include "concepts.hpp"
#include <vector>
#include <algorithm>
#include <utility>
#include <stdexcept>

namespace poly {

/**
 * Sparse polynomial representation.
 *
 * Stores (degree, coefficient) pairs sorted by degree.
 * Zero coefficients are not stored.
 *
 * @tparam T Coefficient type (must satisfy ring concept)
 */
template<typename T>
    requires ring<T>
class polynomial {
public:
    using value_type = T;
    using degree_type = int;
    using term = std::pair<degree_type, value_type>;
    using container_type = std::vector<term>;
    using iterator = typename container_type::iterator;
    using const_iterator = typename container_type::const_iterator;

private:
    container_type terms_;

    // Keep terms sorted and remove zeros
    void normalize() {
        // Remove zero coefficients
        terms_.erase(
            std::remove_if(terms_.begin(), terms_.end(),
                          [](const term& t) { return t.second == T(0); }),
            terms_.end());
        // Sort by degree
        std::sort(terms_.begin(), terms_.end(),
                  [](const term& a, const term& b) { return a.first < b.first; });
    }

public:
    // Constructors
    polynomial() = default;

    /**
     * Create constant polynomial.
     */
    explicit polynomial(T constant) {
        if (constant != T(0)) {
            terms_.emplace_back(0, constant);
        }
    }

    /**
     * Create polynomial from dense coefficient array.
     * coeffs[i] is the coefficient of x^i.
     */
    explicit polynomial(std::initializer_list<T> coeffs) {
        degree_type deg = 0;
        for (const auto& c : coeffs) {
            if (c != T(0)) {
                terms_.emplace_back(deg, c);
            }
            ++deg;
        }
    }

    /**
     * Create polynomial from sparse terms.
     */
    polynomial(std::initializer_list<term> ts) : terms_(ts) {
        normalize();
    }

    /**
     * Create monomial: coefficient * x^degree.
     */
    static polynomial monomial(T coefficient, degree_type degree) {
        polynomial p;
        if (coefficient != T(0)) {
            p.terms_.emplace_back(degree, coefficient);
        }
        return p;
    }

    /**
     * Create the polynomial x (the variable itself).
     */
    static polynomial x() {
        return monomial(T(1), 1);
    }

    // Accessors

    /**
     * Degree of polynomial. Returns -1 for zero polynomial.
     */
    degree_type degree() const {
        return terms_.empty() ? -1 : terms_.back().first;
    }

    /**
     * Check if polynomial is zero.
     */
    bool is_zero() const {
        return terms_.empty();
    }

    /**
     * Check if polynomial is constant (degree <= 0).
     */
    bool is_constant() const {
        return terms_.empty() || (terms_.size() == 1 && terms_[0].first == 0);
    }

    /**
     * Get coefficient of x^d.
     */
    T coefficient(degree_type d) const {
        auto it = std::lower_bound(terms_.begin(), terms_.end(), d,
            [](const term& t, degree_type deg) { return t.first < deg; });
        return (it != terms_.end() && it->first == d) ? it->second : T(0);
    }

    /**
     * Get coefficient of x^d (alias for coefficient).
     */
    T operator[](degree_type d) const {
        return coefficient(d);
    }

    /**
     * Leading coefficient (coefficient of highest degree term).
     */
    T leading_coefficient() const {
        return terms_.empty() ? T(0) : terms_.back().second;
    }

    /**
     * Constant term (coefficient of x^0).
     */
    T constant_term() const {
        return coefficient(0);
    }

    /**
     * Number of non-zero terms.
     */
    std::size_t num_terms() const {
        return terms_.size();
    }

    // Iterators over non-zero terms
    const_iterator begin() const { return terms_.begin(); }
    const_iterator end() const { return terms_.end(); }
    iterator begin() { return terms_.begin(); }
    iterator end() { return terms_.end(); }

    // Evaluation operator (defined after evaluate() in evaluation.hpp)
    // Caller should use evaluate(p, x) function instead

    // Arithmetic operations

    polynomial operator-() const {
        polynomial result;
        result.terms_.reserve(terms_.size());
        for (const auto& [deg, coef] : terms_) {
            result.terms_.emplace_back(deg, -coef);
        }
        return result;
    }

    polynomial& operator+=(const polynomial& other) {
        *this = *this + other;
        return *this;
    }

    polynomial& operator-=(const polynomial& other) {
        *this = *this - other;
        return *this;
    }

    polynomial& operator*=(const polynomial& other) {
        *this = *this * other;
        return *this;
    }

    polynomial& operator*=(const T& scalar) {
        if (scalar == T(0)) {
            terms_.clear();
        } else {
            for (auto& [deg, coef] : terms_) {
                coef = coef * scalar;
            }
        }
        return *this;
    }

    // Comparison
    bool operator==(const polynomial& other) const {
        if (terms_.size() != other.terms_.size()) return false;
        for (std::size_t i = 0; i < terms_.size(); ++i) {
            if (terms_[i].first != other.terms_[i].first ||
                terms_[i].second != other.terms_[i].second) {
                return false;
            }
        }
        return true;
    }

    bool operator!=(const polynomial& other) const {
        return !(*this == other);
    }

    // Friend declarations for operations
    template<typename U>
    friend polynomial<U> operator+(const polynomial<U>& p, const polynomial<U>& q);

    template<typename U>
    friend polynomial<U> operator-(const polynomial<U>& p, const polynomial<U>& q);

    template<typename U>
    friend polynomial<U> operator*(const polynomial<U>& p, const polynomial<U>& q);
};

// =============================================================================
// Arithmetic operations
// =============================================================================

template<typename T>
polynomial<T> operator+(const polynomial<T>& p, const polynomial<T>& q) {
    polynomial<T> result;
    auto& r = result.terms_;

    auto pi = p.begin(), pe = p.end();
    auto qi = q.begin(), qe = q.end();

    while (pi != pe && qi != qe) {
        if (pi->first < qi->first) {
            r.push_back(*pi++);
        } else if (qi->first < pi->first) {
            r.push_back(*qi++);
        } else {
            auto sum = pi->second + qi->second;
            if (sum != T(0)) {
                r.emplace_back(pi->first, sum);
            }
            ++pi; ++qi;
        }
    }

    r.insert(r.end(), pi, pe);
    r.insert(r.end(), qi, qe);

    return result;
}

template<typename T>
polynomial<T> operator-(const polynomial<T>& p, const polynomial<T>& q) {
    polynomial<T> result;
    auto& r = result.terms_;

    auto pi = p.begin(), pe = p.end();
    auto qi = q.begin(), qe = q.end();

    while (pi != pe && qi != qe) {
        if (pi->first < qi->first) {
            r.push_back(*pi++);
        } else if (qi->first < pi->first) {
            r.emplace_back(qi->first, -qi->second);
            ++qi;
        } else {
            auto diff = pi->second - qi->second;
            if (diff != T(0)) {
                r.emplace_back(pi->first, diff);
            }
            ++pi; ++qi;
        }
    }

    r.insert(r.end(), pi, pe);
    while (qi != qe) {
        r.emplace_back(qi->first, -qi->second);
        ++qi;
    }

    return result;
}

template<typename T>
polynomial<T> operator*(const polynomial<T>& p, const polynomial<T>& q) {
    if (p.is_zero() || q.is_zero()) {
        return polynomial<T>{};
    }

    // Use map to accumulate coefficients, then convert to sorted vector
    std::vector<std::pair<int, T>> temp;

    for (const auto& [deg_p, coef_p] : p) {
        for (const auto& [deg_q, coef_q] : q) {
            int new_deg = deg_p + deg_q;
            T new_coef = coef_p * coef_q;

            // Find or insert
            auto it = std::lower_bound(temp.begin(), temp.end(), new_deg,
                [](const auto& t, int d) { return t.first < d; });

            if (it != temp.end() && it->first == new_deg) {
                it->second = it->second + new_coef;
            } else {
                temp.insert(it, {new_deg, new_coef});
            }
        }
    }

    polynomial<T> result;
    for (const auto& [deg, coef] : temp) {
        if (coef != T(0)) {
            result.terms_.emplace_back(deg, coef);
        }
    }

    return result;
}

// Scalar operations

template<typename T>
polynomial<T> operator*(const T& scalar, const polynomial<T>& p) {
    polynomial<T> result = p;
    result *= scalar;
    return result;
}

template<typename T>
polynomial<T> operator*(const polynomial<T>& p, const T& scalar) {
    return scalar * p;
}

template<typename T>
    requires field<T>
polynomial<T> operator/(const polynomial<T>& p, const T& scalar) {
    polynomial<T> result = p;
    for (auto& [deg, coef] : result) {
        coef = coef / scalar;
    }
    return result;
}

} // namespace poly
