#pragma once

#include <vector>
#include <algorithm>
#include <utility>
#include <cmath>
#include <limits>
#include <span>
#include <ranges>
#include "concepts.hpp"

namespace stepanov {

/**
 * Sparse polynomial representation following generic programming principles
 * Coefficients stored as (degree, value) pairs, sorted by degree
 * Zero coefficients are not stored - efficient for sparse polynomials
 */
template <typename T>
    requires field<T>
class polynomial {
public:
    using value_type = T;
    using degree_type = int;
    using coefficient_pair = std::pair<degree_type, value_type>;
    using container_type = std::vector<coefficient_pair>;
    using iterator = typename container_type::iterator;
    using const_iterator = typename container_type::const_iterator;

private:
    container_type coefficients;

    // Comparator for coefficient pairs
    struct degree_less {
        bool operator()(const coefficient_pair& a, const coefficient_pair& b) const {
            return a.first < b.first;
        }
        bool operator()(const coefficient_pair& a, degree_type d) const {
            return a.first < d;
        }
        bool operator()(degree_type d, const coefficient_pair& b) const {
            return d < b.first;
        }
    };

    // Remove zero coefficients and maintain sorted order
    void normalize() {
        coefficients.erase(
            std::remove_if(coefficients.begin(), coefficients.end(),
                          [](const auto& p) { return p.second == T(0); }),
            coefficients.end());
        std::sort(coefficients.begin(), coefficients.end(), degree_less{});
    }

public:
    // Constructors
    polynomial() = default;

    explicit polynomial(T constant) {
        if (constant != T(0)) {
            coefficients.emplace_back(0, constant);
        }
    }

    template <std::input_iterator InputIt>
    polynomial(InputIt first, InputIt last) : coefficients(first, last) {
        normalize();
    }

    polynomial(std::initializer_list<coefficient_pair> init)
        : coefficients(init) {
        normalize();
    }

    // Degree of the polynomial (-1 for zero polynomial)
    degree_type degree() const {
        return coefficients.empty() ? -1 : coefficients.back().first;
    }

    // Check if polynomial is zero
    bool is_zero() const {
        return coefficients.empty();
    }

    // Access coefficient by degree (const version)
    value_type operator[](degree_type d) const {
        auto it = std::lower_bound(coefficients.begin(), coefficients.end(),
                                   d, degree_less{});
        return (it != coefficients.end() && it->first == d) ? it->second : T(0);
    }

    // Coefficient proxy for non-const access
    class coefficient_proxy {
    private:
        polynomial& poly;
        degree_type deg;
        iterator pos;

    public:
        coefficient_proxy(polynomial& p, degree_type d, iterator it)
            : poly(p), deg(d), pos(it) {}

        operator value_type() const {
            if (pos != poly.coefficients.end() && pos->first == deg) {
                return pos->second;
            }
            return T(0);
        }

        coefficient_proxy& operator=(const value_type& val) {
            if (val == T(0)) {
                // Remove zero coefficient if it exists
                if (pos != poly.coefficients.end() && pos->first == deg) {
                    poly.coefficients.erase(pos);
                }
            } else {
                // Insert or update coefficient
                if (pos != poly.coefficients.end() && pos->first == deg) {
                    pos->second = val;
                } else {
                    poly.coefficients.insert(pos, {deg, val});
                }
            }
            return *this;
        }

        coefficient_proxy& operator+=(const value_type& val) {
            *this = value_type(*this) + val;
            return *this;
        }

        coefficient_proxy& operator-=(const value_type& val) {
            *this = value_type(*this) - val;
            return *this;
        }

        coefficient_proxy& operator*=(const value_type& val) {
            *this = value_type(*this) * val;
            return *this;
        }
    };

    // Access coefficient by degree (non-const version)
    coefficient_proxy operator[](degree_type d) {
        auto it = std::lower_bound(coefficients.begin(), coefficients.end(),
                                   d, degree_less{});
        return coefficient_proxy(*this, d, it);
    }

    // Evaluate polynomial at point x
    value_type operator()(const value_type& x) const {
        if (coefficients.empty()) return T(0);

        value_type result = T(0);

        // Simple evaluation: sum of coef * x^degree
        for (const auto& [deg, coef] : coefficients) {
            value_type term = coef;
            for (degree_type i = 0; i < deg; ++i) {
                term = term * x;
            }
            result = result + term;
        }

        return result;
    }

    // Iterator support
    const_iterator begin() const { return coefficients.begin(); }
    const_iterator end() const { return coefficients.end(); }
    iterator begin() { return coefficients.begin(); }
    iterator end() { return coefficients.end(); }

    // Range support for C++20
    auto nonzero_coefficients() const {
        return coefficients | std::views::all;
    }

    // Leading coefficient
    value_type leading_coefficient() const {
        return coefficients.empty() ? T(0) : coefficients.back().second;
    }
};

// Arithmetic operations

template <typename T>
polynomial<T> operator+(const polynomial<T>& p, const polynomial<T>& q) {
    polynomial<T> result;
    auto& r = result.coefficients;

    auto pi = p.begin(), pend = p.end();
    auto qi = q.begin(), qend = q.end();

    while (pi != pend && qi != qend) {
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

    r.insert(r.end(), pi, pend);
    r.insert(r.end(), qi, qend);

    return result;
}

template <typename T>
polynomial<T> operator-(const polynomial<T>& p, const polynomial<T>& q) {
    polynomial<T> result;
    auto& r = result.coefficients;

    auto pi = p.begin(), pend = p.end();
    auto qi = q.begin(), qend = q.end();

    while (pi != pend && qi != qend) {
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

    r.insert(r.end(), pi, pend);
    while (qi != qend) {
        r.emplace_back(qi->first, -qi->second);
        ++qi;
    }

    return result;
}

template <typename T>
polynomial<T> operator*(const polynomial<T>& p, const polynomial<T>& q) {
    polynomial<T> result;

    for (const auto& [deg_p, coef_p] : p) {
        for (const auto& [deg_q, coef_q] : q) {
            result[deg_p + deg_q] += coef_p * coef_q;
        }
    }

    return result;
}

template <typename T>
polynomial<T> operator*(const T& scalar, const polynomial<T>& p) {
    if (scalar == T(0)) return polynomial<T>{};

    polynomial<T> result;
    for (const auto& [deg, coef] : p) {
        result.coefficients.emplace_back(deg, scalar * coef);
    }
    return result;
}

template <typename T>
polynomial<T> operator*(const polynomial<T>& p, const T& scalar) {
    return scalar * p;
}

template <typename T>
polynomial<T> operator/(const polynomial<T>& p, const T& scalar) {
    return (T(1) / scalar) * p;
}

// Polynomial division (quotient and remainder)
template <typename T>
std::pair<polynomial<T>, polynomial<T>> divmod(const polynomial<T>& dividend,
                                               const polynomial<T>& divisor) {
    if (divisor.is_zero()) {
        throw std::domain_error("Division by zero polynomial");
    }

    polynomial<T> quotient;
    polynomial<T> remainder = dividend;

    while (!remainder.is_zero() && remainder.degree() >= divisor.degree()) {
        auto deg_diff = remainder.degree() - divisor.degree();
        auto coef = remainder.leading_coefficient() / divisor.leading_coefficient();

        quotient[deg_diff] = coef;

        // Subtract divisor * coef * x^deg_diff from remainder
        for (const auto& [deg, val] : divisor) {
            remainder[deg + deg_diff] -= coef * val;
        }
    }

    return {quotient, remainder};
}

// Comparison operators
template <typename T>
bool operator==(const polynomial<T>& p, const polynomial<T>& q) {
    return std::ranges::equal(p, q);
}

template <typename T>
bool operator!=(const polynomial<T>& p, const polynomial<T>& q) {
    return !(p == q);
}

// Calculus operations

template <typename T>
polynomial<T> derivative(const polynomial<T>& p) {
    polynomial<T> result;

    for (const auto& [deg, coef] : p) {
        if (deg > 0) {
            result[deg - 1] = coef * T(deg);
        }
    }

    return result;
}

template <typename T>
polynomial<T> antiderivative(const polynomial<T>& p, T constant = T(0)) {
    polynomial<T> result;
    result[0] = constant;

    for (const auto& [deg, coef] : p) {
        result[deg + 1] = coef / T(deg + 1);
    }

    return result;
}

// Root finding using Newton-Raphson method
template <typename T>
    requires ordered_field<T>
class newton_solver {
private:
    static constexpr int max_iterations = 1000;
    T epsilon;

public:
    explicit newton_solver(T eps = T(1e-10)) : epsilon(eps) {}

    std::optional<T> find_root(const polynomial<T>& p, T initial_guess) const {
        auto dp = derivative(p);
        T x = initial_guess;

        for (int i = 0; i < max_iterations; ++i) {
            T fx = p(x);
            if (abs(fx) < epsilon) {
                return x;
            }

            T fpx = dp(x);
            if (abs(fpx) < epsilon) {
                return std::nullopt;  // Derivative too small
            }

            T x_new = x - fx / fpx;
            if (abs(x_new - x) < epsilon) {
                return x_new;
            }

            x = x_new;
        }

        return std::nullopt;  // Failed to converge
    }

    std::vector<T> find_all_roots(const polynomial<T>& p,
                                  T lower_bound, T upper_bound,
                                  int num_samples = 20) const {
        std::vector<T> roots;
        T step = (upper_bound - lower_bound) / T(num_samples);

        for (int i = 0; i <= num_samples; ++i) {
            T initial = lower_bound + T(i) * step;
            if (auto root = find_root(p, initial)) {
                // Check if this root is new (not a duplicate)
                bool is_new = true;
                for (const auto& r : roots) {
                    if (abs(*root - r) < epsilon * T(10)) {
                        is_new = false;
                        break;
                    }
                }
                if (is_new) {
                    roots.push_back(*root);
                }
            }
        }

        std::sort(roots.begin(), roots.end());
        return roots;
    }
};

// Helper functions
template <typename T>
std::vector<T> stationary_points(const polynomial<T>& p) {
    newton_solver<T> solver;
    return solver.find_all_roots(derivative(p),
                                 T(-100), T(100));  // Default bounds
}

template <typename T>
std::vector<T> inflection_points(const polynomial<T>& p) {
    newton_solver<T> solver;
    return solver.find_all_roots(derivative(derivative(p)),
                                 T(-100), T(100));  // Default bounds
}

// Global degree function for concept compliance
template <typename T>
typename polynomial<T>::degree_type degree(const polynomial<T>& p) {
    return p.degree();
}

} // namespace stepanov