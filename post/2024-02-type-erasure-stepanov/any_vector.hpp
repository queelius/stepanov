#pragma once

/**
 * @file any_vector.hpp
 * @brief Type-Erased Vector Spaces
 *
 * A vector space over a field F is a set V with:
 *   - Addition: v + w (associative, commutative)
 *   - Zero: a neutral element for addition
 *   - Negation: -v such that v + (-v) = 0
 *   - Scalar multiplication: a * v for a in F
 *
 * This file demonstrates type erasure for algebraic structures.
 * Any type satisfying VectorSpace can be stored in any_vector<F>.
 *
 * Examples that model VectorSpace<double>:
 *   - vec2, vec3 (geometric vectors)
 *   - polynomials (coefficient vectors)
 *   - matrices (under elementwise operations)
 *   - functions (pointwise operations)
 */

#include <concepts>
#include <memory>
#include <utility>
#include <cmath>
#include <array>
#include <vector>
#include <stdexcept>

namespace vector_space {

// =============================================================================
// VectorSpace Concept
// =============================================================================

/**
 * A type V is a VectorSpace over field F if it supports:
 *   - zero(v) returns the zero vector
 *   - v + w returns vector sum
 *   - -v returns additive inverse
 *   - a * v returns scalar multiple (a is type F)
 */
template<typename V, typename F = double>
concept VectorSpace = requires(V v, V w, F a) {
    zero(v);
    v + w;
    -v;
    a * v;
    v * a;
};

// =============================================================================
// Forward declarations for example types
// =============================================================================

struct vec2;
struct vec3;
class polynomial;
struct complex_vec;

// =============================================================================
// any_vector<F> - Type-Erased Vector Space Element
// =============================================================================

// Forward declaration
template<typename F> class any_vector;

// Helper trait to detect any_vector (avoids constraint recursion)
template<typename T>
struct is_any_vector : std::false_type {};

template<typename F>
struct is_any_vector<any_vector<F>> : std::true_type {};

template<typename T>
inline constexpr bool is_any_vector_v = is_any_vector<std::remove_cvref_t<T>>::value;

/**
 * Type-erased container for any vector space element.
 *
 * Uses the "concept-model" pattern (Sean Parent):
 * 1. concept_t: abstract interface
 * 2. model_t<T>: concrete implementation wrapping T
 * 3. any_vector: handle with value semantics
 */
template<typename F = double>
class any_vector {
public:
    // Concept: the abstract interface
    struct concept_t {
        virtual ~concept_t() = default;
        virtual std::unique_ptr<concept_t> clone() const = 0;
        virtual std::unique_ptr<concept_t> zero_impl() const = 0;
        virtual std::unique_ptr<concept_t> add(const concept_t& other) const = 0;
        virtual std::unique_ptr<concept_t> negate() const = 0;
        virtual std::unique_ptr<concept_t> scale(F scalar) const = 0;
        virtual bool equals(const concept_t& other) const = 0;
    };

    // Model: wraps any T satisfying VectorSpace
    template<typename T>
    struct model_t final : concept_t {
        T data_;

        explicit model_t(T x) : data_(std::move(x)) {}

        std::unique_ptr<concept_t> clone() const override {
            return std::make_unique<model_t>(data_);
        }

        std::unique_ptr<concept_t> zero_impl() const override {
            return std::make_unique<model_t>(zero(data_));
        }

        std::unique_ptr<concept_t> add(const concept_t& other) const override {
            return std::make_unique<model_t>(
                data_ + static_cast<const model_t&>(other).data_
            );
        }

        std::unique_ptr<concept_t> negate() const override {
            return std::make_unique<model_t>(-data_);
        }

        std::unique_ptr<concept_t> scale(F scalar) const override {
            return std::make_unique<model_t>(scalar * data_);
        }

        bool equals(const concept_t& other) const override {
            auto* p = dynamic_cast<const model_t*>(&other);
            return p && data_ == p->data_;
        }
    };

private:
    std::unique_ptr<concept_t> self_;

    // Private constructor for internal use
    struct private_tag {};
    any_vector(private_tag, std::unique_ptr<concept_t> ptr) : self_(std::move(ptr)) {}

public:
    // Default: empty (invalid state, but needed for containers)
    any_vector() = default;

    // Construct from any VectorSpace type (excluding any_vector itself)
    template<typename T>
        requires (!is_any_vector_v<T>) && VectorSpace<T, F>
    any_vector(T x) : self_(std::make_unique<model_t<T>>(std::move(x))) {}

    // Copy
    any_vector(const any_vector& other)
        : self_(other.self_ ? other.self_->clone() : nullptr) {}

    any_vector& operator=(const any_vector& other) {
        if (this != &other) {
            self_ = other.self_ ? other.self_->clone() : nullptr;
        }
        return *this;
    }

    // Move
    any_vector(any_vector&&) noexcept = default;
    any_vector& operator=(any_vector&&) noexcept = default;

    // Check if valid
    explicit operator bool() const { return self_ != nullptr; }

    // Vector space operations (implemented as member functions to avoid ADL issues)
    any_vector get_zero() const {
        return any_vector(private_tag{}, self_->zero_impl());
    }

    any_vector add(const any_vector& other) const {
        return any_vector(private_tag{}, self_->add(*other.self_));
    }

    any_vector neg() const {
        return any_vector(private_tag{}, self_->negate());
    }

    any_vector scale(F scalar) const {
        return any_vector(private_tag{}, self_->scale(scalar));
    }

    bool equals(const any_vector& other) const {
        if (!self_ || !other.self_) return !self_ && !other.self_;
        return self_->equals(*other.self_);
    }
};

// Free functions for any_vector (defined outside class to avoid ADL issues during concept checking)
template<typename F>
any_vector<F> zero(const any_vector<F>& v) {
    return v.get_zero();
}

template<typename F>
any_vector<F> operator+(const any_vector<F>& a, const any_vector<F>& b) {
    return a.add(b);
}

template<typename F>
any_vector<F> operator-(const any_vector<F>& v) {
    return v.neg();
}

template<typename F>
any_vector<F> operator-(const any_vector<F>& a, const any_vector<F>& b) {
    return a + (-b);
}

template<typename F>
any_vector<F> operator*(F scalar, const any_vector<F>& v) {
    return v.scale(scalar);
}

template<typename F>
any_vector<F> operator*(const any_vector<F>& v, F scalar) {
    return scalar * v;
}

template<typename F>
bool operator==(const any_vector<F>& a, const any_vector<F>& b) {
    return a.equals(b);
}

// =============================================================================
// Example 1: vec2 - 2D Geometric Vector
// =============================================================================

struct vec2 {
    double x = 0, y = 0;

    constexpr vec2() = default;
    constexpr vec2(double x_, double y_) : x(x_), y(y_) {}

    constexpr bool operator==(const vec2&) const = default;
};

// Vector space operations via ADL
constexpr vec2 zero(const vec2&) { return {0, 0}; }
constexpr vec2 operator+(vec2 a, vec2 b) { return {a.x + b.x, a.y + b.y}; }
constexpr vec2 operator-(vec2 v) { return {-v.x, -v.y}; }
constexpr vec2 operator*(double s, vec2 v) { return {s * v.x, s * v.y}; }
constexpr vec2 operator*(vec2 v, double s) { return s * v; }

// Additional operations (not required for VectorSpace)
constexpr double dot(vec2 a, vec2 b) { return a.x * b.x + a.y * b.y; }
inline double norm(vec2 v) { return std::sqrt(dot(v, v)); }

// =============================================================================
// Example 2: vec3 - 3D Geometric Vector
// =============================================================================

struct vec3 {
    double x = 0, y = 0, z = 0;

    constexpr vec3() = default;
    constexpr vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    constexpr bool operator==(const vec3&) const = default;
};

constexpr vec3 zero(const vec3&) { return {0, 0, 0}; }
constexpr vec3 operator+(vec3 a, vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
constexpr vec3 operator-(vec3 v) { return {-v.x, -v.y, -v.z}; }
constexpr vec3 operator*(double s, vec3 v) { return {s * v.x, s * v.y, s * v.z}; }
constexpr vec3 operator*(vec3 v, double s) { return s * v; }

constexpr double dot(vec3 a, vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
constexpr vec3 cross(vec3 a, vec3 b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}
inline double norm(vec3 v) { return std::sqrt(dot(v, v)); }

// =============================================================================
// Example 3: polynomial - Polynomial as Coefficient Vector
// =============================================================================

/**
 * Polynomials form a vector space: (a + bx + cx^2) + (d + ex) = (a+d) + (b+e)x + cx^2
 * Scalar multiplication: k * (a + bx) = ka + kbx
 */
class polynomial {
    std::vector<double> coeffs_;  // coeffs_[i] = coefficient of x^i

    void trim() {
        while (!coeffs_.empty() && coeffs_.back() == 0)
            coeffs_.pop_back();
    }

public:
    polynomial() = default;
    polynomial(std::initializer_list<double> c) : coeffs_(c) { trim(); }
    explicit polynomial(std::vector<double> c) : coeffs_(std::move(c)) { trim(); }

    std::size_t degree() const { return coeffs_.empty() ? 0 : coeffs_.size() - 1; }
    double operator[](std::size_t i) const { return i < coeffs_.size() ? coeffs_[i] : 0; }

    double operator()(double x) const {  // Horner evaluation
        double result = 0;
        for (auto it = coeffs_.rbegin(); it != coeffs_.rend(); ++it)
            result = result * x + *it;
        return result;
    }

    bool operator==(const polynomial& other) const {
        if (coeffs_.size() != other.coeffs_.size()) return false;
        for (std::size_t i = 0; i < coeffs_.size(); ++i)
            if (coeffs_[i] != other.coeffs_[i]) return false;
        return true;
    }

    friend polynomial zero(const polynomial&) { return {}; }

    friend polynomial operator+(const polynomial& a, const polynomial& b) {
        std::vector<double> result(std::max(a.coeffs_.size(), b.coeffs_.size()));
        for (std::size_t i = 0; i < result.size(); ++i)
            result[i] = a[i] + b[i];
        return polynomial(std::move(result));
    }

    friend polynomial operator-(const polynomial& p) {
        std::vector<double> result(p.coeffs_.size());
        for (std::size_t i = 0; i < p.coeffs_.size(); ++i)
            result[i] = -p.coeffs_[i];
        return polynomial(std::move(result));
    }

    friend polynomial operator*(double s, const polynomial& p) {
        std::vector<double> result(p.coeffs_.size());
        for (std::size_t i = 0; i < p.coeffs_.size(); ++i)
            result[i] = s * p.coeffs_[i];
        return polynomial(std::move(result));
    }

    friend polynomial operator*(const polynomial& p, double s) { return s * p; }
};

// =============================================================================
// Example 4: complex_vec - Complex Numbers as R^2 Vector Space
// =============================================================================

/**
 * Complex numbers form a 2D real vector space.
 * (a + bi) + (c + di) = (a+c) + (b+d)i
 * r * (a + bi) = ra + rbi
 */
struct complex_vec {
    double re = 0, im = 0;

    constexpr complex_vec() = default;
    constexpr complex_vec(double r, double i = 0) : re(r), im(i) {}

    constexpr bool operator==(const complex_vec&) const = default;

    // Complex multiplication (not required for vector space, but useful)
    constexpr complex_vec operator*(complex_vec other) const {
        return {re * other.re - im * other.im, re * other.im + im * other.re};
    }
};

constexpr complex_vec zero(const complex_vec&) { return {0, 0}; }
constexpr complex_vec operator+(complex_vec a, complex_vec b) { return {a.re + b.re, a.im + b.im}; }
constexpr complex_vec operator-(complex_vec v) { return {-v.re, -v.im}; }
constexpr complex_vec operator*(double s, complex_vec v) { return {s * v.re, s * v.im}; }
constexpr complex_vec operator*(complex_vec v, double s) { return s * v; }

inline double abs(complex_vec v) { return std::sqrt(v.re * v.re + v.im * v.im); }
constexpr complex_vec conj(complex_vec v) { return {v.re, -v.im}; }

// =============================================================================
// Verify concepts are satisfied
// =============================================================================

static_assert(VectorSpace<vec2, double>);
static_assert(VectorSpace<vec3, double>);
static_assert(VectorSpace<polynomial, double>);
static_assert(VectorSpace<complex_vec, double>);

} // namespace vector_space
