// geometric_algebra.hpp
// Geometric Algebra (Clifford Algebra) implementation
// Provides unified framework for geometry, rotations, and physics
// Based on David Hestenes' work and Geometric Algebra for Computer Science

#pragma once

#include <array>
#include <bitset>
#include <cmath>
#include <concepts>
#include <algorithm>
#include <numeric>
#include <bit>

namespace stepanov::math {

// Grade and blade utilities
constexpr size_t grade(size_t blade_index) {
    return std::popcount(blade_index);
}

constexpr int reorder_sign(size_t i, size_t j) {
    // Count swaps needed to reorder basis vectors
    int swaps = 0;
    size_t combined = i & j;
    while (combined) {
        size_t bit = combined & -combined;  // Lowest set bit
        swaps += std::popcount((i & (bit - 1)) ^ (j & (bit - 1)));
        combined &= combined - 1;
    }
    return (swaps & 1) ? -1 : 1;
}

// Multivector in geometric algebra of dimension N
template<size_t N, std::floating_point T = double>
    requires (N <= 5)  // Limit for reasonable memory usage (2^5 = 32 components)
class multivector {
private:
    static constexpr size_t SIZE = 1 << N;  // 2^N components
    std::array<T, SIZE> components_{};

    // Metric signature (default Euclidean)
    std::array<int, N> metric_ = []() {
        std::array<int, N> m;
        m.fill(1);  // All positive (Euclidean)
        return m;
    }();

public:
    // Constructors
    multivector() = default;

    // Scalar constructor
    explicit multivector(T scalar) {
        components_[0] = scalar;
    }

    // Component constructor
    multivector(std::initializer_list<std::pair<size_t, T>> init) {
        for (const auto& [blade, value] : init) {
            if (blade < SIZE) {
                components_[blade] = value;
            }
        }
    }

    // Vector constructor (grade-1 element)
    static multivector vector(const std::array<T, N>& v) {
        multivector result;
        for (size_t i = 0; i < N; ++i) {
            result.components_[1 << i] = v[i];
        }
        return result;
    }

    // Bivector constructor (grade-2 element)
    static multivector bivector(size_t i, size_t j, T value) {
        if (i >= N || j >= N || i == j) return multivector{};
        multivector result;
        size_t blade = (1 << i) | (1 << j);
        result.components_[blade] = value;
        return result;
    }

    // Rotor from rotation plane and angle
    static multivector rotor(size_t i, size_t j, T angle) {
        T half_angle = angle / 2;
        multivector result(std::cos(half_angle));
        size_t blade = (1 << i) | (1 << j);
        result.components_[blade] = -std::sin(half_angle);
        return result;
    }

    // Access components
    T& operator[](size_t blade) { return components_[blade]; }
    const T& operator[](size_t blade) const { return components_[blade]; }

    T scalar() const { return components_[0]; }

    // Grade projection
    multivector grade_project(size_t g) const {
        multivector result;
        for (size_t i = 0; i < SIZE; ++i) {
            if (grade(i) == g) {
                result.components_[i] = components_[i];
            }
        }
        return result;
    }

    // Get specific grade parts
    multivector grade_0() const { return grade_project(0); }  // Scalar
    multivector grade_1() const { return grade_project(1); }  // Vector
    multivector grade_2() const { return grade_project(2); }  // Bivector
    multivector grade_3() const { return grade_project(3); }  // Trivector

    // Addition and subtraction
    multivector operator+(const multivector& other) const {
        multivector result;
        for (size_t i = 0; i < SIZE; ++i) {
            result.components_[i] = components_[i] + other.components_[i];
        }
        return result;
    }

    multivector operator-(const multivector& other) const {
        multivector result;
        for (size_t i = 0; i < SIZE; ++i) {
            result.components_[i] = components_[i] - other.components_[i];
        }
        return result;
    }

    multivector operator-() const {
        multivector result;
        for (size_t i = 0; i < SIZE; ++i) {
            result.components_[i] = -components_[i];
        }
        return result;
    }

    // Scalar multiplication
    multivector operator*(T scalar) const {
        multivector result;
        for (size_t i = 0; i < SIZE; ++i) {
            result.components_[i] = components_[i] * scalar;
        }
        return result;
    }

    friend multivector operator*(T scalar, const multivector& m) {
        return m * scalar;
    }

    // Geometric product (fundamental operation)
    multivector operator*(const multivector& other) const {
        multivector result;

        for (size_t i = 0; i < SIZE; ++i) {
            if (std::abs(components_[i]) < std::numeric_limits<T>::epsilon())
                continue;

            for (size_t j = 0; j < SIZE; ++j) {
                if (std::abs(other.components_[j]) < std::numeric_limits<T>::epsilon())
                    continue;

                // Compute product of basis blades
                size_t common = i & j;
                size_t blade = i ^ j;  // XOR gives resulting blade

                // Sign from reordering and metric
                int sign = reorder_sign(i, j);

                // Apply metric for squared basis vectors
                for (size_t k = 0; k < N; ++k) {
                    if (common & (1 << k)) {
                        sign *= metric_[k];
                    }
                }

                result.components_[blade] += sign * components_[i] * other.components_[j];
            }
        }

        return result;
    }

    // Outer product (wedge product) - antisymmetric part
    multivector operator^(const multivector& other) const {
        multivector result;

        for (size_t i = 0; i < SIZE; ++i) {
            if (std::abs(components_[i]) < std::numeric_limits<T>::epsilon())
                continue;

            for (size_t j = 0; j < SIZE; ++j) {
                if (std::abs(other.components_[j]) < std::numeric_limits<T>::epsilon())
                    continue;

                // Outer product is zero if blades share basis vectors
                if ((i & j) != 0) continue;

                size_t blade = i | j;
                int sign = reorder_sign(i, j);
                result.components_[blade] += sign * components_[i] * other.components_[j];
            }
        }

        return result;
    }

    // Inner product (dot product) - symmetric part
    multivector operator|(const multivector& other) const {
        // Left contraction: a|b = <ab>_{grade(b) - grade(a)}
        multivector result;

        for (size_t i = 0; i < SIZE; ++i) {
            if (std::abs(components_[i]) < std::numeric_limits<T>::epsilon())
                continue;

            size_t grade_i = grade(i);

            for (size_t j = 0; j < SIZE; ++j) {
                if (std::abs(other.components_[j]) < std::numeric_limits<T>::epsilon())
                    continue;

                size_t grade_j = grade(j);
                if (grade_j < grade_i) continue;

                size_t blade = i ^ j;
                if (grade(blade) != grade_j - grade_i) continue;

                int sign = reorder_sign(i, j);

                // Apply metric
                size_t common = i & j;
                for (size_t k = 0; k < N; ++k) {
                    if (common & (1 << k)) {
                        sign *= metric_[k];
                    }
                }

                result.components_[blade] += sign * components_[i] * other.components_[j];
            }
        }

        return result;
    }

    // Reverse (reverses order of basis vectors)
    multivector reverse() const {
        multivector result;
        for (size_t i = 0; i < SIZE; ++i) {
            size_t g = grade(i);
            int sign = ((g * (g - 1)) / 2) & 1 ? -1 : 1;
            result.components_[i] = sign * components_[i];
        }
        return result;
    }

    // Grade involution (changes sign of odd grades)
    multivector grade_involution() const {
        multivector result;
        for (size_t i = 0; i < SIZE; ++i) {
            int sign = (grade(i) & 1) ? -1 : 1;
            result.components_[i] = sign * components_[i];
        }
        return result;
    }

    // Clifford conjugation (combination of reverse and grade involution)
    multivector conjugate() const {
        multivector result;
        for (size_t i = 0; i < SIZE; ++i) {
            size_t g = grade(i);
            int sign = ((g * (g + 1)) / 2) & 1 ? -1 : 1;
            result.components_[i] = sign * components_[i];
        }
        return result;
    }

    // Magnitude squared
    T magnitude_squared() const {
        return (*this * reverse()).scalar();
    }

    // Magnitude
    T magnitude() const {
        return std::sqrt(std::abs(magnitude_squared()));
    }

    // Normalize
    multivector normalized() const {
        T mag = magnitude();
        if (mag < std::numeric_limits<T>::epsilon()) {
            return multivector{};
        }
        return *this * (T(1) / mag);
    }

    // Inverse (for rotors and versors)
    multivector inverse() const {
        multivector rev = reverse();
        T mag_sq = (*this * rev).scalar();
        if (std::abs(mag_sq) < std::numeric_limits<T>::epsilon()) {
            return multivector{};  // No inverse
        }
        return rev * (T(1) / mag_sq);
    }

    // Dual (multiplication by pseudoscalar)
    multivector dual() const {
        multivector I;
        I.components_[SIZE - 1] = 1;  // Pseudoscalar
        return *this * I;
    }

    // Sandwich product (for rotations: R v R^-1)
    multivector sandwich(const multivector& v) const {
        return *this * v * inverse();
    }

    // Exponential (for bivectors gives rotors)
    multivector exp() const {
        // For pure grade-2 (bivector), use Euler's formula
        multivector B = grade_project(2);
        T B_mag = B.magnitude();

        if (B_mag < std::numeric_limits<T>::epsilon()) {
            return multivector(1) + *this;  // First-order approximation
        }

        multivector B_norm = B * (T(1) / B_mag);
        return multivector(std::cos(B_mag)) + B_norm * std::sin(B_mag);
    }

    // Logarithm (for rotors)
    multivector log() const {
        T scalar_part = scalar();
        multivector bivector_part = grade_project(2);
        T biv_mag = bivector_part.magnitude();

        if (biv_mag < std::numeric_limits<T>::epsilon()) {
            return multivector{};  // Pure scalar
        }

        T angle = std::atan2(biv_mag, scalar_part);
        return bivector_part * (angle / biv_mag);
    }

    // Check if this is a blade (can be factored into outer product of vectors)
    bool is_blade() const {
        // A multivector is a blade if it has only one grade component
        int found_grade = -1;
        for (size_t i = 0; i < SIZE; ++i) {
            if (std::abs(components_[i]) > std::numeric_limits<T>::epsilon()) {
                int g = grade(i);
                if (found_grade == -1) {
                    found_grade = g;
                } else if (found_grade != g) {
                    return false;
                }
            }
        }
        return true;
    }

    // Set custom metric (for non-Euclidean spaces)
    void set_metric(const std::array<int, N>& metric) {
        metric_ = metric;
    }

    // Equality
    bool operator==(const multivector& other) const {
        for (size_t i = 0; i < SIZE; ++i) {
            if (std::abs(components_[i] - other.components_[i]) >
                std::numeric_limits<T>::epsilon()) {
                return false;
            }
        }
        return true;
    }
};

// Common geometric algebra spaces
template<typename T = double>
using GA2 = multivector<2, T>;  // 2D geometric algebra

template<typename T = double>
using GA3 = multivector<3, T>;  // 3D geometric algebra (most common)

template<typename T = double>
using GA4 = multivector<4, T>;  // Spacetime algebra

// Conformal Geometric Algebra (CGA) for 3D Euclidean space
// Maps 3D points to 5D null vectors
template<std::floating_point T = double>
class conformal_point {
private:
    multivector<5, T> mv_;

public:
    // Embed 3D point into conformal space
    conformal_point(T x, T y, T z) {
        // e0 = (e+ - e-)/2, e∞ = e+ + e-
        T r2 = x*x + y*y + z*z;
        mv_ = multivector<5, T>::vector({x, y, z, (1 + r2)/2, (1 - r2)/2});
    }

    // Extract 3D coordinates
    std::array<T, 3> to_3d() const {
        T e4 = mv_[1 << 4];  // e- component
        if (std::abs(e4) < std::numeric_limits<T>::epsilon()) {
            return {0, 0, 0};  // Point at infinity
        }

        T scale = T(1) / e4;
        return {mv_[1] * scale, mv_[2] * scale, mv_[4] * scale};
    }

    multivector<5, T> as_multivector() const { return mv_; }
};

// Utility functions for common operations
template<size_t N, typename T>
multivector<N, T> commutator(const multivector<N, T>& a, const multivector<N, T>& b) {
    return (a * b - b * a) * T(0.5);
}

template<size_t N, typename T>
multivector<N, T> anticommutator(const multivector<N, T>& a, const multivector<N, T>& b) {
    return (a * b + b * a) * T(0.5);
}

// Reflection of vector v in hyperplane with normal n
template<size_t N, typename T>
multivector<N, T> reflect(const multivector<N, T>& v, const multivector<N, T>& n) {
    return -(n * v * n.inverse());
}

// Rotation using rotor (generalized quaternion)
template<size_t N, typename T>
multivector<N, T> rotate(const multivector<N, T>& v, const multivector<N, T>& rotor) {
    return rotor.sandwich(v);
}

} // namespace stepanov::math