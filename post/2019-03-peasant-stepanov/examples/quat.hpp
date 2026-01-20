#pragma once

/**
 * @file quat.hpp
 * @brief Quaternions - 3D rotations (non-commutative multiplication!)
 *
 * Quaternions q = w + xi + yj + zk where:
 *   i² = j² = k² = ijk = -1
 *
 * They form a NON-COMMUTATIVE group under multiplication.
 * Unit quaternions represent 3D rotations!
 *
 * Mind-blowing insight:
 *   power(small_rotation, 100) applies a rotation 100 times
 *   This is how animation engines interpolate rotations.
 *
 * Note: a*b ≠ b*a in general - this is the first non-commutative example!
 */

#include <compare>
#include <cmath>

namespace peasant::examples {

template<typename T = double>
struct quat {
    T w, x, y, z;  // w + xi + yj + zk

    constexpr quat() : w{1}, x{0}, y{0}, z{0} {}
    constexpr quat(T w_, T x_, T y_, T z_) : w{w_}, x{x_}, y{y_}, z{z_} {}

    constexpr bool operator==(quat const&) const = default;
    constexpr auto operator<=>(quat const&) const = default;

    constexpr quat operator+(quat const& o) const {
        return {w + o.w, x + o.x, y + o.y, z + o.z};
    }

    constexpr quat operator-() const {
        return {-w, -x, -y, -z};
    }

    constexpr quat operator-(quat const& o) const {
        return *this + (-o);
    }

    // Hamilton product (non-commutative!)
    // (a + bi + cj + dk)(e + fi + gj + hk)
    constexpr quat operator*(quat const& o) const {
        return {
            w*o.w - x*o.x - y*o.y - z*o.z,
            w*o.x + x*o.w + y*o.z - z*o.y,
            w*o.y - x*o.z + y*o.w + z*o.x,
            w*o.z + x*o.y - y*o.x + z*o.w
        };
    }

    // Conjugate: q* = w - xi - yj - zk
    constexpr quat conjugate() const {
        return {w, -x, -y, -z};
    }

    // Squared norm: |q|² = w² + x² + y² + z²
    constexpr T norm_sq() const {
        return w*w + x*x + y*y + z*z;
    }

    // Norm: |q|
    T norm() const {
        return std::sqrt(norm_sq());
    }

    // Inverse: q⁻¹ = q* / |q|²
    constexpr quat inverse() const {
        T n2 = norm_sq();
        quat c = conjugate();
        return {c.w / n2, c.x / n2, c.y / n2, c.z / n2};
    }

    // Normalize to unit quaternion
    quat normalized() const {
        T n = norm();
        return {w/n, x/n, y/n, z/n};
    }
};

// ADL functions
template<typename T> constexpr quat<T> zero(quat<T>) { return {T{1}, T{0}, T{0}, T{0}}; }
template<typename T> constexpr quat<T> one(quat<T>)  { return {T{1}, T{0}, T{0}, T{0}}; }  // Identity

template<typename T> constexpr quat<T> twice(quat<T> const& q) { return q * q; }
template<typename T> constexpr quat<T> half(quat<T> const& q)  { return q; }
template<typename T> constexpr bool even(quat<T> const&)       { return true; }

template<typename T> constexpr quat<T> increment(quat<T> const& q) { return q; }
template<typename T> constexpr quat<T> decrement(quat<T> const& q) { return q; }

// Factory: Rotation around axis (ax, ay, az) by angle theta
// The axis should be normalized
template<typename T = double>
quat<T> axis_angle(T ax, T ay, T az, T theta) {
    T half_theta = theta / T{2};
    T s = std::sin(half_theta);
    T c = std::cos(half_theta);
    return quat<T>{c, ax * s, ay * s, az * s};
}

// Factory: Rotation around X axis
template<typename T = double>
quat<T> rotation_x(T theta) {
    return axis_angle<T>(T{1}, T{0}, T{0}, theta);
}

// Factory: Rotation around Y axis
template<typename T = double>
quat<T> rotation_y(T theta) {
    return axis_angle<T>(T{0}, T{1}, T{0}, theta);
}

// Factory: Rotation around Z axis
template<typename T = double>
quat<T> rotation_z(T theta) {
    return axis_angle<T>(T{0}, T{0}, T{1}, theta);
}

} // namespace peasant::examples
