// quaternion.hpp
// Quaternions for 3D rotations and mathematical operations
// Provides efficient rotation representation without gimbal lock

#pragma once

#include <cmath>
#include <array>
#include <concepts>
#include <numbers>
#include <ostream>

namespace stepanov::math {

template<std::floating_point T = double>
class quaternion {
private:
    T w_, x_, y_, z_;  // w + xi + yj + zk

public:
    // Constructors
    quaternion() : w_(1), x_(0), y_(0), z_(0) {}  // Identity quaternion
    quaternion(T w, T x, T y, T z) : w_(w), x_(x), y_(y), z_(z) {}

    // Scalar + vector constructor
    quaternion(T scalar, const std::array<T, 3>& vec)
        : w_(scalar), x_(vec[0]), y_(vec[1]), z_(vec[2]) {}

    // From axis-angle representation
    static quaternion from_axis_angle(const std::array<T, 3>& axis, T angle) {
        T half_angle = angle / 2;
        T s = std::sin(half_angle);
        T norm = std::sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);

        if (norm < std::numeric_limits<T>::epsilon()) {
            return quaternion{};  // Identity for zero axis
        }

        return quaternion(
            std::cos(half_angle),
            s * axis[0] / norm,
            s * axis[1] / norm,
            s * axis[2] / norm
        );
    }

    // From Euler angles (ZYX convention)
    static quaternion from_euler(T roll, T pitch, T yaw) {
        T cy = std::cos(yaw * 0.5);
        T sy = std::sin(yaw * 0.5);
        T cp = std::cos(pitch * 0.5);
        T sp = std::sin(pitch * 0.5);
        T cr = std::cos(roll * 0.5);
        T sr = std::sin(roll * 0.5);

        return quaternion(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        );
    }

    // From rotation matrix
    static quaternion from_rotation_matrix(const std::array<std::array<T, 3>, 3>& m) {
        T trace = m[0][0] + m[1][1] + m[2][2];

        if (trace > 0) {
            T s = 0.5 / std::sqrt(trace + 1.0);
            return quaternion(
                0.25 / s,
                (m[2][1] - m[1][2]) * s,
                (m[0][2] - m[2][0]) * s,
                (m[1][0] - m[0][1]) * s
            );
        } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
            T s = 2.0 * std::sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]);
            return quaternion(
                (m[2][1] - m[1][2]) / s,
                0.25 * s,
                (m[0][1] + m[1][0]) / s,
                (m[0][2] + m[2][0]) / s
            );
        } else if (m[1][1] > m[2][2]) {
            T s = 2.0 * std::sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]);
            return quaternion(
                (m[0][2] - m[2][0]) / s,
                (m[0][1] + m[1][0]) / s,
                0.25 * s,
                (m[1][2] + m[2][1]) / s
            );
        } else {
            T s = 2.0 * std::sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]);
            return quaternion(
                (m[1][0] - m[0][1]) / s,
                (m[0][2] + m[2][0]) / s,
                (m[1][2] + m[2][1]) / s,
                0.25 * s
            );
        }
    }

    // Accessors
    T w() const noexcept { return w_; }
    T x() const noexcept { return x_; }
    T y() const noexcept { return y_; }
    T z() const noexcept { return z_; }

    T& w() noexcept { return w_; }
    T& x() noexcept { return x_; }
    T& y() noexcept { return y_; }
    T& z() noexcept { return z_; }

    // Scalar and vector parts
    T scalar() const noexcept { return w_; }
    std::array<T, 3> vector() const noexcept { return {x_, y_, z_}; }

    // Basic operations
    quaternion operator+(const quaternion& q) const {
        return quaternion(w_ + q.w_, x_ + q.x_, y_ + q.y_, z_ + q.z_);
    }

    quaternion operator-(const quaternion& q) const {
        return quaternion(w_ - q.w_, x_ - q.x_, y_ - q.y_, z_ - q.z_);
    }

    // Hamilton product (non-commutative multiplication)
    quaternion operator*(const quaternion& q) const {
        return quaternion(
            w_*q.w_ - x_*q.x_ - y_*q.y_ - z_*q.z_,
            w_*q.x_ + x_*q.w_ + y_*q.z_ - z_*q.y_,
            w_*q.y_ - x_*q.z_ + y_*q.w_ + z_*q.x_,
            w_*q.z_ + x_*q.y_ - y_*q.x_ + z_*q.w_
        );
    }

    // Scalar multiplication
    quaternion operator*(T s) const {
        return quaternion(w_*s, x_*s, y_*s, z_*s);
    }

    friend quaternion operator*(T s, const quaternion& q) {
        return q * s;
    }

    quaternion operator/(T s) const {
        return quaternion(w_/s, x_/s, y_/s, z_/s);
    }

    // Conjugate (q* = w - xi - yj - zk)
    quaternion conjugate() const {
        return quaternion(w_, -x_, -y_, -z_);
    }

    // Norm squared
    T norm_squared() const {
        return w_*w_ + x_*x_ + y_*y_ + z_*z_;
    }

    // Norm
    T norm() const {
        return std::sqrt(norm_squared());
    }

    // Normalize
    quaternion normalized() const {
        T n = norm();
        if (n < std::numeric_limits<T>::epsilon()) {
            return quaternion{};  // Return identity for zero quaternion
        }
        return *this / n;
    }

    void normalize() {
        *this = normalized();
    }

    // Inverse (q^-1 = q* / |q|^2)
    quaternion inverse() const {
        T ns = norm_squared();
        if (ns < std::numeric_limits<T>::epsilon()) {
            return quaternion{};  // Undefined, return identity
        }
        return conjugate() / ns;
    }

    // Division (q1 / q2 = q1 * q2^-1)
    quaternion operator/(const quaternion& q) const {
        return *this * q.inverse();
    }

    // Exponential map (for unit quaternions)
    quaternion exp() const {
        T vn = std::sqrt(x_*x_ + y_*y_ + z_*z_);
        T ew = std::exp(w_);

        if (vn < std::numeric_limits<T>::epsilon()) {
            return quaternion(ew, 0, 0, 0);
        }

        T s = ew * std::sin(vn) / vn;
        return quaternion(ew * std::cos(vn), s*x_, s*y_, s*z_);
    }

    // Logarithm (for unit quaternions)
    quaternion log() const {
        T n = norm();
        if (std::abs(n - 1) > std::numeric_limits<T>::epsilon()) {
            // Not a unit quaternion, normalize first
            return normalized().log();
        }

        T vn = std::sqrt(x_*x_ + y_*y_ + z_*z_);
        if (vn < std::numeric_limits<T>::epsilon()) {
            return quaternion(0, 0, 0, 0);
        }

        T theta = std::atan2(vn, w_);
        T s = theta / vn;
        return quaternion(0, s*x_, s*y_, s*z_);
    }

    // Power
    quaternion pow(T t) const {
        // q^t = exp(t * log(q))
        return (log() * t).exp();
    }

    // Spherical linear interpolation (SLERP)
    static quaternion slerp(const quaternion& q0, const quaternion& q1, T t) {
        quaternion q0n = q0.normalized();
        quaternion q1n = q1.normalized();

        T dot = q0n.w_*q1n.w_ + q0n.x_*q1n.x_ + q0n.y_*q1n.y_ + q0n.z_*q1n.z_;

        // If quaternions are nearly parallel
        if (dot > 0.9995) {
            return (q0n + (q1n - q0n) * t).normalized();
        }

        // Ensure shortest path
        if (dot < 0) {
            q1n = q1n * T(-1);
            dot = -dot;
        }

        dot = std::clamp(dot, T(-1), T(1));
        T theta = std::acos(dot);
        T theta_t = theta * t;

        quaternion q2 = (q1n - q0n * dot).normalized();
        return q0n * std::cos(theta_t) + q2 * std::sin(theta_t);
    }

    // Rotate a 3D vector
    std::array<T, 3> rotate(const std::array<T, 3>& v) const {
        // v' = q * v * q^-1 (where v is treated as pure quaternion)
        quaternion vq(0, v[0], v[1], v[2]);
        quaternion result = *this * vq * conjugate();
        return {result.x_, result.y_, result.z_};
    }

    // Convert to rotation matrix
    std::array<std::array<T, 3>, 3> to_rotation_matrix() const {
        quaternion q = normalized();
        T xx = q.x_ * q.x_;
        T xy = q.x_ * q.y_;
        T xz = q.x_ * q.z_;
        T xw = q.x_ * q.w_;
        T yy = q.y_ * q.y_;
        T yz = q.y_ * q.z_;
        T yw = q.y_ * q.w_;
        T zz = q.z_ * q.z_;
        T zw = q.z_ * q.w_;

        return {{
            {1 - 2*(yy + zz), 2*(xy - zw), 2*(xz + yw)},
            {2*(xy + zw), 1 - 2*(xx + zz), 2*(yz - xw)},
            {2*(xz - yw), 2*(yz + xw), 1 - 2*(xx + yy)}
        }};
    }

    // Convert to axis-angle
    std::pair<std::array<T, 3>, T> to_axis_angle() const {
        quaternion q = normalized();
        T angle = 2 * std::acos(q.w_);
        T s = std::sqrt(1 - q.w_*q.w_);

        if (s < std::numeric_limits<T>::epsilon()) {
            return {{1, 0, 0}, 0};  // No rotation
        }

        return {{q.x_/s, q.y_/s, q.z_/s}, angle};
    }

    // Dot product
    T dot(const quaternion& q) const {
        return w_*q.w_ + x_*q.x_ + y_*q.y_ + z_*q.z_;
    }

    // Equality
    bool operator==(const quaternion& q) const {
        constexpr T eps = std::numeric_limits<T>::epsilon();
        return std::abs(w_ - q.w_) < eps &&
               std::abs(x_ - q.x_) < eps &&
               std::abs(y_ - q.y_) < eps &&
               std::abs(z_ - q.z_) < eps;
    }

    // Output
    friend std::ostream& operator<<(std::ostream& os, const quaternion& q) {
        return os << "(" << q.w_ << " + " << q.x_ << "i + "
                  << q.y_ << "j + " << q.z_ << "k)";
    }
};

// Type aliases
using quaternionf = quaternion<float>;
using quaterniond = quaternion<double>;

// Utility functions
template<std::floating_point T>
T angle_between(const quaternion<T>& q1, const quaternion<T>& q2) {
    T dot = q1.dot(q2);
    return 2 * std::acos(std::clamp(std::abs(dot), T(0), T(1)));
}

// Squad (Spherical Quadrangle Interpolation) for smoother paths
template<std::floating_point T>
quaternion<T> squad(const quaternion<T>& q0, const quaternion<T>& q1,
                   const quaternion<T>& q2, const quaternion<T>& q3, T t) {
    // Intermediate control points
    auto intermediate = [](const quaternion<T>& qn_minus_1,
                          const quaternion<T>& qn,
                          const quaternion<T>& qn_plus_1) {
        quaternion<T> qn_inv = qn.inverse();
        quaternion<T> ln_minus = (qn_inv * qn_minus_1).log();
        quaternion<T> ln_plus = (qn_inv * qn_plus_1).log();
        return qn * ((ln_minus + ln_plus) * T(-0.25)).exp();
    };

    quaternion<T> a = intermediate(q0, q1, q2);
    quaternion<T> b = intermediate(q1, q2, q3);

    return quaternion<T>::slerp(
        quaternion<T>::slerp(q1, q2, t),
        quaternion<T>::slerp(a, b, t),
        2 * t * (1 - t)
    );
}

} // namespace stepanov::math