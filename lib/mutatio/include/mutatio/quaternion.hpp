/**
 * Quaternions - 3D Rotations Without Gimbal Lock
 * 
 * Transform: SO(3) → ℍ (unit quaternions)
 * Representation: q = w + xi + yj + zk
 * 
 * Trade-off:
 *   Gain: Smooth interpolation, no gimbal lock, efficient composition
 *   Loss: Less intuitive than Euler angles, 4 components for 3 DOF
 * 
 * Applications:
 *   - 3D graphics and animation
 *   - Robotics and aerospace
 *   - Physics simulations
 *   - Crystallography
 */

#pragma once
#include <cmath>
#include <iostream>
#include <array>

namespace mutatio {

template<typename T>
class quaternion {
    static_assert(std::is_floating_point_v<T>, "quaternion requires floating-point type");
    
private:
    T w_, x_, y_, z_;  // w + xi + yj + zk
    
public:
    // Constructors
    quaternion() : w_(1), x_(0), y_(0), z_(0) {}  // Identity
    quaternion(T w, T x, T y, T z) : w_(w), x_(x), y_(y), z_(z) {}
    
    // Factory methods
    static quaternion identity() {
        return quaternion(1, 0, 0, 0);
    }
    
    // Create from axis-angle representation
    static quaternion from_axis_angle(T x, T y, T z, T angle) {
        T half_angle = angle / 2;
        T sin_half = std::sin(half_angle);
        T cos_half = std::cos(half_angle);
        
        // Normalize axis
        T norm = std::sqrt(x*x + y*y + z*z);
        if (norm > 0) {
            x /= norm;
            y /= norm;
            z /= norm;
        }
        
        return quaternion(cos_half, x * sin_half, y * sin_half, z * sin_half);
    }
    
    // Create from Euler angles (ZYX convention)
    static quaternion from_euler(T roll, T pitch, T yaw) {
        T cr = std::cos(roll / 2);
        T sr = std::sin(roll / 2);
        T cp = std::cos(pitch / 2);
        T sp = std::sin(pitch / 2);
        T cy = std::cos(yaw / 2);
        T sy = std::sin(yaw / 2);
        
        return quaternion(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        );
    }
    
    // Getters
    T w() const { return w_; }
    T x() const { return x_; }
    T y() const { return y_; }
    T z() const { return z_; }
    
    // Quaternion operations
    T norm() const {
        return std::sqrt(w_*w_ + x_*x_ + y_*y_ + z_*z_);
    }
    
    T norm_squared() const {
        return w_*w_ + x_*x_ + y_*y_ + z_*z_;
    }
    
    quaternion normalized() const {
        T n = norm();
        if (n == 0) return *this;
        return quaternion(w_/n, x_/n, y_/n, z_/n);
    }
    
    quaternion conjugate() const {
        return quaternion(w_, -x_, -y_, -z_);
    }
    
    quaternion inverse() const {
        T norm_sq = norm_squared();
        if (norm_sq == 0) {
            throw std::runtime_error("Cannot invert zero quaternion");
        }
        return conjugate() / norm_sq;
    }
    
    // Arithmetic
    quaternion operator+(const quaternion& other) const {
        return quaternion(w_ + other.w_, x_ + other.x_, 
                         y_ + other.y_, z_ + other.z_);
    }
    
    quaternion operator-(const quaternion& other) const {
        return quaternion(w_ - other.w_, x_ - other.x_, 
                         y_ - other.y_, z_ - other.z_);
    }
    
    // Hamilton product (non-commutative!)
    quaternion operator*(const quaternion& q) const {
        return quaternion(
            w_*q.w_ - x_*q.x_ - y_*q.y_ - z_*q.z_,
            w_*q.x_ + x_*q.w_ + y_*q.z_ - z_*q.y_,
            w_*q.y_ - x_*q.z_ + y_*q.w_ + z_*q.x_,
            w_*q.z_ + x_*q.y_ - y_*q.x_ + z_*q.w_
        );
    }
    
    quaternion operator*(T scalar) const {
        return quaternion(w_ * scalar, x_ * scalar, 
                         y_ * scalar, z_ * scalar);
    }
    
    quaternion operator/(T scalar) const {
        return quaternion(w_ / scalar, x_ / scalar, 
                         y_ / scalar, z_ / scalar);
    }
    
    // Rotate a 3D vector
    std::array<T, 3> rotate(T vx, T vy, T vz) const {
        quaternion v(0, vx, vy, vz);
        quaternion result = (*this) * v * conjugate();
        return {result.x_, result.y_, result.z_};
    }
    
    // Spherical linear interpolation (SLERP)
    quaternion slerp(const quaternion& other, T t) const {
        quaternion q1 = normalized();
        quaternion q2 = other.normalized();
        
        T dot = q1.w_*q2.w_ + q1.x_*q2.x_ + q1.y_*q2.y_ + q1.z_*q2.z_;
        
        // If dot < 0, negate one quaternion to take shorter path
        if (dot < 0) {
            q2 = q2 * -1;
            dot = -dot;
        }
        
        // If quaternions are very close, use linear interpolation
        if (dot > 0.9995) {
            return (q1 * (1-t) + q2 * t).normalized();
        }
        
        T theta = std::acos(dot);
        T sin_theta = std::sin(theta);
        T a = std::sin((1-t) * theta) / sin_theta;
        T b = std::sin(t * theta) / sin_theta;
        
        return q1 * a + q2 * b;
    }
    
    // Convert to axis-angle
    void to_axis_angle(T& x, T& y, T& z, T& angle) const {
        quaternion q = normalized();
        angle = 2 * std::acos(q.w_);
        T sin_half = std::sqrt(1 - q.w_ * q.w_);
        
        if (sin_half < 0.001) {
            x = q.x_;
            y = q.y_;
            z = q.z_;
        } else {
            x = q.x_ / sin_half;
            y = q.y_ / sin_half;
            z = q.z_ / sin_half;
        }
    }
    
    // Comparison
    bool operator==(const quaternion& other) const {
        return w_ == other.w_ && x_ == other.x_ && 
               y_ == other.y_ && z_ == other.z_;
    }
    
    // Output
    friend std::ostream& operator<<(std::ostream& os, const quaternion& q) {
        os << q.w_;
        if (q.x_ >= 0) os << "+";
        os << q.x_ << "i";
        if (q.y_ >= 0) os << "+";
        os << q.y_ << "j";
        if (q.z_ >= 0) os << "+";
        os << q.z_ << "k";
        return os;
    }
};

// Type aliases
using quatf = quaternion<float>;
using quatd = quaternion<double>;

} // namespace mutatio