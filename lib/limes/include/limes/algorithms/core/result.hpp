#pragma once

#include <cstddef>
#include <optional>
#include <vector>
#include "../concepts/concepts.hpp"

namespace limes::algorithms {

// Integration result with comprehensive metadata
template<concepts::Field T>
struct integration_result {
    using value_type = T;

    T value_{};
    T error_{};
    std::size_t iterations_{};
    std::size_t evaluations_{};
    bool converged_{true};

    // Optional additional information
    std::optional<T> variance_{};
    std::optional<std::vector<T>> intermediate_values_{};

    constexpr integration_result() noexcept = default;

    constexpr integration_result(T val, T err, std::size_t iters) noexcept
        : value_{val}, error_{err}, iterations_{iters}, evaluations_{iters} {}

    constexpr integration_result(T val, T err, std::size_t iters, std::size_t evals) noexcept
        : value_{val}, error_{err}, iterations_{iters}, evaluations_{evals} {}

    // Accessors
    constexpr T value() const noexcept { return value_; }
    constexpr T error() const noexcept { return error_; }
    constexpr std::size_t iterations() const noexcept { return iterations_; }
    constexpr std::size_t evaluations() const noexcept { return evaluations_; }
    constexpr bool converged() const noexcept { return converged_; }

    // Conversion operators
    constexpr operator T() const noexcept { return value_; }
    constexpr explicit operator bool() const noexcept { return converged_; }

    // Relative error
    constexpr T relative_error() const noexcept {
        if (value_ == T{}) return error_;
        return error_ / std::abs(value_);
    }

    // Combine results (for composite integration)
    constexpr integration_result& operator+=(const integration_result& other) noexcept {
        value_ += other.value_;
        error_ += other.error_;
        iterations_ += other.iterations_;
        evaluations_ += other.evaluations_;
        converged_ = converged_ && other.converged_;
        return *this;
    }

    constexpr integration_result operator+(const integration_result& other) const noexcept {
        integration_result result = *this;
        result += other;
        return result;
    }

    // Scale result
    constexpr integration_result& operator*=(T scale) noexcept {
        value_ *= scale;
        error_ *= std::abs(scale);
        return *this;
    }

    constexpr integration_result operator*(T scale) const noexcept {
        integration_result result = *this;
        result *= scale;
        return result;
    }

    friend constexpr integration_result operator*(T scale, const integration_result& r) noexcept {
        return r * scale;
    }
};

// Deduction guide
template<typename T>
integration_result(T, T, std::size_t) -> integration_result<T>;

template<typename T>
integration_result(T, T, std::size_t, std::size_t) -> integration_result<T>;

} // namespace limes::algorithms
