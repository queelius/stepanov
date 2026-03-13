#pragma once

#include <cmath>
#include <memory>
#include "../concepts/concepts.hpp"

namespace limes::algorithms::accumulators {

using concepts::Field;
using concepts::Accumulator;

// Simple accumulator - no error compensation
template<Field T>
class simple_accumulator {
public:
    using value_type = T;

    constexpr simple_accumulator() noexcept : sum_{} {}
    constexpr explicit simple_accumulator(T init) noexcept : sum_{init} {}

    constexpr simple_accumulator& operator+=(T value) noexcept {
        sum_ += value;
        return *this;
    }

    constexpr simple_accumulator& operator-=(T value) noexcept {
        sum_ -= value;
        return *this;
    }

    constexpr T operator()() const noexcept { return sum_; }
    constexpr operator T() const noexcept { return sum_; }

    constexpr void reset() noexcept { sum_ = T{}; }

private:
    T sum_;
};

// Kahan-Babuska accumulator with error compensation
template<Field T>
class kahan_accumulator {
public:
    using value_type = T;

    constexpr kahan_accumulator() noexcept : sum_{}, correction_{} {}
    constexpr explicit kahan_accumulator(T init) noexcept : sum_{init}, correction_{} {}

    constexpr kahan_accumulator& operator+=(T value) noexcept {
        T y = value - correction_;
        T t = sum_ + y;
        correction_ = (t - sum_) - y;
        sum_ = t;
        return *this;
    }

    constexpr T operator()() const noexcept { return sum_; }
    constexpr operator T() const noexcept { return sum_; }

    constexpr T error() const noexcept { return correction_; }
    constexpr T correction() const noexcept { return correction_; }

    constexpr void reset() noexcept {
        sum_ = T{};
        correction_ = T{};
    }

private:
    T sum_;
    T correction_;
};

// Neumaier accumulator - improved Kahan for all magnitudes
template<Field T>
class neumaier_accumulator {
public:
    using value_type = T;

    constexpr neumaier_accumulator() noexcept : sum_{}, correction_{} {}
    constexpr explicit neumaier_accumulator(T init) noexcept : sum_{init}, correction_{} {}

    constexpr neumaier_accumulator& operator+=(T value) noexcept {
        T t = sum_ + value;
        if (std::abs(sum_) >= std::abs(value)) {
            correction_ += (sum_ - t) + value;
        } else {
            correction_ += (value - t) + sum_;
        }
        sum_ = t;
        return *this;
    }

    constexpr T operator()() const noexcept { return sum_ + correction_; }
    constexpr operator T() const noexcept { return sum_ + correction_; }

    constexpr T error() const noexcept { return correction_; }
    constexpr T correction() const noexcept { return correction_; }

    constexpr void reset() noexcept {
        sum_ = T{};
        correction_ = T{};
    }

private:
    T sum_;
    T correction_;
};

// Klein accumulator - second-order error compensation
template<Field T>
class klein_accumulator {
public:
    using value_type = T;

    constexpr klein_accumulator() noexcept : sum_{}, cs_{}, ccs_{} {}
    constexpr explicit klein_accumulator(T init) noexcept : sum_{init}, cs_{}, ccs_{} {}

    constexpr klein_accumulator& operator+=(T value) noexcept {
        T t = sum_ + value;
        T c = T{};

        if (std::abs(sum_) >= std::abs(value)) {
            c = (sum_ - t) + value;
        } else {
            c = (value - t) + sum_;
        }

        sum_ = t;
        t = cs_ + c;

        if (std::abs(cs_) >= std::abs(c)) {
            ccs_ += (cs_ - t) + c;
        } else {
            ccs_ += (c - t) + cs_;
        }

        cs_ = t;
        return *this;
    }

    constexpr T operator()() const noexcept { return sum_ + cs_ + ccs_; }
    constexpr operator T() const noexcept { return sum_ + cs_ + ccs_; }

    constexpr T error() const noexcept { return cs_ + ccs_; }
    constexpr T correction() const noexcept { return cs_; }
    constexpr T second_correction() const noexcept { return ccs_; }

    constexpr void reset() noexcept {
        sum_ = T{};
        cs_ = T{};
        ccs_ = T{};
    }

private:
    T sum_;
    T cs_;   // first-order correction
    T ccs_;  // second-order correction
};

// Pairwise accumulator for better accuracy with many terms
template<Field T, std::size_t ChunkSize = 128>
class pairwise_accumulator {
public:
    using value_type = T;

    constexpr pairwise_accumulator() noexcept : buffer_{}, count_{0}, sum_{} {}

    constexpr pairwise_accumulator& operator+=(T value) noexcept {
        if (count_ < ChunkSize) {
            buffer_[count_++] = value;
        } else {
            sum_ += pairwise_sum(buffer_, ChunkSize);
            buffer_[0] = value;
            count_ = 1;
        }
        return *this;
    }

    constexpr T operator()() const noexcept {
        if (count_ == 0) return sum_;
        return sum_ + pairwise_sum(buffer_, count_);
    }

    constexpr operator T() const noexcept { return operator()(); }

    constexpr void reset() noexcept {
        count_ = 0;
        sum_ = T{};
    }

private:
    T buffer_[ChunkSize];
    std::size_t count_;
    T sum_;

    static constexpr T pairwise_sum(const T* data, std::size_t n) noexcept {
        if (n <= 1) return n ? data[0] : T{};
        if (n == 2) return data[0] + data[1];

        std::size_t mid = n / 2;
        return pairwise_sum(data, mid) + pairwise_sum(data + mid, n - mid);
    }
};

// Type-erased accumulator wrapper
template<Field T>
class any_accumulator {
public:
    using value_type = T;

    template<Accumulator<T> A>
    explicit any_accumulator(A acc)
        : impl_{std::make_unique<accumulator_impl<A>>(std::move(acc))} {}

    any_accumulator(const any_accumulator& other)
        : impl_{other.impl_->clone()} {}

    any_accumulator& operator=(const any_accumulator& other) {
        impl_ = other.impl_->clone();
        return *this;
    }

    any_accumulator& operator+=(T value) {
        impl_->add(value);
        return *this;
    }

    T operator()() const { return impl_->get(); }
    operator T() const { return impl_->get(); }

    void reset() { impl_->reset(); }

private:
    struct accumulator_base {
        virtual ~accumulator_base() = default;
        virtual void add(T value) = 0;
        virtual T get() const = 0;
        virtual void reset() = 0;
        virtual std::unique_ptr<accumulator_base> clone() const = 0;
    };

    template<typename A>
    struct accumulator_impl : accumulator_base {
        A acc;

        explicit accumulator_impl(A a) : acc{std::move(a)} {}

        void add(T value) override { acc += value; }
        T get() const override { return acc(); }
        void reset() override { acc.reset(); }

        std::unique_ptr<accumulator_base> clone() const override {
            return std::make_unique<accumulator_impl>(acc);
        }
    };

    std::unique_ptr<accumulator_base> impl_;
};

} // namespace limes::algorithms::accumulators
