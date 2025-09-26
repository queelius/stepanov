#pragma once

#include "../concepts.hpp"
#include <cmath>
#include <limits>
#include <algorithm>
#include <tuple>
#include <type_traits>

namespace stepanov {
namespace statistics {

// Kahan-Babuška-Neumaier summation for numerical stability
template<field T>
class kahan_accumulator {
public:
    using value_type = T;

private:
    T sum_;
    T correction_;
    std::size_t count_;

public:
    kahan_accumulator() : sum_(T(0)), correction_(T(0)), count_(0) {}

    kahan_accumulator& operator+=(T value) {
        T t = sum_ + value;
        if (std::abs(sum_) >= std::abs(value)) {
            correction_ += (sum_ - t) + value;
        } else {
            correction_ += (value - t) + sum_;
        }
        sum_ = t;
        ++count_;
        return *this;
    }

    kahan_accumulator& operator+=(const kahan_accumulator& other) {
        operator+=(other.sum_ + other.correction_);
        count_ += other.count_ - 1;  // Adjust for double counting
        return *this;
    }

    T eval() const { return sum_ + correction_; }
    operator T() const { return eval(); }

    T sum() const { return sum_; }
    T correction() const { return correction_; }
    std::size_t count() const { return count_; }

    void reset() {
        sum_ = T(0);
        correction_ = T(0);
        count_ = 0;
    }
};

// Welford's algorithm for variance computation
template<field T>
class welford_accumulator {
public:
    using value_type = T;

private:
    std::size_t count_;
    T mean_;
    T m2_;  // Sum of squared differences from mean

public:
    welford_accumulator() : count_(0), mean_(T(0)), m2_(T(0)) {}

    welford_accumulator& operator+=(T value) {
        ++count_;
        T delta = value - mean_;
        mean_ += delta / T(count_);
        T delta2 = value - mean_;
        m2_ += delta * delta2;
        return *this;
    }

    welford_accumulator& operator+=(const welford_accumulator& other) {
        if (other.count_ == 0) return *this;
        if (count_ == 0) {
            *this = other;
            return *this;
        }

        std::size_t new_count = count_ + other.count_;
        T delta = other.mean_ - mean_;

        T new_mean = (mean_ * T(count_) + other.mean_ * T(other.count_)) / T(new_count);
        T new_m2 = m2_ + other.m2_ + delta * delta * T(count_) * T(other.count_) / T(new_count);

        count_ = new_count;
        mean_ = new_mean;
        m2_ = new_m2;
        return *this;
    }

    T eval() const { return mean_; }
    operator T() const { return mean_; }

    T mean() const { return mean_; }

    T variance() const {
        return count_ > 1 ? m2_ / T(count_ - 1) : T(0);
    }

    T population_variance() const {
        return count_ > 0 ? m2_ / T(count_) : T(0);
    }

    T stddev() const {
        return std::sqrt(variance());
    }

    T population_stddev() const {
        return std::sqrt(population_variance());
    }

    std::size_t count() const { return count_; }

    void reset() {
        count_ = 0;
        mean_ = T(0);
        m2_ = T(0);
    }
};

// Min/Max accumulator
template<totally_ordered T>
class minmax_accumulator {
public:
    using value_type = T;

private:
    T min_;
    T max_;
    bool empty_;
    std::size_t count_;

public:
    minmax_accumulator()
        : min_(std::numeric_limits<T>::max()),
          max_(std::numeric_limits<T>::lowest()),
          empty_(true),
          count_(0) {}

    minmax_accumulator& operator+=(T value) {
        if (empty_) {
            min_ = max_ = value;
            empty_ = false;
        } else {
            min_ = std::min(min_, value);
            max_ = std::max(max_, value);
        }
        ++count_;
        return *this;
    }

    minmax_accumulator& operator+=(const minmax_accumulator& other) {
        if (!other.empty_) {
            if (empty_) {
                *this = other;
            } else {
                min_ = std::min(min_, other.min_);
                max_ = std::max(max_, other.max_);
                count_ += other.count_;
            }
        }
        return *this;
    }

    T eval() const { return max_ - min_; }  // Range
    operator T() const { return eval(); }

    T min() const { return empty_ ? T{} : min_; }
    T max() const { return empty_ ? T{} : max_; }
    T range() const { return empty_ ? T{} : max_ - min_; }
    bool empty() const { return empty_; }
    std::size_t count() const { return count_; }

    void reset() {
        min_ = std::numeric_limits<T>::max();
        max_ = std::numeric_limits<T>::lowest();
        empty_ = true;
        count_ = 0;
    }
};

// Histogram accumulator
template<totally_ordered T>
class histogram_accumulator {
public:
    using value_type = T;

private:
    T min_;
    T max_;
    std::size_t num_bins_;
    std::vector<std::size_t> bins_;
    std::size_t underflow_;
    std::size_t overflow_;
    std::size_t total_count_;

    std::size_t get_bin(T value) const {
        if (value < min_) return std::size_t(-1);  // Underflow
        if (value >= max_) return num_bins_;  // Overflow

        T range = max_ - min_;
        T normalized = (value - min_) / range;
        std::size_t bin = static_cast<std::size_t>(normalized * T(num_bins_));
        return std::min(bin, num_bins_ - 1);
    }

public:
    histogram_accumulator(T min, T max, std::size_t num_bins)
        : min_(min), max_(max), num_bins_(num_bins),
          bins_(num_bins, 0), underflow_(0), overflow_(0), total_count_(0) {}

    histogram_accumulator& operator+=(T value) {
        std::size_t bin = get_bin(value);
        if (bin == std::size_t(-1)) {
            ++underflow_;
        } else if (bin >= num_bins_) {
            ++overflow_;
        } else {
            ++bins_[bin];
        }
        ++total_count_;
        return *this;
    }

    histogram_accumulator& operator+=(const histogram_accumulator& other) {
        if (min_ == other.min_ && max_ == other.max_ && num_bins_ == other.num_bins_) {
            for (std::size_t i = 0; i < num_bins_; ++i) {
                bins_[i] += other.bins_[i];
            }
            underflow_ += other.underflow_;
            overflow_ += other.overflow_;
            total_count_ += other.total_count_;
        }
        return *this;
    }

    std::size_t eval() const { return total_count_; }
    operator std::size_t() const { return total_count_; }

    std::size_t bin_count(std::size_t i) const {
        return i < num_bins_ ? bins_[i] : 0;
    }

    T bin_lower(std::size_t i) const {
        if (i >= num_bins_) return max_;
        T range = max_ - min_;
        return min_ + (range * T(i)) / T(num_bins_);
    }

    T bin_upper(std::size_t i) const {
        if (i >= num_bins_) return max_;
        T range = max_ - min_;
        return min_ + (range * T(i + 1)) / T(num_bins_);
    }

    std::size_t underflow() const { return underflow_; }
    std::size_t overflow() const { return overflow_; }
    std::size_t total_count() const { return total_count_; }
    std::size_t num_bins() const { return num_bins_; }

    void reset() {
        std::fill(bins_.begin(), bins_.end(), 0);
        underflow_ = 0;
        overflow_ = 0;
        total_count_ = 0;
    }
};

// Moving average accumulator
template<field T>
class moving_average_accumulator {
public:
    using value_type = T;

private:
    std::vector<T> window_;
    std::size_t window_size_;
    std::size_t current_pos_;
    T sum_;
    bool full_;

public:
    explicit moving_average_accumulator(std::size_t window_size)
        : window_(window_size), window_size_(window_size),
          current_pos_(0), sum_(T(0)), full_(false) {}

    moving_average_accumulator& operator+=(T value) {
        if (full_) {
            sum_ -= window_[current_pos_];
        }
        window_[current_pos_] = value;
        sum_ += value;

        current_pos_ = (current_pos_ + 1) % window_size_;
        if (current_pos_ == 0) {
            full_ = true;
        }

        return *this;
    }

    T eval() const {
        std::size_t count = full_ ? window_size_ : current_pos_;
        return count > 0 ? sum_ / T(count) : T(0);
    }

    operator T() const { return eval(); }

    std::size_t size() const {
        return full_ ? window_size_ : current_pos_;
    }

    void reset() {
        std::fill(window_.begin(), window_.end(), T(0));
        current_pos_ = 0;
        sum_ = T(0);
        full_ = false;
    }
};

// Exponentially weighted moving average
template<field T>
class ewma_accumulator {
public:
    using value_type = T;

private:
    T alpha_;  // Smoothing factor [0, 1]
    T value_;
    bool initialized_;

public:
    explicit ewma_accumulator(T alpha = T(0.1))
        : alpha_(alpha), value_(T(0)), initialized_(false) {}

    ewma_accumulator& operator+=(T value) {
        if (!initialized_) {
            value_ = value;
            initialized_ = true;
        } else {
            value_ = alpha_ * value + (T(1) - alpha_) * value_;
        }
        return *this;
    }

    ewma_accumulator& operator+=(const ewma_accumulator& other) {
        if (other.initialized_) {
            operator+=(other.value_);
        }
        return *this;
    }

    T eval() const { return value_; }
    operator T() const { return value_; }

    void set_alpha(T alpha) { alpha_ = alpha; }
    T alpha() const { return alpha_; }

    void reset() {
        value_ = T(0);
        initialized_ = false;
    }
};

// Composite accumulator combining multiple statistics
template<field T>
class composite_accumulator {
public:
    using value_type = T;

private:
    kahan_accumulator<T> sum_;
    welford_accumulator<T> moments_;
    minmax_accumulator<T> minmax_;

public:
    composite_accumulator() = default;

    composite_accumulator& operator+=(T value) {
        sum_ += value;
        moments_ += value;
        minmax_ += value;
        return *this;
    }

    composite_accumulator& operator+=(const composite_accumulator& other) {
        sum_ += other.sum_;
        moments_ += other.moments_;
        minmax_ += other.minmax_;
        return *this;
    }

    T eval() const { return moments_.mean(); }
    operator T() const { return eval(); }

    // Statistical measures
    T sum() const { return sum_.eval(); }
    T mean() const { return moments_.mean(); }
    T variance() const { return moments_.variance(); }
    T stddev() const { return moments_.stddev(); }
    T min() const { return minmax_.min(); }
    T max() const { return minmax_.max(); }
    T range() const { return minmax_.range(); }
    std::size_t count() const { return moments_.count(); }

    void reset() {
        sum_.reset();
        moments_.reset();
        minmax_.reset();
    }
};

// Quantile estimation using P² algorithm
template<field T>
class quantile_accumulator {
public:
    using value_type = T;

private:
    static constexpr std::size_t markers = 5;
    T positions_[markers];
    T heights_[markers];
    T desired_positions_[markers];
    T increments_[markers];
    std::size_t count_;
    T p_;  // Desired quantile [0, 1]

    void update_markers() {
        // P² algorithm marker update
        for (std::size_t i = 1; i < markers - 1; ++i) {
            T d = desired_positions_[i] - positions_[i];
            if ((d >= T(1) && positions_[i + 1] - positions_[i] > T(1)) ||
                (d <= T(-1) && positions_[i - 1] - positions_[i] < T(-1))) {

                T d_sign = d > T(0) ? T(1) : T(-1);

                // Parabolic prediction
                T qi = parabolic_prediction(i, d_sign);

                if (heights_[i - 1] < qi && qi < heights_[i + 1]) {
                    heights_[i] = qi;
                } else {
                    // Linear prediction
                    heights_[i] = linear_prediction(i, d_sign);
                }

                positions_[i] += d_sign;
            }
        }
    }

    T parabolic_prediction(std::size_t i, T d) const {
        return heights_[i] +
               d / (positions_[i + 1] - positions_[i - 1]) *
               ((positions_[i] - positions_[i - 1] + d) *
                (heights_[i + 1] - heights_[i]) / (positions_[i + 1] - positions_[i]) +
                (positions_[i + 1] - positions_[i] - d) *
                (heights_[i] - heights_[i - 1]) / (positions_[i] - positions_[i - 1]));
    }

    T linear_prediction(std::size_t i, T d) const {
        std::size_t j = d > T(0) ? i + 1 : i - 1;
        return heights_[i] +
               d * (heights_[j] - heights_[i]) / (positions_[j] - positions_[i]);
    }

public:
    explicit quantile_accumulator(T p = T(0.5)) : count_(0), p_(p) {
        // Initialize marker positions for quantile p
        positions_[0] = T(1);
        positions_[1] = T(1) + T(2) * p_;
        positions_[2] = T(1) + T(4) * p_;
        positions_[3] = T(3) + T(2) * p_;
        positions_[4] = T(5);

        desired_positions_[0] = T(1);
        desired_positions_[1] = T(1) + T(2) * p_;
        desired_positions_[2] = T(1) + T(4) * p_;
        desired_positions_[3] = T(3) + T(2) * p_;
        desired_positions_[4] = T(5);

        increments_[0] = T(0);
        increments_[1] = p_ / T(2);
        increments_[2] = p_;
        increments_[3] = (T(1) + p_) / T(2);
        increments_[4] = T(1);
    }

    quantile_accumulator& operator+=(T value) {
        if (count_ < markers) {
            heights_[count_] = value;
            ++count_;

            if (count_ == markers) {
                std::sort(heights_, heights_ + markers);
            }
        } else {
            // Find cell k
            std::size_t k;
            if (value < heights_[0]) {
                heights_[0] = value;
                k = 0;
            } else if (value >= heights_[markers - 1]) {
                heights_[markers - 1] = value;
                k = markers - 1;
            } else {
                k = 1;
                while (k < markers - 1 && heights_[k] <= value) {
                    ++k;
                }
                --k;
            }

            // Update positions
            for (std::size_t i = k + 1; i < markers; ++i) {
                positions_[i] += T(1);
            }

            for (std::size_t i = 0; i < markers; ++i) {
                desired_positions_[i] += increments_[i];
            }

            update_markers();
            ++count_;
        }

        return *this;
    }

    T eval() const {
        if (count_ < markers) {
            // Not enough data, use sorted array
            if (count_ == 0) return T(0);
            std::size_t idx = static_cast<std::size_t>(p_ * T(count_ - 1));
            return heights_[idx];
        }
        return heights_[2];  // Middle marker for quantile
    }

    operator T() const { return eval(); }

    T quantile() const { return eval(); }
    std::size_t count() const { return count_; }

    void reset() {
        count_ = 0;
        // Re-initialize positions
        positions_[0] = T(1);
        positions_[1] = T(1) + T(2) * p_;
        positions_[2] = T(1) + T(4) * p_;
        positions_[3] = T(3) + T(2) * p_;
        positions_[4] = T(5);
    }
};

// Accumulator composition utilities
template<typename... Accumulators>
class accumulator_tuple {
public:
    using value_type = std::common_type_t<typename Accumulators::value_type...>;

private:
    std::tuple<Accumulators...> accumulators_;

    template<std::size_t... Is>
    void add_impl(value_type value, std::index_sequence<Is...>) {
        ((std::get<Is>(accumulators_) += value), ...);
    }

public:
    accumulator_tuple() = default;

    accumulator_tuple& operator+=(value_type value) {
        add_impl(value, std::index_sequence_for<Accumulators...>{});
        return *this;
    }

    template<std::size_t I>
    auto& get() { return std::get<I>(accumulators_); }

    template<std::size_t I>
    const auto& get() const { return std::get<I>(accumulators_); }

    template<typename Acc>
    Acc& get() { return std::get<Acc>(accumulators_); }

    template<typename Acc>
    const Acc& get() const { return std::get<Acc>(accumulators_); }
};

// Convenience function to create accumulator tuple
template<typename... Accumulators>
auto make_accumulator_tuple(Accumulators... accs) {
    return accumulator_tuple<Accumulators...>();
}

} // namespace statistics
} // namespace stepanov