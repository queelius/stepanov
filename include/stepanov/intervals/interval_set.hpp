#pragma once

#include "../concepts.hpp"
#include <algorithm>
#include <vector>
#include <optional>
#include <iterator>
#include <functional>

namespace stepanov {
namespace intervals {

// Interval type representing [a, b) half-open intervals by default
template<totally_ordered T>
class interval {
public:
    using value_type = T;

    enum class boundary_type {
        open,
        closed
    };

private:
    T lower_;
    T upper_;
    boundary_type lower_boundary_;
    boundary_type upper_boundary_;
    bool empty_;

public:
    // Constructors
    interval() : empty_(true) {}

    // Half-open interval [a, b)
    interval(const T& a, const T& b)
        : lower_(a), upper_(b),
          lower_boundary_(boundary_type::closed),
          upper_boundary_(boundary_type::open),
          empty_(!(a < b)) {}

    // Explicit boundary types
    interval(const T& a, const T& b,
             boundary_type lower_bound, boundary_type upper_bound)
        : lower_(a), upper_(b),
          lower_boundary_(lower_bound),
          upper_boundary_(upper_bound),
          empty_(false) {
        if (a > b || (a == b && (lower_bound == boundary_type::open ||
                                  upper_bound == boundary_type::open))) {
            empty_ = true;
        }
    }

    // Singleton interval [a, a]
    static interval singleton(const T& a) {
        return interval(a, a, boundary_type::closed, boundary_type::closed);
    }

    // Empty interval
    static interval empty() {
        return interval();
    }

    // Open interval (a, b)
    static interval open(const T& a, const T& b) {
        return interval(a, b, boundary_type::open, boundary_type::open);
    }

    // Closed interval [a, b]
    static interval closed(const T& a, const T& b) {
        return interval(a, b, boundary_type::closed, boundary_type::closed);
    }

    // Accessors
    bool is_empty() const { return empty_; }
    const T& lower() const { return lower_; }
    const T& upper() const { return upper_; }
    boundary_type lower_boundary() const { return lower_boundary_; }
    boundary_type upper_boundary() const { return upper_boundary_; }

    // Membership test
    bool contains(const T& x) const {
        if (empty_) return false;

        bool above_lower = (lower_boundary_ == boundary_type::closed) ?
            (lower_ <= x) : (lower_ < x);
        bool below_upper = (upper_boundary_ == boundary_type::closed) ?
            (x <= upper_) : (x < upper_);

        return above_lower && below_upper;
    }

    // Check if this interval contains another interval
    bool contains(const interval& other) const {
        if (other.empty_) return true;
        if (empty_) return false;

        bool lower_ok = (lower_ < other.lower_) ||
            (lower_ == other.lower_ &&
             (lower_boundary_ == boundary_type::closed ||
              other.lower_boundary_ == boundary_type::open));

        bool upper_ok = (upper_ > other.upper_) ||
            (upper_ == other.upper_ &&
             (upper_boundary_ == boundary_type::closed ||
              other.upper_boundary_ == boundary_type::open));

        return lower_ok && upper_ok;
    }

    // Intersection
    interval intersect(const interval& other) const {
        if (empty_ || other.empty_) return empty();

        T new_lower = std::max(lower_, other.lower_);
        T new_upper = std::min(upper_, other.upper_);

        if (new_lower > new_upper) return empty();

        boundary_type new_lower_boundary = lower_boundary_;
        if (other.lower_ > lower_ ||
            (other.lower_ == lower_ && other.lower_boundary_ == boundary_type::open)) {
            new_lower_boundary = other.lower_boundary_;
        }

        boundary_type new_upper_boundary = upper_boundary_;
        if (other.upper_ < upper_ ||
            (other.upper_ == upper_ && other.upper_boundary_ == boundary_type::open)) {
            new_upper_boundary = other.upper_boundary_;
        }

        return interval(new_lower, new_upper, new_lower_boundary, new_upper_boundary);
    }

    // Check adjacency (intervals touch without overlapping)
    bool adjacent_to(const interval& other) const {
        if (empty_ || other.empty_) return false;

        // Check if upper boundary of this touches lower boundary of other
        if (upper_ == other.lower_) {
            return (upper_boundary_ == boundary_type::closed &&
                    other.lower_boundary_ == boundary_type::open) ||
                   (upper_boundary_ == boundary_type::open &&
                    other.lower_boundary_ == boundary_type::closed);
        }

        // Check if lower boundary of this touches upper boundary of other
        if (lower_ == other.upper_) {
            return (lower_boundary_ == boundary_type::closed &&
                    other.upper_boundary_ == boundary_type::open) ||
                   (lower_boundary_ == boundary_type::open &&
                    other.upper_boundary_ == boundary_type::closed);
        }

        return false;
    }

    // Check overlap
    bool overlaps(const interval& other) const {
        return !intersect(other).is_empty();
    }

    // Hull (smallest interval containing both)
    interval hull(const interval& other) const {
        if (empty_) return other;
        if (other.empty_) return *this;

        T new_lower = std::min(lower_, other.lower_);
        T new_upper = std::max(upper_, other.upper_);

        boundary_type new_lower_boundary = lower_boundary_;
        if (other.lower_ < lower_ ||
            (other.lower_ == lower_ && other.lower_boundary_ == boundary_type::closed)) {
            new_lower_boundary = other.lower_boundary_;
        }

        boundary_type new_upper_boundary = upper_boundary_;
        if (other.upper_ > upper_ ||
            (other.upper_ == upper_ && other.upper_boundary_ == boundary_type::closed)) {
            new_upper_boundary = other.upper_boundary_;
        }

        return interval(new_lower, new_upper, new_lower_boundary, new_upper_boundary);
    }

    // Comparison operators
    bool operator==(const interval& other) const {
        if (empty_ && other.empty_) return true;
        if (empty_ || other.empty_) return false;

        return lower_ == other.lower_ && upper_ == other.upper_ &&
               lower_boundary_ == other.lower_boundary_ &&
               upper_boundary_ == other.upper_boundary_;
    }

    bool operator!=(const interval& other) const {
        return !(*this == other);
    }

    bool operator<(const interval& other) const {
        if (empty_ && !other.empty_) return true;
        if (!empty_ && other.empty_) return false;
        if (empty_ && other.empty_) return false;

        if (lower_ != other.lower_) return lower_ < other.lower_;
        if (lower_boundary_ != other.lower_boundary_)
            return lower_boundary_ == boundary_type::open;
        if (upper_ != other.upper_) return upper_ < other.upper_;
        return upper_boundary_ == boundary_type::open &&
               other.upper_boundary_ == boundary_type::closed;
    }
};

// Disjoint interval set: collection of non-overlapping intervals
template<totally_ordered T>
class disjoint_interval_set {
public:
    using value_type = T;
    using interval_type = interval<T>;
    using container_type = std::vector<interval_type>;
    using iterator = typename container_type::iterator;
    using const_iterator = typename container_type::const_iterator;

private:
    container_type intervals_;

    // Maintain sorted, non-overlapping intervals
    void normalize() {
        if (intervals_.empty()) return;

        // Sort intervals
        std::sort(intervals_.begin(), intervals_.end());

        // Merge overlapping and adjacent intervals
        container_type merged;
        merged.push_back(intervals_[0]);

        for (std::size_t i = 1; i < intervals_.size(); ++i) {
            auto& last = merged.back();
            const auto& current = intervals_[i];

            if (current.is_empty()) continue;

            if (last.overlaps(current) || last.adjacent_to(current)) {
                // Merge intervals
                last = last.hull(current);
            } else {
                merged.push_back(current);
            }
        }

        intervals_ = std::move(merged);
    }

public:
    // Constructors
    disjoint_interval_set() = default;

    explicit disjoint_interval_set(const interval_type& i) {
        if (!i.is_empty()) {
            intervals_.push_back(i);
        }
    }

    disjoint_interval_set(std::initializer_list<interval_type> intervals)
        : intervals_(intervals) {
        normalize();
    }

    template<typename InputIt>
    disjoint_interval_set(InputIt first, InputIt last)
        : intervals_(first, last) {
        normalize();
    }

    // Size and capacity
    std::size_t size() const { return intervals_.size(); }
    bool empty() const { return intervals_.empty(); }
    void clear() { intervals_.clear(); }
    void reserve(std::size_t n) { intervals_.reserve(n); }

    // Iterators
    iterator begin() { return intervals_.begin(); }
    iterator end() { return intervals_.end(); }
    const_iterator begin() const { return intervals_.begin(); }
    const_iterator end() const { return intervals_.end(); }
    const_iterator cbegin() const { return intervals_.cbegin(); }
    const_iterator cend() const { return intervals_.cend(); }

    // Element access
    const interval_type& operator[](std::size_t i) const { return intervals_[i]; }
    const interval_type& at(std::size_t i) const { return intervals_.at(i); }

    // Membership test
    bool contains(const T& x) const {
        // Binary search for efficiency
        auto it = std::lower_bound(intervals_.begin(), intervals_.end(), x,
            [](const interval_type& i, const T& val) {
                return i.upper() <= val;
            });

        return it != intervals_.end() && it->contains(x);
    }

    bool contains(const interval_type& i) const {
        if (i.is_empty()) return true;

        // Find the interval that might contain i
        for (const auto& interval : intervals_) {
            if (interval.contains(i)) return true;
            if (interval.lower() > i.upper()) break;
        }
        return false;
    }

    // Insert interval
    void insert(const interval_type& i) {
        if (!i.is_empty()) {
            intervals_.push_back(i);
            normalize();
        }
    }

    void insert(const T& a, const T& b) {
        insert(interval_type(a, b));
    }

    // Erase interval
    void erase(const interval_type& i) {
        if (i.is_empty()) return;

        container_type new_intervals;

        for (const auto& interval : intervals_) {
            if (!interval.overlaps(i)) {
                new_intervals.push_back(interval);
            } else {
                // Split interval if necessary
                if (interval.lower() < i.lower()) {
                    new_intervals.push_back(interval_type(
                        interval.lower(), i.lower(),
                        interval.lower_boundary(),
                        i.lower_boundary() == interval_type::boundary_type::closed ?
                            interval_type::boundary_type::open :
                            interval_type::boundary_type::closed
                    ));
                }
                if (i.upper() < interval.upper()) {
                    new_intervals.push_back(interval_type(
                        i.upper(), interval.upper(),
                        i.upper_boundary() == interval_type::boundary_type::closed ?
                            interval_type::boundary_type::open :
                            interval_type::boundary_type::closed,
                        interval.upper_boundary()
                    ));
                }
            }
        }

        intervals_ = std::move(new_intervals);
        normalize();
    }

    // Set operations
    disjoint_interval_set union_with(const disjoint_interval_set& other) const {
        disjoint_interval_set result(*this);
        for (const auto& interval : other.intervals_) {
            result.insert(interval);
        }
        return result;
    }

    disjoint_interval_set intersect_with(const disjoint_interval_set& other) const {
        disjoint_interval_set result;

        for (const auto& i1 : intervals_) {
            for (const auto& i2 : other.intervals_) {
                auto intersection = i1.intersect(i2);
                if (!intersection.is_empty()) {
                    result.intervals_.push_back(intersection);
                }
            }
        }

        result.normalize();
        return result;
    }

    disjoint_interval_set difference(const disjoint_interval_set& other) const {
        disjoint_interval_set result(*this);
        for (const auto& interval : other.intervals_) {
            result.erase(interval);
        }
        return result;
    }

    disjoint_interval_set symmetric_difference(const disjoint_interval_set& other) const {
        auto union_set = union_with(other);
        auto intersection_set = intersect_with(other);
        return union_set.difference(intersection_set);
    }

    // Complement (requires bounded domain)
    disjoint_interval_set complement(const T& min_val, const T& max_val) const {
        disjoint_interval_set result;

        if (intervals_.empty()) {
            result.insert(interval_type::closed(min_val, max_val));
            return result;
        }

        // Add interval before first
        if (min_val < intervals_[0].lower()) {
            result.insert(interval_type(
                min_val, intervals_[0].lower(),
                interval_type::boundary_type::closed,
                intervals_[0].lower_boundary() == interval_type::boundary_type::closed ?
                    interval_type::boundary_type::open :
                    interval_type::boundary_type::closed
            ));
        }

        // Add intervals between existing ones
        for (std::size_t i = 0; i < intervals_.size() - 1; ++i) {
            result.insert(interval_type(
                intervals_[i].upper(), intervals_[i + 1].lower(),
                intervals_[i].upper_boundary() == interval_type::boundary_type::closed ?
                    interval_type::boundary_type::open :
                    interval_type::boundary_type::closed,
                intervals_[i + 1].lower_boundary() == interval_type::boundary_type::closed ?
                    interval_type::boundary_type::open :
                    interval_type::boundary_type::closed
            ));
        }

        // Add interval after last
        if (intervals_.back().upper() < max_val) {
            result.insert(interval_type(
                intervals_.back().upper(), max_val,
                intervals_.back().upper_boundary() == interval_type::boundary_type::closed ?
                    interval_type::boundary_type::open :
                    interval_type::boundary_type::closed,
                interval_type::boundary_type::closed
            ));
        }

        return result;
    }

    // Hull - smallest interval containing all intervals
    std::optional<interval_type> hull() const {
        if (intervals_.empty()) return std::nullopt;

        return interval_type(
            intervals_.front().lower(),
            intervals_.back().upper(),
            intervals_.front().lower_boundary(),
            intervals_.back().upper_boundary()
        );
    }

    // Measure (total length)
    template<typename U = T>
        requires requires(U a, U b) { a - b; }
    U measure() const {
        U total = U{};
        for (const auto& interval : intervals_) {
            total += interval.upper() - interval.lower();
        }
        return total;
    }

    // Operators
    disjoint_interval_set& operator|=(const disjoint_interval_set& other) {
        *this = union_with(other);
        return *this;
    }

    disjoint_interval_set& operator&=(const disjoint_interval_set& other) {
        *this = intersect_with(other);
        return *this;
    }

    disjoint_interval_set& operator-=(const disjoint_interval_set& other) {
        *this = difference(other);
        return *this;
    }

    disjoint_interval_set& operator^=(const disjoint_interval_set& other) {
        *this = symmetric_difference(other);
        return *this;
    }

    friend disjoint_interval_set operator|(const disjoint_interval_set& a,
                                           const disjoint_interval_set& b) {
        return a.union_with(b);
    }

    friend disjoint_interval_set operator&(const disjoint_interval_set& a,
                                           const disjoint_interval_set& b) {
        return a.intersect_with(b);
    }

    friend disjoint_interval_set operator-(const disjoint_interval_set& a,
                                           const disjoint_interval_set& b) {
        return a.difference(b);
    }

    friend disjoint_interval_set operator^(const disjoint_interval_set& a,
                                           const disjoint_interval_set& b) {
        return a.symmetric_difference(b);
    }

    bool operator==(const disjoint_interval_set& other) const {
        return intervals_ == other.intervals_;
    }

    bool operator!=(const disjoint_interval_set& other) const {
        return intervals_ != other.intervals_;
    }
};

// Convenience aliases
template<typename T>
using interval_set = disjoint_interval_set<T>;

} // namespace intervals
} // namespace stepanov