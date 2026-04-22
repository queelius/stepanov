#pragma once

/**
 * @file intervals.hpp
 * @brief Disjoint intervals as a Boolean algebra
 *
 * An interval on a totally ordered set T is parameterized by two endpoints
 * and an open/closed flag at each end. Single intervals are closed under
 * intersection but not under union (the union of disjoint intervals is not
 * an interval) or complement (the complement of an interval is generally two
 * intervals). To recover closure we need a *set* of intervals.
 *
 * Keep the set in a canonical form, sorted by lower endpoint with all
 * overlapping or touching intervals merged, and closure becomes free:
 *
 *   Union, intersection, complement, difference, symmetric difference
 *   all preserve the canonical form.
 *
 * The resulting structure is a Boolean algebra:
 *
 *   join  (union)         op|     A ∪ B = B ∪ A
 *   meet  (intersection)  op&     A ∩ B = B ∩ A
 *   complement            op~     A ∪ ~A = U,   A ∩ ~A = ∅
 *   bottom (empty)        ∅
 *   top    (universe)     U = [lowest, highest]
 *
 *                 ~(A ∪ B) = ~A ∩ ~B   (De Morgan)
 *                 ~(A ∩ B) = ~A ∪ ~B
 *
 * This closure is a property of one-dimensional intervals; it does not
 * survive the move to two dimensions. Axis-aligned rectangles are closed
 * under intersection but the complement of a rectangle set has holes that
 * cannot in general be expressed as a finite union of rectangles. The
 * algebra of disjoint intervals is thus the deepest thing 1D buys you.
 *
 * Design notes:
 *
 * - `interval<T>` is a regular type over any totally_ordered T.
 * - `disjoint_intervals<T>` holds the canonical form as an invariant.
 * - Operations are free functions (ADL) following the series convention.
 * - The openness of each endpoint is data: it participates in the algebra.
 *   Complement of [3, 7] is (-∞, 3) ∪ (7, ∞), not [−∞, 3) ∪ (7, ∞].
 *   In other words, complement flips open/closed at every finite boundary.
 *
 * Reference: Stepanov & Rose, "From Mathematics to Generic Programming"
 *            ch. 3 on regular types and ch. 5 on algebraic structure.
 */

#include <algorithm>
#include <concepts>
#include <initializer_list>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace stepanov {

// =============================================================================
// Endpoint concept
// =============================================================================
//
// The underlying value type of an interval must be regular (supports copy,
// equality, default construction) and totally ordered (operator<).
//
// For infinity handling we distinguish `has_infinity` types (float, double)
// from integer types. The functions `lowest_endpoint<T>()` and
// `highest_endpoint<T>()` return the appropriate universe bounds.

template<typename T>
concept endpoint = std::regular<T> && std::totally_ordered<T>;

template<endpoint T>
constexpr T lowest_endpoint() noexcept {
    if constexpr (std::numeric_limits<T>::has_infinity)
        return -std::numeric_limits<T>::infinity();
    else
        return std::numeric_limits<T>::lowest();
}

template<endpoint T>
constexpr T highest_endpoint() noexcept {
    if constexpr (std::numeric_limits<T>::has_infinity)
        return std::numeric_limits<T>::infinity();
    else
        return std::numeric_limits<T>::max();
}

// =============================================================================
// interval<T>
// =============================================================================
//
// An interval records: two endpoints and two bits of openness. An interval
// is *empty* if it contains no points. Two ways that happens:
//
//   lower > upper                    (reversed bounds)
//   lower == upper with either side open   (0 < x < 0 has no solutions)
//
// The default-constructed interval is empty. All empty intervals compare
// equal.

template<endpoint T>
class interval {
public:
    using value_type = T;

    // Default construction yields the canonical empty interval.
    constexpr interval() noexcept = default;

    // General construction. Normalizes to the canonical empty form if the
    // input is degenerate, so that `empty(i)` depends only on whether the
    // set contains any points, not on the particular bits stored.
    constexpr interval(T lower, T upper,
                       bool left_closed = true, bool right_closed = true) noexcept
        : lower_{lower}, upper_{upper},
          left_closed_{left_closed}, right_closed_{right_closed},
          empty_{false}
    {
        if (lower_ > upper_ ||
            (lower_ == upper_ && (!left_closed_ || !right_closed_))) {
            *this = interval{};
        }
    }

    // Named constructors for readability at call sites.
    static constexpr interval closed(T a, T b)     { return interval{a, b, true,  true};  }
    static constexpr interval open(T a, T b)       { return interval{a, b, false, false}; }
    static constexpr interval left_open(T a, T b)  { return interval{a, b, false, true};  }
    static constexpr interval right_open(T a, T b) { return interval{a, b, true,  false}; }
    static constexpr interval point(T v)           { return interval{v, v, true,  true};  }

    // Accessors. Returning std::optional makes the empty case explicit:
    // you cannot silently read an uninitialized bound.
    constexpr std::optional<T> lower_bound() const noexcept {
        return empty_ ? std::nullopt : std::optional<T>{lower_};
    }
    constexpr std::optional<T> upper_bound() const noexcept {
        return empty_ ? std::nullopt : std::optional<T>{upper_};
    }
    constexpr bool left_closed() const noexcept  { return !empty_ && left_closed_;  }
    constexpr bool right_closed() const noexcept { return !empty_ && right_closed_; }
    constexpr bool empty_set() const noexcept    { return empty_; }

    // Equality. Two empty intervals are equal regardless of stored bits.
    // Two non-empty intervals are equal iff bounds and openness agree.
    friend constexpr bool operator==(const interval& a, const interval& b) noexcept {
        if (a.empty_ && b.empty_) return true;
        if (a.empty_ != b.empty_) return false;
        return a.lower_ == b.lower_ && a.upper_ == b.upper_ &&
               a.left_closed_ == b.left_closed_ && a.right_closed_ == b.right_closed_;
    }

private:
    T lower_{};
    T upper_{};
    bool left_closed_{false};
    bool right_closed_{false};
    bool empty_{true};
};

// =============================================================================
// Free-function interface for interval<T>
// =============================================================================

template<endpoint T>
constexpr bool empty(const interval<T>& i) noexcept { return i.empty_set(); }

template<endpoint T>
constexpr bool contains(const interval<T>& i, const T& x) noexcept {
    if (empty(i)) return false;
    const T l = *i.lower_bound();
    const T u = *i.upper_bound();
    const bool above_lower = i.left_closed()  ? (x >= l) : (x > l);
    const bool below_upper = i.right_closed() ? (x <= u) : (x < u);
    return above_lower && below_upper;
}

// Intersection is the one operation that stays within `interval`.
template<endpoint T>
constexpr interval<T> intersect(const interval<T>& a, const interval<T>& b) noexcept {
    if (empty(a) || empty(b)) return {};
    const T al = *a.lower_bound(), au = *a.upper_bound();
    const T bl = *b.lower_bound(), bu = *b.upper_bound();

    // Take the maximum of the two lower bounds (the stricter left).
    T lower; bool left_c;
    if (al > bl)      { lower = al; left_c = a.left_closed(); }
    else if (bl > al) { lower = bl; left_c = b.left_closed(); }
    else              { lower = al; left_c = a.left_closed() && b.left_closed(); }

    // And the minimum of the two upper bounds (the stricter right).
    T upper; bool right_c;
    if (au < bu)      { upper = au; right_c = a.right_closed(); }
    else if (bu < au) { upper = bu; right_c = b.right_closed(); }
    else              { upper = au; right_c = a.right_closed() && b.right_closed(); }

    return interval<T>{lower, upper, left_c, right_c};
}

// =============================================================================
// disjoint_intervals<T>
// =============================================================================
//
// A sorted vector of non-overlapping, non-touching-with-a-gap intervals.
// The class maintains this canonical form as an invariant: every public
// operation returns a value already in canonical form.
//
// Two inputs "touch" if one ends exactly where the next begins. They merge
// only when at least one endpoint is closed, because otherwise the join
// point is excluded from both and there is a genuine gap.

template<endpoint T>
class disjoint_intervals {
public:
    using interval_type = interval<T>;
    using value_type = T;
    using storage_type = std::vector<interval_type>;
    using const_iterator = typename storage_type::const_iterator;

    constexpr disjoint_intervals() = default;

    // Construct from an initializer list of intervals, canonicalized.
    disjoint_intervals(std::initializer_list<interval_type> xs);

    // Construct from an arbitrary range of intervals, canonicalized.
    explicit disjoint_intervals(std::vector<interval_type> xs);

    const_iterator begin() const noexcept { return intervals_.begin(); }
    const_iterator end()   const noexcept { return intervals_.end();   }

    std::size_t size()  const noexcept { return intervals_.size(); }
    bool        empty() const noexcept { return intervals_.empty(); }

    friend constexpr bool operator==(const disjoint_intervals& a,
                                     const disjoint_intervals& b) noexcept {
        return a.intervals_ == b.intervals_;
    }

private:
    storage_type intervals_;
};

// =============================================================================
// canonicalize: the sweep-line merge
// =============================================================================
//
// This is the heart of the library. Take any vector of intervals, return
// its canonical form: sorted by lower endpoint, with overlapping and
// touching-yet-closed intervals merged, with empty intervals dropped.
//
// Two pedagogical traps wait here.
//
// Trap 1: the sort comparator. The natural partial order on intervals is
// CONTAINMENT (A < B iff A ⊆ B). This is not a strict weak ordering, and
// passing it to std::sort is undefined behavior: the sort may loop or
// corrupt memory. We must sort by a proper linear key. Sorting by lower
// endpoint (tie-breaking: closed-left before open-left to make the wider
// interval appear first) is a strict weak order and is what we want.
//
// Trap 2: losing the openness flags on merge. A naive implementation
// produces closed merged output regardless of input. That is wrong:
//
//   (1, 3] ∪ [3, 5)  =  (1, 5)      (merges, touches at 3)
//   (1, 3) ∪ (3, 5)  =  (1, 3) ∪ (3, 5)   (gap at 3, does not merge)
//
// The merged interval inherits the left-openness of whichever operand
// contributed the leftmost endpoint, and the right-openness of whichever
// contributed the rightmost. At ties, closed wins.

template<endpoint T>
std::vector<interval<T>> canonicalize(std::vector<interval<T>> xs) {
    // Drop empty intervals. infimum/supremum do not exist for them and
    // the sweep below assumes non-empty inputs.
    const auto end = std::remove_if(xs.begin(), xs.end(),
        [](const interval<T>& i) { return empty(i); });
    xs.erase(end, xs.end());
    if (xs.empty()) return xs;

    // Sort by lower endpoint; closed-left sorts before open-left at ties.
    // Using std::less<interval<T>>{} would be tempting but fatal.
    std::sort(xs.begin(), xs.end(), [](const interval<T>& a, const interval<T>& b) {
        const T al = *a.lower_bound(), bl = *b.lower_bound();
        if (al != bl) return al < bl;
        return a.left_closed() && !b.left_closed();
    });

    // Sweep. Maintain a running (lower, upper, lc, rc) quadruple that
    // represents the growing merged run. Emit when a new interval cannot
    // extend the run.
    std::vector<interval<T>> out;
    out.reserve(xs.size());

    T     lo = *xs.front().lower_bound();
    bool  lc = xs.front().left_closed();
    T     hi = *xs.front().upper_bound();
    bool  rc = xs.front().right_closed();

    for (std::size_t k = 1; k < xs.size(); ++k) {
        const T    il = *xs[k].lower_bound();
        const T    iu = *xs[k].upper_bound();
        const bool ilc = xs[k].left_closed();
        const bool iuc = xs[k].right_closed();

        // Separated from the current run if the next lower endpoint is
        // strictly past hi, or equal to hi with both sides open there.
        const bool separate = (il > hi) || (il == hi && !rc && !ilc);
        if (separate) {
            out.push_back(interval<T>{lo, hi, lc, rc});
            lo = il; lc = ilc;
            hi = iu; rc = iuc;
            continue;
        }

        // Extend the right side.
        if (iu > hi) {
            hi = iu;
            rc = iuc;
        } else if (iu == hi) {
            rc = rc || iuc;  // closed wins at a tie
        }
        // At a left-endpoint tie, closed wins there too.
        if (il == lo) {
            lc = lc || ilc;
        }
    }
    out.push_back(interval<T>{lo, hi, lc, rc});
    return out;
}

// Constructor bodies (out-of-class so canonicalize is visible).

template<endpoint T>
disjoint_intervals<T>::disjoint_intervals(std::initializer_list<interval<T>> xs)
    : intervals_{canonicalize(std::vector<interval<T>>{xs})} {}

template<endpoint T>
disjoint_intervals<T>::disjoint_intervals(std::vector<interval<T>> xs)
    : intervals_{canonicalize(std::move(xs))} {}

// =============================================================================
// Boolean algebra on disjoint_intervals<T>
// =============================================================================

// Union is concatenate-and-canonicalize.
template<endpoint T>
disjoint_intervals<T> unite(const disjoint_intervals<T>& a,
                            const disjoint_intervals<T>& b) {
    std::vector<interval<T>> merged;
    merged.reserve(a.size() + b.size());
    merged.insert(merged.end(), a.begin(), a.end());
    merged.insert(merged.end(), b.begin(), b.end());
    return disjoint_intervals<T>{std::move(merged)};
}

// Intersection is the O(n+m) linear scan familiar from merge algorithms.
template<endpoint T>
disjoint_intervals<T> intersect(const disjoint_intervals<T>& a,
                                const disjoint_intervals<T>& b) {
    std::vector<interval<T>> out;
    auto i = a.begin(), j = b.begin();
    while (i != a.end() && j != b.end()) {
        auto piece = intersect(*i, *j);
        if (!empty(piece)) out.push_back(piece);
        // Advance whichever ends first.
        if (*i->upper_bound() < *j->upper_bound())       ++i;
        else if (*j->upper_bound() < *i->upper_bound())  ++j;
        else { ++i; ++j; }
    }
    return disjoint_intervals<T>{std::move(out)};
}

// Complement within a universe [lo, hi] (closed by convention at the
// artificial boundary). Each finite internal boundary flips open/closed:
// a closed input endpoint becomes an open complement endpoint, and vice
// versa. This is the endpoint-flip rule the docstring names.
template<endpoint T>
disjoint_intervals<T> complement(const disjoint_intervals<T>& s,
                                 T lo = lowest_endpoint<T>(),
                                 T hi = highest_endpoint<T>()) {
    // We walk left-to-right maintaining a cursor (cur, cur_lc) for the start
    // of the next gap to emit. The universe endpoints lo, hi are treated as
    // closed by convention. At every finite internal boundary the openness
    // flips: the complement is closed where the input was open, and open
    // where the input was closed.
    std::vector<interval<T>> out;
    T    cur    = lo;
    bool cur_lc = true;  // next complement-interval's left-closed flag

    for (const auto& iv : s) {
        const T il = *iv.lower_bound();
        const T iu = *iv.upper_bound();
        if (iu <= cur) continue;    // iv precedes the cursor entirely
        if (il >= hi) break;         // iv is past the universe upper bound

        if (il > cur) {
            // Emit [cur, il] with openness flipped from iv's left side.
            const bool rc = !iv.left_closed();
            out.push_back(interval<T>{cur, il, cur_lc, rc});
        }
        cur    = iu;
        cur_lc = !iv.right_closed();  // flip
        if (cur >= hi) { cur = hi; cur_lc = true; break; }
    }
    if (cur < hi) {
        out.push_back(interval<T>{cur, hi, cur_lc, true});
    }
    return disjoint_intervals<T>{std::move(out)};
}

// Difference: A - B = A ∩ ~B. We define it in terms of complement to
// make the algebraic identity visible in the code.
template<endpoint T>
disjoint_intervals<T> difference(const disjoint_intervals<T>& a,
                                 const disjoint_intervals<T>& b) {
    return intersect(a, complement(b));
}

// Symmetric difference: (A ∪ B) - (A ∩ B), equivalently (A - B) ∪ (B - A).
template<endpoint T>
disjoint_intervals<T> symmetric_difference(const disjoint_intervals<T>& a,
                                           const disjoint_intervals<T>& b) {
    return unite(difference(a, b), difference(b, a));
}

// Point-set containment.
template<endpoint T>
bool contains(const disjoint_intervals<T>& s, const T& x) {
    return std::any_of(s.begin(), s.end(),
        [&](const interval<T>& i) { return contains(i, x); });
}

// Total length covered. For integer T this is a count with endpoints;
// for floating T this is Lebesgue measure. The library is agnostic.
template<endpoint T>
T measure(const disjoint_intervals<T>& s) {
    T total{};
    for (const auto& i : s) {
        total = total + (*i.upper_bound() - *i.lower_bound());
    }
    return total;
}

// =============================================================================
// Operator sugar (optional, for call-site readability)
// =============================================================================
//
// Boolean algebra reads best with symbolic operators. We overload the bitwise
// ones, which is the standard C++ convention for set operations.

template<endpoint T>
disjoint_intervals<T> operator|(const disjoint_intervals<T>& a,
                                const disjoint_intervals<T>& b) {
    return unite(a, b);
}

template<endpoint T>
disjoint_intervals<T> operator&(const disjoint_intervals<T>& a,
                                const disjoint_intervals<T>& b) {
    return intersect(a, b);
}

template<endpoint T>
disjoint_intervals<T> operator~(const disjoint_intervals<T>& s) {
    return complement(s);
}

template<endpoint T>
disjoint_intervals<T> operator-(const disjoint_intervals<T>& a,
                                const disjoint_intervals<T>& b) {
    return difference(a, b);
}

template<endpoint T>
disjoint_intervals<T> operator^(const disjoint_intervals<T>& a,
                                const disjoint_intervals<T>& b) {
    return symmetric_difference(a, b);
}

}  // namespace stepanov
