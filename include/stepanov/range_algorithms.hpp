#pragma once

#include <concepts>
#include <utility>
#include <functional>
#include <algorithm>
#include <numeric>
#include <random>
#include <execution>
#include <optional>
#include <cmath>
#include "iterators.hpp"
#include "iterators_enhanced.hpp"
#include "ranges.hpp"
#include "ranges_enhanced.hpp"
#include "concepts.hpp"

namespace stepanov {

// ================ Fundamental Range Algorithms (EoP Style) ================

// Find mismatch - returns iterators to first differing positions
template<input_iterator I0, sentinel_for<I0> S0,
         input_iterator I1, sentinel_for<I1> S1,
         typename BinaryPred = std::equal_to<>>
constexpr std::pair<I0, I1> find_mismatch(I0 first0, S0 last0,
                                          I1 first1, S1 last1,
                                          BinaryPred pred = BinaryPred{}) {
    while (first0 != last0 && first1 != last1 && pred(*first0, *first1)) {
        ++first0;
        ++first1;
    }
    return {first0, first1};
}

// Find adjacent mismatch - first non-matching adjacent pair
template<iterator_forward I, sentinel_for<I> S, typename BinaryPred = std::equal_to<>>
constexpr I find_adjacent_mismatch(I first, S last, BinaryPred pred = BinaryPred{}) {
    if (first == last) return first;

    I next = first;
    ++next;
    while (next != last) {
        if (!pred(*first, *next)) {
            return first;
        }
        first = next;
        ++next;
    }
    return next;
}

// Check if relation preserving over range
template<input_iterator I, sentinel_for<I> S, typename Relation>
    requires std::predicate<Relation, std::iter_value_t<I>, std::iter_value_t<I>>
constexpr bool relation_preserving(I first, S last, Relation rel) {
    if (first == last) return true;

    I prev = first;
    ++first;
    while (first != last) {
        if (!rel(*prev, *first)) {
            return false;
        }
        prev = first;
        ++first;
    }
    return true;
}

// Check if strictly increasing
template<input_iterator I, sentinel_for<I> S, typename Compare = std::less<>>
constexpr bool strictly_increasing_range(I first, S last, Compare comp = Compare{}) {
    return relation_preserving(first, last, comp);
}

// Check if weakly increasing (non-decreasing)
template<input_iterator I, sentinel_for<I> S, typename Compare = std::less<>>
constexpr bool weakly_increasing_range(I first, S last, Compare comp = Compare{}) {
    return relation_preserving(first, last, [&comp](const auto& a, const auto& b) {
        return !comp(b, a);
    });
}

// Partition point - find boundary in partitioned range
template<iterator_forward I, sentinel_for<I> S, typename UnaryPred>
constexpr I partition_point(I first, S last, UnaryPred pred) {
    if constexpr (iterator_random_access<I> && sized_sentinel_for<S, I>) {
        auto len = distance(first, last);

        while (len > 0) {
            auto half = len / 2;
            I mid = next(first, half);
            if (pred(*mid)) {
                first = next(mid);
                len = len - half - 1;
            } else {
                len = half;
            }
        }
        return first;
    } else {
        // Linear search for forward iterators
        while (first != last && pred(*first)) {
            ++first;
        }
        return first;
    }
}

// Rotate - rearranges elements
template<iterator_forward I, sentinel_for<I> S>
constexpr I rotate(I first, I middle, S last) {
    if (first == middle) return first;
    if (middle == last) return first;

    if constexpr (iterator_bidirectional<I>) {
        reverse_iterator rFirst(last);
        reverse_iterator rMiddle(middle);
        reverse_iterator rLast(first);

        // Three reverses algorithm
        while (rFirst != rMiddle && rMiddle != rLast) {
            iter_swap(rFirst++, --rMiddle);
        }

        if (rFirst == rMiddle) {
            while (rMiddle != rLast) {
                iter_swap(--rMiddle, --rLast);
            }
            return I(rLast.base());
        } else {
            while (rFirst != rMiddle) {
                iter_swap(rFirst++, --rLast);
            }
            return I(rMiddle.base());
        }
    } else {
        // Forward iterator version - cycle leader algorithm
        I write = first;
        I next_read = first;

        for (I read = middle; read != last; ++write, ++read) {
            if (write == next_read) next_read = read;
            iter_swap(write, read);
        }

        rotate(write, next_read, last);
        return write;
    }
}

// Random shuffle using Knuth shuffle
template<iterator_random_access I, typename URBG>
constexpr void random_shuffle(I first, I last, URBG&& g) {
    using diff_t = difference_type<I>;
    diff_t n = distance(first, last);

    for (diff_t i = n - 1; i > 0; --i) {
        std::uniform_int_distribution<diff_t> dist(0, i);
        iter_swap(first + i, first + dist(g));
    }
}

// Is heap - verify heap property
template<iterator_random_access I, sentinel_for<I> S, typename Compare = std::less<>>
constexpr bool is_heap(I first, S last, Compare comp = Compare{}) {
    auto n = distance(first, last);
    decltype(n) parent = 0;

    for (decltype(n) child = 1; child < n; ++child) {
        if (comp(first[parent], first[child])) {
            return false;
        }
        if ((child & 1) == 0) {
            ++parent;
        }
    }
    return true;
}

// Make heap - construct heap from range
template<iterator_random_access I, sentinel_for<I> S, typename Compare = std::less<>>
constexpr void make_heap(I first, S last, Compare comp = Compare{}) {
    auto n = distance(first, last);
    if (n < 2) return;

    auto parent = (n - 2) / 2;

    while (true) {
        // Sift down
        auto node = parent;
        auto child = 2 * node + 1;

        while (child < n) {
            if (child + 1 < n && comp(first[child], first[child + 1])) {
                ++child;
            }
            if (comp(first[node], first[child])) {
                iter_swap(first + node, first + child);
                node = child;
                child = 2 * node + 1;
            } else {
                break;
            }
        }

        if (parent == 0) break;
        --parent;
    }
}

// Push heap - add element to heap
template<iterator_random_access I, sentinel_for<I> S, typename Compare = std::less<>>
constexpr void push_heap(I first, S last, Compare comp = Compare{}) {
    auto n = distance(first, last);
    if (n < 2) return;

    auto child = n - 1;
    auto parent = (child - 1) / 2;

    while (child > 0 && comp(first[parent], first[child])) {
        iter_swap(first + parent, first + child);
        child = parent;
        parent = (child - 1) / 2;
    }
}

// Pop heap - remove max element
template<iterator_random_access I, sentinel_for<I> S, typename Compare = std::less<>>
constexpr void pop_heap(I first, S last, Compare comp = Compare{}) {
    auto n = distance(first, last);
    if (n < 2) return;

    iter_swap(first, first + n - 1);
    --n;

    // Sift down
    decltype(n) node = 0;
    decltype(n) child = 1;

    while (child < n) {
        if (child + 1 < n && comp(first[child], first[child + 1])) {
            ++child;
        }
        if (comp(first[node], first[child])) {
            iter_swap(first + node, first + child);
            node = child;
            child = 2 * node + 1;
        } else {
            break;
        }
    }
}

// Sort heap - sort heap into ascending order
template<iterator_random_access I, sentinel_for<I> S, typename Compare = std::less<>>
constexpr void sort_heap(I first, S last, Compare comp = Compare{}) {
    while (distance(first, last) > 1) {
        pop_heap(first, last, comp);
        --last;
    }
}

// ================ Numeric Range Algorithms ================

// Accumulate with binary operation
template<input_iterator I, sentinel_for<I> S, typename T, typename BinaryOp = std::plus<>>
constexpr T accumulate(I first, S last, T init, BinaryOp op = BinaryOp{}) {
    for (; first != last; ++first) {
        init = op(std::move(init), *first);
    }
    return init;
}

// Inner product
template<input_iterator I1, sentinel_for<I1> S1,
         input_iterator I2,
         typename T,
         typename BinaryOp1 = std::plus<>,
         typename BinaryOp2 = std::multiplies<>>
constexpr T inner_product(I1 first1, S1 last1, I2 first2, T init,
                          BinaryOp1 op1 = BinaryOp1{},
                          BinaryOp2 op2 = BinaryOp2{}) {
    for (; first1 != last1; ++first1, ++first2) {
        init = op1(std::move(init), op2(*first1, *first2));
    }
    return init;
}

// Adjacent difference
template<input_iterator I, sentinel_for<I> S, typename O, typename BinaryOp = std::minus<>>
    requires writable_iterator<O, std::iter_value_t<I>>
constexpr O adjacent_difference(I first, S last, O result, BinaryOp op = BinaryOp{}) {
    if (first == last) return result;

    auto val = *first;
    *result = val;
    ++result;
    ++first;

    while (first != last) {
        auto tmp = *first;
        *result = op(tmp, val);
        val = std::move(tmp);
        ++result;
        ++first;
    }
    return result;
}

// Partial sum / Inclusive scan
template<input_iterator I, sentinel_for<I> S, typename O, typename BinaryOp = std::plus<>>
    requires writable_iterator<O, std::iter_value_t<I>>
constexpr O partial_sum(I first, S last, O result, BinaryOp op = BinaryOp{}) {
    if (first == last) return result;

    auto sum = *first;
    *result = sum;
    ++result;
    ++first;

    while (first != last) {
        sum = op(std::move(sum), *first);
        *result = sum;
        ++result;
        ++first;
    }
    return result;
}

// Exclusive scan
template<input_iterator I, sentinel_for<I> S, typename O,
         typename T, typename BinaryOp = std::plus<>>
    requires writable_iterator<O, T>
constexpr O exclusive_scan(I first, S last, O result, T init, BinaryOp op = BinaryOp{}) {
    while (first != last) {
        auto val = init;
        init = op(std::move(init), *first);
        *result = std::move(val);
        ++first;
        ++result;
    }
    return result;
}

// Inclusive scan
template<input_iterator I, sentinel_for<I> S, typename O, typename BinaryOp = std::plus<>>
    requires writable_iterator<O, std::iter_value_t<I>>
constexpr O inclusive_scan(I first, S last, O result, BinaryOp op = BinaryOp{}) {
    return partial_sum(first, last, result, op);
}

// Transform reduce - map-reduce pattern
template<input_iterator I, sentinel_for<I> S,
         typename T,
         typename BinaryOp = std::plus<>,
         typename UnaryOp = std::identity>
constexpr T transform_reduce(I first, S last, T init,
                            BinaryOp binary_op = BinaryOp{},
                            UnaryOp unary_op = UnaryOp{}) {
    for (; first != last; ++first) {
        init = binary_op(std::move(init), unary_op(*first));
    }
    return init;
}

// Reduce with binary operation
template<input_iterator I, sentinel_for<I> S,
         typename T = std::iter_value_t<I>,
         typename BinaryOp = std::plus<>>
constexpr T reduce(I first, S last, T init = T{}, BinaryOp op = BinaryOp{}) {
    return accumulate(first, last, std::move(init), op);
}

// ================ Partitioned Range Operations ================

// Partition - rearrange elements based on predicate
template<iterator_forward I, sentinel_for<I> S, typename UnaryPred>
constexpr I partition(I first, S last, UnaryPred pred) {
    first = std::find_if_not(first, last, pred);
    if (first == last) return first;

    for (I i = next(first); i != last; ++i) {
        if (pred(*i)) {
            iter_swap(i, first);
            ++first;
        }
    }
    return first;
}

// Stable partition - maintains relative order
template<iterator_bidirectional I, sentinel_for<I> S, typename UnaryPred>
constexpr I stable_partition(I first, S last, UnaryPred pred) {
    if (first == last) return first;

    if constexpr (iterator_random_access<I>) {
        auto n = distance(first, last);
        if (n == 1) {
            return pred(*first) ? last : first;
        }

        I mid = first;
        advance(mid, n / 2);

        I left_end = stable_partition(first, mid, pred);
        I right_begin = stable_partition(mid, last, pred);

        return rotate(left_end, mid, right_begin);
    } else {
        // Simpler version for bidirectional iterators
        I write = first;
        for (I read = first; read != last; ++read) {
            if (pred(*read)) {
                if (write != read) {
                    iter_swap(write, read);
                }
                ++write;
            }
        }
        return write;
    }
}

// Is partitioned - check if range is partitioned
template<input_iterator I, sentinel_for<I> S, typename UnaryPred>
constexpr bool is_partitioned(I first, S last, UnaryPred pred) {
    for (; first != last; ++first) {
        if (!pred(*first)) {
            break;
        }
    }
    for (; first != last; ++first) {
        if (pred(*first)) {
            return false;
        }
    }
    return true;
}

// ================ Set Operations on Sorted Ranges ================

// Set union
template<input_iterator I1, sentinel_for<I1> S1,
         input_iterator I2, sentinel_for<I2> S2,
         typename O, typename Compare = std::less<>>
    requires writable_iterator<O, std::iter_value_t<I1>> &&
             writable_iterator<O, std::iter_value_t<I2>>
constexpr O set_union(I1 first1, S1 last1, I2 first2, S2 last2,
                     O result, Compare comp = Compare{}) {
    while (first1 != last1 && first2 != last2) {
        if (comp(*first1, *first2)) {
            *result++ = *first1++;
        } else if (comp(*first2, *first1)) {
            *result++ = *first2++;
        } else {
            *result++ = *first1++;
            ++first2;
        }
    }
    return std::copy(first2, last2, std::copy(first1, last1, result));
}

// Set intersection
template<input_iterator I1, sentinel_for<I1> S1,
         input_iterator I2, sentinel_for<I2> S2,
         typename O, typename Compare = std::less<>>
    requires writable_iterator<O, std::iter_value_t<I1>>
constexpr O set_intersection(I1 first1, S1 last1, I2 first2, S2 last2,
                            O result, Compare comp = Compare{}) {
    while (first1 != last1 && first2 != last2) {
        if (comp(*first1, *first2)) {
            ++first1;
        } else if (comp(*first2, *first1)) {
            ++first2;
        } else {
            *result++ = *first1++;
            ++first2;
        }
    }
    return result;
}

// Set difference
template<input_iterator I1, sentinel_for<I1> S1,
         input_iterator I2, sentinel_for<I2> S2,
         typename O, typename Compare = std::less<>>
    requires writable_iterator<O, std::iter_value_t<I1>>
constexpr O set_difference(I1 first1, S1 last1, I2 first2, S2 last2,
                          O result, Compare comp = Compare{}) {
    while (first1 != last1 && first2 != last2) {
        if (comp(*first1, *first2)) {
            *result++ = *first1++;
        } else if (comp(*first2, *first1)) {
            ++first2;
        } else {
            ++first1;
            ++first2;
        }
    }
    return std::copy(first1, last1, result);
}

// Set symmetric difference
template<input_iterator I1, sentinel_for<I1> S1,
         input_iterator I2, sentinel_for<I2> S2,
         typename O, typename Compare = std::less<>>
    requires writable_iterator<O, std::iter_value_t<I1>> &&
             writable_iterator<O, std::iter_value_t<I2>>
constexpr O set_symmetric_difference(I1 first1, S1 last1, I2 first2, S2 last2,
                                    O result, Compare comp = Compare{}) {
    while (first1 != last1 && first2 != last2) {
        if (comp(*first1, *first2)) {
            *result++ = *first1++;
        } else if (comp(*first2, *first1)) {
            *result++ = *first2++;
        } else {
            ++first1;
            ++first2;
        }
    }
    return std::copy(first2, last2, std::copy(first1, last1, result));
}

// ================ Sampling and Selection ================

// Sample - random sampling from range
template<input_iterator I, sentinel_for<I> S, typename O, typename URBG>
    requires writable_iterator<O, std::iter_value_t<I>>
O sample(I first, S last, O out, std::iter_difference_t<O> n, URBG&& g) {
    using dist_t = std::uniform_int_distribution<std::iter_difference_t<I>>;
    using param_t = typename dist_t::param_type;

    dist_t dist{};
    std::iter_difference_t<I> sample_size = 0;

    while (first != last && sample_size < n) {
        out[sample_size++] = *first++;
    }

    for (std::iter_difference_t<I> pop_size = sample_size;
         first != last; ++first, ++pop_size) {
        const auto k = dist(g, param_t{0, pop_size});
        if (k < n) {
            out[k] = *first;
        }
    }

    return out + sample_size;
}

// Nth element - partial sorting
template<iterator_random_access I, sentinel_for<I> S, typename Compare = std::less<>>
constexpr void nth_element(I first, I nth, S last, Compare comp = Compare{}) {
    if (first == last || nth == last) return;

    while (distance(first, last) > 1) {
        // Partition around pivot
        I pivot = first;
        I left = first + 1;
        I right = last - 1;

        while (left <= right) {
            while (left <= right && !comp(*pivot, *left)) {
                ++left;
            }
            while (left <= right && comp(*pivot, *right)) {
                --right;
            }
            if (left < right) {
                iter_swap(left, right);
                ++left;
                --right;
            }
        }

        iter_swap(first, right);

        if (right == nth) {
            return;
        } else if (nth < right) {
            last = right;
        } else {
            first = right + 1;
        }
    }
}

// ================ Permutation Operations ================

// Next permutation
template<iterator_bidirectional I, sentinel_for<I> S, typename Compare = std::less<>>
constexpr bool next_permutation(I first, S last, Compare comp = Compare{}) {
    if (first == last) return false;
    I i = last;
    if (first == --i) return false;

    while (true) {
        I i1 = i;
        if (comp(*--i, *i1)) {
            I i2 = last;
            while (!comp(*i, *--i2));
            iter_swap(i, i2);
            std::reverse(i1, last);
            return true;
        }
        if (i == first) {
            std::reverse(first, last);
            return false;
        }
    }
}

// Previous permutation
template<iterator_bidirectional I, sentinel_for<I> S, typename Compare = std::less<>>
constexpr bool prev_permutation(I first, S last, Compare comp = Compare{}) {
    if (first == last) return false;
    I i = last;
    if (first == --i) return false;

    while (true) {
        I i1 = i;
        if (comp(*i1, *--i)) {
            I i2 = last;
            while (!comp(*--i2, *i));
            iter_swap(i, i2);
            std::reverse(i1, last);
            return true;
        }
        if (i == first) {
            std::reverse(first, last);
            return false;
        }
    }
}

// Is permutation - check if ranges are permutations
template<iterator_forward I1, sentinel_for<I1> S1,
         iterator_forward I2, sentinel_for<I2> S2,
         typename BinaryPred = std::equal_to<>>
constexpr bool is_permutation(I1 first1, S1 last1, I2 first2, S2 last2,
                              BinaryPred pred = BinaryPred{}) {
    // First, check if they have the same length
    auto [end1, end2] = find_mismatch(first1, last1, first2, last2, pred);
    if (end1 == last1 && end2 == last2) {
        return true;
    }
    if (end1 == last1 || end2 == last2) {
        return false;
    }

    // Check remaining elements
    for (I1 it1 = end1; it1 != last1; ++it1) {
        // Count occurrences in first range
        auto count1 = std::count_if(first1, last1,
            [&](const auto& x) { return pred(x, *it1); });

        // Count occurrences in second range
        auto count2 = std::count_if(first2, last2,
            [&](const auto& x) { return pred(x, *it1); });

        if (count1 != count2) {
            return false;
        }
    }

    return true;
}

// ================ Advanced Searching ================

// Boyer-Moore-Horspool search
template<iterator_forward I1, sentinel_for<I1> S1,
         iterator_forward I2, sentinel_for<I2> S2>
constexpr I1 search_bmh(I1 first1, S1 last1, I2 first2, S2 last2) {
    auto pattern_length = distance(first2, last2);
    if (pattern_length == 0) return first1;
    if (pattern_length == 1) return std::find(first1, last1, *first2);

    // Build bad character table
    constexpr std::size_t table_size = 256;
    std::array<decltype(pattern_length), table_size> bad_char{};
    bad_char.fill(pattern_length);

    auto it = first2;
    for (decltype(pattern_length) i = 0; i < pattern_length - 1; ++i, ++it) {
        bad_char[static_cast<unsigned char>(*it)] = pattern_length - 1 - i;
    }

    // Search
    while (distance(first1, last1) >= pattern_length) {
        auto text_it = first1;
        auto pattern_it = first2;

        advance(text_it, pattern_length - 1);
        advance(pattern_it, pattern_length - 1);

        while (*text_it == *pattern_it) {
            if (pattern_it == first2) {
                return first1;
            }
            --text_it;
            --pattern_it;
        }

        advance(first1, bad_char[static_cast<unsigned char>(*text_it)]);
    }

    return last1;
}

// ================ Mathematical Range Operations ================

// GCD of range
template<input_iterator I, sentinel_for<I> S>
    requires euclidean_domain<std::iter_value_t<I>>
constexpr auto gcd_range(I first, S last) {
    if (first == last) return std::iter_value_t<I>{0};

    auto result = *first;
    ++first;

    while (first != last) {
        result = gcd(result, *first);
        if (result == std::iter_value_t<I>{1}) {
            break;  // Early termination for GCD = 1
        }
        ++first;
    }

    return result;
}

// LCM of range
template<input_iterator I, sentinel_for<I> S>
    requires euclidean_domain<std::iter_value_t<I>>
constexpr auto lcm_range(I first, S last) {
    if (first == last) return std::iter_value_t<I>{0};

    auto result = *first;
    ++first;

    while (first != last) {
        auto g = gcd(result, *first);
        result = (result / g) * (*first);  // Avoid overflow
        ++first;
    }

    return result;
}

// Product of range
template<input_iterator I, sentinel_for<I> S,
         typename T = std::iter_value_t<I>>
constexpr T product(I first, S last, T init = T{1}) {
    return accumulate(first, last, init, std::multiplies<>{});
}

// ================ Statistical Operations ================

// Mean
template<input_iterator I, sentinel_for<I> S>
    requires std::is_arithmetic_v<std::iter_value_t<I>>
constexpr double mean(I first, S last) {
    double sum = 0.0;
    std::size_t count = 0;

    for (; first != last; ++first) {
        sum += *first;
        ++count;
    }

    return count > 0 ? sum / count : 0.0;
}

// Variance
template<input_iterator I, sentinel_for<I> S>
    requires std::is_arithmetic_v<std::iter_value_t<I>>
constexpr double variance(I first, S last) {
    auto m = mean(first, last);
    double sum_sq_diff = 0.0;
    std::size_t count = 0;

    for (; first != last; ++first) {
        double diff = *first - m;
        sum_sq_diff += diff * diff;
        ++count;
    }

    return count > 1 ? sum_sq_diff / (count - 1) : 0.0;
}

// Standard deviation
template<input_iterator I, sentinel_for<I> S>
    requires std::is_arithmetic_v<std::iter_value_t<I>>
constexpr double stddev(I first, S last) {
    return std::sqrt(variance(first, last));
}

} // namespace stepanov