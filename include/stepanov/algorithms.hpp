#pragma once

#include <concepts>
#include <iterator>
#include <functional>
#include <optional>
#include <utility>
#include <vector>
#include <span>
#include "concepts.hpp"

namespace stepanov {

/**
 * Generic algorithms following Alex Stepanov's principles
 *
 * These algorithms are designed to be:
 * - Maximally general (work with any type satisfying minimal requirements)
 * - Efficient (no unnecessary operations)
 * - Composable (can be combined to build more complex algorithms)
 * - Mathematically grounded (based on algebraic structures)
 */

// =============================================================================
// Orbit and cycle detection algorithms
// =============================================================================

/**
 * Apply transformation f repeatedly n times
 * power(x, n, f) computes f^n(x)
 */
template <typename T, typename F, typename N>
    requires transformation<F, T> && std::integral<N>
constexpr T power(T x, N n, F f) {
    while (n != 0) {
        x = f(x);
        --n;
    }
    return x;
}

/**
 * Find the distance from x to y under transformation f
 * Returns nullopt if y is not reachable from x
 */
template <typename T, typename F>
    requires transformation<F, T> && regular<T>
constexpr std::optional<size_t> distance(T x, T y, F f, size_t limit = 1000000) {
    size_t n = 0;
    while (x != y && n < limit) {
        x = f(x);
        ++n;
    }
    return (x == y) ? std::optional(n) : std::nullopt;
}

/**
 * Collision point - find where two orbits meet
 * Floyd's tortoise and hare algorithm
 */
template <typename T>
struct collision_point_result {
    T collision;
    bool has_collision;
    size_t steps_slow;
    size_t steps_fast;
};

template <typename T, typename F>
    requires transformation<F, T> && regular<T>
constexpr collision_point_result<T> collision_point(T x, F f) {
    T slow = x;
    T fast = x;
    size_t steps_slow = 0;
    size_t steps_fast = 0;

    do {
        slow = f(slow);
        ++steps_slow;
        fast = f(f(fast));
        steps_fast += 2;
    } while (slow != fast);

    return {slow, true, steps_slow, steps_fast};
}

/**
 * Cycle detection - find cycle length and tail length
 * Brent's algorithm (more efficient than Floyd's)
 */
template <typename T>
struct cycle_info {
    bool has_cycle;
    size_t cycle_length;
    size_t tail_length;
    T cycle_start;
};

template <typename T, typename F>
    requires transformation<F, T> && regular<T>
constexpr cycle_info<T> detect_cycle(T x, F f) {
    // Brent's algorithm
    size_t power_of_2 = 1;
    size_t length = 1;
    T tortoise = x;
    T hare = f(x);

    // Find cycle length
    while (tortoise != hare) {
        if (power_of_2 == length) {
            tortoise = hare;
            power_of_2 *= 2;
            length = 0;
        }
        hare = f(hare);
        ++length;
    }

    // Find tail length
    T mu = x;
    T nu = x;
    for (size_t i = 0; i < length; ++i) {
        nu = f(nu);
    }

    size_t tail = 0;
    while (mu != nu) {
        mu = f(mu);
        nu = f(nu);
        ++tail;
    }

    return {true, length, tail, mu};
}

// =============================================================================
// Accumulation and reduction algorithms
// =============================================================================

/**
 * Generic accumulate with binary operation
 * More general than std::accumulate - works with any binary operation
 */
template <typename InputIt, typename T, typename BinaryOp>
    requires std::input_iterator<InputIt> &&
             binary_operation<BinaryOp, T>
constexpr T accumulate(InputIt first, InputIt last, T init, BinaryOp op) {
    while (first != last) {
        init = op(init, *first);
        ++first;
    }
    return init;
}

/**
 * Accumulate with early termination predicate
 */
template <typename InputIt, typename T, typename BinaryOp, typename Pred>
    requires std::input_iterator<InputIt> &&
             binary_operation<BinaryOp, T> &&
             predicate<Pred, T>
constexpr T accumulate_until(InputIt first, InputIt last, T init,
                             BinaryOp op, Pred should_stop) {
    while (first != last && !should_stop(init)) {
        init = op(init, *first);
        ++first;
    }
    return init;
}

/**
 * Parallel reduce for associative operations
 * Can be parallelized efficiently
 */
template <typename RandomIt, typename BinaryOp>
    requires std::random_access_iterator<RandomIt> &&
             associative_operation<BinaryOp, std::iter_value_t<RandomIt>>
constexpr auto reduce(RandomIt first, RandomIt last, BinaryOp op)
    -> std::iter_value_t<RandomIt>
{
    using T = std::iter_value_t<RandomIt>;

    auto n = std::distance(first, last);
    if (n == 0) return T{};
    if (n == 1) return *first;

    // Divide and conquer approach
    auto mid = first + n / 2;
    return op(reduce(first, mid, op), reduce(mid, last, op));
}

// =============================================================================
// Partitioning algorithms
// =============================================================================

/**
 * Stable partition with minimal moves
 * Returns iterator to partition point
 */
template <typename BidirIt, typename Pred>
    requires std::bidirectional_iterator<BidirIt> &&
             predicate<Pred, std::iter_value_t<BidirIt>>
constexpr BidirIt stable_partition(BidirIt first, BidirIt last, Pred p) {
    auto n = std::distance(first, last);
    if (n == 0) return first;
    if (n == 1) return p(*first) ? last : first;

    auto mid = first;
    std::advance(mid, n / 2);

    return std::rotate(stable_partition(first, mid, p),
                      mid,
                      stable_partition(mid, last, p));
}

/**
 * Three-way partition (Dutch national flag problem)
 * Partitions into: less than pivot, equal to pivot, greater than pivot
 */
template <typename RandomIt>
struct partition_3way_result {
    RandomIt less_end;
    RandomIt equal_end;
};

template <typename RandomIt, typename T>
    requires std::random_access_iterator<RandomIt> &&
             totally_ordered<std::iter_value_t<RandomIt>> &&
             std::same_as<T, std::iter_value_t<RandomIt>>
constexpr partition_3way_result<RandomIt>
partition_3way(RandomIt first, RandomIt last, const T& pivot) {
    RandomIt less = first;
    RandomIt equal = first;
    RandomIt greater = last;

    while (equal != greater) {
        if (*equal < pivot) {
            std::iter_swap(less, equal);
            ++less;
            ++equal;
        } else if (*equal == pivot) {
            ++equal;
        } else {
            --greater;
            std::iter_swap(equal, greater);
        }
    }

    return {less, equal};
}

// =============================================================================
// Permutation algorithms
// =============================================================================

/**
 * Apply permutation defined by indices
 * Cycles through permutation in O(n) time with O(1) extra space
 */
template <typename RandomIt, typename IndexIt>
    requires std::random_access_iterator<RandomIt> &&
             std::random_access_iterator<IndexIt> &&
             std::integral<std::iter_value_t<IndexIt>>
constexpr void apply_permutation(RandomIt first, RandomIt last,
                                 IndexIt index_first) {
    using diff_t = typename std::iterator_traits<RandomIt>::difference_type;
    diff_t n = std::distance(first, last);

    for (diff_t i = 0; i < n; ++i) {
        auto current = i;
        while (index_first[current] != i) {
            auto next = index_first[current];
            std::iter_swap(first + current, first + next);
            index_first[current] = current;
            current = next;
        }
        index_first[current] = current;
    }
}

/**
 * Rotate by GCD algorithm (most efficient)
 * Rotates [first, last) by n positions to the left
 */
template <typename ForwardIt>
    requires forward_iterator<ForwardIt>
constexpr ForwardIt rotate_left(ForwardIt first, ForwardIt last, size_t n) {
    auto distance = std::distance(first, last);
    if (distance == 0 || n % distance == 0) return first;

    n = n % distance;
    auto new_first = first;
    std::advance(new_first, n);

    // GCD algorithm
    auto gcd_val = binary_gcd(n, distance - n);
    for (size_t i = 0; i < gcd_val; ++i) {
        auto temp = std::move(*(first + i));
        auto j = i;
        auto k = j + n;

        while (k != i) {
            *(first + j) = std::move(*(first + k));
            j = k;
            k = (j + n) % distance;
        }
        *(first + j) = std::move(temp);
    }

    return new_first;
}

// =============================================================================
// Search algorithms
// =============================================================================

/**
 * Binary search with transformation
 * Finds first element where f(element) >= value
 */
template <typename ForwardIt, typename T, typename F>
    requires forward_iterator<ForwardIt> &&
             transformation<F, std::iter_value_t<ForwardIt>> &&
             totally_ordered<T>
constexpr ForwardIt binary_search_transformed(ForwardIt first, ForwardIt last,
                                             const T& value, F f) {
    while (first != last) {
        auto mid = first;
        std::advance(mid, std::distance(first, last) / 2);

        if (f(*mid) < value) {
            first = ++mid;
        } else {
            last = mid;
        }
    }
    return first;
}

/**
 * Exponential search - useful when the target is near the beginning
 */
template <typename ForwardIt, typename T>
    requires forward_iterator<ForwardIt> &&
             totally_ordered<std::iter_value_t<ForwardIt>>
constexpr ForwardIt exponential_search(ForwardIt first, ForwardIt last,
                                       const T& value) {
    if (first == last) return last;

    auto bound = first;
    size_t jump = 1;

    while (bound != last && *bound < value) {
        auto prev = bound;
        size_t step = std::min(jump,
                              static_cast<size_t>(std::distance(bound, last)));
        std::advance(bound, step);
        if (bound == last || *bound >= value) {
            return std::lower_bound(prev, bound, value);
        }
        jump *= 2;
    }

    return bound;
}

// =============================================================================
// Merge algorithms
// =============================================================================

/**
 * K-way merge using tournament tree (heap)
 * Merges k sorted ranges into output
 */
template <typename OutputIt, typename... InputRanges>
    requires std::output_iterator<OutputIt, std::common_type_t<
        typename std::ranges::range_value_t<InputRanges>...>>
constexpr OutputIt merge_k_way(OutputIt result, InputRanges&&... ranges) {
    using value_type = std::common_type_t<
        typename std::ranges::range_value_t<InputRanges>...>;

    struct source {
        value_type value;
        size_t index;
        bool operator>(const source& other) const {
            return value > other.value;
        }
    };

    std::vector<source> heap;
    std::vector<std::ranges::iterator_t<InputRanges>...> iterators;
    std::vector<std::ranges::sentinel_t<InputRanges>...> ends;

    // Initialize heap with first element from each range
    size_t idx = 0;
    ((void)(
        iterators.push_back(std::ranges::begin(ranges)),
        ends.push_back(std::ranges::end(ranges)),
        (iterators[idx] != ends[idx] ?
            (heap.push_back({*iterators[idx], idx}), ++iterators[idx]) : void()),
        ++idx
    ), ...);

    std::make_heap(heap.begin(), heap.end(), std::greater<>{});

    // Merge
    while (!heap.empty()) {
        std::pop_heap(heap.begin(), heap.end(), std::greater<>{});
        auto [value, index] = heap.back();
        heap.pop_back();

        *result++ = value;

        if (iterators[index] != ends[index]) {
            heap.push_back({*iterators[index], index});
            ++iterators[index];
            std::push_heap(heap.begin(), heap.end(), std::greater<>{});
        }
    }

    return result;
}

// =============================================================================
// Functional composition
// =============================================================================

/**
 * Function composition: compose(f, g)(x) = f(g(x))
 */
template <typename F, typename G>
class composed_function {
private:
    F f;
    G g;

public:
    constexpr composed_function(F f, G g) : f(f), g(g) {}

    template <typename T>
    constexpr auto operator()(T&& x) const {
        return f(g(std::forward<T>(x)));
    }
};

template <typename F, typename G>
constexpr auto compose(F f, G g) {
    return composed_function<F, G>(f, g);
}

/**
 * Iterate a function n times and collect results
 */
template <typename T, typename F, typename N>
    requires transformation<F, T> && std::integral<N>
constexpr std::vector<T> orbit(T x, N n, F f) {
    std::vector<T> result;
    result.reserve(n + 1);
    result.push_back(x);

    for (N i = 0; i < n; ++i) {
        x = f(x);
        result.push_back(x);
    }

    return result;
}

} // namespace stepanov