#pragma once

#include <vector>
#include <functional>
#include <algorithm>
#include "concepts.hpp"

namespace stepanov {

/**
 * Binary Indexed Tree (Fenwick Tree) - Generic Implementation
 *
 * A data structure that efficiently supports:
 * - Point updates: O(log n)
 * - Prefix queries: O(log n)
 * - Range queries: O(log n)
 *
 * Key insight: Works for any invertible monoid operation, not just addition.
 * This generalizes beyond the typical sum queries to any associative operation
 * with an inverse.
 *
 * Trade-offs vs segment tree:
 * - More memory efficient (n vs 4n)
 * - Simpler implementation
 * - Requires invertible operation (segment tree doesn't)
 * - Cannot handle arbitrary range updates efficiently
 */

template<typename T>
concept invertible_monoid = requires(T a, T b) {
    { a + b } -> std::convertible_to<T>;      // binary operation
    { T() } -> std::convertible_to<T>;        // identity
    { inverse(a, b) } -> std::convertible_to<T>;  // inverse operation: a - b
};

// Generic Fenwick tree for any group operation
template<typename T, typename BinaryOp = std::plus<T>, typename InverseOp = std::minus<T>>
class fenwick_tree {
private:
    std::vector<T> tree;
    size_t n;
    BinaryOp combine;
    InverseOp inverse_op;
    T identity;

    // Get least significant bit
    static constexpr size_t lowbit(size_t x) {
        return x & (-x);
    }

public:
    // Construct from size with identity element
    fenwick_tree(size_t size, T id = T(), BinaryOp op = BinaryOp(), InverseOp inv = InverseOp())
        : tree(size + 1, id), n(size), combine(op), inverse_op(inv), identity(id) {}

    // Construct from initial values
    fenwick_tree(const std::vector<T>& values, T id = T(), BinaryOp op = BinaryOp(), InverseOp inv = InverseOp())
        : tree(values.size() + 1, id), n(values.size()), combine(op), inverse_op(inv), identity(id) {
        for (size_t i = 0; i < n; ++i) {
            update(i, values[i]);
        }
    }

    // Update value at index i by delta
    void update(size_t index, const T& delta) {
        index++;  // 1-indexed internally
        while (index <= n) {
            tree[index] = combine(tree[index], delta);
            index += lowbit(index);
        }
    }

    // Query prefix [0, index]
    T query_prefix(size_t index) const {
        index++;  // 1-indexed internally
        T result = identity;
        while (index > 0) {
            result = combine(result, tree[index]);
            index -= lowbit(index);
        }
        return result;
    }

    // Query range [left, right]
    T query_range(size_t left, size_t right) const {
        if (left == 0) {
            return query_prefix(right);
        }
        return inverse_op(query_prefix(right), query_prefix(left - 1));
    }

    // Get single element (using inverse)
    T get(size_t index) const {
        return query_range(index, index);
    }

    // Set value at index (not just update)
    void set(size_t index, const T& value) {
        T current = get(index);
        T delta = inverse_op(value, current);
        update(index, delta);
    }

    size_t size() const { return n; }
};

// Specialized 2D Fenwick Tree for rectangular queries
template<typename T, typename BinaryOp = std::plus<T>, typename InverseOp = std::minus<T>>
class fenwick_tree_2d {
private:
    std::vector<std::vector<T>> tree;
    size_t rows, cols;
    BinaryOp combine;
    InverseOp inverse_op;
    T identity;

    static constexpr size_t lowbit(size_t x) {
        return x & (-x);
    }

public:
    fenwick_tree_2d(size_t r, size_t c, T id = T(), BinaryOp op = BinaryOp(), InverseOp inv = InverseOp())
        : tree(r + 1, std::vector<T>(c + 1, id)), rows(r), cols(c),
          combine(op), inverse_op(inv), identity(id) {}

    void update(size_t row, size_t col, const T& delta) {
        row++; col++;  // 1-indexed
        for (size_t i = row; i <= rows; i += lowbit(i)) {
            for (size_t j = col; j <= cols; j += lowbit(j)) {
                tree[i][j] = combine(tree[i][j], delta);
            }
        }
    }

    T query_prefix(size_t row, size_t col) const {
        row++; col++;  // 1-indexed
        T result = identity;
        for (size_t i = row; i > 0; i -= lowbit(i)) {
            for (size_t j = col; j > 0; j -= lowbit(j)) {
                result = combine(result, tree[i][j]);
            }
        }
        return result;
    }

    // Query rectangle from (r1, c1) to (r2, c2) inclusive
    T query_rectangle(size_t r1, size_t c1, size_t r2, size_t c2) const {
        T total = query_prefix(r2, c2);
        if (r1 > 0) total = inverse_op(total, query_prefix(r1 - 1, c2));
        if (c1 > 0) total = inverse_op(total, query_prefix(r2, c1 - 1));
        if (r1 > 0 && c1 > 0) total = combine(total, query_prefix(r1 - 1, c1 - 1));
        return total;
    }
};

// Range Fenwick Tree - supports range updates and point queries
template<typename T>
class range_fenwick_tree {
private:
    fenwick_tree<T> tree1;
    fenwick_tree<T> tree2;

public:
    range_fenwick_tree(size_t n) : tree1(n), tree2(n) {}

    // Update range [left, right] by delta
    void update_range(size_t left, size_t right, const T& delta) {
        tree1.update(left, delta);
        if (right + 1 < tree1.size()) {
            tree1.update(right + 1, -delta);
        }
        tree2.update(left, delta * T(left));
        if (right + 1 < tree2.size()) {
            tree2.update(right + 1, -delta * T(right + 1));
        }
    }

    // Query prefix sum [0, index]
    T query_prefix(size_t index) const {
        return tree1.query_prefix(index) * T(index + 1) - tree2.query_prefix(index);
    }

    // Query range sum [left, right]
    T query_range(size_t left, size_t right) const {
        if (left == 0) return query_prefix(right);
        return query_prefix(right) - query_prefix(left - 1);
    }
};

// Order Statistics Tree using Fenwick Tree
// Efficiently find kth smallest element in dynamic multiset
template<typename T>
requires totally_ordered<T>
class order_statistics_tree {
private:
    std::vector<T> values;  // sorted unique values
    fenwick_tree<int> counts;  // count of each value

    size_t value_index(const T& val) const {
        auto it = std::lower_bound(values.begin(), values.end(), val);
        if (it != values.end() && *it == val) {
            return std::distance(values.begin(), it);
        }
        return values.size();  // not found
    }

public:
    order_statistics_tree(const std::vector<T>& unique_values)
        : values(unique_values), counts(unique_values.size()) {
        std::sort(values.begin(), values.end());
    }

    void insert(const T& val) {
        size_t idx = value_index(val);
        if (idx < values.size()) {
            counts.update(idx, 1);
        }
    }

    void erase(const T& val) {
        size_t idx = value_index(val);
        if (idx < values.size() && counts.get(idx) > 0) {
            counts.update(idx, -1);
        }
    }

    // Find kth smallest element (0-indexed)
    T kth_smallest(size_t k) const {
        size_t left = 0, right = values.size() - 1;
        while (left < right) {
            size_t mid = left + (right - left) / 2;
            if (counts.query_prefix(mid) > static_cast<int>(k)) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return values[left];
    }

    // Count elements less than val
    size_t count_less(const T& val) const {
        size_t idx = value_index(val);
        if (idx == 0) return 0;
        return counts.query_prefix(idx - 1);
    }

    // Count elements in range [low, high]
    size_t count_range(const T& low, const T& high) const {
        size_t low_idx = value_index(low);
        size_t high_idx = value_index(high);
        if (low_idx > high_idx) return 0;
        return counts.query_range(low_idx, high_idx);
    }
};

// Fenwick tree with lazy propagation for range updates
template<typename T>
class lazy_fenwick_tree {
private:
    struct node {
        T value;
        T lazy;
        node() : value(T()), lazy(T()) {}
    };

    std::vector<node> tree;
    size_t n;

    void push(size_t idx, size_t len) {
        if (tree[idx].lazy != T()) {
            tree[idx].value += tree[idx].lazy * T(len);
            if (2 * idx + 1 < tree.size()) {
                tree[2 * idx].lazy += tree[idx].lazy;
                tree[2 * idx + 1].lazy += tree[idx].lazy;
            }
            tree[idx].lazy = T();
        }
    }

public:
    lazy_fenwick_tree(size_t size) : tree(4 * size), n(size) {}

    void update_range(size_t left, size_t right, const T& delta) {
        update_range_impl(1, 0, n - 1, left, right, delta);
    }

    T query_range(size_t left, size_t right) {
        return query_range_impl(1, 0, n - 1, left, right);
    }

private:
    void update_range_impl(size_t idx, size_t l, size_t r, size_t ql, size_t qr, const T& delta) {
        push(idx, r - l + 1);
        if (ql > r || qr < l) return;
        if (ql <= l && r <= qr) {
            tree[idx].lazy += delta;
            push(idx, r - l + 1);
            return;
        }
        size_t mid = l + (r - l) / 2;
        update_range_impl(2 * idx, l, mid, ql, qr, delta);
        update_range_impl(2 * idx + 1, mid + 1, r, ql, qr, delta);
        tree[idx].value = tree[2 * idx].value + tree[2 * idx + 1].value;
    }

    T query_range_impl(size_t idx, size_t l, size_t r, size_t ql, size_t qr) {
        if (ql > r || qr < l) return T();
        push(idx, r - l + 1);
        if (ql <= l && r <= qr) return tree[idx].value;
        size_t mid = l + (r - l) / 2;
        return query_range_impl(2 * idx, l, mid, ql, qr) +
               query_range_impl(2 * idx + 1, mid + 1, r, ql, qr);
    }
};

} // namespace stepanov