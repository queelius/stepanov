#pragma once

/**
 * @file bool_matrix.hpp
 * @brief Boolean matrices - graph reachability in exactly n steps
 *
 * A boolean matrix A represents a directed graph: A[i][j] = 1 iff edge i→j exists.
 *
 * Matrix multiplication with OR-of-ANDs:
 *   (A*B)[i][j] = OR_k (A[i][k] AND B[k][j])
 *
 * Mind-blowing insight:
 *   A^n[i][j] = 1 iff there's a path of EXACTLY n edges from i to j
 *
 * power(A, n) computes n-step reachability in O(N³ log n).
 */

#include <array>
#include <bitset>
#include <cstddef>
#include <compare>

namespace peasant::examples {

template<size_t N>
struct bool_matrix {
    std::array<std::bitset<N>, N> rows;

    constexpr bool_matrix() {
        for (auto& row : rows) row.reset();
    }

    bool operator==(bool_matrix const& o) const {
        for (size_t i = 0; i < N; ++i) {
            if (rows[i] != o.rows[i]) return false;
        }
        return true;
    }

    auto operator<=>(bool_matrix const& o) const {
        for (size_t i = 0; i < N; ++i) {
            if (rows[i].to_ulong() < o.rows[i].to_ulong()) return std::strong_ordering::less;
            if (rows[i].to_ulong() > o.rows[i].to_ulong()) return std::strong_ordering::greater;
        }
        return std::strong_ordering::equal;
    }

    bool get(size_t i, size_t j) const { return rows[i][j]; }
    void set(size_t i, size_t j, bool v = true) { rows[i][j] = v; }

    // Boolean matrix multiplication: OR of ANDs
    bool_matrix operator*(bool_matrix const& o) const {
        bool_matrix result;
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                bool any = false;
                for (size_t k = 0; k < N; ++k) {
                    if (rows[i][k] && o.rows[k][j]) {
                        any = true;
                        break;
                    }
                }
                result.rows[i][j] = any;
            }
        }
        return result;
    }

    // Required for algebraic concept
    bool_matrix operator+(bool_matrix const& o) const { return *this * o; }
    bool_matrix operator-() const { return *this; }
};

// Identity: I[i][j] = (i == j)
template<size_t N>
bool_matrix<N> bool_identity() {
    bool_matrix<N> result;
    for (size_t i = 0; i < N; ++i) {
        result.rows[i][i] = true;
    }
    return result;
}

// ADL functions
template<size_t N> bool_matrix<N> zero(bool_matrix<N>) { return bool_identity<N>(); }
template<size_t N> bool_matrix<N> one(bool_matrix<N>)  { return bool_identity<N>(); }

template<size_t N> bool_matrix<N> twice(bool_matrix<N> const& m) { return m * m; }
template<size_t N> bool_matrix<N> half(bool_matrix<N> const& m)  { return m; }
template<size_t N> bool even(bool_matrix<N> const&)              { return true; }

template<size_t N> bool_matrix<N> increment(bool_matrix<N> const& m) { return m; }
template<size_t N> bool_matrix<N> decrement(bool_matrix<N> const& m) { return m; }

// Factory: Create adjacency matrix from edge list
template<size_t N>
bool_matrix<N> from_edges(std::initializer_list<std::pair<size_t, size_t>> edges) {
    bool_matrix<N> result;
    for (auto [i, j] : edges) {
        result.set(i, j);
    }
    return result;
}

} // namespace peasant::examples
