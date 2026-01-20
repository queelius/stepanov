#pragma once

/**
 * @file tropical.hpp
 * @brief Tropical semiring - shortest paths via matrix power!
 *
 * In the tropical semiring:
 *   "addition" is min (or max)
 *   "multiplication" is +
 *
 * Mind-blowing fact: If A is an adjacency matrix with edge weights,
 * then A^n in tropical semiring gives n-hop shortest paths!
 *
 * power(A, n) computes shortest paths using at most n edges.
 * power(A, V-1) gives all-pairs shortest paths (Bellman-Ford via algebra).
 */

#include <limits>
#include <algorithm>
#include <compare>

namespace peasant::examples {

// A tropical number: "infinity" represents "no path"
struct tropical {
    double v;

    static constexpr double inf = std::numeric_limits<double>::infinity();

    constexpr tropical() : v(inf) {}
    constexpr tropical(double x) : v(x) {}

    constexpr bool operator==(tropical const&) const = default;
    constexpr auto operator<=>(tropical const&) const = default;

    // Tropical "addition" is min
    constexpr tropical operator+(tropical o) const {
        return tropical{std::min(v, o.v)};
    }

    // Tropical "multiplication" is +
    constexpr tropical operator*(tropical o) const {
        return tropical{v + o.v};
    }

    constexpr tropical operator-() const { return *this; }
};

// ADL functions
constexpr tropical zero(tropical) { return tropical{tropical::inf}; }  // min identity
constexpr tropical one(tropical)  { return tropical{0.0}; }            // + identity

constexpr tropical twice(tropical x)     { return x * x; }  // In tropical: x + x = 2x
constexpr tropical half(tropical x)      { return tropical{x.v / 2}; }
constexpr bool even(tropical)            { return true; }   // Force multiply path
constexpr tropical increment(tropical x) { return x; }
constexpr tropical decrement(tropical x) { return x; }

// A simple NxN tropical matrix for shortest paths
template<size_t N>
struct trop_matrix {
    tropical data[N][N];

    constexpr bool operator==(trop_matrix const&) const = default;
    constexpr auto operator<=>(trop_matrix const& o) const {
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j)
                if (auto c = data[i][j] <=> o.data[i][j]; c != 0) return c;
        return std::strong_ordering::equal;
    }

    // Tropical matrix multiplication: (A*B)[i][j] = min_k(A[i][k] + B[k][j])
    constexpr trop_matrix operator*(trop_matrix const& o) const {
        trop_matrix result;
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                tropical sum = zero(tropical{});
                for (size_t k = 0; k < N; ++k) {
                    sum = sum + (data[i][k] * o.data[k][j]);
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    constexpr trop_matrix operator+(trop_matrix const& o) const { return *this * o; }
    constexpr trop_matrix operator-() const { return *this; }
};

// Identity tropical matrix
template<size_t N>
constexpr trop_matrix<N> trop_identity() {
    trop_matrix<N> result;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result.data[i][j] = (i == j) ? tropical{0} : tropical{};
        }
    }
    return result;
}

// ADL for tropical matrices
template<size_t N> constexpr trop_matrix<N> zero(trop_matrix<N>) { return trop_identity<N>(); }
template<size_t N> constexpr trop_matrix<N> one(trop_matrix<N>)  { return trop_identity<N>(); }

template<size_t N> constexpr trop_matrix<N> twice(trop_matrix<N> const& m) { return m * m; }
template<size_t N> constexpr trop_matrix<N> half(trop_matrix<N> const& m)  { return m; }
template<size_t N> constexpr bool even(trop_matrix<N> const&) { return true; }
template<size_t N> constexpr trop_matrix<N> increment(trop_matrix<N> const& m) { return m; }
template<size_t N> constexpr trop_matrix<N> decrement(trop_matrix<N> const& m) { return m; }

} // namespace peasant::examples
