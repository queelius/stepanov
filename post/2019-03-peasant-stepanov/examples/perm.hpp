#pragma once

/**
 * @file perm.hpp
 * @brief Permutations - apply σⁿ in O(log n) compositions
 *
 * A permutation σ composed with itself n times: σⁿ
 * Useful for: cycle detection, group theory, cryptographic shuffles.
 *
 * power(shuffle, 1000000) applies a shuffle a million times instantly.
 */

#include <array>
#include <cstddef>
#include <compare>

namespace peasant::examples {

template<size_t N>
struct perm {
    std::array<size_t, N> p;  // p[i] = where i goes

    constexpr bool operator==(perm const&) const = default;
    constexpr auto operator<=>(perm const&) const = default;

    // Composition: (σ ∘ τ)(i) = σ(τ(i))
    constexpr perm operator*(perm const& other) const {
        perm result;
        for (size_t i = 0; i < N; ++i) {
            result.p[i] = p[other.p[i]];
        }
        return result;
    }

    // For the algebraic requirements (not meaningful for permutations)
    constexpr perm operator+(perm const& other) const { return *this * other; }
    constexpr perm operator-() const { return inverse(); }

    // Inverse permutation
    constexpr perm inverse() const {
        perm result;
        for (size_t i = 0; i < N; ++i) {
            result.p[p[i]] = i;
        }
        return result;
    }

    // Apply to a value
    constexpr size_t operator()(size_t i) const { return p[i]; }
};

// Identity permutation
template<size_t N>
constexpr perm<N> identity_perm() {
    perm<N> result;
    for (size_t i = 0; i < N; ++i) result.p[i] = i;
    return result;
}

// ADL functions
template<size_t N> constexpr perm<N> zero(perm<N>) { return identity_perm<N>(); }
template<size_t N> constexpr perm<N> one(perm<N>)  { return identity_perm<N>(); }

template<size_t N> constexpr perm<N> twice(perm<N> const& p) { return p * p; }
template<size_t N> constexpr perm<N> half(perm<N> const& p)  { return p; }  // Not meaningful
template<size_t N> constexpr bool even(perm<N> const&) { return true; }     // Force multiply path

template<size_t N> constexpr perm<N> increment(perm<N> const& p) { return p; }
template<size_t N> constexpr perm<N> decrement(perm<N> const& p) { return p; }

// Example: a simple rotation
template<size_t N>
constexpr perm<N> rotate_perm() {
    perm<N> result;
    for (size_t i = 0; i < N; ++i) {
        result.p[i] = (i + 1) % N;
    }
    return result;
}

} // namespace peasant::examples
