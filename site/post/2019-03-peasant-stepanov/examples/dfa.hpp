#pragma once

/**
 * @file dfa.hpp
 * @brief DFA transitions - fast-forward an automaton 10^6 steps
 *
 * A DFA transition function δ maps states to states.
 * Composition: (δ₁ ∘ δ₂)(s) = δ₁(δ₂(s))
 *
 * This forms a monoid under composition!
 *
 * Mind-blowing use case:
 *   power(transition, 1000000) computes the millionth state in O(log n)
 *
 * Applications:
 * - Regex matching: precompute character class transitions
 * - Game theory: compute n-move outcomes
 * - Any finite state machine fast-forwarding
 */

#include <array>
#include <cstddef>
#include <compare>

namespace peasant::examples {

template<size_t NumStates>
struct dfa_transition {
    std::array<size_t, NumStates> next;  // next[s] = state after transition

    constexpr bool operator==(dfa_transition const&) const = default;
    constexpr auto operator<=>(dfa_transition const&) const = default;

    // Composition: (f * g)(s) = f(g(s))
    constexpr dfa_transition operator*(dfa_transition const& g) const {
        dfa_transition result;
        for (size_t s = 0; s < NumStates; ++s) {
            result.next[s] = next[g.next[s]];
        }
        return result;
    }

    // Apply to a state
    constexpr size_t operator()(size_t state) const {
        return next[state];
    }

    // Required for algebraic concept
    constexpr dfa_transition operator+(dfa_transition const& o) const { return *this * o; }
    constexpr dfa_transition operator-() const { return *this; }
};

// Identity transition: every state maps to itself
template<size_t N>
constexpr dfa_transition<N> identity_transition() {
    dfa_transition<N> result;
    for (size_t i = 0; i < N; ++i) {
        result.next[i] = i;
    }
    return result;
}

// ADL functions
template<size_t N> constexpr dfa_transition<N> zero(dfa_transition<N>) { return identity_transition<N>(); }
template<size_t N> constexpr dfa_transition<N> one(dfa_transition<N>)  { return identity_transition<N>(); }

template<size_t N> constexpr dfa_transition<N> twice(dfa_transition<N> const& t) { return t * t; }
template<size_t N> constexpr dfa_transition<N> half(dfa_transition<N> const& t)  { return t; }
template<size_t N> constexpr bool even(dfa_transition<N> const&)                 { return true; }

template<size_t N> constexpr dfa_transition<N> increment(dfa_transition<N> const& t) { return t; }
template<size_t N> constexpr dfa_transition<N> decrement(dfa_transition<N> const& t) { return t; }

// Factory: Simple cyclic transition (state i → (i+1) mod N)
template<size_t N>
constexpr dfa_transition<N> cyclic_transition() {
    dfa_transition<N> result;
    for (size_t i = 0; i < N; ++i) {
        result.next[i] = (i + 1) % N;
    }
    return result;
}

// Factory: Create from array
template<size_t N>
constexpr dfa_transition<N> make_transition(std::array<size_t, N> next) {
    dfa_transition<N> result;
    result.next = next;
    return result;
}

} // namespace peasant::examples
