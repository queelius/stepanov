#pragma once

/**
 * @file mod_int.hpp
 * @brief Modular integers - the foundation of modern cryptography
 *
 * power(g, secret, p) is literally Diffie-Hellman key exchange.
 * power(m, e, n) is RSA encryption.
 * power(base, n, mod) computes polynomial rolling hashes.
 *
 * All of cryptography rests on efficient modular exponentiation.
 */

#include <cstdint>
#include <compare>

namespace peasant::examples {

template<int64_t Mod>
struct mod_int {
    int64_t v;

    constexpr mod_int() : v(0) {}
    constexpr mod_int(int64_t x) : v(((x % Mod) + Mod) % Mod) {}

    constexpr bool operator==(mod_int const&) const = default;
    constexpr auto operator<=>(mod_int const&) const = default;

    constexpr mod_int operator+(mod_int o) const { return mod_int{v + o.v}; }
    constexpr mod_int operator-(mod_int o) const { return mod_int{v - o.v}; }
    constexpr mod_int operator-() const { return mod_int{-v}; }
    constexpr mod_int operator*(mod_int o) const { return mod_int{v * o.v}; }
};

// ADL functions
template<int64_t M> constexpr mod_int<M> zero(mod_int<M>) { return mod_int<M>{0}; }
template<int64_t M> constexpr mod_int<M> one(mod_int<M>)  { return mod_int<M>{1}; }

template<int64_t M> constexpr mod_int<M> twice(mod_int<M> x)     { return x + x; }
template<int64_t M> constexpr mod_int<M> half(mod_int<M> x)      { return mod_int<M>{x.v / 2}; }
template<int64_t M> constexpr bool even(mod_int<M> x)            { return (x.v & 1) == 0; }
template<int64_t M> constexpr mod_int<M> increment(mod_int<M> x) { return mod_int<M>{x.v + 1}; }
template<int64_t M> constexpr mod_int<M> decrement(mod_int<M> x) { return mod_int<M>{x.v - 1}; }

// Common moduli
using mod_1e9_7 = mod_int<1000000007>;  // Common competitive programming prime
using mod_1e9_9 = mod_int<1000000009>;  // Another common prime

} // namespace peasant::examples
