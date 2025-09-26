#pragma once

#include <type_traits>
#include <concepts>

namespace stepanov {

// Operations for built-in integer types
template<std::integral T>
inline T twice(T x) { return x * 2; }

template<std::integral T>
inline T half(T x) { return x / 2; }

template<std::integral T>
inline bool even(T x) { return (x & 1) == 0; }

template<std::integral T>
inline T increment(T x) { return x + 1; }

template<std::integral T>
inline T decrement(T x) { return x - 1; }

// Operations for floating point types
template<std::floating_point T>
inline T twice(T x) { return x * 2.0; }

template<std::floating_point T>
inline T half(T x) { return x / 2.0; }

template<std::floating_point T>
inline bool even(T x) {
    // For floats, check if the integer part is even
    return even(static_cast<long long>(x));
}

template<std::floating_point T>
inline T increment(T x) { return x + 1.0; }

template<std::floating_point T>
inline T decrement(T x) { return x - 1.0; }

// Generic iterator-based sum for ranges
template<typename Iterator, typename T>
T sum(Iterator first, Iterator last, T init) {
    T result = init;
    while (first != last) {
        result = result + *first;
        ++first;
    }
    return result;
}

// Generic iterator-based product for ranges
template<typename Iterator, typename T>
T product(Iterator first, Iterator last, T init) {
    T result = init;
    while (first != last) {
        result = result * *first;
        ++first;
    }
    return result;
}

// Specialized power for integers with integer exponents
template<typename T, typename I>
    requires std::integral<T> && std::integral<I>
T power(T base, I exp) {
    T result = 1;
    while (exp > 0) {
        if (exp & 1)
            result *= base;
        base *= base;
        exp >>= 1;
    }
    return result;
}

// power_mod is defined in math.hpp

// Generic multiply_accumulate
template<typename T>
T multiply_accumulate(T a, T b, T acc) {
    while (b != 0) {
        if (b & 1)
            acc += a;
        a *= 2;
        b /= 2;
    }
    return acc;
}

// Generic power_accumulate
template<typename T, typename I>
    requires std::integral<I>
T power_accumulate(T base, I exp, T result) {
    while (exp != 0) {
        if (exp & 1)
            result = result * base;
        base = base * base;
        exp >>= 1;
    }
    return result;
}

} // namespace stepanov