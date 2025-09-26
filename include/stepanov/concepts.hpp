#pragma once

#include <concepts>
#include <type_traits>
#include <utility>
#include <iterator>

namespace stepanov {

// Fundamental algebraic operations - the building blocks
// These concepts check if the required functions can be found via ADL
template<typename T>
concept has_twice = requires(T a) {
    { twice(a) } -> std::convertible_to<T>;
};

template<typename T>
concept has_half = requires(T a) {
    { half(a) } -> std::convertible_to<T>;
};

template<typename T>
concept has_even = requires(const T& a) {
    { even(a) } -> std::convertible_to<bool>;
};

template<typename T>
concept has_increment = requires(T a) {
    { increment(a) } -> std::convertible_to<T>;
};

template<typename T>
concept has_decrement = requires(T a) {
    { decrement(a) } -> std::convertible_to<T>;
};

// Regular type concept - fundamental for generic programming
template<typename T>
concept regular = std::semiregular<T> && std::equality_comparable<T>;

// Arithmetic operations with proper constraints
template<typename T>
concept additive_group = regular<T> && requires(T a, T b) {
    { a + b } -> std::convertible_to<T>;
    { a - b } -> std::convertible_to<T>;
    { -a } -> std::convertible_to<T>;
    { T(0) } -> std::convertible_to<T>;  // additive identity
};

template<typename T>
concept multiplicative_monoid = regular<T> && requires(T a, T b) {
    { a * b } -> std::convertible_to<T>;
    { T(1) } -> std::convertible_to<T>;  // multiplicative identity
};

template<typename T>
concept ring = additive_group<T> && multiplicative_monoid<T>;

template<typename T>
concept integral_domain = ring<T>;
// Integral domain is a semantic requirement (no zero divisors)
// Cannot be checked syntactically

// Euclidean domain - for GCD algorithms
template<typename T>
concept euclidean_domain = integral_domain<T> && requires(T a, T b) {
    { quotient(a, b) } -> std::convertible_to<T>;
    { remainder(a, b) } -> std::convertible_to<T>;
    { norm(a) } -> std::integral;
};

// Field concept - division ring
template<typename T>
concept field = ring<T> && requires(T a, T b) {
    { a / b } -> std::convertible_to<T>;  // b != 0
};

// Ordered algebraic structures
template<typename T>
concept totally_ordered = regular<T> && std::totally_ordered<T>;

template<typename T>
concept ordered_ring = ring<T> && totally_ordered<T>;

template<typename T>
concept ordered_field = field<T> && totally_ordered<T>;

// Power operation requirements
template<typename T>
concept power_semigroup = requires(T a, T b) {
    { power(a, b) } -> std::convertible_to<T>;
};

// Iterator concepts for generic algorithms
template<typename I>
concept readable = std::input_iterator<I>;

template<typename I>
concept writable = std::output_iterator<I, std::iter_value_t<I>>;

template<typename I>
concept forward_iterator = std::forward_iterator<I>;

template<typename I>
concept bidirectional_iterator = std::bidirectional_iterator<I>;

template<typename I>
concept random_access_iterator = std::random_access_iterator<I>;

// Function object concepts
template<typename F, typename... Args>
concept predicate = std::predicate<F, Args...>;

template<typename F, typename T>
concept unary_function = std::invocable<F, T> && requires(F f, T t) {
    typename std::invoke_result_t<F, T>;
};

template<typename F, typename T>
concept binary_operation = std::invocable<F, T, T> && requires(F f, T a, T b) {
    { f(a, b) } -> std::convertible_to<T>;
};

// Associative binary operation
template<typename F, typename T>
concept associative_operation = binary_operation<F, T> && requires {
    // Semantic requirement: f(a, f(b, c)) == f(f(a, b), c)
    typename F::associative_tag;
};

// Commutative binary operation
template<typename F, typename T>
concept commutative_operation = binary_operation<F, T> && requires {
    // Semantic requirement: f(a, b) == f(b, a)
    typename F::commutative_tag;
};

// Group theory concepts
template<typename G>
concept group = regular<G> && requires(G a, G b) {
    { compose(a, b) } -> std::convertible_to<G>;
    { inverse(a) } -> std::convertible_to<G>;
    { identity(a) } -> std::convertible_to<G>;
};

template<typename G>
concept abelian_group = group<G>;
// Abelian group is a semantic requirement (commutative)
// Cannot be checked syntactically

// Algebraic operation concept for generic algorithms
template<typename T>
concept algebraic =
    has_twice<T> &&
    has_half<T> &&
    has_even<T> &&
    has_increment<T> &&
    has_decrement<T> &&
    regular<T> &&
    requires(T a) {
        { T(0) } -> std::convertible_to<T>;
        { T(1) } -> std::convertible_to<T>;
    };

// Distance and size concepts
template<typename T>
concept has_distance = requires(T a, T b) {
    { distance(a, b) } -> std::integral;
};

template<typename T>
concept has_successor = requires(T a) {
    { successor(a) } -> std::convertible_to<T>;
};

template<typename T>
concept has_predecessor = requires(T a) {
    { predecessor(a) } -> std::convertible_to<T>;
};

// Transformation concepts
template<typename F, typename T>
concept transformation = unary_function<F, T> && requires(F f, T x) {
    { f(x) } -> std::convertible_to<T>;
};

template<typename F, typename T>
concept distance_preserving = transformation<F, T> && requires {
    // Semantic: distance(f(x), f(y)) == distance(x, y)
    typename F::distance_preserving_tag;
};

// Memory and resource concepts
template<typename T>
concept memory_pool = requires(T pool, std::size_t n) {
    { pool.allocate(n) } -> std::convertible_to<void*>;
    { pool.deallocate(std::declval<void*>(), n) } -> std::same_as<void>;
};

// Polynomial concept
template<typename P>
concept polynomial_like = requires(P p, typename P::value_type x, int n) {
    typename P::value_type;
    typename P::degree_type;
    { p[n] } -> std::convertible_to<typename P::value_type>;
    { p(x) } -> std::convertible_to<typename P::value_type>;  // evaluation
    { degree(p) } -> std::convertible_to<typename P::degree_type>;
};

} // namespace stepanov