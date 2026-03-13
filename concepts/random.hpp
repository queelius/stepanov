// random.hpp - Distribution and random element concepts
//
// Pure concept definitions for probability distributions. Zero implementation.
// Extracted from alea's RandomElement concept and generalized.
//
// The key algebraic insight: distributions are types that generate samples,
// and composition of distributions preserves distributional structure.

#pragma once

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace stepanov {

// =============================================================================
// URBG concept — Uniform Random Bit Generator
// =============================================================================

/// Mirrors the C++ standard's UniformRandomBitGenerator requirements.
/// Any type satisfying this works with our distributions.
template <typename G>
concept URBG = requires(G& g) {
    typename G::result_type;
    { G::min() } -> std::same_as<typename G::result_type>;
    { G::max() } -> std::same_as<typename G::result_type>;
    { g() } -> std::same_as<typename G::result_type>;
} && std::unsigned_integral<typename G::result_type>;

// =============================================================================
// Distribution concept
// =============================================================================

/// A Distribution can produce random samples from its probability law.
///
/// This is the fundamental concept: a distribution is a type that
/// generates values according to some probability law.
template <typename D>
concept Distribution = requires(const D d) {
    typename D::value_type;
    typename D::scalar_type;
    { d.mean() } -> std::convertible_to<typename D::value_type>;
    { d.variance() } -> std::convertible_to<typename D::value_type>;
};

// =============================================================================
// ContinuousDistribution — has PDF and CDF
// =============================================================================

/// A continuous distribution over the reals with density function.
template <typename D>
concept ContinuousDistribution = Distribution<D> && requires(const D d, typename D::value_type x) {
    { d.pdf(x) } -> std::convertible_to<typename D::scalar_type>;
    { d.cdf(x) } -> std::convertible_to<typename D::scalar_type>;
};

// =============================================================================
// DiscreteDistribution — has PMF
// =============================================================================

/// A discrete distribution with probability mass function.
template <typename D>
concept DiscreteDistribution = Distribution<D> && requires(const D d, typename D::value_type x) {
    { d.pmf(x) } -> std::convertible_to<typename D::scalar_type>;
};

} // namespace stepanov
