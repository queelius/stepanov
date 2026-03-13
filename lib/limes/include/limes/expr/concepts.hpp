#pragma once

#include <concepts>
#include <span>
#include <cstddef>

namespace limes::expr::concepts {

// ExprNode concept: types that represent expression tree nodes
// Each node has a compile-time arity and can be evaluated with arguments
template<typename E, typename T>
concept ExprNode = requires(E const& expr, std::span<T const> args) {
    { E::arity_v } -> std::convertible_to<std::size_t>;
    { expr.evaluate(args) } -> std::convertible_to<T>;
};

// DifferentiableNode: nodes that can compute their derivative with respect to a dimension
template<typename E, typename T, std::size_t Dim>
concept DifferentiableNode = ExprNode<E, T> && requires(E const& expr) {
    { expr.template derivative<Dim>() };
};

// StringableNode: nodes that can produce a string representation
template<typename E>
concept StringableNode = requires(E const& expr) {
    { expr.to_string() } -> std::convertible_to<std::string>;
};

} // namespace limes::expr::concepts
