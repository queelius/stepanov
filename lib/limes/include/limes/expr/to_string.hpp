#pragma once

#include <string>
#include <sstream>
#include "nodes/binary.hpp"

namespace limes::expr {

// Free function to_string for any expression node
// Uses ADL to call the node's to_string() method
template<typename E>
    requires is_expr_node_v<E>
[[nodiscard]] std::string to_string(E const& expr) {
    return expr.to_string();
}

// Pretty print an expression (indented format)
namespace detail {

template<typename E>
void pretty_print_impl(std::ostream& os, E const& expr, int indent) {
    std::string padding(indent * 2, ' ');
    os << padding << expr.to_string();
}

} // namespace detail

template<typename E>
    requires is_expr_node_v<E>
[[nodiscard]] std::string pretty_print(E const& expr) {
    std::ostringstream oss;
    detail::pretty_print_impl(oss, expr, 0);
    return oss.str();
}

// Stream output operator
template<typename E>
    requires is_expr_node_v<E>
std::ostream& operator<<(std::ostream& os, E const& expr) {
    return os << expr.to_string();
}

} // namespace limes::expr
