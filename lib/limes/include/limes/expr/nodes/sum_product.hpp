#pragma once

#include <span>
#include <string>
#include <cstddef>
#include <vector>
#include <sstream>
#include "binary.hpp"

namespace limes::expr {

namespace detail {

// Build an extended argument vector with a placeholder inserted at IndexDim
template<typename T>
std::vector<T> make_extended_args(std::span<T const> args, std::size_t index_dim) {
    std::vector<T> ext_args(args.begin(), args.end());
    if (ext_args.size() <= index_dim) {
        ext_args.resize(index_dim + 1);
    } else {
        ext_args.insert(ext_args.begin() + static_cast<std::ptrdiff_t>(index_dim), T(0));
    }
    return ext_args;
}

} // namespace detail

// FiniteSum<Expr, IndexDim>: Sum over integer index, i.e. sum(i=lo..hi) body(x, i)
template<typename Expr, std::size_t IndexDim>
struct FiniteSum {
    using value_type = typename Expr::value_type;
    using body_type = Expr;

    static constexpr std::size_t arity_v =
        (Expr::arity_v > IndexDim) ? (Expr::arity_v - 1) : Expr::arity_v;

    Expr body;
    int lo;
    int hi;

    constexpr FiniteSum(Expr b, int l, int h) noexcept : body{b}, lo{l}, hi{h} {}

    [[nodiscard]] constexpr value_type eval(std::span<value_type const> args) const {
        value_type sum = value_type(0);
        auto ext_args = detail::make_extended_args(args, IndexDim);

        for (int i = lo; i <= hi; ++i) {
            ext_args[IndexDim] = static_cast<value_type>(i);
            sum += body.eval(std::span<value_type const>(ext_args));
        }
        return sum;
    }

    [[nodiscard]] [[deprecated("use eval() instead")]]
    constexpr value_type evaluate(std::span<value_type const> args) const {
        return eval(args);
    }

    // d/dx[sum f(x,i)] = sum df/dx(x,i)
    template<std::size_t Dim>
    [[nodiscard]] constexpr auto derivative() const {
        constexpr std::size_t AdjustedDim = (Dim >= IndexDim) ? (Dim + 1) : Dim;
        auto dbody = body.template derivative<AdjustedDim>();
        return FiniteSum<decltype(dbody), IndexDim>{dbody, lo, hi};
    }

    [[nodiscard]] std::string to_string() const {
        std::ostringstream oss;
        oss << "(sum[i" << IndexDim << "=" << lo << ".." << hi << "] " << body.to_string() << ")";
        return oss.str();
    }
};

// FiniteProduct<Expr, IndexDim>: Product over integer index, i.e. prod(i=lo..hi) body(x, i)
template<typename Expr, std::size_t IndexDim>
struct FiniteProduct {
    using value_type = typename Expr::value_type;
    using body_type = Expr;

    static constexpr std::size_t arity_v =
        (Expr::arity_v > IndexDim) ? (Expr::arity_v - 1) : Expr::arity_v;

    Expr body;
    int lo;
    int hi;

    constexpr FiniteProduct(Expr b, int l, int h) noexcept : body{b}, lo{l}, hi{h} {}

    [[nodiscard]] constexpr value_type eval(std::span<value_type const> args) const {
        value_type prod = value_type(1);
        auto ext_args = detail::make_extended_args(args, IndexDim);

        for (int i = lo; i <= hi; ++i) {
            ext_args[IndexDim] = static_cast<value_type>(i);
            prod *= body.eval(std::span<value_type const>(ext_args));
        }
        return prod;
    }

    [[nodiscard]] [[deprecated("use eval() instead")]]
    constexpr value_type evaluate(std::span<value_type const> args) const {
        return eval(args);
    }

    // Logarithmic derivative: d/dx[prod f_i] = (prod f_i) * sum(f_i'/f_i)
    template<std::size_t Dim>
    [[nodiscard]] constexpr auto derivative() const {
        constexpr std::size_t AdjustedDim = (Dim >= IndexDim) ? (Dim + 1) : Dim;
        auto dbody = body.template derivative<AdjustedDim>();
        auto term = dbody / body;
        auto sum_of_terms = FiniteSum<decltype(term), IndexDim>{term, lo, hi};
        return (*this) * sum_of_terms;
    }

    [[nodiscard]] std::string to_string() const {
        std::ostringstream oss;
        oss << "(prod[i" << IndexDim << "=" << lo << ".." << hi << "] " << body.to_string() << ")";
        return oss.str();
    }
};

// Type traits

template<typename T>
struct is_finite_sum : std::false_type {};

template<typename E, std::size_t I>
struct is_finite_sum<FiniteSum<E, I>> : std::true_type {};

template<typename T>
inline constexpr bool is_finite_sum_v = is_finite_sum<T>::value;

template<typename T>
struct is_finite_product : std::false_type {};

template<typename E, std::size_t I>
struct is_finite_product<FiniteProduct<E, I>> : std::true_type {};

template<typename T>
inline constexpr bool is_finite_product_v = is_finite_product<T>::value;

// Factory functions

template<std::size_t IndexDim, typename E>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto sum(E body, int lo, int hi) {
    return FiniteSum<E, IndexDim>{body, lo, hi};
}

template<std::size_t IndexDim, typename E>
    requires is_expr_node_v<E>
[[nodiscard]] constexpr auto product(E body, int lo, int hi) {
    return FiniteProduct<E, IndexDim>{body, lo, hi};
}

} // namespace limes::expr
