#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace limes::expr {

// Forward declarations for node types
template<typename T> struct Const;
template<std::size_t N, typename T> struct Var;
template<typename Op, typename L, typename R> struct Binary;
template<typename Op, typename E> struct Unary;
template<typename Tag, typename E> struct UnaryFunc;
template<typename E, std::size_t Dim, typename Lo, typename Hi> struct Integral;
template<typename E, std::size_t Dim, typename BoundValue> struct BoundExpr;
template<typename T> struct ConstBound;
template<typename E> struct ExprBound;

// =============================================================================
// variable_set: Compile-time bitset of dimensions an expression depends on
// =============================================================================

/// Primary template - default to 0 (no dependencies).
/// Covers Const<T>, ConstBound<T>, and any unknown types.
template<typename E, typename = void>
struct variable_set {
    static constexpr std::uint64_t value = 0;
};

/// Var<N, T>: Variable N depends on dimension N
template<std::size_t N, typename T>
struct variable_set<Var<N, T>> {
    static constexpr std::uint64_t value = (1ULL << N);
};

/// Binary<Op, L, R>: Union of both children's dependencies
template<typename Op, typename L, typename R>
struct variable_set<Binary<Op, L, R>> {
    static constexpr std::uint64_t value = variable_set<L>::value | variable_set<R>::value;
};

/// Unary<Op, E>: Same dependencies as child
template<typename Op, typename E>
struct variable_set<Unary<Op, E>> {
    static constexpr std::uint64_t value = variable_set<E>::value;
};

/// UnaryFunc<Tag, E>: Same dependencies as child
template<typename Tag, typename E>
struct variable_set<UnaryFunc<Tag, E>> {
    static constexpr std::uint64_t value = variable_set<E>::value;
};

/// ExprBound<E>: Same dependencies as the expression
template<typename E>
struct variable_set<ExprBound<E>> {
    static constexpr std::uint64_t value = variable_set<E>::value;
};

/// Integral<E, Dim, Lo, Hi>: Remove integrated dimension, add bound dependencies
/// The integrand's dependency on Dim is consumed; outer variables and bounds remain
template<typename E, std::size_t Dim, typename Lo, typename Hi>
struct variable_set<Integral<E, Dim, Lo, Hi>> {
    // Remove the integrated dimension from the integrand's dependencies
    // Add any dependencies from variable bounds
    static constexpr std::uint64_t integrand_deps = variable_set<E>::value & ~(1ULL << Dim);
    static constexpr std::uint64_t bound_deps = variable_set<Lo>::value | variable_set<Hi>::value;
    static constexpr std::uint64_t value = integrand_deps | bound_deps;
};

/// BoundExpr: Remove the bound dimension, keep remaining dependencies
template<typename E, std::size_t Dim, typename BoundValue>
struct variable_set<BoundExpr<E, Dim, BoundValue>> {
    // Remove the bound dimension from the expression's dependencies
    static constexpr std::uint64_t value = variable_set<E>::value & ~(1ULL << Dim);
};

// =============================================================================
// Convenience aliases and helper functions
// =============================================================================

/// Helper variable template for cleaner syntax
template<typename E>
inline constexpr std::uint64_t variable_set_v = variable_set<E>::value;

/// Check if an expression depends on a specific dimension
template<typename E, std::size_t Dim>
inline constexpr bool depends_on_v = (variable_set<E>::value & (1ULL << Dim)) != 0;

/// Check if an expression depends on any of the given dimensions (mask)
template<typename E, std::uint64_t Mask>
inline constexpr bool depends_on_any_v = (variable_set<E>::value & Mask) != 0;

/// Check if an expression depends on all of the given dimensions (mask)
template<typename E, std::uint64_t Mask>
inline constexpr bool depends_on_all_v = (variable_set<E>::value & Mask) == Mask;

/// Check if an expression is constant (depends on no variables)
template<typename E>
inline constexpr bool is_constant_v = (variable_set<E>::value == 0);

/// Count the number of dimensions an expression depends on
template<typename E>
inline constexpr std::size_t dependency_count_v = []() constexpr {
    std::uint64_t bits = variable_set<E>::value;
    std::size_t count = 0;
    while (bits) {
        count += bits & 1;
        bits >>= 1;
    }
    return count;
}();

/// Get the maximum dimension index an expression depends on (0 if constant)
template<typename E>
inline constexpr std::size_t max_dimension_v = []() constexpr {
    std::uint64_t bits = variable_set<E>::value;
    if (bits == 0) return std::size_t{0};
    std::size_t max_dim = 0;
    for (std::size_t i = 0; i < 64; ++i) {
        if (bits & (1ULL << i)) {
            max_dim = i;
        }
    }
    return max_dim;
}();

// =============================================================================
// Separability Detection
// =============================================================================

// Forward declarations for operation tags (defined in nodes/binary.hpp)
struct Add;
struct Sub;
struct Mul;

namespace detail {

/// Helper to check if an expression depends only on certain dimensions
template<typename E, std::uint64_t AllowedMask>
inline constexpr bool depends_only_on_v = (variable_set<E>::value & ~AllowedMask) == 0;

/// Primary template: not separable by default
template<typename E, std::size_t D1, std::size_t D2, typename = void>
struct is_separable_impl : std::false_type {};

/// Const<T>: Always separable (trivially, as g(x)*1 or 1*h(y))
template<typename T, std::size_t D1, std::size_t D2>
struct is_separable_impl<Const<T>, D1, D2> : std::true_type {};

/// Var<N, T>: Separable only if it depends on exactly one of D1 or D2
/// (then it can be written as var*1 or 1*var)
template<std::size_t N, typename T, std::size_t D1, std::size_t D2>
struct is_separable_impl<Var<N, T>, D1, D2>
    : std::bool_constant<(N == D1) || (N == D2)> {};

/// Binary<Mul, L, R>: Separable if L depends only on D1 and R depends only on D2
/// (or vice versa)
template<typename L, typename R, std::size_t D1, std::size_t D2>
struct is_separable_impl<Binary<Mul, L, R>, D1, D2> {
    static constexpr std::uint64_t mask_d1 = 1ULL << D1;
    static constexpr std::uint64_t mask_d2 = 1ULL << D2;

    // Check if L depends only on D1 and R depends only on D2
    static constexpr bool lr_separable =
        depends_only_on_v<L, mask_d1> && depends_only_on_v<R, mask_d2>;

    // Check if L depends only on D2 and R depends only on D1
    static constexpr bool rl_separable =
        depends_only_on_v<L, mask_d2> && depends_only_on_v<R, mask_d1>;

    static constexpr bool value = lr_separable || rl_separable;
};

/// Binary<Add/Sub, L, R>: Separable if both L and R are separable
/// (this is a conservative check - true separability requires more analysis)
template<typename L, typename R, std::size_t D1, std::size_t D2>
struct is_separable_impl<Binary<Add, L, R>, D1, D2>
    : std::bool_constant<
        is_separable_impl<L, D1, D2>::value &&
        is_separable_impl<R, D1, D2>::value
    > {};

template<typename L, typename R, std::size_t D1, std::size_t D2>
struct is_separable_impl<Binary<Sub, L, R>, D1, D2>
    : std::bool_constant<
        is_separable_impl<L, D1, D2>::value &&
        is_separable_impl<R, D1, D2>::value
    > {};

/// Single-child expressions (Unary, UnaryFunc) are separable if they
/// depend on at most one of D1 or D2
template<typename E, std::size_t D1, std::size_t D2>
inline constexpr bool single_child_separable_v =
    depends_only_on_v<E, (1ULL << D1)> || depends_only_on_v<E, (1ULL << D2)>;

template<typename Op, typename E, std::size_t D1, std::size_t D2>
struct is_separable_impl<Unary<Op, E>, D1, D2>
    : std::bool_constant<single_child_separable_v<Unary<Op, E>, D1, D2>> {};

template<typename Tag, typename E, std::size_t D1, std::size_t D2>
struct is_separable_impl<UnaryFunc<Tag, E>, D1, D2>
    : std::bool_constant<single_child_separable_v<UnaryFunc<Tag, E>, D1, D2>> {};

} // namespace detail

/// Check if an expression f(x, y) is separable as g(x) * h(y)
/// with respect to dimensions D1 and D2
template<typename E, std::size_t D1, std::size_t D2>
struct is_separable : detail::is_separable_impl<E, D1, D2> {};

template<typename E, std::size_t D1, std::size_t D2>
inline constexpr bool is_separable_v = is_separable<E, D1, D2>::value;

// =============================================================================
// Separation: Extract g and h from f(x,y) = g(x) * h(y)
// =============================================================================

namespace detail {

/// Extract factor from a multiplicative expression
/// Returns the factor that depends only on the specified dimension
template<typename E, std::size_t TargetDim, std::size_t OtherDim>
struct extract_factor;

/// For Const: return the constant (as factor 1 for the non-matching dimension)
template<typename T, std::size_t TargetDim, std::size_t OtherDim>
struct extract_factor<Const<T>, TargetDim, OtherDim> {
    using type = Const<T>;
    static constexpr auto extract(Const<T> const& c) { return c; }
};

/// For Var: return if it matches TargetDim, otherwise return Const<T>{1}
template<std::size_t N, typename T, std::size_t TargetDim, std::size_t OtherDim>
struct extract_factor<Var<N, T>, TargetDim, OtherDim> {
    using type = std::conditional_t<(N == TargetDim), Var<N, T>, Const<T>>;
    static constexpr auto extract(Var<N, T> const& v) {
        if constexpr (N == TargetDim) {
            return v;
        } else {
            return Const<T>{T(1)};
        }
    }
};

/// For Binary<Mul, L, R>: extract the appropriate factor
template<typename L, typename R, std::size_t TargetDim, std::size_t OtherDim>
struct extract_factor<Binary<Mul, L, R>, TargetDim, OtherDim> {
    static constexpr std::uint64_t target_mask = 1ULL << TargetDim;

    // If L depends only on TargetDim, L is our factor
    // If R depends only on TargetDim, R is our factor
    static constexpr bool l_is_target = depends_only_on_v<L, target_mask>;
    static constexpr bool r_is_target = depends_only_on_v<R, target_mask>;

    using type = std::conditional_t<l_is_target, L, R>;

    static constexpr auto extract(Binary<Mul, L, R> const& expr) {
        if constexpr (l_is_target) {
            return expr.left;
        } else {
            return expr.right;
        }
    }
};

/// Generic extract_factor for single-child nodes (Unary, UnaryFunc):
/// Return the expression if it depends only on TargetDim, otherwise Const{1}
template<typename Expr, std::size_t TargetDim, std::size_t OtherDim>
    requires requires { typename Expr::value_type; }
struct extract_single_child_factor {
    static constexpr bool is_target = depends_only_on_v<Expr, (1ULL << TargetDim)>;

    using value_type = typename Expr::value_type;
    using type = std::conditional_t<is_target, Expr, Const<value_type>>;

    static constexpr auto extract(Expr const& expr) {
        if constexpr (is_target) {
            return expr;
        } else {
            return Const<value_type>{value_type(1)};
        }
    }
};

template<typename Op, typename E, std::size_t TargetDim, std::size_t OtherDim>
struct extract_factor<Unary<Op, E>, TargetDim, OtherDim>
    : extract_single_child_factor<Unary<Op, E>, TargetDim, OtherDim> {};

template<typename Tag, typename E, std::size_t TargetDim, std::size_t OtherDim>
struct extract_factor<UnaryFunc<Tag, E>, TargetDim, OtherDim>
    : extract_single_child_factor<UnaryFunc<Tag, E>, TargetDim, OtherDim> {};

} // namespace detail

/// Separate a multiplicative expression f(x,y) = g(x) * h(y)
/// Returns a pair (g, h) where g depends on D1 and h depends on D2
///
/// Precondition: is_separable_v<E, D1, D2> must be true
template<std::size_t D1, std::size_t D2, typename E>
constexpr auto separate(E const& expr) {
    static_assert(is_separable_v<E, D1, D2>,
        "Expression is not separable with respect to the given dimensions");

    auto g = detail::extract_factor<E, D1, D2>::extract(expr);
    auto h = detail::extract_factor<E, D2, D1>::extract(expr);

    return std::make_pair(g, h);
}

} // namespace limes::expr
