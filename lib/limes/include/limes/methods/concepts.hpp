#pragma once

#include <concepts>
#include <functional>
#include "../algorithms/core/result.hpp"

namespace limes::methods {

/// A 1D integration method: callable with (function, lower, upper) returning integration_result
template<typename M, typename T>
concept IntegrationMethod = requires(M const& m, std::function<T(T)> f, T a, T b) {
    { m(f, a, b) } -> std::convertible_to<algorithms::integration_result<T>>;
};

/// Type trait to tag types as integration methods (specialized per method struct)
template<typename M>
struct is_integration_method : std::false_type {};

template<typename M>
inline constexpr bool is_integration_method_v = is_integration_method<M>::value;

} // namespace limes::methods
