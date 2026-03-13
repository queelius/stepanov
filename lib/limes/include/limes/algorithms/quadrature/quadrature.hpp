#pragma once

#include <array>
#include <cmath>
#include <numbers>
#include "../concepts/concepts.hpp"

namespace limes::algorithms::quadrature {

// Base quadrature rule: stores N nodes and weights on [-1, 1]
template<concepts::Field T, std::size_t N>
struct quadrature_rule {
    using value_type = T;
    using size_type = std::size_t;

    static constexpr size_type size() noexcept { return N; }

    std::array<T, N> weights;
    std::array<T, N> abscissas;

    constexpr T weight(size_type i) const noexcept { return weights[i]; }
    constexpr T abscissa(size_type i) const noexcept { return abscissas[i]; }
};

// Gauss-Legendre quadrature (primary template requires specialization)
template<concepts::Field T, std::size_t N>
struct gauss_legendre : quadrature_rule<T, N> {
    static_assert(N == 2 || N == 3 || N == 5 || N == 7 || N == 15,
        "gauss_legendre is only specialized for N = 2, 3, 5, 7, 15");
};

// Specialized implementations for common orders
template<concepts::Field T>
struct gauss_legendre<T, 2> : quadrature_rule<T, 2> {
    constexpr gauss_legendre() noexcept {
        constexpr T sqrt3 = T(0.5773502691896257645091487805019574556);
        this->abscissas = {-sqrt3, sqrt3};
        this->weights = {T(1), T(1)};
    }
};

template<concepts::Field T>
struct gauss_legendre<T, 3> : quadrature_rule<T, 3> {
    constexpr gauss_legendre() noexcept {
        constexpr T sqrt35 = T(0.7745966692414833770358530799564799221);
        this->abscissas = {-sqrt35, T(0), sqrt35};
        this->weights = {T(5)/T(9), T(8)/T(9), T(5)/T(9)};
    }
};

template<concepts::Field T>
struct gauss_legendre<T, 5> : quadrature_rule<T, 5> {
    constexpr gauss_legendre() noexcept {
        constexpr T x1 = T(0.9061798459386639927976268782993929651);
        constexpr T x2 = T(0.5384693101056830910363144207002088049);
        constexpr T w1 = T(0.2369268850561890875142640407199173626);
        constexpr T w2 = T(0.4786286704993664680412915148356381929);
        constexpr T w3 = T(0.5688888888888888888888888888888888889);

        this->abscissas = {-x1, -x2, T(0), x2, x1};
        this->weights = {w1, w2, w3, w2, w1};
    }
};

template<concepts::Field T>
struct gauss_legendre<T, 7> : quadrature_rule<T, 7> {
    constexpr gauss_legendre() noexcept {
        constexpr T x1 = T(0.9491079123427585245261896840478512624);
        constexpr T x2 = T(0.7415311855993944398638647732807884070);
        constexpr T x3 = T(0.4058451513773971669066064120769614633);
        constexpr T w1 = T(0.1294849661688696932706114326790820183);
        constexpr T w2 = T(0.2797053914892766679014677714237795824);
        constexpr T w3 = T(0.3818300505051189449503697754889751338);
        constexpr T w4 = T(0.4179591836734693877551020408163265306);

        this->abscissas = {-x1, -x2, -x3, T(0), x3, x2, x1};
        this->weights = {w1, w2, w3, w4, w3, w2, w1};
    }
};

template<concepts::Field T>
struct gauss_legendre<T, 15> : quadrature_rule<T, 15> {
    constexpr gauss_legendre() noexcept {
        this->abscissas = {
            T(-0.9879925180204854284895657185866125811),
            T(-0.9372733924007059043077589477102094712),
            T(-0.8482065834104272076884935534726048004),
            T(-0.7244177313601700474161860546139379979),
            T(-0.5709721726085388475372267372539106413),
            T(-0.3941513470775633698972073709810454684),
            T(-0.2011940939974345223006283033945962078),
            T(0.0),
            T(0.2011940939974345223006283033945962078),
            T(0.3941513470775633698972073709810454684),
            T(0.5709721726085388475372267372539106413),
            T(0.7244177313601700474161860546139379979),
            T(0.8482065834104272076884935534726048004),
            T(0.9372733924007059043077589477102094712),
            T(0.9879925180204854284895657185866125811)
        };
        this->weights = {
            T(0.0307532419961172683546283935772044177),
            T(0.0703660474881081247092674164506673384),
            T(0.1071592204671719350118695466858693034),
            T(0.1395706779261543144478047945110283225),
            T(0.1662692058169939335532008604812088111),
            T(0.1861610000155622110268005618664228245),
            T(0.1984314853271115764561183264438393248),
            T(0.2025782419255612728806201999675193148),
            T(0.1984314853271115764561183264438393248),
            T(0.1861610000155622110268005618664228245),
            T(0.1662692058169939335532008604812088111),
            T(0.1395706779261543144478047945110283225),
            T(0.1071592204671719350118695466858693034),
            T(0.0703660474881081247092674164506673384),
            T(0.0307532419961172683546283935772044177)
        };
    }
};

// Gauss-Kronrod 7-15 rule (extends Gauss-Legendre with embedded error estimation)
template<concepts::Field T>
struct gauss_kronrod_15 : quadrature_rule<T, 15> {
    constexpr gauss_kronrod_15() noexcept {
        this->abscissas = {
            T(-0.9914553711208126), T(-0.9491079123427585), T(-0.8648644233597691),
            T(-0.7415311855993944), T(-0.5860872354676911), T(-0.4058451513773972),
            T(-0.2077849550078985), T(0),
            T(0.2077849550078985), T(0.4058451513773972), T(0.5860872354676911),
            T(0.7415311855993944), T(0.8648644233597691), T(0.9491079123427585),
            T(0.9914553711208126)
        };

        this->weights = {
            T(0.0229353220105292), T(0.0630920926299785), T(0.1047900103222502),
            T(0.1406532597155259), T(0.1690047266392679), T(0.1903505780647854),
            T(0.2044329400752989), T(0.2094821410847278),
            T(0.2044329400752989), T(0.1903505780647854), T(0.1690047266392679),
            T(0.1406532597155259), T(0.1047900103222502), T(0.0630920926299785),
            T(0.0229353220105292)
        };
    }

    // Embedded 7-point Gauss rule for error estimation
    static constexpr std::size_t gauss_size = 7;
    std::array<T, gauss_size> gauss_weights = {
        T(0.1294849661688697), T(0.2797053914892767), T(0.3818300505051189),
        T(0.4179591836734694),
        T(0.3818300505051189), T(0.2797053914892767), T(0.1294849661688697)
    };
    std::array<std::size_t, gauss_size> gauss_indices = {1, 3, 5, 7, 9, 11, 13};
};

// Clenshaw-Curtis quadrature
template<concepts::Field T, std::size_t N>
struct clenshaw_curtis : quadrature_rule<T, N> {
    constexpr clenshaw_curtis() noexcept {
        compute_nodes();
    }

private:
    constexpr void compute_nodes() noexcept {
        constexpr T pi = std::numbers::pi_v<T>;
        constexpr std::size_t n = N - 1;

        for (std::size_t i = 0; i < N; ++i) {
            T theta = pi * T(i) / T(n);
            this->abscissas[i] = -std::cos(theta);

            T w = T(0);
            std::size_t max_j = n / 2;
            for (std::size_t j = 0; j <= max_j; ++j) {
                T cos_term = std::cos(T(2) * j * theta);
                T b_j = (j == 0 || j == max_j) ? T(1) : T(2);

                if (j == 0) {
                    w += b_j * cos_term;
                } else {
                    w -= b_j * cos_term / (T(4) * j * j - T(1));
                }
            }

            w *= T(2) / T(n);
            if (i == 0 || i == n) {
                w /= T(2);
            }

            this->weights[i] = w;
        }
    }
};

// Simpson's rule (3-point, exact for cubics)
template<concepts::Field T>
struct simpson_rule : quadrature_rule<T, 3> {
    constexpr simpson_rule() noexcept {
        this->abscissas = {T(-1), T(0), T(1)};
        this->weights = {T(1)/T(3), T(4)/T(3), T(1)/T(3)};
    }
};

// Trapezoidal rule (2-point, exact for linear)
template<concepts::Field T>
struct trapezoidal_rule : quadrature_rule<T, 2> {
    constexpr trapezoidal_rule() noexcept {
        this->abscissas = {T(-1), T(1)};
        this->weights = {T(1), T(1)};
    }
};

// Midpoint rule (1-point, exact for linear)
template<concepts::Field T>
struct midpoint_rule : quadrature_rule<T, 1> {
    constexpr midpoint_rule() noexcept {
        this->abscissas = {T(0)};
        this->weights = {T(2)};
    }
};

// Tanh-sinh (double exponential) quadrature nodes
template<concepts::Field T>
class tanh_sinh_nodes {
public:
    using value_type = T;

    static constexpr std::size_t max_level = 10;

    constexpr T abscissa(std::size_t level, std::size_t index) const noexcept {
        T t = node_parameter(level, index);
        return transform_abscissa(t);
    }

    constexpr T weight(std::size_t level, std::size_t index) const noexcept {
        T h = T(1) / std::pow(T(2), level);
        T t = node_parameter(level, index);
        return h * transform_weight(t);
    }

private:
    static constexpr T node_parameter(std::size_t level, std::size_t index) noexcept {
        T h = T(1) / std::pow(T(2), level);
        return (index == 0) ? T(0) : h * T(index);
    }

    static constexpr T transform_abscissa(T t) noexcept {
        T expmt = std::exp(-t);
        T u = std::exp(t - expmt);
        return (u - T(1)/u) / (u + T(1)/u);
    }

    static constexpr T transform_weight(T t) noexcept {
        T expmt = std::exp(-t);
        T u = std::exp(t - expmt);
        T cosh_val = (u + T(1)/u) / T(2);
        return (T(1) + expmt) / (cosh_val * cosh_val);
    }
};

} // namespace limes::algorithms::quadrature
