#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <vector>
#include <mutatio.hpp>

using namespace mutatio;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ── Logarithmic ──────────────────────────────────────

TEST_CASE("lg: multiplication becomes addition", "[mutatio][lg]") {
    lgd a(2.0), b(3.0);
    auto product = a * b;
    REQUIRE_THAT(product.value(), WithinAbs(6.0, 1e-10));
}

TEST_CASE("lg: division becomes subtraction", "[mutatio][lg]") {
    lgd a(6.0), b(2.0);
    REQUIRE_THAT((a / b).value(), WithinAbs(3.0, 1e-10));
}

TEST_CASE("lg: power becomes multiplication", "[mutatio][lg]") {
    lgd a(2.0);
    REQUIRE_THAT(a.pow(10).value(), WithinAbs(1024.0, 1e-6));
}

TEST_CASE("lg: extreme range without overflow", "[mutatio][lg]") {
    lgd tiny(1e-100), huge(1e100);
    REQUIRE_THAT((tiny * huge).value(), WithinAbs(1.0, 1e-10));
}

TEST_CASE("lg: chain of small multiplications", "[mutatio][lg]") {
    lgd result(1.0);
    for (int i = 0; i < 1000; ++i)
        result = result * lgd(1.001);
    // Should not overflow or underflow
    REQUIRE(std::isfinite(result.log()));
}

// ── Odds Ratio ───────────────────────────────────────

TEST_CASE("odds_ratio: probability round-trip", "[mutatio][odds]") {
    auto o = odds_ratio<double>::from_probability(0.75);
    REQUIRE_THAT(o.to_probability(), WithinAbs(0.75, 1e-10));
}

TEST_CASE("odds_ratio: 50/50 gives odds 1", "[mutatio][odds]") {
    auto o = odds_ratio<double>::from_probability(0.5);
    REQUIRE_THAT(o.value(), WithinAbs(1.0, 1e-10));
}

TEST_CASE("odds_ratio: Bayesian update via multiplication", "[mutatio][odds]") {
    auto prior = odds_ratio<double>::from_probability(0.1);
    auto lr = odds_ratio<double>(2.0);
    auto posterior = prior * lr;
    double expected = 0.2 / (0.2 + 0.9);
    REQUIRE_THAT(posterior.to_probability(), WithinAbs(expected, 1e-10));
}

TEST_CASE("odds_ratio: inverse update recovers prior", "[mutatio][odds]") {
    auto prior = odds_ratio<double>::from_probability(0.1);
    auto lr = odds_ratio<double>(2.0);
    auto posterior = prior * lr;
    auto recovered = posterior / lr;
    REQUIRE_THAT(recovered.to_probability(), WithinAbs(0.1, 1e-10));
}

TEST_CASE("log_odds: zero log-odds at 50/50", "[mutatio][odds]") {
    auto lo = log_odds<double>::from_probability(0.5);
    REQUIRE_THAT(lo.value(), WithinAbs(0.0, 1e-10));
}

// ── Stern-Brocot ─────────────────────────────────────

TEST_CASE("stern_brocot: exact addition", "[mutatio][rational]") {
    stern_brocot<int> a(1, 2), b(1, 3);
    auto sum = a + b;
    REQUIRE(sum.numerator() == 5);
    REQUIRE(sum.denominator() == 6);
}

TEST_CASE("stern_brocot: exact multiplication", "[mutatio][rational]") {
    stern_brocot<int> a(2, 3), b(3, 4);
    auto prod = a * b;
    REQUIRE(prod.numerator() == 1);
    REQUIRE(prod.denominator() == 2);
}

TEST_CASE("stern_brocot: pi approximation 355/113", "[mutatio][rational]") {
    auto pi = stern_brocot<int>::approximate(3.141592653589793, 1000);
    REQUIRE(pi.numerator() == 355);
    REQUIRE(pi.denominator() == 113);
}

TEST_CASE("stern_brocot: negative fractions", "[mutatio][rational]") {
    stern_brocot<int> neg(-1, 2), pos(1, 3);
    auto sum = neg + pos;
    REQUIRE(sum.numerator() == -1);
    REQUIRE(sum.denominator() == 6);
}

// ── Tropical ─────────────────────────────────────────

TEST_CASE("tropical_min: addition is min", "[mutatio][tropical]") {
    tropical_min<double> a(3.0), b(5.0);
    REQUIRE_THAT((a + b).value(), WithinAbs(3.0, 1e-10));
}

TEST_CASE("tropical_min: multiplication is plus", "[mutatio][tropical]") {
    tropical_min<double> a(3.0), b(5.0);
    REQUIRE_THAT((a * b).value(), WithinAbs(8.0, 1e-10));
}

TEST_CASE("tropical_min: identity elements", "[mutatio][tropical]") {
    tropical_min<double> a(7.0);
    REQUIRE_THAT((a + tropical_min<double>::zero()).value(), WithinAbs(7.0, 1e-10));
    REQUIRE_THAT((a * tropical_min<double>::one()).value(), WithinAbs(7.0, 1e-10));
}

TEST_CASE("tropical_min: infinity absorbs multiplication", "[mutatio][tropical]") {
    auto inf = tropical_min<double>::zero();
    tropical_min<double> a(5.0);
    REQUIRE((inf * a).is_infinite());
}

TEST_CASE("tropical_max: addition is max", "[mutatio][tropical]") {
    tropical_max<double> a(3.0), b(5.0);
    REQUIRE_THAT((a + b).value(), WithinAbs(5.0, 1e-10));
}

// ── Modular ──────────────────────────────────────────

TEST_CASE("modular: wrapping", "[mutatio][modular]") {
    modular<int, 7> a(10);
    REQUIRE(a.value() == 3);
}

TEST_CASE("modular: arithmetic", "[mutatio][modular]") {
    modular<int, 7> a(3), b(5);
    REQUIRE((a + b).value() == 1);
    REQUIRE((a * b).value() == 1);
    REQUIRE((b - a).value() == 2);
}

TEST_CASE("modular: inverse mod prime", "[mutatio][modular]") {
    modular<int, 31> x(5);
    auto inv = x.inverse();
    REQUIRE((x * inv).value() == 1);
}

TEST_CASE("modular: Fermat's little theorem", "[mutatio][modular]") {
    modular<int, 31> a(3);
    REQUIRE(a.pow(30).value() == 1);
}

TEST_CASE("modular: negative wrapping", "[mutatio][modular]") {
    modular<int, 7> neg(-3);
    REQUIRE(neg.value() == 4);
}

// ── Quaternion ───────────────────────────────────────

TEST_CASE("quaternion: conjugate negates imaginary", "[mutatio][quaternion]") {
    quaternion<double> q(1, 2, 3, 4);
    auto c = q.conjugate();
    REQUIRE_THAT(c.w(), WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(c.x(), WithinAbs(-2.0, 1e-10));
    REQUIRE_THAT(c.y(), WithinAbs(-3.0, 1e-10));
    REQUIRE_THAT(c.z(), WithinAbs(-4.0, 1e-10));
}

TEST_CASE("quaternion: normalized has unit norm", "[mutatio][quaternion]") {
    quaternion<double> q(1, 2, 3, 4);
    REQUIRE_THAT(q.normalized().norm(), WithinAbs(1.0, 1e-10));
}

TEST_CASE("quaternion: inverse gives identity", "[mutatio][quaternion]") {
    quaternion<double> q(1, 2, 3, 4);
    auto n = q.normalized();
    auto id = n * n.inverse();
    REQUIRE_THAT(id.w(), WithinAbs(1.0, 1e-9));
    REQUIRE_THAT(id.x(), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(id.y(), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(id.z(), WithinAbs(0.0, 1e-9));
}

TEST_CASE("quaternion: 90-degree rotation around z", "[mutatio][quaternion]") {
    auto rot = quaternion<double>::from_axis_angle(0, 0, 1, M_PI / 2);
    auto result = rot.rotate(1.0, 0.0, 0.0);
    REQUIRE_THAT(result[0], WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(result[1], WithinAbs(1.0, 1e-9));
    REQUIRE_THAT(result[2], WithinAbs(0.0, 1e-9));
}

TEST_CASE("quaternion: SLERP midpoint", "[mutatio][quaternion]") {
    auto q0 = quaternion<double>::from_axis_angle(0, 0, 1, 0);
    auto q1 = quaternion<double>::from_axis_angle(0, 0, 1, M_PI / 2);
    auto mid = q0.slerp(q1, 0.5);
    double angle = 2 * std::acos(mid.w());
    REQUIRE_THAT(angle, WithinAbs(M_PI / 4, 1e-6));
}

TEST_CASE("quaternion: 1000 rotations stay unit", "[mutatio][quaternion]") {
    auto q = quaternion<double>::from_axis_angle(1, 1, 1, 0.001).normalized();
    auto acc = q;
    for (int i = 0; i < 1000; ++i) {
        acc = (acc * q).normalized();
    }
    REQUIRE_THAT(acc.norm(), WithinAbs(1.0, 1e-6));
}
