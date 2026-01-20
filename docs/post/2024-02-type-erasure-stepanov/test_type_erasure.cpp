#include <gtest/gtest.h>
#include "any_regular.hpp"
#include <string>
#include <vector>

using namespace type_erasure;

// =============================================================================
// Construction
// =============================================================================

TEST(AnyRegularTest, DefaultConstruction) {
    any_regular a;
    EXPECT_FALSE(a.has_value());
    EXPECT_FALSE(static_cast<bool>(a));
}

TEST(AnyRegularTest, IntConstruction) {
    any_regular a(42);
    EXPECT_TRUE(a.has_value());
    EXPECT_TRUE(a.holds<int>());
    EXPECT_EQ(*a.get<int>(), 42);
}

TEST(AnyRegularTest, StringConstruction) {
    any_regular a(std::string("hello"));
    EXPECT_TRUE(a.has_value());
    EXPECT_TRUE(a.holds<std::string>());
    EXPECT_EQ(*a.get<std::string>(), "hello");
}

TEST(AnyRegularTest, VectorConstruction) {
    any_regular a(std::vector<int>{1, 2, 3});
    EXPECT_TRUE(a.has_value());
    EXPECT_TRUE(a.holds<std::vector<int>>());
    EXPECT_EQ(a.get<std::vector<int>>()->size(), 3);
}

// =============================================================================
// Copy semantics
// =============================================================================

TEST(AnyRegularTest, CopyConstruction) {
    any_regular a(42);
    any_regular b(a);

    EXPECT_EQ(*a.get<int>(), 42);
    EXPECT_EQ(*b.get<int>(), 42);
}

TEST(AnyRegularTest, CopyIndependence) {
    any_regular a(std::string("hello"));
    any_regular b(a);

    // Modifying original shouldn't affect copy
    a = std::string("world");

    EXPECT_EQ(*a.get<std::string>(), "world");
    EXPECT_EQ(*b.get<std::string>(), "hello");
}

TEST(AnyRegularTest, CopyAssignment) {
    any_regular a(42);
    any_regular b(std::string("hello"));

    b = a;

    EXPECT_TRUE(b.holds<int>());
    EXPECT_EQ(*b.get<int>(), 42);
}

// =============================================================================
// Type checking
// =============================================================================

TEST(AnyRegularTest, HoldsCorrectType) {
    any_regular a(42);

    EXPECT_TRUE(a.holds<int>());
    EXPECT_FALSE(a.holds<double>());
    EXPECT_FALSE(a.holds<std::string>());
}

TEST(AnyRegularTest, TypeInfo) {
    any_regular a(42);
    EXPECT_EQ(a.type(), typeid(int));

    any_regular b(std::string("hello"));
    EXPECT_EQ(b.type(), typeid(std::string));

    any_regular c;
    EXPECT_EQ(c.type(), typeid(void));
}

// =============================================================================
// Access
// =============================================================================

TEST(AnyRegularTest, GetReturnsNullForWrongType) {
    any_regular a(42);
    EXPECT_EQ(a.get<double>(), nullptr);
    EXPECT_EQ(a.get<std::string>(), nullptr);
}

TEST(AnyRegularTest, GetOrDefault) {
    any_regular a(42);
    EXPECT_EQ(a.get_or<int>(0), 42);
    EXPECT_EQ(a.get_or<double>(3.14), 3.14);  // Wrong type, returns default

    any_regular empty;
    EXPECT_EQ(empty.get_or<int>(99), 99);
}

// =============================================================================
// Equality
// =============================================================================

TEST(AnyRegularTest, EmptyEquality) {
    any_regular a;
    any_regular b;
    EXPECT_EQ(a, b);
}

TEST(AnyRegularTest, SameTypeEquality) {
    any_regular a(42);
    any_regular b(42);
    any_regular c(43);

    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

TEST(AnyRegularTest, DifferentTypeInequality) {
    any_regular a(42);
    any_regular b(42.0);  // double, not int

    EXPECT_NE(a, b);
}

TEST(AnyRegularTest, EmptyVsNonEmpty) {
    any_regular a;
    any_regular b(42);

    EXPECT_NE(a, b);
}

// =============================================================================
// Assignment
// =============================================================================

TEST(AnyRegularTest, DirectValueAssignment) {
    any_regular a;
    a = 42;

    EXPECT_TRUE(a.holds<int>());
    EXPECT_EQ(*a.get<int>(), 42);
}

TEST(AnyRegularTest, TypeChange) {
    any_regular a(42);
    EXPECT_TRUE(a.holds<int>());

    a = std::string("hello");
    EXPECT_TRUE(a.holds<std::string>());
    EXPECT_EQ(*a.get<std::string>(), "hello");
}

// =============================================================================
// Heterogeneous container
// =============================================================================

TEST(AnyRegularTest, HeterogeneousVector) {
    std::vector<any_regular> v;
    v.push_back(42);
    v.push_back(std::string("hello"));
    v.push_back(3.14);

    EXPECT_TRUE(v[0].holds<int>());
    EXPECT_TRUE(v[1].holds<std::string>());
    EXPECT_TRUE(v[2].holds<double>());

    EXPECT_EQ(*v[0].get<int>(), 42);
    EXPECT_EQ(*v[1].get<std::string>(), "hello");
    EXPECT_DOUBLE_EQ(*v[2].get<double>(), 3.14);
}

// =============================================================================
// any_vector Tests - Type-Erased Vector Spaces
// =============================================================================

#include "any_vector.hpp"

using namespace vector_space;

// -----------------------------------------------------------------------------
// vec2 Tests
// -----------------------------------------------------------------------------

TEST(Vec2Test, Construction) {
    vec2 v{3, 4};
    EXPECT_EQ(v.x, 3);
    EXPECT_EQ(v.y, 4);
}

TEST(Vec2Test, Addition) {
    vec2 a{1, 2};
    vec2 b{3, 4};
    vec2 c = a + b;
    EXPECT_EQ(c.x, 4);
    EXPECT_EQ(c.y, 6);
}

TEST(Vec2Test, ScalarMultiplication) {
    vec2 v{2, 3};
    vec2 scaled = 2.0 * v;
    EXPECT_EQ(scaled.x, 4);
    EXPECT_EQ(scaled.y, 6);
}

TEST(Vec2Test, Negation) {
    vec2 v{1, -2};
    vec2 neg = -v;
    EXPECT_EQ(neg.x, -1);
    EXPECT_EQ(neg.y, 2);
}

TEST(Vec2Test, Zero) {
    vec2 v{5, 5};
    vec2 z = zero(v);
    EXPECT_EQ(z.x, 0);
    EXPECT_EQ(z.y, 0);
}

// -----------------------------------------------------------------------------
// vec3 Tests
// -----------------------------------------------------------------------------

TEST(Vec3Test, CrossProduct) {
    vec3 i{1, 0, 0};
    vec3 j{0, 1, 0};
    vec3 k = cross(i, j);
    EXPECT_EQ(k.x, 0);
    EXPECT_EQ(k.y, 0);
    EXPECT_EQ(k.z, 1);
}

// -----------------------------------------------------------------------------
// polynomial Tests
// -----------------------------------------------------------------------------

TEST(PolynomialTest, Construction) {
    polynomial p{1, 2, 3};  // 1 + 2x + 3x^2
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 2);
    EXPECT_EQ(p[2], 3);
    EXPECT_EQ(p.degree(), 2);
}

TEST(PolynomialTest, Evaluation) {
    polynomial p{1, 2, 1};  // 1 + 2x + x^2 = (1+x)^2
    EXPECT_DOUBLE_EQ(p(0), 1);
    EXPECT_DOUBLE_EQ(p(1), 4);
    EXPECT_DOUBLE_EQ(p(2), 9);
}

TEST(PolynomialTest, Addition) {
    polynomial p{1, 2};      // 1 + 2x
    polynomial q{3, 0, 1};   // 3 + x^2
    polynomial r = p + q;    // 4 + 2x + x^2
    EXPECT_EQ(r[0], 4);
    EXPECT_EQ(r[1], 2);
    EXPECT_EQ(r[2], 1);
}

TEST(PolynomialTest, ScalarMultiplication) {
    polynomial p{1, 2, 3};
    polynomial q = 2.0 * p;
    EXPECT_EQ(q[0], 2);
    EXPECT_EQ(q[1], 4);
    EXPECT_EQ(q[2], 6);
}

// -----------------------------------------------------------------------------
// complex_vec Tests
// -----------------------------------------------------------------------------

TEST(ComplexVecTest, Addition) {
    complex_vec a{1, 2};
    complex_vec b{3, 4};
    complex_vec c = a + b;
    EXPECT_EQ(c.re, 4);
    EXPECT_EQ(c.im, 6);
}

TEST(ComplexVecTest, Multiplication) {
    complex_vec i{0, 1};
    complex_vec i2 = i * i;
    EXPECT_EQ(i2.re, -1);
    EXPECT_EQ(i2.im, 0);
}

// -----------------------------------------------------------------------------
// any_vector Tests
// -----------------------------------------------------------------------------

TEST(AnyVectorTest, ConstructFromVec2) {
    any_vector<double> v(vec2{3, 4});
    EXPECT_TRUE(static_cast<bool>(v));
}

TEST(AnyVectorTest, Addition) {
    any_vector<double> a(vec2{1, 2});
    any_vector<double> b(vec2{3, 4});
    any_vector<double> c = a + b;
    EXPECT_TRUE(c == any_vector<double>(vec2{4, 6}));
}

TEST(AnyVectorTest, ScalarMultiplication) {
    any_vector<double> v(vec2{2, 3});
    any_vector<double> scaled = 2.0 * v;
    EXPECT_TRUE(scaled == any_vector<double>(vec2{4, 6}));
}

TEST(AnyVectorTest, Negation) {
    any_vector<double> v(vec2{1, -2});
    any_vector<double> neg = -v;
    EXPECT_TRUE(neg == any_vector<double>(vec2{-1, 2}));
}

TEST(AnyVectorTest, Zero) {
    any_vector<double> v(vec2{5, 5});
    any_vector<double> z = zero(v);
    EXPECT_TRUE(z == any_vector<double>(vec2{0, 0}));
}

TEST(AnyVectorTest, PolynomialVector) {
    // Polynomials are vector spaces too
    any_vector<double> p1(polynomial{1, 2});     // 1 + 2x
    any_vector<double> p2(polynomial{3, 0, 1});  // 3 + x^2
    any_vector<double> sum = p1 + p2;            // 4 + 2x + x^2
    EXPECT_TRUE(sum == any_vector<double>(polynomial{4, 2, 1}));
}

TEST(AnyVectorTest, LinearCombination) {
    // 2*(1,0) + 3*(0,1) = (2,3)
    any_vector<double> e1(vec2{1, 0});
    any_vector<double> e2(vec2{0, 1});
    any_vector<double> result = 2.0 * e1 + 3.0 * e2;
    EXPECT_TRUE(result == any_vector<double>(vec2{2, 3}));
}

TEST(AnyVectorTest, Subtraction) {
    any_vector<double> a(vec2{5, 7});
    any_vector<double> b(vec2{2, 3});
    any_vector<double> diff = a - b;
    EXPECT_TRUE(diff == any_vector<double>(vec2{3, 4}));
}

TEST(AnyVectorTest, VectorSpaceAxioms) {
    // Verify vector space axioms hold for any_vector

    any_vector<double> v(vec2{1, 2});
    any_vector<double> w(vec2{3, 4});
    any_vector<double> u(vec2{5, 6});
    any_vector<double> z = zero(v);

    // Additive identity: v + 0 = v
    EXPECT_TRUE(v + z == v);

    // Additive inverse: v + (-v) = 0
    EXPECT_TRUE(v + (-v) == z);

    // Commutativity: v + w = w + v
    EXPECT_TRUE(v + w == w + v);

    // Scalar identity: 1 * v = v
    EXPECT_TRUE(1.0 * v == v);

    // Scalar zero: 0 * v = 0
    EXPECT_TRUE(0.0 * v == z);
}
