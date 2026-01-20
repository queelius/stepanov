---
title: "Exact Rational Arithmetic"
date: 2020-02-18
draft: false
tags:
  - C++
  - generic-programming
  - number-theory
  - fractions
  - exact-arithmetic
categories:
  - Computer Science
  - Mathematics
series: ["stepanov"]
series_weight: 4
math: true
description: "Rational numbers give us exact arithmetic where floating-point fails. The implementation reveals deep connections to GCD, the Stern-Brocot tree, and the algebraic structure of fields."
---

*Fractions that never lose precision*

## The Problem with Floating-Point

```cpp
double x = 0.1 + 0.2;
std::cout << (x == 0.3);  // Prints 0 (false!)
```

Floating-point represents numbers as m x 2^e. The number 0.1 has no exact finite representation in binary, just as 1/3 has no exact finite representation in decimal.

Rational arithmetic solves this: 1/3 stays exactly 1/3.

## The Implementation

A rational is a pair (numerator, denominator) kept in lowest terms:

```cpp
template<std::integral T>
class rat {
    T num_;  // numerator (carries sign)
    T den_;  // denominator (always positive)

    void reduce() {
        T g = std::gcd(abs(num_), den_);
        num_ /= g;
        den_ /= g;
    }
};
```

Key invariants:
1. Denominator is always positive (sign lives in numerator)
2. GCD(|num|, den) = 1 (always reduced)
3. Zero is uniquely 0/1

## Arithmetic

Addition requires a common denominator:

$$\frac{a}{b} + \frac{c}{d} = \frac{ad + bc}{bd}$$

Then reduce. The implementation:

```cpp
rat operator+(rat const& rhs) const {
    return rat(num_ * rhs.den_ + rhs.num_ * den_,
               den_ * rhs.den_);
}
```

The constructor automatically reduces.

Multiplication is simpler:

$$\frac{a}{b} \times \frac{c}{d} = \frac{ac}{bd}$$

Division multiplies by the reciprocal:

$$\frac{a/b}{c/d} = \frac{ad}{bc}$$

## Exact Comparison

No floating-point fuzziness. Two reduced rationals are equal iff their numerators and denominators match:

```cpp
bool operator==(rat const& rhs) const {
    return num_ == rhs.num_ && den_ == rhs.den_;
}
```

For ordering, cross-multiply (valid when denominators are positive):

$$\frac{a}{b} < \frac{c}{d} \iff ad < cb$$

## The Mediant

The mediant of a/b and c/d is (a+c)/(b+d). Unlike the average, it's *not* halfway between, but it has remarkable properties:

```cpp
rat mediant(rat const& a, rat const& b) {
    return rat(a.numerator() + b.numerator(),
               a.denominator() + b.denominator());
}
```

Properties:
- If a/b < c/d, then a/b < mediant < c/d
- The mediant is always in lowest terms if a/b and c/d are neighbors in the Stern-Brocot tree
- Mediants generate all positive rationals exactly once

## Stern-Brocot Tree

Start with 0/1 and 1/0 (infinity). Repeatedly take mediants:

```
Level 0:     0/1                     1/0
Level 1:     0/1       1/1           1/0
Level 2:     0/1   1/2   1/1   2/1   1/0
Level 3:  0/1 1/3 1/2 2/3 1/1 3/2 2/1 3/1 1/0
```

Every positive rational appears exactly once. This connects to:
- Continued fractions (path from root encodes CF)
- Best rational approximations
- Farey sequences

## GCD: The Unifying Thread

Reducing fractions uses GCD. The GCD algorithm is:

```cpp
T gcd(T a, T b) {
    while (b != 0) {
        a = a % b;
        std::swap(a, b);
    }
    return a;
}
```

This is Euclid's algorithm from ~300 BCE---the same algorithm that appears for any Euclidean domain.

Rational numbers form a **field**: every non-zero element has a multiplicative inverse (the reciprocal). The denominator requirement (non-zero, reduced) comes from the algebraic structure.

## Overflow Considerations

For large computations, numerator and denominator can grow quickly. Options:
1. Use `int64_t` and accept limits
2. Use arbitrary-precision integers
3. Use interval arithmetic to track bounds

This implementation uses the first approach for simplicity.

## Further Reading

- Graham, Knuth, Patashnik, *Concrete Mathematics*, Chapter 4
- Stern, "Uber eine zahlentheoretische Funktion" (1858)
